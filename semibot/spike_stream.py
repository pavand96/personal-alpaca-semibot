"""
Real-time spike detector using Alpaca WebSocket trade + news streams.

Three detection layers, each faster than the last:

  1. Startup snapshot scan
     On launch, immediately snapshots all watchlist symbols.  Any symbol
     already gapping >= threshold is queued even before the first live trade.
     Catches stocks that moved while the process was starting up.

  2. News WebSocket (NewsDataStream, ~seconds after headline)
     Subscribes to all news for watchlist symbols.  When a catalyst headline
     arrives (earnings, upgrade, guidance, acquisition, …) a snapshot is
     fetched for that symbol immediately.
     - Gap >= earnings_bypass_gap_pct AND earnings/catalyst keywords
       → order fires right away (no sector confirmation required).
     - Smaller gaps → queued; fires when sector_confirm threshold is met.

  3. Trade WebSocket (StockDataStream, ~200ms after first trade)
     Every live trade for every watchlist symbol is checked.  Orders fire
     once sector_confirm distinct symbols have all hit the gap threshold.

All three share the same _pending dict, _already_ordered set, and _order_lock,
so there are no duplicate orders regardless of which layer triggers first.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import NewsDataStream, StockDataStream
from alpaca.data.requests import StockSnapshotRequest

from semibot import ledger as _ledger
from semibot.bot import SemiMomentumBot, floor_order_qty, limit_price_for_side
from semibot.events import append_event
from semibot.spike_tracker import record_spike_entry

log = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
_REGULAR_CLOSE_HOUR = 16  # 4:00 PM ET

# Broader net: triggers a snapshot check when found in any news headline/summary
_CATALYST_KEYWORDS = frozenset({
    # earnings
    "earnings", "eps", "quarterly", "quarter",
    "q1", "q2", "q3", "q4", "beats", "beat", "misses", "missed",
    "revenue", "results", "guidance", "outlook", "profit",
    # analyst actions
    "upgrade", "upgraded", "raises", "raised", "price target",
    "overweight", "outperform", "buy rating",
    # corporate events
    "acquisition", "merger", "buyout", "contract", "partnership",
    # regulatory / product
    "fda", "approval", "approved", "breakthrough",
})

# Narrower set: earnings-type events that may bypass sector confirmation
_EARNINGS_KEYWORDS = frozenset({
    "earnings", "eps", "quarterly", "quarter",
    "q1", "q2", "q3", "q4", "beats", "beat", "misses", "missed",
    "revenue", "results", "profit",
})


def _best_price_from_snap(snap: Any) -> float | None:
    """Most current price from a snapshot.

    Prefers latest_trade if it occurred today (pre-market counts).
    Falls back to quote midpoint when latest_trade is from a prior session —
    quotes arrive before trades print, so we can detect a 4 AM gap even when
    no ECN trade has executed yet.
    """
    if not snap:
        return None
    today = datetime.now(MARKET_TZ).date()

    latest = getattr(snap, "latest_trade", None)
    if latest:
        ts = getattr(latest, "timestamp", None)
        try:
            trade_date = ts.astimezone(MARKET_TZ).date() if ts is not None else None
        except Exception:
            trade_date = None
        if trade_date == today:
            p = float(latest.price)
            if p > 0:
                return p

    # latest_trade is stale — use NBBO midpoint
    quote = getattr(snap, "latest_quote", None)
    if quote:
        ask = float(getattr(quote, "ask_price", 0) or 0)
        bid = float(getattr(quote, "bid_price", 0) or 0)
        if ask > 0 and bid > 0:
            return (ask + bid) / 2.0
        if ask > 0:
            return ask
        if bid > 0:
            return bid

    # Last resort: stale trade price (gap ≈ 0%, filtered downstream)
    if latest:
        p = float(latest.price)
        if p > 0:
            return p
    return None


def _spread_ok(snap: Any, max_spread_pct: float) -> tuple[bool, float]:
    """Check bid-ask spread from snapshot.

    Returns (ok, spread_pct).  Fails open (True) when quote data is unavailable —
    a missing quote just means we cannot verify, not that the market is bad.
    Fails closed when one side of the book is missing (withdrawn quote = risk signal).
    """
    if not snap:
        return True, 0.0
    quote = getattr(snap, "latest_quote", None)
    if not quote:
        return True, 0.0
    ask = float(getattr(quote, "ask_price", 0) or 0)
    bid = float(getattr(quote, "bid_price", 0) or 0)
    if ask <= 0 or bid <= 0:
        # One side missing — can't assess; block it
        return False, 0.0
    mid = (ask + bid) / 2.0
    if mid <= 0:
        return True, 0.0
    spread_pct = (ask - bid) / mid * 100.0
    return spread_pct <= max_spread_pct, spread_pct


def _liquidity_ok(snap: Any, min_dollar_vol: float, min_trades: int) -> tuple[bool, str]:
    """Check pre-market session liquidity from snapshot daily_bar.

    Fails open when no pre-market activity has accumulated yet — this is expected at
    3:45 AM before the 4 AM open and should not block the startup scan.
    Fails closed when some activity exists but is too thin to trust: a gap caused
    by 1–2 trades or a $5K print is likely noise, not real demand.
    Returns (ok, reason_if_rejected).
    """
    if not snap:
        return True, ""
    daily = getattr(snap, "daily_bar", None)
    if not daily:
        return True, ""

    trade_count = int(getattr(daily, "trade_count", 0) or 0)
    volume = float(getattr(daily, "volume", 0) or 0)

    # No pre-market activity yet — fail open (pre-market may not have opened)
    if trade_count == 0:
        return True, ""

    if trade_count < min_trades:
        return False, f"trade_count={trade_count} < {min_trades}"

    if min_dollar_vol > 0:
        price = 0.0
        lt = getattr(snap, "latest_trade", None)
        if lt:
            price = float(getattr(lt, "price", 0) or 0)
        if price <= 0:
            q = getattr(snap, "latest_quote", None)
            if q:
                ask = float(getattr(q, "ask_price", 0) or 0)
                bid = float(getattr(q, "bid_price", 0) or 0)
                if ask > 0 and bid > 0:
                    price = (ask + bid) / 2.0
        if price > 0:
            dollar_vol = volume * price
            if dollar_vol < min_dollar_vol:
                return False, f"dollar_vol=${dollar_vol:,.0f} < ${min_dollar_vol:,.0f}"

    return True, ""


def _liquidity_stats(snap: Any, price: float | None = None) -> tuple[float, int]:
    """Return approximate session dollar volume and trade count from a snapshot."""
    if not snap:
        return 0.0, 0
    daily = getattr(snap, "daily_bar", None)
    if not daily:
        return 0.0, 0

    trade_count = int(getattr(daily, "trade_count", 0) or 0)
    volume = float(getattr(daily, "volume", 0) or 0)
    mark = float(price or 0)
    if mark <= 0:
        latest = getattr(snap, "latest_trade", None)
        if latest:
            mark = float(getattr(latest, "price", 0) or 0)
    if mark <= 0:
        quote = getattr(snap, "latest_quote", None)
        if quote:
            ask = float(getattr(quote, "ask_price", 0) or 0)
            bid = float(getattr(quote, "bid_price", 0) or 0)
            if ask > 0 and bid > 0:
                mark = (ask + bid) / 2.0
    return (volume * mark if mark > 0 else 0.0), trade_count


def _fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def _parse_feed(config: dict[str, Any]) -> DataFeed:
    raw = config["alpaca"].get("data_feed", "iex").upper()
    return DataFeed[raw] if raw in DataFeed.__members__ else DataFeed.IEX


def fetch_reference_closes(
    config: dict[str, Any],
    api_key: str,
    secret_key: str,
    symbols: list[str] | None = None,
) -> dict[str, float]:
    """Return the most recent regular-session close for each watchlist symbol.

    Pre-market  (before 16:00 ET): uses previous_daily_bar.close (yesterday's 4pm)
    After-hours (after  16:00 ET): uses daily_bar.close (today's 4pm, now locked)
    """
    now_et = datetime.now(MARKET_TZ)
    use_prev = now_et.hour < _REGULAR_CLOSE_HOUR

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    feed = _parse_feed(config)
    if symbols is None:
        symbols = config["watchlist"]

    snapshots = client.get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols=symbols, feed=feed)
    )

    closes: dict[str, float] = {}
    for symbol in symbols:
        snap = snapshots.get(symbol)
        if not snap:
            continue
        day_bar = getattr(snap, "daily_bar", None)
        prev_bar = getattr(snap, "previous_daily_bar", None)
        ref_bar = prev_bar if use_prev else day_bar
        fallback = day_bar if use_prev else prev_bar
        close = None
        if ref_bar and getattr(ref_bar, "close", None):
            close = float(ref_bar.close)
        elif fallback and getattr(fallback, "close", None):
            close = float(fallback.close)
        if close and close > 0:
            closes[symbol] = close

    window = "pre-market (vs yesterday's close)" if use_prev else "after-hours (vs today's close)"
    print(f"Reference closes loaded for {len(closes)} symbols [{window}]")
    for sym, c in sorted(closes.items()):
        print(f"  {sym:6}  ref={c:.2f}")
    return closes


class SpikeStreamScanner:
    """Three-layer spike detector: startup snapshot + news WebSocket + trade WebSocket."""

    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        self._settings = config["adaptive_semis_allocator"]
        self._min_gap = float(self._settings.get("spike_min_gap_pct", 5.0))
        self._max_gap = float(self._settings.get("spike_max_gap_pct", 20.0))
        self._notional = float(self._settings.get("spike_notional_per_trade", 1000.0))
        self._sector_confirm = int(self._settings.get("spike_sector_confirm", 1))
        self._tracker_path = str(self._settings.get("spike_tracker_path", "logs/spike_tracker.json"))
        self._earnings_multiplier = float(self._settings.get("earnings_notional_multiplier", 1.0))
        self._earnings_bypass_gap = float(self._settings.get("earnings_bypass_gap_pct", 8.0))
        self._dry_run = bool(config["risk"].get("dry_run", True))
        self._offset_bps = float(config.get("orders", {}).get("premarket_limit_offset_bps", 25.0))

        self._sell_gap_threshold = float(self._settings.get("spike_sell_gap_pct", 5.0))
        self._max_spread_pct = float(self._settings.get("spike_max_spread_pct", 2.0))
        self._min_dollar_volume = float(self._settings.get("spike_min_dollar_volume", 50_000.0))
        self._min_trade_count = int(self._settings.get("spike_min_trade_count", 2))
        self._min_single_trade_value = float(self._settings.get("spike_min_single_trade_value", 2_000.0))

        # Speculative bucket — separate thresholds, smaller notional, no earnings bypass
        _s = self._settings
        self._spec_symbols: frozenset[str] = frozenset(_s.get("speculative_watchlist", []))
        self._spec_min_gap = float(_s.get("spec_spike_min_gap_pct", 10.0))
        self._spec_max_gap = float(_s.get("spec_spike_max_gap_pct", 60.0))
        self._spec_notional = float(_s.get("spec_spike_notional", 50.0))
        self._spec_max_spread = float(_s.get("spec_spike_max_spread_pct", 1.0))
        self._spec_min_dollar_vol = float(_s.get("spec_spike_min_dollar_volume", 100_000.0))
        self._spec_min_trade_count = int(_s.get("spec_spike_min_trade_count", 5))
        self._spec_sector_confirm = int(_s.get("spec_spike_sector_confirm", 2))
        self._spec_min_single_trade_value = float(_s.get("spec_spike_min_single_trade_value", 500.0))

        self._reference_closes: dict[str, float] = {}
        self._already_ordered: set[str] = set()
        self._already_sold: set[str] = set()
        # Two pending queues — liquid semis and speculative names.
        # Only liquid names get the earnings-based sector-confirm timeout flush.
        self._pending_liquid: dict[str, tuple[float, float]] = {}  # symbol → (gap_pct, price)
        self._pending_spec: dict[str, tuple[float, float]] = {}
        self._pending_details: dict[str, dict[str, Any]] = {}
        self._order_details: dict[str, dict[str, Any]] = {}
        self._decision_seen: set[tuple[str, str, str, str]] = set()
        self._pending_ts: dict[str, float] = {}                    # symbol → monotonic entry time
        self._order_lock = threading.Lock()
        self._earnings_symbols: set[str] = set()
        # Symbols with earnings TODAY — bidirectional (buy beat, sell miss)
        self._earnings_today: set[str] = set()
        # Held-symbols REST cache (refreshed at most every 30s)
        self._held_cache: set[str] = set()
        self._held_cache_ts: float = 0.0
        self._ledger_db: str = str(config.get("runtime", {}).get("ledger_db", _ledger.DEFAULT_DB))

    def load_references(self) -> None:
        liquid = self.config["watchlist"]
        spec = list(self._spec_symbols)
        combined = list(dict.fromkeys(liquid + spec))  # dedupe, preserve order
        self._reference_closes = fetch_reference_closes(
            self.config, self.api_key, self.secret_key, symbols=combined
        )

    def load_earnings_calendar(self) -> None:
        """Fetch today's earnings reporters — enables sell-on-miss in addition to buy-on-beat."""
        from semibot.earnings_calendar import get_reporting_soon, get_reporting_today

        lookahead = int(self._settings.get("earnings_lookahead_days", 7))
        symbols = self.config["watchlist"]

        today_reporters = get_reporting_today(symbols)
        self._earnings_today = set(today_reporters)

        soon = get_reporting_soon(symbols, days=lookahead)
        if today_reporters:
            print(f"  *** EARNINGS TODAY (bidirectional): {', '.join(sorted(today_reporters))} ***")
            print("      Beat → buy immediately | Miss → sell position immediately")
        elif soon:
            print("  Upcoming earnings: " + "  ".join(f"{s}:{d}" for s, d in sorted(soon.items(), key=lambda x: x[1])))
        else:
            print(f"  No earnings in next {lookahead} days")

    def load_earnings_symbols(self) -> None:
        """Load catalyst signals for notional boost context.

        News signals mark which symbols have recent catalyst headlines.  They are
        used ONLY to boost notional size (earnings_notional_multiplier) — not to
        lower bypass thresholds or skip sector confirmation.

        Bypass of sector confirmation is driven exclusively by the calendar
        (_earnings_today), not by news keywords.  A headline can look bullish but
        already be priced in; the gap price is the only reliable signal.
        """
        from semibot.news_monitor import load_signals_file, scan_news

        settings = self._settings
        signals_path = str(settings.get("news_signals_file", "logs/news_signals.json"))

        # 1. Pre-detected signals from the overnight cron
        pre_signals = load_signals_file(signals_path)
        if pre_signals:
            print(f"Pre-detected signals from monitor ({len(pre_signals)}): "
                  f"{', '.join(sorted(pre_signals))}")
            for sym, sig in sorted(pre_signals.items(), key=lambda x: -x[1].get("score", 0)):
                print(f"  {sym:6}  score={sig.get('score', '?')}  {sig.get('headline', '')[:70]}")

        # 2. Fresh 24h news scan to catch anything missed between cron runs
        try:
            fresh = scan_news(self.config, self.api_key, self.secret_key, lookback_minutes=24 * 60)
        except Exception as exc:
            log.warning("Fresh news scan failed: %s", exc)
            fresh = {}

        merged = {**pre_signals, **fresh}
        self._earnings_symbols = set(merged)

        if self._earnings_symbols:
            print(f"Catalyst context loaded for notional boost (×{self._earnings_multiplier:.1f}): "
                  f"{', '.join(sorted(self._earnings_symbols))}")
        else:
            print("No catalyst signals detected")

    _HELD_CACHE_TTL = 30.0  # seconds between position REST fetches

    def _get_held_symbols(self) -> set[str]:
        now = time.monotonic()
        if now - self._held_cache_ts < self._HELD_CACHE_TTL:
            return self._held_cache
        try:
            bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
            positions = bot.get_positions()
            self._held_cache = {s for s, p in positions.items() if float(getattr(p, "qty", 0)) > 0}
            self._held_cache_ts = now
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
        return self._held_cache

    def _bucket(self, symbol: str) -> str:
        return "speculative" if symbol in self._spec_symbols else "liquid_semi"

    def _quality_details(
        self,
        symbol: str,
        *,
        gap_pct: float | None = None,
        spread_pct: float | None = None,
        max_spread_pct: float | None = None,
        dollar_volume: float | None = None,
        min_dollar_volume: float | None = None,
        trade_count: int | None = None,
        min_trade_count: int | None = None,
        single_trade_value: float | None = None,
        min_single_trade_value: float | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {"bucket": self._bucket(symbol)}
        if source:
            details["source"] = source
        if gap_pct is not None:
            details["gap"] = f"{gap_pct:+.2f}%"
            details["_gap_pct"] = gap_pct
        if spread_pct is not None and max_spread_pct is not None:
            suffix = "OK" if spread_pct <= max_spread_pct else f"> max {max_spread_pct:.2f}%"
            details["spread"] = f"{spread_pct:.2f}% {suffix}"
            details["_spread_pct"] = spread_pct
            details["_max_spread_pct"] = max_spread_pct
        if dollar_volume is not None and min_dollar_volume is not None:
            suffix = "OK" if dollar_volume >= min_dollar_volume or trade_count == 0 else f"< min {_fmt_money(min_dollar_volume)}"
            details["dollar volume"] = f"{_fmt_money(dollar_volume)} {suffix}"
            details["_dollar_volume"] = dollar_volume
            details["_min_dollar_volume"] = min_dollar_volume
        if trade_count is not None and min_trade_count is not None:
            suffix = "OK" if trade_count >= min_trade_count or trade_count == 0 else f"< min {min_trade_count}"
            details["trade count"] = f"{trade_count} {suffix}"
            details["_trade_count"] = trade_count
            details["_min_trade_count"] = min_trade_count
        if single_trade_value is not None:
            if min_single_trade_value is not None:
                suffix = "OK" if single_trade_value >= min_single_trade_value else f"< min {_fmt_money(min_single_trade_value)}"
                details["single trade value"] = f"{_fmt_money(single_trade_value)} {suffix}"
            else:
                details["single trade value"] = _fmt_money(single_trade_value)
            details["_single_trade_value"] = single_trade_value
        return details

    def _print_decision(
        self,
        symbol: str,
        action: str,
        details: dict[str, Any],
        *,
        dedupe: bool = False,
    ) -> None:
        reason = str(details.get("reason", ""))
        source = str(details.get("source", ""))
        if dedupe:
            key = (source, symbol, action, reason)
            if key in self._decision_seen:
                return
            self._decision_seen.add(key)

        ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
        display = [
            f"{key}: {value}"
            for key, value in details.items()
            if not key.startswith("_") and value not in (None, "")
        ]
        print(f"[{ts}] {symbol} {action}  " + "  ".join(display))

        try:
            append_event(self.config["runtime"]["log_file"], {
                "event": "spike_decision",
                "symbol": symbol,
                "action": action.lower(),
                "decision": action,
                "source": details.get("source", ""),
                "bucket": details.get("bucket", ""),
                "reason": details.get("reason", ""),
                "skip_reason": details.get("reason", "") if action == "SKIPPED" else "",
                "gap_pct": details.get("_gap_pct", ""),
                "spread_pct": details.get("_spread_pct", ""),
                "max_spread_pct": details.get("_max_spread_pct", ""),
                "dollar_volume": details.get("_dollar_volume", ""),
                "min_dollar_volume": details.get("_min_dollar_volume", ""),
                "trade_count": details.get("_trade_count", ""),
                "min_trade_count": details.get("_min_trade_count", ""),
                "single_trade_value": details.get("_single_trade_value", ""),
                "confirmation_count": details.get("_confirmation_count", ""),
                "confirmation_required": details.get("_confirmation_required", ""),
                "limit_price": details.get("_limit_price", ""),
                "qty": details.get("_qty", ""),
                "notional": details.get("_notional", ""),
                "strategy": details.get("strategy", ""),
                "price": details.get("_price", ""),
            })
        except Exception as exc:
            log.warning("Could not append spike decision event: %s", exc)

    def _place_order(self, symbol: str, gap_pct: float, current_price: float) -> None:
        """Place an extended-hours limit order.  Safe to call from any thread."""
        from uuid import uuid4

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        is_spec = symbol in self._spec_symbols
        notional = self._spec_notional if is_spec else self._notional
        if not is_spec and self._earnings_multiplier > 1.0 and symbol in self._earnings_symbols:
            notional *= self._earnings_multiplier
            print(f"  [catalyst] {symbol}: notional boosted to ${notional:.0f} (×{self._earnings_multiplier:.1f})")

        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        limit_price = limit_price_for_side(current_price, "buy", self._offset_bps)
        qty = floor_order_qty(notional, limit_price)

        parts = [f"spike gap={gap_pct:+.1f}%"]
        if symbol in self._earnings_today:
            parts.append("earnings_today")
        if is_spec:
            parts.append("spec_bucket")
        reason = " ".join(parts)
        strategy = "spike_spec" if is_spec else "spike_liquid"
        details = self._order_details.pop(symbol, self._quality_details(symbol, gap_pct=gap_pct))
        details.update({
            "limit price": f"${limit_price:.2f}",
            "qty": f"{qty:.4f}",
            "notional": f"${notional:.2f}",
            "strategy": strategy,
            "mode": "DRY-RUN" if self._dry_run else "PAPER",
            "reason": details.get("reason", "confirmed spike"),
            "_limit_price": limit_price,
            "_qty": qty,
            "_notional": notional,
            "_price": current_price,
        })

        if self._dry_run:
            self._print_decision(symbol, "ORDER", details)
            _ledger.record_fill(
                self._ledger_db, symbol, "buy", qty, limit_price, notional,
                reason, strategy=strategy, dry_run=True,
            )
            return

        order = LimitOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            qty=qty,
            limit_price=limit_price,
            extended_hours=True,
            client_order_id=f"spike-{symbol}-{uuid4().hex[:10]}",
        )
        result = bot.trading.submit_order(order_data=order)
        order_id = str(result.id)  # type: ignore[union-attr]
        details["order id"] = order_id
        self._print_decision(symbol, "ORDER", details)
        try:
            record_spike_entry(self._tracker_path, symbol, gap_pct, current_price)
        except Exception as exc:
            log.warning("spike_tracker record failed for %s: %s", symbol, exc)
        try:
            append_event(self.config["runtime"]["log_file"], {
                "event": "spike_order_submitted",
                "symbol": symbol,
                "gap_pct": round(gap_pct, 2),
                "price": current_price,
                "limit_price": limit_price,
                "qty": qty,
                "order_id": order_id,
            })
        except Exception as exc:
            log.warning("Could not append spike order event: %s", exc)
        _ledger.record_fill(
            self._ledger_db, symbol, "buy", qty, limit_price, notional,
            reason, strategy=strategy, order_id=order_id, dry_run=False,
        )

    def _place_sell_order(self, symbol: str, gap_pct: float, current_price: float) -> None:
        """Sell existing position via extended-hours limit order on earnings miss."""
        from uuid import uuid4

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        positions = bot.get_positions()
        position = positions.get(symbol)
        if not position:
            log.warning("_place_sell_order: no position found for %s", symbol)
            self._print_decision(symbol, "SKIPPED", {
                **self._quality_details(symbol, gap_pct=gap_pct),
                "reason": "earnings miss sell skipped; no position found",
            })
            return
        qty = float(getattr(position, "qty", 0.0))
        if qty <= 0:
            self._print_decision(symbol, "SKIPPED", {
                **self._quality_details(symbol, gap_pct=gap_pct),
                "reason": "earnings miss sell skipped; zero position quantity",
            })
            return

        limit_price = limit_price_for_side(current_price, "sell", self._offset_bps)
        reason = f"earnings_miss gap={gap_pct:+.1f}%"
        notional_sell = qty * limit_price
        details = self._quality_details(symbol, gap_pct=gap_pct)
        details.update({
            "limit price": f"${limit_price:.2f}",
            "qty": f"{qty:.4f}",
            "notional": f"${notional_sell:.2f}",
            "strategy": "spike_sell",
            "mode": "DRY-RUN" if self._dry_run else "PAPER",
            "reason": "earnings miss sell triggered",
            "_limit_price": limit_price,
            "_qty": qty,
            "_notional": notional_sell,
            "_price": current_price,
        })

        if self._dry_run:
            self._print_decision(symbol, "SELL", details)
            _ledger.record_fill(
                self._ledger_db, symbol, "sell", qty, limit_price, notional_sell,
                reason, strategy="spike_sell", dry_run=True,
            )
            return

        order = LimitOrderRequest(
            symbol=symbol,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            qty=qty,
            limit_price=limit_price,
            extended_hours=True,
            client_order_id=f"spike-sell-{symbol}-{uuid4().hex[:10]}",
        )
        result = bot.trading.submit_order(order_data=order)
        order_id = str(result.id)  # type: ignore[union-attr]
        details["order id"] = order_id
        self._print_decision(symbol, "SELL", details)
        try:
            from semibot.spike_tracker import remove_spike_entry
            remove_spike_entry(self._tracker_path, symbol)
        except Exception as exc:
            log.warning("spike_tracker remove failed for %s: %s", symbol, exc)
        try:
            append_event(self.config["runtime"]["log_file"], {
                "event": "earnings_miss_sell",
                "symbol": symbol,
                "gap_pct": round(gap_pct, 2),
                "price": current_price,
                "limit_price": limit_price,
                "qty": qty,
                "order_id": order_id,
            })
        except Exception as exc:
            log.warning("Could not append sell event: %s", exc)
        _ledger.record_fill(
            self._ledger_db, symbol, "sell", qty, limit_price, notional_sell,
            reason, strategy="spike_sell", order_id=order_id, dry_run=False,
        )

    def _check_earnings_miss(self, symbol: str, gap_pct: float, price: float, source: str) -> bool:
        """Check if this is an earnings-miss negative gap and sell if we hold the position.

        Returns True if a sell was triggered (caller should skip buy logic).
        Must NOT be called with _order_lock held.
        """
        if gap_pct > -self._sell_gap_threshold:
            return False
        if symbol not in self._earnings_today:
            return False
        if symbol in self._already_sold:
            return False

        with self._order_lock:
            if symbol in self._already_sold:
                return False
            held = self._get_held_symbols()
            if symbol not in held:
                self._print_decision(symbol, "SKIPPED", {
                    **self._quality_details(symbol, gap_pct=gap_pct, source=source),
                    "reason": "earnings miss detected but no position is open",
                }, dedupe=True)
                return False
            self._already_sold.add(symbol)

        self._place_sell_order(symbol, gap_pct, price)
        return True

    def _try_queue_or_flush(
        self,
        symbol: str,
        gap_pct: float,
        price: float,
        source: str,
        pending: dict[str, tuple[float, float]] | None = None,
        confirm: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, float]]:
        """Thread-safe: add symbol to pending; return list of (sym, gap, price) to order if threshold met.

        Must be called with _order_lock held by the caller.
        Pass pending=self._pending_spec and confirm=self._spec_sector_confirm for spec names.
        """
        if pending is None:
            pending = self._pending_liquid
        if confirm is None:
            confirm = self._sector_confirm
        pending[symbol] = (gap_pct, price)
        decision_details = dict(details or self._quality_details(symbol, gap_pct=gap_pct))
        self._pending_ts[symbol] = time.monotonic()
        n = len(pending)
        decision_details.update({
            "source": source,
            "confirmation": f"{n}/{confirm} sector names gapping",
            "reason": "waiting for sector/spec bucket confirmation",
            "_confirmation_count": n,
            "_confirmation_required": confirm,
            "_price": price,
        })
        self._pending_details[symbol] = decision_details
        self._print_decision(symbol, "PENDING", decision_details)
        if n >= confirm:
            to_flush = list(pending.items())
            for sym, _ in to_flush:
                self._already_ordered.add(sym)
                self._pending_ts.pop(sym, None)
                order_details = dict(self._pending_details.pop(sym, {}))
                order_details.update({
                    "sector confirmation": f"{n}/{confirm} passed",
                    "reason": "confirmed spec spike" if sym in self._spec_symbols else "confirmed semi spike",
                    "_confirmation_count": n,
                    "_confirmation_required": confirm,
                })
                self._order_details[sym] = order_details
            pending.clear()
            return [(sym, g, p) for sym, (g, p) in to_flush]
        return []

    # ── Layer 1: startup snapshot ─────────────────────────────────────────────

    def _startup_gap_scan(self) -> None:
        """Snapshot all symbols right after subscription — catches pre-existing gaps."""
        if not self._reference_closes:
            return
        liquid = self.config["watchlist"]
        spec = list(self._spec_symbols)
        all_symbols = list(dict.fromkeys(liquid + spec))
        print("Startup gap scan (checking symbols already moving)...")
        try:
            client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
            snapshots = client.get_stock_snapshot(
                StockSnapshotRequest(
                    symbol_or_symbols=all_symbols,
                    feed=_parse_feed(self.config),
                )
            )
        except Exception as exc:
            log.warning("Startup gap scan failed: %s", exc)
            return

        held = self._get_held_symbols()
        to_order: list[tuple[str, float, float]] = []

        # Build price map first (no lock needed).
        # _best_price_from_snap falls back to NBBO midpoint when latest_trade is stale,
        # so we can detect gaps even if no pre-market trade has printed yet.
        sym_data: list[tuple[str, float, float, dict[str, Any]]] = []
        for symbol in all_symbols:
            is_spec = symbol in self._spec_symbols
            max_spread = self._spec_max_spread if is_spec else self._max_spread_pct
            min_dollar_vol = self._spec_min_dollar_vol if is_spec else self._min_dollar_volume
            min_trades = self._spec_min_trade_count if is_spec else self._min_trade_count

            snap = snapshots.get(symbol)
            price = _best_price_from_snap(snap)
            if price is None:
                continue
            ref = self._reference_closes.get(symbol)
            if not ref or ref <= 0:
                continue
            gap_pct = ((price / ref) - 1) * 100
            spread_pass, spread = _spread_ok(snap, max_spread)
            dollar_volume, trade_count = _liquidity_stats(snap, price)
            details = self._quality_details(
                symbol,
                gap_pct=gap_pct,
                spread_pct=spread,
                max_spread_pct=max_spread,
                dollar_volume=dollar_volume,
                min_dollar_volume=min_dollar_vol,
                trade_count=trade_count,
                min_trade_count=min_trades,
                source="STARTUP",
            )
            if not spread_pass:
                self._print_decision(symbol, "SKIPPED", {
                    **details,
                    "reason": "spread too wide",
                }, dedupe=True)
                continue
            liq_pass, liq_reason = _liquidity_ok(snap, min_dollar_vol, min_trades)
            if not liq_pass:
                self._print_decision(symbol, "SKIPPED", {
                    **details,
                    "liquidity": liq_reason,
                    "reason": "thin pre-market liquidity",
                }, dedupe=True)
                continue
            sym_data.append((symbol, gap_pct, price, details))

        # Check earnings misses outside the lock — _check_earnings_miss acquires its own
        # lock internally, so calling it inside _order_lock would deadlock.
        # Spec names skip earnings-miss logic — no position to sell, no calendar bypass.
        sold_symbols: set[str] = set()
        for symbol, gap_pct, price, _details in sym_data:
            if symbol not in self._spec_symbols:
                if self._check_earnings_miss(symbol, gap_pct, price, "STARTUP"):
                    sold_symbols.add(symbol)

        with self._order_lock:
            for symbol, gap_pct, price, details in sym_data:
                if symbol in sold_symbols:
                    continue
                is_spec = symbol in self._spec_symbols
                pending = self._pending_spec if is_spec else self._pending_liquid
                min_gap = self._spec_min_gap if is_spec else self._min_gap
                max_gap = self._spec_max_gap if is_spec else self._max_gap
                confirm = self._spec_sector_confirm if is_spec else self._sector_confirm
                if symbol in self._already_ordered or symbol in pending or symbol in held:
                    continue
                if gap_pct < min_gap or gap_pct > max_gap:
                    if abs(gap_pct) >= 1.0:
                        reason = "gap below threshold" if gap_pct < min_gap else "gap above max threshold"
                        self._print_decision(symbol, "SKIPPED", {
                            **self._quality_details(symbol, gap_pct=gap_pct, source="STARTUP"),
                            "required gap": f"{min_gap:.2f}% to {max_gap:.2f}%",
                            "reason": reason,
                        }, dedupe=True)
                    continue
                to_order.extend(self._try_queue_or_flush(
                    symbol,
                    gap_pct,
                    price,
                    "STARTUP",
                    pending,
                    confirm,
                    details,
                ))

        for sym, gap, price in to_order:
            self._place_order(sym, gap, price)

        if not to_order and not self._pending_liquid and not self._pending_spec:
            print("  No pre-existing gaps above threshold")

    # ── Layer 2: news WebSocket ───────────────────────────────────────────────

    def _check_news_gaps(self, symbols: list[str], news_text: str) -> None:
        """Snapshot news-mentioned symbols and fire/queue immediately.  Runs in a thread.

        News is used as an attention trigger — it tells us to look at a symbol NOW.
        The actual buy/sell decision is based on the gap (price), not the headline text.
        Bypass of sector confirmation is granted only when the earnings calendar confirms
        the symbol is reporting today AND the gap is large.  Keywords alone are not enough.
        """
        try:
            client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
            snapshots = client.get_stock_snapshot(
                StockSnapshotRequest(
                    symbol_or_symbols=symbols,
                    feed=_parse_feed(self.config),
                )
            )
        except Exception as exc:
            log.warning("News gap snapshot failed: %s", exc)
            return

        held = self._get_held_symbols()
        for symbol in symbols:
            if symbol in self._already_ordered:
                continue
            is_spec = symbol in self._spec_symbols
            max_spread = self._spec_max_spread if is_spec else self._max_spread_pct
            min_dollar_vol = self._spec_min_dollar_vol if is_spec else self._min_dollar_volume
            min_trades = self._spec_min_trade_count if is_spec else self._min_trade_count
            min_gap = self._spec_min_gap if is_spec else self._min_gap
            max_gap = self._spec_max_gap if is_spec else self._max_gap
            pending = self._pending_spec if is_spec else self._pending_liquid
            confirm = self._spec_sector_confirm if is_spec else self._sector_confirm

            snap = snapshots.get(symbol)
            price = _best_price_from_snap(snap)
            if price is None:
                continue
            ref = self._reference_closes.get(symbol)
            if not ref or ref <= 0:
                continue
            gap_pct = ((price / ref) - 1) * 100
            spread_pass, spread = _spread_ok(snap, max_spread)
            dollar_volume, trade_count = _liquidity_stats(snap, price)
            details = self._quality_details(
                symbol,
                gap_pct=gap_pct,
                spread_pct=spread,
                max_spread_pct=max_spread,
                dollar_volume=dollar_volume,
                min_dollar_volume=min_dollar_vol,
                trade_count=trade_count,
                min_trade_count=min_trades,
                source="NEWS",
            )
            if not spread_pass:
                self._print_decision(symbol, "SKIPPED", {
                    **details,
                    "reason": "spread too wide",
                }, dedupe=True)
                continue
            liq_pass, liq_reason = _liquidity_ok(snap, min_dollar_vol, min_trades)
            if not liq_pass:
                self._print_decision(symbol, "SKIPPED", {
                    **details,
                    "liquidity": liq_reason,
                    "reason": "thin pre-market liquidity",
                }, dedupe=True)
                continue
            if not is_spec and self._check_earnings_miss(symbol, gap_pct, price, "NEWS"):
                continue
            if gap_pct < min_gap or gap_pct > max_gap:
                reason = "gap below threshold" if gap_pct < min_gap else "gap above max threshold"
                self._print_decision(symbol, "SKIPPED", {
                    **details,
                    "required gap": f"{min_gap:.2f}% to {max_gap:.2f}%",
                    "reason": reason,
                }, dedupe=True)
                continue

            # Bypass sector confirmation only when the calendar confirms earnings today
            # AND the gap is large.  Spec names never get bypass — no calendar tracking.
            bypass = (
                not is_spec
                and symbol in self._earnings_today
                and gap_pct >= self._earnings_bypass_gap
            )
            to_order: list[tuple[str, float, float]] = []

            with self._order_lock:
                if symbol in self._already_ordered or symbol in pending:
                    continue
                if symbol in held:
                    self._already_ordered.add(symbol)
                    continue
                if bypass:
                    self._already_ordered.add(symbol)
                    self._order_details[symbol] = {
                        **details,
                        "sector confirmation": "earnings bypass",
                        "reason": "earnings gap bypassed sector confirmation",
                        "_confirmation_count": confirm,
                        "_confirmation_required": confirm,
                    }
                    to_order.append((symbol, gap_pct, price))
                else:
                    to_order.extend(self._try_queue_or_flush(
                        symbol,
                        gap_pct,
                        price,
                        "NEWS",
                        pending,
                        confirm,
                        details,
                    ))

            for sym, gap, p in to_order:
                self._place_order(sym, gap, p)

    async def _on_news(self, news: Any) -> None:
        """News WebSocket handler — fires on every incoming article."""
        symbols: list[str] = getattr(news, "symbols", []) or []
        headline = (getattr(news, "headline", "") or "").lower()
        summary = (getattr(news, "summary", "") or "").lower()
        text = headline + " " + summary

        if not any(kw in text for kw in _CATALYST_KEYWORDS):
            return

        watchlist_set = set(self.config["watchlist"]) | self._spec_symbols
        relevant = [s for s in symbols if s in watchlist_set]
        if not relevant:
            return

        ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
        headline_preview = (getattr(news, "headline", "") or "")[:90]
        print(f"[{ts}] NEWS [{', '.join(relevant)}]: {headline_preview}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._check_news_gaps, relevant, text)

    # ── Layer 3: trade WebSocket ──────────────────────────────────────────────

    async def _on_trade(self, trade: Any) -> None:
        symbol = getattr(trade, "symbol", None)
        if not symbol:
            return
        ref = self._reference_closes.get(symbol)
        if not ref:
            return

        price = float(trade.price)
        gap_pct = ((price / ref) - 1) * 100

        is_spec = symbol in self._spec_symbols
        min_single = self._spec_min_single_trade_value if is_spec else self._min_single_trade_value
        min_gap = self._spec_min_gap if is_spec else self._min_gap
        max_gap = self._spec_max_gap if is_spec else self._max_gap
        pending = self._pending_spec if is_spec else self._pending_liquid
        confirm = self._spec_sector_confirm if is_spec else self._sector_confirm

        # Reject tiny prints — a 1-share test trade should not trigger an order
        trade_value = price * float(getattr(trade, "size", 0) or 0)
        details = self._quality_details(
            symbol,
            gap_pct=gap_pct,
            single_trade_value=trade_value,
            min_single_trade_value=min_single,
            source="TRADE",
        )
        if trade_value < min_single:
            self._print_decision(symbol, "SKIPPED", {
                **details,
                "reason": "single trade value too small",
            }, dedupe=True)
            return

        # Check earnings miss (negative gap) before buy logic; spec names skip this
        if not is_spec and self._check_earnings_miss(symbol, gap_pct, price, "TRADE"):
            await asyncio.sleep(0)  # yield to event loop
            return

        if gap_pct < min_gap or gap_pct > max_gap:
            if abs(gap_pct) >= 1.0 or gap_pct > max_gap:
                reason = "gap below threshold" if gap_pct < min_gap else "gap above max threshold"
                self._print_decision(symbol, "SKIPPED", {
                    **details,
                    "required gap": f"{min_gap:.2f}% to {max_gap:.2f}%",
                    "reason": reason,
                }, dedupe=True)
            return

        if symbol in self._already_ordered:
            return

        # Fetch held symbols outside the lock — occasional stale read is acceptable and
        # far better than a blocking REST call that holds _order_lock for ~200ms.
        held = self._get_held_symbols()
        to_order: list[tuple[str, float, float]] = []
        with self._order_lock:
            if symbol in self._already_ordered or symbol in pending:
                return
            if symbol in held:
                self._already_ordered.add(symbol)
                return
            to_order = self._try_queue_or_flush(symbol, gap_pct, price, "TRADE", pending, confirm, details)

        if to_order:
            loop = asyncio.get_running_loop()
            for sym, gap, p in to_order:
                await loop.run_in_executor(None, self._place_order, sym, gap, p)

    # ── Earnings pending flusher ──────────────────────────────────────────────

    async def _earnings_pending_flusher(self) -> None:
        """Every 60s, force-flush earnings-day symbols stuck in _pending.

        sector_confirm=3 is great for filtering noise, but a single stock reporting
        a blowout quarter should not stay queued forever waiting for two more symbols
        to gap.  Any symbol in _earnings_today that has been pending for >60s is
        ordered immediately regardless of confirm count.
        """
        FLUSH_AFTER = 60.0
        while True:
            await asyncio.sleep(60)
            to_order: list[tuple[str, float, float]] = []
            now_ts = time.monotonic()
            with self._order_lock:
                for sym in list(self._pending_liquid):
                    if sym not in self._earnings_today:
                        continue
                    if now_ts - self._pending_ts.get(sym, now_ts) < FLUSH_AFTER:
                        continue
                    gap_pct, price = self._pending_liquid.pop(sym)
                    self._pending_ts.pop(sym, None)
                    self._already_ordered.add(sym)
                    to_order.append((sym, gap_pct, price))
                    details = dict(self._pending_details.pop(sym, {}))
                    details.update({
                        "sector confirmation": "earnings timeout bypass",
                        "reason": "earnings pending timeout; ordering without full sector confirmation",
                        "_confirmation_count": 1,
                        "_confirmation_required": self._sector_confirm,
                    })
                    self._order_details[sym] = details
            for sym, gap, p in to_order:
                self._place_order(sym, gap, p)

    # ── Session snapshot ──────────────────────────────────────────────────────

    def _snapshot_session_start(self) -> None:
        """Record equity + reference closes as benchmark at stream startup."""
        try:
            from alpaca.trading.client import TradingClient
            trading = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=bool(self.config["alpaca"].get("paper", True)),
            )
            account = trading.get_account()
            _ledger.record_equity(
                self._ledger_db,
                cash=float(getattr(account, "cash", 0) or 0),
                position_market_value=float(getattr(account, "long_market_value", 0) or 0),
                total_equity=float(getattr(account, "equity", 0) or 0),
                unrealized_pnl=float(getattr(account, "unrealized_pl", 0) or 0),
            )
        except Exception as exc:
            log.warning("Could not record equity snapshot: %s", exc)

        if self._reference_closes:
            _ledger.record_benchmark(self._ledger_db, dict(self._reference_closes))

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, execute: bool = False) -> None:
        """Start all three detection layers.  Blocks until market opens or 7:45pm."""
        self._dry_run = self._dry_run or not execute
        liquid = self.config["watchlist"]
        spec = list(self._spec_symbols)
        all_symbols = list(dict.fromkeys(liquid + spec))
        feed = _parse_feed(self.config)

        print("── Earnings calendar ──")
        self.load_earnings_calendar()

        print("── Catalyst / news signals ──")
        self.load_earnings_symbols()

        print(f"── Starting spike scanner ──  "
              f"liquid={len(liquid)} semis  spec={len(spec)} names  "
              f"min_gap={self._min_gap:.1f}%  spec_min_gap={self._spec_min_gap:.1f}%  "
              f"sell_gap=-{self._sell_gap_threshold:.1f}%  "
              f"notional=${self._notional:.0f}  spec_notional=${self._spec_notional:.0f}  "
              f"confirm={self._sector_confirm}  spec_confirm={self._spec_sector_confirm}  "
              f"{'DRY-RUN' if self._dry_run else 'LIVE'}")
        if spec:
            print(f"  Speculative bucket: {', '.join(sorted(spec))}")

        trade_stream = StockDataStream(api_key=self.api_key, secret_key=self.secret_key, feed=feed)
        trade_stream.subscribe_trades(self._on_trade, *all_symbols)

        news_stream = NewsDataStream(api_key=self.api_key, secret_key=self.secret_key)
        news_stream.subscribe_news(self._on_news, *all_symbols)

        async def _main() -> None:
            from alpaca.trading.client import TradingClient
            trading = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=bool(self.config["alpaca"].get("paper", True)),
            )

            # Layer 1: check for pre-existing gaps before the first trade arrives
            self._startup_gap_scan()
            self._snapshot_session_start()

            async def _watchdog() -> None:
                while True:
                    await asyncio.sleep(30)
                    now = datetime.now(MARKET_TZ)
                    if (now.hour, now.minute) >= (19, 45):
                        print(f"[{now.strftime('%H:%M:%S')}] Extended-hours window closed (7:45pm) — stopping")
                        trade_stream.stop()
                        news_stream.stop()
                        return
                    try:
                        clock = trading.get_clock()
                        if getattr(clock, "is_open", False):
                            print(f"[{now.strftime('%H:%M:%S')}] Market opened — stopping spike stream")
                            trade_stream.stop()
                            news_stream.stop()
                            return
                    except Exception as exc:
                        log.warning("Clock check failed: %s", exc)

            # Layers 2+3 run concurrently; news stream failure is non-fatal
            try:
                await asyncio.gather(
                    trade_stream._run_forever(),
                    news_stream._run_forever(),
                    _watchdog(),
                    self._earnings_pending_flusher(),
                )
            except Exception as exc:
                log.warning("News stream error (%s) — continuing with trade stream only", exc)
                await asyncio.gather(
                    trade_stream._run_forever(),
                    _watchdog(),
                    self._earnings_pending_flusher(),
                )

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            print("Spike stream stopped.")
        finally:
            trade_stream.stop()
            try:
                news_stream.stop()
            except Exception:
                pass
