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


def _parse_feed(config: dict[str, Any]) -> DataFeed:
    raw = config["alpaca"].get("data_feed", "iex").upper()
    return DataFeed[raw] if raw in DataFeed.__members__ else DataFeed.IEX


def fetch_reference_closes(config: dict[str, Any], api_key: str, secret_key: str) -> dict[str, float]:
    """Return the most recent regular-session close for each watchlist symbol.

    Pre-market  (before 16:00 ET): uses previous_daily_bar.close (yesterday's 4pm)
    After-hours (after  16:00 ET): uses daily_bar.close (today's 4pm, now locked)
    """
    now_et = datetime.now(MARKET_TZ)
    use_prev = now_et.hour < _REGULAR_CLOSE_HOUR

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    feed = _parse_feed(config)
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

        self._reference_closes: dict[str, float] = {}
        self._already_ordered: set[str] = set()
        self._already_sold: set[str] = set()
        # Qualified but waiting for sector_confirm threshold
        self._pending: dict[str, tuple[float, float]] = {}  # symbol → (gap_pct, price)
        self._pending_ts: dict[str, float] = {}             # symbol → monotonic time of entry
        self._order_lock = threading.Lock()
        self._earnings_symbols: set[str] = set()
        # Per-symbol bypass gap threshold — lowered for high-score pre-detected signals
        self._bypass_gap_by_symbol: dict[str, float] = {}
        # Symbols with earnings TODAY — bidirectional (buy beat, sell miss)
        self._earnings_today: set[str] = set()
        # Held-symbols REST cache (refreshed at most every 30s)
        self._held_cache: set[str] = set()
        self._held_cache_ts: float = 0.0

    def load_references(self) -> None:
        self._reference_closes = fetch_reference_closes(self.config, self.api_key, self.secret_key)

    def load_earnings_calendar(self) -> None:
        """Fetch today's earnings reporters — enables sell-on-miss in addition to buy-on-beat."""
        from semibot.earnings_calendar import get_reporting_today, get_reporting_soon

        lookahead = int(self._settings.get("earnings_lookahead_days", 7))
        symbols = self.config["watchlist"]

        today_reporters = get_reporting_today(symbols)
        self._earnings_today = set(today_reporters)

        soon = get_reporting_soon(symbols, days=lookahead)
        if today_reporters:
            print(f"  *** EARNINGS TODAY (bidirectional): {', '.join(sorted(today_reporters))} ***")
            print(f"      Beat → buy immediately | Miss → sell position immediately")
        elif soon:
            print(f"  Upcoming earnings: " + "  ".join(f"{s}:{d}" for s, d in sorted(soon.items(), key=lambda x: x[1])))
        else:
            print(f"  No earnings in next {lookahead} days")

    def load_earnings_symbols(self) -> None:
        """Load catalyst signals: pre-detected signals file first, then fresh 24h news scan.

        Pre-detected signals (written by the news monitor cron) are merged with a
        fresh 24h news fetch so the stream starts with the richest possible context.
        High-score signals (earnings score >= 4) lower the bypass threshold so even
        a 6% gap triggers an immediate order without waiting for sector confirmation.
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
                bypass_note = "  [BYPASS]" if sig.get("bypass_confirm") else ""
                print(f"  {sym:6}  score={sig.get('score', '?')}  {sig.get('headline', '')[:65]}{bypass_note}")

        # 2. Fresh 24h news scan to catch anything missed between cron runs
        try:
            fresh = scan_news(self.config, self.api_key, self.secret_key, lookback_minutes=24 * 60)
        except Exception as exc:
            log.warning("Fresh news scan failed: %s", exc)
            fresh = {}

        # Merge: pre-detected + fresh; fresh overwrites for the same symbol
        merged = {**pre_signals, **fresh}

        found: set[str] = set()
        bypass_overrides: dict[str, float] = {}
        for sym, sig in merged.items():
            found.add(sym)
            score = int(sig.get("score", 1))
            bypass_confirm = bool(sig.get("bypass_confirm", False))
            # High-confidence earnings (score >= 4): lower bypass gap to 6%
            if bypass_confirm and score >= 4:
                bypass_overrides[sym] = 6.0
            # Standard earnings: use configured threshold
            elif bypass_confirm:
                bypass_overrides[sym] = self._earnings_bypass_gap

        self._earnings_symbols = found
        self._bypass_gap_by_symbol = bypass_overrides

        if found:
            print(f"Catalyst signals active (notional ×{self._earnings_multiplier:.1f}): "
                  f"{', '.join(sorted(found))}")
            if bypass_overrides:
                for sym, thresh in sorted(bypass_overrides.items()):
                    print(f"  {sym}: bypass at ≥{thresh:.0f}% gap")
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

    def _place_order(self, symbol: str, gap_pct: float, current_price: float) -> None:
        """Place an extended-hours limit order.  Safe to call from any thread."""
        from uuid import uuid4

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        notional = self._notional
        if self._earnings_multiplier > 1.0 and symbol in self._earnings_symbols:
            notional *= self._earnings_multiplier
            print(f"  [catalyst] {symbol}: notional boosted to ${notional:.0f} (×{self._earnings_multiplier:.1f})")

        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        limit_price = limit_price_for_side(current_price, "buy", self._offset_bps)
        qty = floor_order_qty(notional, limit_price)

        label = f"spike gap={gap_pct:+.1f}%"
        if self._dry_run:
            print(f"  DRY-RUN  BUY {symbol}  {qty:.4f}sh @ ${limit_price:.2f}  [{label}]")
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
        print(f"  ORDER    BUY {symbol}  {qty:.4f}sh @ ${limit_price:.2f}  [{label}]  id={result.id}")  # type: ignore[union-attr]
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
                "order_id": str(result.id),  # type: ignore[union-attr]
            })
        except Exception as exc:
            log.warning("Could not append spike order event: %s", exc)

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
            return
        qty = float(getattr(position, "qty", 0.0))
        if qty <= 0:
            return

        limit_price = limit_price_for_side(current_price, "sell", self._offset_bps)
        label = f"earnings miss gap={gap_pct:+.1f}%"

        if self._dry_run:
            print(f"  DRY-RUN  SELL {symbol}  {qty:.4f}sh @ ${limit_price:.2f}  [{label}]")
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
        print(f"  SELL     {symbol}  {qty:.4f}sh @ ${limit_price:.2f}  [{label}]  id={result.id}")  # type: ignore[union-attr]
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
                "order_id": str(result.id),  # type: ignore[union-attr]
            })
        except Exception as exc:
            log.warning("Could not append sell event: %s", exc)

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
                return False
            self._already_sold.add(symbol)

        ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
        print(f"[{ts}] *** EARNINGS MISS {symbol} [{source}]: {gap_pct:+.2f}% → SELLING position ***")
        self._place_sell_order(symbol, gap_pct, price)
        return True

    def _try_queue_or_flush(self, symbol: str, gap_pct: float, price: float, source: str) -> list[tuple[str, float, float]]:
        """Thread-safe: add symbol to pending; return list of (sym, gap, price) to order if threshold met.

        Must be called with _order_lock held by the caller.
        """
        self._pending[symbol] = (gap_pct, price)
        self._pending_ts[symbol] = time.monotonic()
        ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
        n = len(self._pending)
        print(f"[{ts}] {source} PENDING {symbol}: {gap_pct:+.2f}%  [{n}/{self._sector_confirm} for confirm]")
        if n >= self._sector_confirm:
            to_flush = list(self._pending.items())
            for sym, _ in to_flush:
                self._already_ordered.add(sym)
            self._pending.clear()
            self._pending_ts.clear()
            return [(sym, g, p) for sym, (g, p) in to_flush]
        return []

    # ── Layer 1: startup snapshot ─────────────────────────────────────────────

    def _startup_gap_scan(self) -> None:
        """Snapshot all symbols right after subscription — catches pre-existing gaps."""
        if not self._reference_closes:
            return
        print("Startup gap scan (checking symbols already moving)...")
        try:
            client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
            snapshots = client.get_stock_snapshot(
                StockSnapshotRequest(
                    symbol_or_symbols=self.config["watchlist"],
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
        sym_data: list[tuple[str, float, float]] = []
        for symbol in self.config["watchlist"]:
            price = _best_price_from_snap(snapshots.get(symbol))
            if price is None:
                continue
            ref = self._reference_closes.get(symbol)
            if not ref or ref <= 0:
                continue
            gap_pct = ((price / ref) - 1) * 100
            sym_data.append((symbol, gap_pct, price))

        # Check earnings misses outside the lock — _check_earnings_miss acquires its own
        # lock internally, so calling it inside _order_lock would deadlock.
        sold_symbols: set[str] = set()
        for symbol, gap_pct, price in sym_data:
            if self._check_earnings_miss(symbol, gap_pct, price, "STARTUP"):
                sold_symbols.add(symbol)

        with self._order_lock:
            for symbol, gap_pct, price in sym_data:
                if symbol in sold_symbols:
                    continue
                if symbol in self._already_ordered or symbol in self._pending or symbol in held:
                    continue
                if gap_pct < self._min_gap or gap_pct > self._max_gap:
                    if abs(gap_pct) >= 1.0:
                        print(f"  {symbol}: {gap_pct:+.2f}% (below threshold)")
                    continue
                to_order.extend(self._try_queue_or_flush(symbol, gap_pct, price, "STARTUP"))

        for sym, gap, price in to_order:
            self._place_order(sym, gap, price)

        if not to_order and not self._pending:
            print("  No pre-existing gaps above threshold")

    # ── Layer 2: news WebSocket ───────────────────────────────────────────────

    def _check_news_gaps(self, symbols: list[str], news_text: str) -> None:
        """Snapshot news-mentioned symbols and fire/queue immediately.  Runs in a thread."""
        is_earnings = any(kw in news_text for kw in _EARNINGS_KEYWORDS)
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
            price = _best_price_from_snap(snapshots.get(symbol))
            if price is None:
                continue
            ref = self._reference_closes.get(symbol)
            if not ref or ref <= 0:
                continue
            gap_pct = ((price / ref) - 1) * 100
            # Check earnings miss before buy logic
            if self._check_earnings_miss(symbol, gap_pct, price, "NEWS"):
                continue
            if gap_pct < self._min_gap or gap_pct > self._max_gap:
                continue

            ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
            # Per-symbol bypass threshold: lowered for high-score pre-detected signals
            bypass_gap = self._bypass_gap_by_symbol.get(symbol, self._earnings_bypass_gap)
            bypass = is_earnings and gap_pct >= bypass_gap
            to_order: list[tuple[str, float, float]] = []

            with self._order_lock:
                if symbol in self._already_ordered or symbol in self._pending:
                    continue
                if symbol in held:
                    self._already_ordered.add(symbol)
                    continue
                if bypass:
                    print(f"[{ts}] NEWS BYPASS {symbol}: {gap_pct:+.2f}% (earnings ≥{bypass_gap:.0f}% → no sector confirm)")
                    self._already_ordered.add(symbol)
                    to_order.append((symbol, gap_pct, price))
                else:
                    to_order.extend(self._try_queue_or_flush(symbol, gap_pct, price, "NEWS"))

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

        watchlist_set = set(self.config["watchlist"])
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

        # Check earnings miss (negative gap) before buy logic
        if self._check_earnings_miss(symbol, gap_pct, price, "TRADE"):
            await asyncio.sleep(0)  # yield to event loop
            return

        if gap_pct < self._min_gap or gap_pct > self._max_gap:
            return

        if symbol in self._already_ordered:
            return

        # Fetch held symbols outside the lock — occasional stale read is acceptable and
        # far better than a blocking REST call that holds _order_lock for ~200ms.
        held = self._get_held_symbols()
        to_order: list[tuple[str, float, float]] = []
        with self._order_lock:
            if symbol in self._already_ordered or symbol in self._pending:
                return
            if symbol in held:
                self._already_ordered.add(symbol)
                return
            to_order = self._try_queue_or_flush(symbol, gap_pct, price, "TRADE")

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
                for sym in list(self._pending):
                    if sym not in self._earnings_today:
                        continue
                    if now_ts - self._pending_ts.get(sym, now_ts) < FLUSH_AFTER:
                        continue
                    gap_pct, price = self._pending.pop(sym)
                    self._pending_ts.pop(sym, None)
                    self._already_ordered.add(sym)
                    to_order.append((sym, gap_pct, price))
                    ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
                    print(f"[{ts}] EARNINGS-FLUSH {sym}: {gap_pct:+.2f}% "
                          f"(sector confirm timeout — ordering without full confirm)")
            for sym, gap, p in to_order:
                self._place_order(sym, gap, p)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, execute: bool = False) -> None:
        """Start all three detection layers.  Blocks until market opens or 7:45pm."""
        self._dry_run = self._dry_run or not execute
        symbols = self.config["watchlist"]
        feed = _parse_feed(self.config)

        print("── Earnings calendar ──")
        self.load_earnings_calendar()

        print("── Catalyst / news signals ──")
        self.load_earnings_symbols()

        print(f"── Starting spike scanner ──  "
              f"min_gap={self._min_gap:.1f}%  sell_gap=-{self._sell_gap_threshold:.1f}%  "
              f"notional=${self._notional:.0f}  sector_confirm={self._sector_confirm}  "
              f"earnings_bypass≥{self._earnings_bypass_gap:.0f}%  "
              f"{'DRY-RUN' if self._dry_run else 'LIVE'}")

        trade_stream = StockDataStream(api_key=self.api_key, secret_key=self.secret_key, feed=feed)
        trade_stream.subscribe_trades(self._on_trade, *symbols)

        news_stream = NewsDataStream(api_key=self.api_key, secret_key=self.secret_key)
        news_stream.subscribe_news(self._on_news, *symbols)

        async def _main() -> None:
            from alpaca.trading.client import TradingClient
            trading = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=bool(self.config["alpaca"].get("paper", True)),
            )

            # Layer 1: check for pre-existing gaps before the first trade arrives
            self._startup_gap_scan()

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
