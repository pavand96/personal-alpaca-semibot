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
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import NewsClient, StockHistoricalDataClient
from alpaca.data.live import NewsDataStream, StockDataStream
from alpaca.data.requests import NewsRequest, StockSnapshotRequest

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

        self._reference_closes: dict[str, float] = {}
        self._already_ordered: set[str] = set()
        # Qualified but waiting for sector_confirm threshold
        self._pending: dict[str, tuple[float, float]] = {}  # symbol → (gap_pct, price)
        self._order_lock = threading.Lock()
        self._earnings_symbols: set[str] = set()

    def load_references(self) -> None:
        self._reference_closes = fetch_reference_closes(self.config, self.api_key, self.secret_key)

    def load_earnings_symbols(self) -> None:
        """Check Alpaca news (last 24h) for earnings/catalyst headlines for each watchlist symbol."""
        now = datetime.now(MARKET_TZ)
        since = now - timedelta(hours=24)
        client = NewsClient(api_key=self.api_key, secret_key=self.secret_key)
        found: set[str] = set()

        for symbol in self.config["watchlist"]:
            try:
                req = NewsRequest(
                    symbols=symbol,
                    start=since.astimezone(UTC),
                    end=now.astimezone(UTC),
                    limit=5,
                )
                resp = client.get_news(req)
                articles = getattr(resp, "news", getattr(resp, "data", []))
                if isinstance(articles, dict):
                    articles = [
                        a for vals in articles.values()
                        for a in (vals if isinstance(vals, list) else [vals])
                    ]
                for article in (articles or []):
                    headline = (getattr(article, "headline", "") or "").lower()
                    summary = (getattr(article, "summary", "") or "").lower()
                    if any(kw in headline + " " + summary for kw in _CATALYST_KEYWORDS):
                        found.add(symbol)
                        break
            except Exception as exc:
                log.warning("News fetch failed for %s: %s", symbol, exc)

        self._earnings_symbols = found
        if found:
            print(f"Catalyst signals detected (notional ×{self._earnings_multiplier:.1f}): "
                  f"{', '.join(sorted(found))}")
        else:
            print("No catalyst signals in last 24h")

    def _get_held_symbols(self) -> set[str]:
        try:
            bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
            positions = bot.get_positions()
            return {s for s, p in positions.items() if float(getattr(p, "qty", 0)) > 0}
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
            return set()

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

    def _try_queue_or_flush(self, symbol: str, gap_pct: float, price: float, source: str) -> list[tuple[str, float, float]]:
        """Thread-safe: add symbol to pending; return list of (sym, gap, price) to order if threshold met.

        Must be called with _order_lock held by the caller.
        """
        self._pending[symbol] = (gap_pct, price)
        ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
        n = len(self._pending)
        print(f"[{ts}] {source} PENDING {symbol}: {gap_pct:+.2f}%  [{n}/{self._sector_confirm} for confirm]")
        if n >= self._sector_confirm:
            to_flush = list(self._pending.items())
            for sym, _ in to_flush:
                self._already_ordered.add(sym)
            self._pending.clear()
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

        with self._order_lock:
            for symbol in self.config["watchlist"]:
                if symbol in self._already_ordered or symbol in self._pending or symbol in held:
                    continue
                snap = snapshots.get(symbol)
                latest = getattr(snap, "latest_trade", None) if snap else None
                if not latest:
                    continue
                price = float(latest.price)
                ref = self._reference_closes.get(symbol)
                if not ref or ref <= 0:
                    continue
                gap_pct = ((price / ref) - 1) * 100
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
            snap = snapshots.get(symbol)
            latest = getattr(snap, "latest_trade", None) if snap else None
            if not latest:
                continue
            price = float(latest.price)
            ref = self._reference_closes.get(symbol)
            if not ref or ref <= 0:
                continue
            gap_pct = ((price / ref) - 1) * 100
            if gap_pct < self._min_gap or gap_pct > self._max_gap:
                continue

            ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
            bypass = is_earnings and gap_pct >= self._earnings_bypass_gap
            to_order: list[tuple[str, float, float]] = []

            with self._order_lock:
                if symbol in self._already_ordered or symbol in self._pending:
                    continue
                if symbol in held:
                    self._already_ordered.add(symbol)
                    continue
                if bypass:
                    print(f"[{ts}] NEWS BYPASS {symbol}: {gap_pct:+.2f}% (earnings ≥{self._earnings_bypass_gap:.0f}% → no sector confirm)")
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

        loop = asyncio.get_event_loop()
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
        if gap_pct < self._min_gap or gap_pct > self._max_gap:
            return

        if symbol in self._already_ordered:
            return

        to_order: list[tuple[str, float, float]] = []
        with self._order_lock:
            if symbol in self._already_ordered or symbol in self._pending:
                return
            held = self._get_held_symbols()
            if symbol in held:
                self._already_ordered.add(symbol)
                return
            to_order = self._try_queue_or_flush(symbol, gap_pct, price, "TRADE")

        if to_order:
            loop = asyncio.get_event_loop()
            for sym, gap, p in to_order:
                await loop.run_in_executor(None, self._place_order, sym, gap, p)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, execute: bool = False) -> None:
        """Start all three detection layers.  Blocks until market opens or 7:45pm."""
        self._dry_run = self._dry_run or not execute
        symbols = self.config["watchlist"]
        feed = _parse_feed(self.config)

        print("Checking catalyst/earnings calendar...")
        self.load_earnings_symbols()

        print(f"Starting spike scanner  "
              f"min_gap={self._min_gap:.1f}%  notional=${self._notional:.0f}  "
              f"sector_confirm={self._sector_confirm}  "
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
                    if now.hour >= 19 and now.minute >= 45:
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
                )
            except Exception as exc:
                log.warning("News stream error (%s) — continuing with trade stream only", exc)
                await asyncio.gather(
                    trade_stream._run_forever(),
                    _watchdog(),
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
