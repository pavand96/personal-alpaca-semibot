"""
Real-time spike detector using Alpaca WebSocket trade stream.

Subscribes to live trades for all watchlist symbols.  On every incoming trade,
computes the gap vs the reference close (yesterday's 4pm close pre-market,
today's 4pm close after-hours) and immediately places an extended-hours limit
order the moment the gap threshold is crossed.

Sector confirmation: orders only fire when spike_sector_confirm distinct
watchlist symbols have all hit the gap threshold in the same session.  When
the threshold is met, all pending symbols are ordered simultaneously.

Earnings boost: recent Alpaca news is checked at startup; symbols with earnings
headlines in the last 24h receive a notional multiplier (earnings_notional_multiplier).

Latency: trade-to-order in ~200ms (stream delivery + one REST call).
No polling loop — the order fires on the first qualifying trade.
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
from alpaca.data.live import StockDataStream
from alpaca.data.requests import NewsRequest, StockSnapshotRequest

from semibot.bot import SemiMomentumBot, floor_order_qty, limit_price_for_side
from semibot.events import append_event
from semibot.spike_tracker import record_spike_entry

log = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
_REGULAR_CLOSE_HOUR = 16  # 4:00 PM ET

_EARNINGS_KEYWORDS = frozenset({
    "earnings", "eps", "quarterly", "quarter",
    "q1", "q2", "q3", "q4", "beats", "beat", "misses", "missed",
    "revenue", "results", "outlook", "guidance", "profit",
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
    """WebSocket-based spike detector with sector confirmation and earnings boost."""

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
        self._dry_run = bool(config["risk"].get("dry_run", True))
        self._offset_bps = float(config.get("orders", {}).get("premarket_limit_offset_bps", 25.0))

        self._reference_closes: dict[str, float] = {}
        self._already_ordered: set[str] = set()
        # Symbols that have qualified but are waiting for sector confirmation
        self._pending: dict[str, tuple[float, float]] = {}  # symbol → (gap_pct, price)
        self._order_lock = threading.Lock()
        self._earnings_symbols: set[str] = set()

    def load_references(self) -> None:
        self._reference_closes = fetch_reference_closes(self.config, self.api_key, self.secret_key)

    def load_earnings_symbols(self) -> None:
        """Check Alpaca news (last 24h) and flag symbols with earnings headlines."""
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
                    if any(kw in headline + " " + summary for kw in _EARNINGS_KEYWORDS):
                        found.add(symbol)
                        break
            except Exception as exc:
                log.warning("News fetch failed for %s: %s", symbol, exc)

        self._earnings_symbols = found
        if found:
            print(f"Earnings signals detected (notional ×{self._earnings_multiplier:.1f}): "
                  f"{', '.join(sorted(found))}")
        else:
            print("No earnings signals in last 24h")

    def _get_held_symbols(self) -> set[str]:
        try:
            bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
            positions = bot.get_positions()
            return {s for s, p in positions.items() if float(getattr(p, "qty", 0)) > 0}
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
            return set()

    def _place_order(self, symbol: str, gap_pct: float, current_price: float) -> None:
        """Place an extended-hours limit order.  Called via run_in_executor — keep it synchronous."""
        from uuid import uuid4

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        notional = self._notional
        if self._earnings_multiplier > 1.0 and symbol in self._earnings_symbols:
            notional *= self._earnings_multiplier
            print(f"  [earnings] {symbol}: notional boosted to ${notional:.0f} (×{self._earnings_multiplier:.1f})")

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

        # Fast-path check before acquiring lock
        if symbol in self._already_ordered:
            return

        to_flush: dict[str, tuple[float, float]] | None = None
        with self._order_lock:
            if symbol in self._already_ordered or symbol in self._pending:
                return
            held = self._get_held_symbols()
            if symbol in held:
                self._already_ordered.add(symbol)
                return
            self._pending[symbol] = (gap_pct, price)
            ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
            n_pending = len(self._pending)
            print(f"[{ts}] SPIKE {symbol}: {gap_pct:+.2f}% vs ref ${ref:.2f} "
                  f"[{n_pending}/{self._sector_confirm} for sector confirm]")
            if n_pending >= self._sector_confirm:
                to_flush = dict(self._pending)
                for sym in to_flush:
                    self._already_ordered.add(sym)
                self._pending.clear()

        if to_flush:
            loop = asyncio.get_event_loop()
            for sym, (sym_gap, sym_price) in to_flush.items():
                await loop.run_in_executor(None, self._place_order, sym, sym_gap, sym_price)

    def run(self, execute: bool = False) -> None:
        """Start the WebSocket stream.  Blocks until stopped or market opens."""
        self._dry_run = self._dry_run or not execute
        symbols = self.config["watchlist"]
        feed = _parse_feed(self.config)

        print("Checking earnings calendar...")
        self.load_earnings_symbols()

        print(f"Starting spike stream for {len(symbols)} symbols  "
              f"min_gap={self._min_gap:.1f}%  notional=${self._notional:.0f}  "
              f"sector_confirm={self._sector_confirm}  "
              f"{'DRY-RUN' if self._dry_run else 'LIVE'}")

        stream = StockDataStream(api_key=self.api_key, secret_key=self.secret_key, feed=feed)
        stream.subscribe_trades(self._on_trade, *symbols)

        async def _main() -> None:
            from alpaca.trading.client import TradingClient
            trading = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=bool(self.config["alpaca"].get("paper", True)),
            )

            async def _watchdog() -> None:
                while True:
                    await asyncio.sleep(30)
                    now = datetime.now(MARKET_TZ)
                    if now.hour >= 19 and now.minute >= 45:
                        print(f"[{now.strftime('%H:%M:%S')}] Extended-hours window closed (7:45pm) — stopping")
                        stream.stop()
                        return
                    try:
                        clock = trading.get_clock()
                        if getattr(clock, "is_open", False):
                            print(f"[{now.strftime('%H:%M:%S')}] Market opened — stopping spike stream")
                            stream.stop()
                            return
                    except Exception as exc:
                        log.warning("Clock check failed: %s", exc)

            await asyncio.gather(
                stream._run_forever(),
                _watchdog(),
            )

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            print("Spike stream stopped.")
        finally:
            stream.stop()
