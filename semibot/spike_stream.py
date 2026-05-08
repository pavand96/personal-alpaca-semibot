"""
Real-time spike detector using Alpaca WebSocket trade stream.

Subscribes to live trades for all watchlist symbols.  On every incoming trade,
computes the gap vs the reference close (yesterday's 4pm close pre-market,
today's 4pm close after-hours) and immediately places an extended-hours limit
order the moment the gap threshold is crossed.

Latency: trade-to-order in ~200ms (stream delivery + one REST call).
No polling loop — the order fires on the first qualifying trade.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockSnapshotRequest

from semibot.bot import SemiMomentumBot, limit_price_for_side, floor_order_qty
from semibot.events import append_event

log = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
_REGULAR_CLOSE_HOUR = 16  # 4:00 PM ET


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
    """WebSocket-based spike detector with immediate order placement."""

    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        self._settings = config["adaptive_semis_allocator"]
        self._min_gap = float(self._settings.get("spike_min_gap_pct", 3.0))
        self._max_gap = float(self._settings.get("spike_max_gap_pct", 20.0))
        self._notional = float(self._settings.get("spike_notional_per_trade", 1000.0))
        self._dry_run = bool(config["risk"].get("dry_run", True))
        self._offset_bps = float(config.get("orders", {}).get("premarket_limit_offset_bps", 25.0))

        self._reference_closes: dict[str, float] = {}
        self._already_ordered: set[str] = set()
        self._order_lock = threading.Lock()  # protect against concurrent order attempts

    def load_references(self) -> None:
        self._reference_closes = fetch_reference_closes(self.config, self.api_key, self.secret_key)

    def _get_held_symbols(self) -> set[str]:
        try:
            bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
            positions = bot.get_positions()
            return {s for s, p in positions.items() if float(getattr(p, "qty", 0)) > 0}
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
            return set()

    def _place_order(self, symbol: str, gap_pct: float, current_price: float) -> None:
        """Place an extended-hours limit order.  Called from the asyncio thread — keep it fast."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        from uuid import uuid4

        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        limit_price = limit_price_for_side(current_price, "buy", self._offset_bps)
        qty = floor_order_qty(self._notional, limit_price)

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
            append_event(self.config["runtime"]["log_file"], {
                "event": "spike_order_submitted",
                "symbol": symbol,
                "gap_pct": round(gap_pct, 2),
                "price": current_price,
                "limit_price": limit_price,
                "qty": qty,
                "order_id": str(result.id),  # type: ignore[union-attr]
            })
        except Exception:
            pass

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

        with self._order_lock:
            if symbol in self._already_ordered:
                return  # double-check after acquiring
            held = self._get_held_symbols()
            if symbol in held:
                self._already_ordered.add(symbol)  # already owned, don't try again
                return
            self._already_ordered.add(symbol)

        ts = datetime.now(MARKET_TZ).strftime("%H:%M:%S")
        print(f"[{ts}] *** SPIKE {symbol}: {gap_pct:+.2f}% vs ref ${ref:.2f} → placing order NOW ***")
        # Run order placement in a thread so we don't block the WebSocket event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._place_order, symbol, gap_pct, price)

    def run(self, execute: bool = False) -> None:
        """Start the WebSocket stream.  Blocks until stopped or market opens."""
        self._dry_run = self._dry_run or not execute
        symbols = self.config["watchlist"]
        feed = _parse_feed(self.config)

        print(f"Starting spike stream for {len(symbols)} symbols (min_gap={self._min_gap:.1f}%, "
              f"notional=${self._notional:.0f}, {'DRY-RUN' if self._dry_run else 'LIVE'})")

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
                stream._run_forever(),  # type: ignore[attr-defined]
                _watchdog(),
            )

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            print("Spike stream stopped.")
        finally:
            stream.stop()
