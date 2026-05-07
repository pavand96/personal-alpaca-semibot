from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from semibot.backtest import parse_adjustment
from semibot.bot import parse_data_feed


MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)


@dataclass(frozen=True)
class IntradayBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class IntradayTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    open_price: float
    vwap: float
    relative_volume: float
    exit_reason: str


@dataclass(frozen=True)
class IntradayBacktestResult:
    start: date
    end: date
    starting_cash: float
    ending_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: list[IntradayTrade]


class IntradayOpeningMomentumBacktester:
    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.feed = parse_data_feed(config["alpaca"].get("data_feed", "iex"))
        self.adjustment = parse_adjustment(config["backtest"].get("bar_adjustment", "split"))

    def run(self, start: date, end: date) -> IntradayBacktestResult:
        settings = self.config["intraday"]
        starting_cash = float(self.config["backtest"]["initial_cash"])
        cash = starting_cash
        peak_equity = starting_cash
        max_drawdown_pct = 0.0
        trades: list[IntradayTrade] = []

        lookback_days = int(settings["average_volume_lookback_days"])
        fetch_start = start - timedelta(days=lookback_days * 3)
        bars_by_symbol = self.fetch_minute_bars(self.config["watchlist"], fetch_start, end)
        bars_by_symbol_day = group_bars_by_symbol_day(bars_by_symbol)

        entry_time = parse_clock(settings["entry_time"])
        exit_time = parse_clock(settings["exit_time"])
        per_trade_notional = float(settings["per_trade_notional"])
        min_gain = float(settings["min_open_gain_pct"]) / 100
        max_gain = float(settings["max_open_gain_pct"]) / 100
        relative_volume_min = float(settings["relative_volume_min"])
        stop_loss = float(settings["stop_loss_pct"]) / 100
        take_profit = float(settings["take_profit_pct"]) / 100

        days = sorted(
            {
                trading_day
                for symbol_days in bars_by_symbol_day.values()
                for trading_day in symbol_days
                if start <= trading_day <= end
            }
        )
        traded_symbols_by_day: dict[date, set[str]] = {}

        for trading_day in days:
            candidates: list[tuple[float, str, IntradayTrade]] = []
            for symbol in self.config["watchlist"]:
                if symbol in traded_symbols_by_day.get(trading_day, set()):
                    continue

                bars = bars_by_symbol_day.get(symbol, {}).get(trading_day, [])
                if not bars:
                    continue

                setup = build_entry_setup(
                    symbol=symbol,
                    trading_day=trading_day,
                    bars=bars,
                    history=bars_by_symbol_day.get(symbol, {}),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    lookback_days=lookback_days,
                    min_gain=min_gain,
                    max_gain=max_gain,
                    relative_volume_min=relative_volume_min,
                    per_trade_notional=per_trade_notional,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                if setup:
                    candidates.append((setup.relative_volume, symbol, setup))

            for _, symbol, trade in sorted(candidates, reverse=True):
                if cash < per_trade_notional:
                    continue
                cash -= per_trade_notional
                cash += per_trade_notional + trade.pnl
                trades.append(trade)
                traded_symbols_by_day.setdefault(trading_day, set()).add(symbol)

                equity = cash
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                    max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)

        ending_equity = cash
        total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100
        return IntradayBacktestResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
        )

    def fetch_minute_bars(self, symbols: list[str], start: date, end: date) -> dict[str, list[IntradayBar]]:
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(start, time.min, tzinfo=timezone.utc),
            end=datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc),
            adjustment=self.adjustment,
            feed=self.feed,
        )
        response = self.data.get_stock_bars(request)
        raw_bars = getattr(response, "data", response)

        bars_by_symbol: dict[str, list[IntradayBar]] = {symbol: [] for symbol in symbols}
        for symbol, bars in raw_bars.items():
            for bar in bars:
                timestamp = bar.timestamp.astimezone(MARKET_TZ)
                if not is_regular_session(timestamp.time()):
                    continue
                bars_by_symbol.setdefault(symbol, []).append(
                    IntradayBar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(getattr(bar, "volume", 0.0) or 0.0),
                    )
                )

        for bars in bars_by_symbol.values():
            bars.sort(key=lambda item: item.timestamp)
        return {symbol: bars for symbol, bars in bars_by_symbol.items() if bars}


def build_entry_setup(
    symbol: str,
    trading_day: date,
    bars: list[IntradayBar],
    history: dict[date, list[IntradayBar]],
    entry_time: time,
    exit_time: time,
    lookback_days: int,
    min_gain: float,
    max_gain: float,
    relative_volume_min: float,
    per_trade_notional: float,
    stop_loss: float,
    take_profit: float,
) -> IntradayTrade | None:
    open_bar = next((bar for bar in bars if bar.timestamp.time() >= SESSION_OPEN), None)
    entry_bar = latest_bar_at_or_before(bars, entry_time)
    if not open_bar or not entry_bar or entry_bar.timestamp.time() < entry_time:
        return None

    open_price = open_bar.open
    current_price = entry_bar.close
    gain_from_open = (current_price / open_price) - 1 if open_price else 0.0
    if gain_from_open < min_gain or gain_from_open > max_gain:
        return None

    cumulative_bars = [bar for bar in bars if SESSION_OPEN <= bar.timestamp.time() <= entry_time]
    current_volume = sum(bar.volume for bar in cumulative_bars)
    average_volume = average_cumulative_volume(history, trading_day, entry_time, lookback_days)
    if average_volume <= 0:
        return None
    relative_volume = current_volume / average_volume
    if relative_volume <= relative_volume_min:
        return None

    vwap = calculate_vwap(cumulative_bars)
    if current_price <= vwap:
        return None

    qty = per_trade_notional / current_price
    stop_price = current_price * (1 - stop_loss)
    target_price = current_price * (1 + take_profit)
    exit_price = current_price
    exit_timestamp = entry_bar.timestamp
    exit_reason = "entry_bar_close"

    for bar in bars:
        bar_time = bar.timestamp.time()
        if bar.timestamp <= entry_bar.timestamp:
            continue
        if bar_time > exit_time:
            break

        if bar.low <= stop_price:
            exit_price = stop_price
            exit_timestamp = bar.timestamp
            exit_reason = "stop_loss"
            break
        if bar.high >= target_price:
            exit_price = target_price
            exit_timestamp = bar.timestamp
            exit_reason = "take_profit"
            break
        if bar_time >= exit_time:
            exit_price = bar.close
            exit_timestamp = bar.timestamp
            exit_reason = "time_exit"
            break

    pnl = (exit_price - current_price) * qty
    pnl_pct = (exit_price / current_price - 1) * 100
    return IntradayTrade(
        entry_time=entry_bar.timestamp,
        exit_time=exit_timestamp,
        symbol=symbol,
        qty=qty,
        entry_price=current_price,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        open_price=open_price,
        vwap=vwap,
        relative_volume=relative_volume,
        exit_reason=exit_reason,
    )


def group_bars_by_symbol_day(
    bars_by_symbol: dict[str, list[IntradayBar]],
) -> dict[str, dict[date, list[IntradayBar]]]:
    grouped: dict[str, dict[date, list[IntradayBar]]] = {}
    for symbol, bars in bars_by_symbol.items():
        for bar in bars:
            grouped.setdefault(symbol, {}).setdefault(bar.timestamp.date(), []).append(bar)
    return grouped


def latest_bar_at_or_before(bars: list[IntradayBar], clock: time) -> IntradayBar | None:
    eligible = [bar for bar in bars if bar.timestamp.time() <= clock]
    return eligible[-1] if eligible else None


def average_cumulative_volume(
    history: dict[date, list[IntradayBar]],
    trading_day: date,
    entry_time: time,
    lookback_days: int,
) -> float:
    volumes: list[float] = []
    for historical_day in sorted(day for day in history if day < trading_day)[-lookback_days:]:
        bars = history[historical_day]
        cumulative = sum(
            bar.volume for bar in bars if SESSION_OPEN <= bar.timestamp.time() <= entry_time
        )
        if cumulative > 0:
            volumes.append(cumulative)
    return sum(volumes) / len(volumes) if volumes else 0.0


def calculate_vwap(bars: list[IntradayBar]) -> float:
    total_volume = sum(bar.volume for bar in bars)
    if total_volume <= 0:
        return 0.0
    return sum(((bar.high + bar.low + bar.close) / 3) * bar.volume for bar in bars) / total_volume


def is_regular_session(clock: time) -> bool:
    return time(9, 30) <= clock <= time(16, 0)


def parse_clock(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(int(hour), int(minute))


def print_intraday_result(result: IntradayBacktestResult) -> None:
    print(f"Intraday backtest {result.start.isoformat()} to {result.end.isoformat()}")
    print(f"Starting cash: ${result.starting_cash:,.2f}")
    print(f"Ending equity: ${result.ending_equity:,.2f}")
    print(f"Total return: {result.total_return_pct:.2f}%")
    print(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Trades: {len(result.trades)}")

    winners = sum(1 for trade in result.trades if trade.pnl > 0)
    win_rate = (winners / len(result.trades) * 100) if result.trades else 0.0
    print(f"Win rate: {win_rate:.1f}%")

    pnl_by_symbol: dict[str, float] = {}
    for trade in result.trades:
        pnl_by_symbol[trade.symbol] = pnl_by_symbol.get(trade.symbol, 0.0) + trade.pnl

    print("\nPer-symbol realized P/L")
    for symbol, pnl in sorted(pnl_by_symbol.items(), key=lambda item: item[1], reverse=True):
        print(f"{symbol:5} ${pnl:,.2f}")

    print("\nRecent trades")
    for trade in result.trades[-10:]:
        print(
            f"{trade.entry_time.date()} {trade.symbol:5} entry=${trade.entry_price:.2f} "
            f"exit=${trade.exit_price:.2f} pnl=${trade.pnl:.2f} "
            f"rv={trade.relative_volume:.2f} reason={trade.exit_reason}"
        )


def write_intraday_trades_csv(path: str | Path, trades: list[IntradayTrade]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "entry_time",
                "exit_time",
                "symbol",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "pnl_pct",
                "open_price",
                "vwap",
                "relative_volume",
                "exit_reason",
            ],
        )
        writer.writeheader()
        for trade in trades:
            writer.writerow(
                {
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "symbol": trade.symbol,
                    "qty": round(trade.qty, 8),
                    "entry_price": round(trade.entry_price, 4),
                    "exit_price": round(trade.exit_price, 4),
                    "pnl": round(trade.pnl, 2),
                    "pnl_pct": round(trade.pnl_pct, 4),
                    "open_price": round(trade.open_price, 4),
                    "vwap": round(trade.vwap, 4),
                    "relative_volume": round(trade.relative_volume, 4),
                    "exit_reason": trade.exit_reason,
                }
            )
