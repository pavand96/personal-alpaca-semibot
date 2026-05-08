from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from semibot.bot import parse_data_feed


@dataclass(frozen=True)
class DailyBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestPosition:
    qty: float = 0.0
    avg_entry: float = 0.0
    peak_price: float = 0.0


@dataclass(frozen=True)
class BacktestTrade:
    timestamp: datetime
    symbol: str
    action: str
    qty: float
    price: float
    notional: float
    cash_after: float
    reason: str


@dataclass(frozen=True)
class BenchmarkResult:
    symbol: str
    start_date: date
    end_date: date
    buy_price: float
    sell_price: float
    ending_equity: float
    total_return_pct: float
    max_drawdown_pct: float


@dataclass(frozen=True)
class BacktestResult:
    start: date
    end: date
    starting_cash: float
    ending_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: list[BacktestTrade]
    per_symbol_pnl: dict[str, float]
    benchmarks: list[BenchmarkResult]


class Backtester:
    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.feed = parse_data_feed(config["alpaca"].get("data_feed", "iex"))
        self.adjustment = parse_adjustment(config["backtest"].get("bar_adjustment", "split"))

    def run(self, start: date, end: date) -> BacktestResult:
        settings = self.config["backtest"]
        starting_cash = float(settings["initial_cash"])
        cash = starting_cash
        positions = {symbol: BacktestPosition() for symbol in self.config["watchlist"]}
        trades: list[BacktestTrade] = []
        per_symbol_pnl = {symbol: 0.0 for symbol in self.config["watchlist"]}

        bars_by_symbol = self.fetch_daily_bars(self.config["watchlist"], start, end)
        dates = sorted({bar.timestamp.date() for bars in bars_by_symbol.values() for bar in bars})
        bars_by_date = {
            bar_date: {
                symbol: bars[index]
                for symbol, bars in bars_by_symbol.items()
                for index in range(len(bars))
                if bars[index].timestamp.date() == bar_date
            }
            for bar_date in dates
        }

        buy_threshold = float(self.config["strategy"]["buy_threshold_pct"])
        sell_threshold = float(self.config["strategy"]["sell_threshold_pct"])
        per_trade_notional = float(self.config["strategy"]["per_trade_notional"])
        max_position_notional = float(self.config["strategy"]["max_position_notional"])
        max_buys = int(self.config["strategy"]["max_symbols_to_buy_per_run"])
        max_orders = int(self.config["risk"]["max_orders_per_run"])
        min_price = float(self.config["strategy"]["min_price"])
        max_price = float(self.config["strategy"]["max_price"])
        slippage_bps = float(settings["slippage_bps"])

        equity_curve: list[float] = []
        peak_equity = starting_cash
        max_drawdown_pct = 0.0
        pending_signals: list[dict[str, Any]] = []

        for day_index, current_date in enumerate(dates):
            today_bars = bars_by_date[current_date]

            if pending_signals:
                orders_used = 0
                for signal in pending_signals:
                    if orders_used >= max_orders:
                        break
                    bar = today_bars.get(signal["symbol"])
                    if not bar:
                        continue
                    trade = self.execute_signal(
                        signal=signal,
                        bar=bar,
                        cash=cash,
                        positions=positions,
                        per_symbol_pnl=per_symbol_pnl,
                        slippage_bps=slippage_bps,
                    )
                    if trade:
                        cash = trade.cash_after
                        trades.append(trade)
                        orders_used += 1
                pending_signals = []

            equity = cash + market_value(positions, today_bars)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
            equity_curve.append(equity)

            if day_index == 0:
                continue

            yesterday_bars = bars_by_date[dates[day_index - 1]]
            signals: list[dict[str, Any]] = []
            buys_used = 0

            ranked_symbols = sorted(
                today_bars,
                key=lambda symbol: pct_change(today_bars[symbol].close, yesterday_bars.get(symbol)),
                reverse=True,
            )
            for symbol in ranked_symbols:
                if symbol not in yesterday_bars:
                    continue
                bar = today_bars[symbol]
                previous = yesterday_bars[symbol]
                change_pct = ((bar.close - previous.close) / previous.close) * 100
                position = positions[symbol]
                held_value = position.qty * bar.close

                if bar.close < min_price or bar.close > max_price:
                    continue
                if position.qty > 0 and change_pct <= sell_threshold:
                    signals.append(
                        {
                            "symbol": symbol,
                            "action": "sell",
                            "qty": position.qty,
                            "reason": f"close change {change_pct:.2f}% <= sell threshold {sell_threshold:.2f}%",
                        }
                    )
                    continue
                if (
                    change_pct >= buy_threshold
                    and held_value + per_trade_notional <= max_position_notional
                    and buys_used < max_buys
                ):
                    buys_used += 1
                    signals.append(
                        {
                            "symbol": symbol,
                            "action": "buy",
                            "notional": per_trade_notional,
                            "reason": f"close change {change_pct:.2f}% >= buy threshold {buy_threshold:.2f}%",
                        }
                    )
            pending_signals = signals

        if settings.get("liquidate_at_end", True) and dates:
            final_bars = bars_by_date[dates[-1]]
            for symbol, position in positions.items():
                if position.qty <= 0 or symbol not in final_bars:
                    continue
                trade = self.execute_signal(
                    signal={"symbol": symbol, "action": "sell", "qty": position.qty, "reason": "final liquidation"},
                    bar=final_bars[symbol],
                    cash=cash,
                    positions=positions,
                    per_symbol_pnl=per_symbol_pnl,
                    slippage_bps=slippage_bps,
                    use_close=True,
                )
                if trade:
                    cash = trade.cash_after
                    trades.append(trade)

        ending_equity = cash
        if not settings.get("liquidate_at_end", True) and dates:
            ending_equity = cash + market_value(positions, bars_by_date[dates[-1]])

        total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100
        benchmarks = self.run_benchmarks(start=start, end=end, starting_cash=starting_cash)
        return BacktestResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            per_symbol_pnl=per_symbol_pnl,
            benchmarks=benchmarks,
        )

    def fetch_daily_bars(self, symbols: list[str], start: date, end: date) -> dict[str, list[DailyBar]]:
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, time.min, tzinfo=UTC),
            end=datetime.combine(end, time.max, tzinfo=UTC),
            adjustment=self.adjustment,
            feed=self.feed,
        )
        response = self.data.get_stock_bars(request)
        raw_bars = getattr(response, "data", response)

        bars_by_symbol: dict[str, list[DailyBar]] = {symbol: [] for symbol in symbols}
        for symbol, bars in raw_bars.items():  # type: ignore[union-attr]
            for bar in bars:
                bars_by_symbol.setdefault(symbol, []).append(
                    DailyBar(
                        symbol=symbol,
                        timestamp=bar.timestamp,
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

    def run_benchmarks(self, start: date, end: date, starting_cash: float) -> list[BenchmarkResult]:
        settings = self.config["backtest"]
        symbols = list(settings.get("benchmark_symbols", []))
        if settings.get("benchmark_include_watchlist", True):
            symbols.extend(self.config["watchlist"])
        symbols = dedupe_symbols(symbols)

        if not symbols and not settings.get("benchmark_equal_weight_watchlist", True):
            return []

        slippage_bps = float(settings["slippage_bps"])
        bars_by_symbol = self.fetch_daily_bars(symbols, start, end)
        results: list[BenchmarkResult] = []

        if settings.get("benchmark_equal_weight_watchlist", True):
            basket = equal_weight_benchmark(
                name="SEMIS_EQ",
                symbols=self.config["watchlist"],
                bars_by_symbol=bars_by_symbol,
                starting_cash=starting_cash,
                slippage_bps=slippage_bps,
            )
            if basket:
                results.append(basket)

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])
            if not bars:
                continue

            buy_price = apply_slippage(bars[0].open, "buy", slippage_bps)
            sell_price = apply_slippage(bars[-1].close, "sell", slippage_bps)
            qty = starting_cash / buy_price
            ending_equity = qty * sell_price
            total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100

            peak_equity = starting_cash
            max_drawdown_pct = 0.0
            for bar in bars:
                equity = qty * bar.close
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                    max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)

            results.append(
                BenchmarkResult(
                    symbol=symbol,
                    start_date=bars[0].timestamp.date(),
                    end_date=bars[-1].timestamp.date(),
                    buy_price=buy_price,
                    sell_price=sell_price,
                    ending_equity=ending_equity,
                    total_return_pct=total_return_pct,
                    max_drawdown_pct=max_drawdown_pct,
                )
            )

        return results

    def execute_signal(
        self,
        signal: dict[str, Any],
        bar: DailyBar,
        cash: float,
        positions: dict[str, BacktestPosition],
        per_symbol_pnl: dict[str, float],
        slippage_bps: float,
        use_close: bool = False,
    ) -> BacktestTrade | None:
        symbol = signal["symbol"]
        action = signal["action"]
        position = positions[symbol]
        base_price = bar.close if use_close else bar.open
        price = apply_slippage(base_price, action, slippage_bps)

        if action == "buy":
            notional = min(float(signal["notional"]), cash)
            if notional <= 0:
                return None
            qty = notional / price
            previous_cost = position.avg_entry * position.qty
            position.qty += qty
            position.avg_entry = (previous_cost + notional) / position.qty
            position.peak_price = max(position.peak_price, price)
            cash_after = cash - notional
        else:
            qty = min(float(signal["qty"]), position.qty)
            if qty <= 0:
                return None
            notional = qty * price
            per_symbol_pnl[symbol] += (price - position.avg_entry) * qty
            position.qty -= qty
            if position.qty <= 0:
                position.qty = 0.0
                position.avg_entry = 0.0
                position.peak_price = 0.0
            cash_after = cash + notional

        return BacktestTrade(
            timestamp=bar.timestamp,
            symbol=symbol,
            action=action,
            qty=qty,
            price=price,
            notional=notional,
            cash_after=cash_after,
            reason=signal["reason"],
        )


def print_backtest_result(result: BacktestResult) -> None:
    print(f"Backtest {result.start.isoformat()} to {result.end.isoformat()}")
    print(f"Starting cash: ${result.starting_cash:,.2f}")
    print(f"Ending equity: ${result.ending_equity:,.2f}")
    print(f"Total return: {result.total_return_pct:.2f}%")
    print(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Trades: {len(result.trades)}")

    winners = sum(1 for value in result.per_symbol_pnl.values() if value > 0)
    traded = sum(1 for value in result.per_symbol_pnl.values() if value != 0)
    win_rate = (winners / traded * 100) if traded else 0.0
    print(f"Winning symbols: {winners}/{traded} ({win_rate:.1f}%)")

    if result.benchmarks:
        print("\nBenchmark buy-and-hold comparison")
        for benchmark in result.benchmarks:
            outperformance = result.total_return_pct - benchmark.total_return_pct
            print(
                f"{benchmark.symbol:5} return={benchmark.total_return_pct:7.2f}% "
                f"ending=${benchmark.ending_equity:,.2f} "
                f"max_dd={benchmark.max_drawdown_pct:7.2f}% "
                f"bot_vs={outperformance:+7.2f}%"
            )

    print("\nPer-symbol realized P/L")
    for symbol, pnl in sorted(result.per_symbol_pnl.items(), key=lambda item: item[1], reverse=True):
        if pnl != 0:
            print(f"{symbol:5} ${pnl:,.2f}")

    print("\nRecent trades")
    for trade in result.trades[-10:]:
        print(
            f"{trade.timestamp.date()} {trade.action.upper():4} {trade.symbol:5} "
            f"qty={trade.qty:.4f} price=${trade.price:.2f} notional=${trade.notional:.2f}"
        )


def write_trades_csv(path: str | Path, trades: list[BacktestTrade]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "symbol", "action", "qty", "price", "notional", "cash_after", "reason"],
        )
        writer.writeheader()
        for trade in trades:
            writer.writerow(
                {
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "qty": round(trade.qty, 8),
                    "price": round(trade.price, 4),
                    "notional": round(trade.notional, 2),
                    "cash_after": round(trade.cash_after, 2),
                    "reason": trade.reason,
                }
            )


def pct_change(close: float, previous: DailyBar | None) -> float:
    if previous is None or previous.close <= 0:
        return float("-inf")
    return ((close - previous.close) / previous.close) * 100


def market_value(positions: dict[str, BacktestPosition], bars: dict[str, DailyBar]) -> float:
    return sum(position.qty * bars[symbol].close for symbol, position in positions.items() if symbol in bars)


def apply_slippage(price: float, action: str, slippage_bps: float) -> float:
    multiplier = 1 + (slippage_bps / 10000)
    if action == "sell":
        multiplier = 1 - (slippage_bps / 10000)
    return price * multiplier


def equal_weight_benchmark(
    name: str,
    symbols: list[str],
    bars_by_symbol: dict[str, list[DailyBar]],
    starting_cash: float,
    slippage_bps: float,
) -> BenchmarkResult | None:
    available_symbols = [symbol for symbol in symbols if bars_by_symbol.get(symbol)]
    if not available_symbols:
        return None

    allocation = starting_cash / len(available_symbols)
    quantities: dict[str, float] = {}
    latest_close: dict[str, float] = {}
    start_dates: list[date] = []
    end_dates: list[date] = []

    for symbol in available_symbols:
        bars = bars_by_symbol[symbol]
        buy_price = apply_slippage(bars[0].open, "buy", slippage_bps)
        quantities[symbol] = allocation / buy_price
        latest_close[symbol] = bars[0].close
        start_dates.append(bars[0].timestamp.date())
        end_dates.append(bars[-1].timestamp.date())

    indexed = {
        symbol: {bar.timestamp.date(): bar for bar in bars_by_symbol[symbol]}
        for symbol in available_symbols
    }
    all_dates = sorted({bar_date for bars in indexed.values() for bar_date in bars})

    peak_equity = starting_cash
    max_drawdown_pct = 0.0
    ending_equity = starting_cash
    for bar_date in all_dates:
        for symbol in available_symbols:
            bar = indexed[symbol].get(bar_date)
            if bar:
                latest_close[symbol] = bar.close
        ending_equity = sum(quantities[symbol] * latest_close[symbol] for symbol in available_symbols)
        peak_equity = max(peak_equity, ending_equity)
        if peak_equity > 0:
            drawdown_pct = ((ending_equity - peak_equity) / peak_equity) * 100
            max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)

    for symbol in available_symbols:
        bars = bars_by_symbol[symbol]
        latest_close[symbol] = apply_slippage(bars[-1].close, "sell", slippage_bps)
    ending_equity = sum(quantities[symbol] * latest_close[symbol] for symbol in available_symbols)
    total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100

    return BenchmarkResult(
        symbol=name,
        start_date=max(start_dates),
        end_date=min(end_dates),
        buy_price=0.0,
        sell_price=0.0,
        ending_equity=ending_equity,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
    )


def dedupe_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for symbol in symbols:
        cleaned = symbol.strip().upper()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def parse_adjustment(value: str) -> Adjustment:
    normalized = str(value).strip().upper()
    try:
        return getattr(Adjustment, normalized)
    except AttributeError as error:
        valid = ", ".join(item.name.lower() for item in Adjustment)
        raise ValueError(f"Unsupported bar_adjustment '{value}'. Choose one of: {valid}") from error
