from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from semibot.backtest import (
    Backtester,
    BacktestPosition,
    BacktestResult,
    BacktestTrade,
    DailyBar,
    apply_slippage,
    market_value,
    print_backtest_result,
    write_trades_csv,
)


@dataclass(frozen=True)
class SwingCandidate:
    symbol: str
    score: float
    return_5d_pct: float
    return_20d_pct: float
    volume_ratio: float


class SectorSwingAllocatorBacktester(Backtester):
    def run(self, start: date, end: date) -> BacktestResult:
        settings = self.config["sector_swing_allocator"]
        starting_cash = float(self.config["backtest"]["initial_cash"])
        cash = starting_cash
        positions = {symbol: BacktestPosition() for symbol in self.config["watchlist"]}
        trades: list[BacktestTrade] = []
        per_symbol_pnl = {symbol: 0.0 for symbol in self.config["watchlist"]}

        momentum_lookback = int(settings["momentum_lookback_days"])
        fast_lookback = int(settings["fast_momentum_days"])
        sector_lookback = int(settings["sector_lookback_days"])
        fetch_start = start - timedelta(days=max(momentum_lookback, sector_lookback, 30) * 3)
        bars_by_symbol = self.fetch_daily_bars(self.config["watchlist"], fetch_start, end)
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

        max_symbols = int(settings["max_symbols"])
        max_total_exposure = float(settings["max_total_exposure"])
        min_sector_return = float(settings["min_sector_return_pct"])
        min_symbol_return = float(settings["min_symbol_momentum_pct"])
        rebalance_days = int(settings["rebalance_days"])
        trailing_stop_pct = float(settings["trailing_stop_pct"]) / 100
        slippage_bps = float(self.config["backtest"]["slippage_bps"])
        last_rebalance_index = -rebalance_days

        equity_curve: list[float] = []
        peak_equity = starting_cash
        max_drawdown_pct = 0.0

        for day_index, current_date in enumerate(dates):
            if current_date < start or current_date > end:
                continue
            today_bars = bars_by_date[current_date]

            for symbol, position in positions.items():
                if position.qty <= 0 or symbol not in today_bars:
                    continue
                bar = today_bars[symbol]
                stop_price = position.peak_price * (1 - trailing_stop_pct)
                if bar.open <= stop_price:
                    cash = sell_position(
                        trades=trades,
                        per_symbol_pnl=per_symbol_pnl,
                        cash=cash,
                        position=position,
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        price=apply_slippage(bar.open, "sell", slippage_bps),
                        reason=f"gap below trailing stop {trailing_stop_pct * 100:.1f}%",
                    )
                    continue
                if bar.low <= stop_price:
                    cash = sell_position(
                        trades=trades,
                        per_symbol_pnl=per_symbol_pnl,
                        cash=cash,
                        position=position,
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        price=apply_slippage(stop_price, "sell", slippage_bps),
                        reason=f"trailing stop {trailing_stop_pct * 100:.1f}%",
                    )
                    continue
                position.peak_price = max(position.peak_price, bar.high)

            should_rebalance = day_index - last_rebalance_index >= rebalance_days
            if should_rebalance:
                candidates = rank_swing_candidates(
                    bars_by_symbol=bars_by_symbol,
                    current_date=current_date,
                    momentum_lookback=momentum_lookback,
                    fast_lookback=fast_lookback,
                    sector_lookback=sector_lookback,
                    min_sector_return=min_sector_return,
                    min_symbol_return=min_symbol_return,
                )
                target_symbols = {candidate.symbol for candidate in candidates[:max_symbols]}

                for symbol, position in positions.items():
                    if position.qty <= 0 or symbol in target_symbols or symbol not in today_bars:
                        continue
                    bar = today_bars[symbol]
                    cash = sell_position(
                        trades=trades,
                        per_symbol_pnl=per_symbol_pnl,
                        cash=cash,
                        position=position,
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        price=apply_slippage(bar.open, "sell", slippage_bps),
                        reason="rebalance out of top sector momentum names",
                    )

                selected = candidates[:max_symbols]
                if selected:
                    target_notional = max_total_exposure / len(selected)
                    for candidate in selected:
                        if candidate.symbol not in today_bars:
                            continue
                        position = positions[candidate.symbol]
                        bar = today_bars[candidate.symbol]
                        price = apply_slippage(bar.open, "buy", slippage_bps)
                        held_value = position.qty * price
                        buy_notional = max(0.0, min(target_notional - held_value, cash))
                        if buy_notional <= 0:
                            continue
                        qty = buy_notional / price
                        previous_cost = position.avg_entry * position.qty
                        position.qty += qty
                        position.avg_entry = (previous_cost + buy_notional) / position.qty
                        position.peak_price = max(position.peak_price, bar.high, price)
                        cash -= buy_notional
                        trades.append(
                            BacktestTrade(
                                timestamp=bar.timestamp,
                                symbol=candidate.symbol,
                                action="buy",
                                qty=qty,
                                price=price,
                                notional=buy_notional,
                                cash_after=cash,
                                reason=(
                                    f"sector swing score={candidate.score:.2f} "
                                    f"ret5={candidate.return_5d_pct:.2f}% ret20={candidate.return_20d_pct:.2f}% "
                                    f"vol={candidate.volume_ratio:.2f}x"
                                ),
                            )
                        )
                last_rebalance_index = day_index

            equity = cash + market_value(positions, today_bars)
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)

        if dates:
            final_bars = bars_by_date[dates[-1]]
            for symbol, position in positions.items():
                if position.qty <= 0 or symbol not in final_bars:
                    continue
                bar = final_bars[symbol]
                cash = sell_position(
                    trades=trades,
                    per_symbol_pnl=per_symbol_pnl,
                    cash=cash,
                    position=position,
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    price=apply_slippage(bar.close, "sell", slippage_bps),
                    reason="liquidate at end",
                )

        ending_equity = cash
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


def rank_swing_candidates(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    momentum_lookback: int,
    fast_lookback: int,
    sector_lookback: int,
    min_sector_return: float,
    min_symbol_return: float,
) -> list[SwingCandidate]:
    sector_return = sector_daily_return_pct(bars_by_symbol, current_date, sector_lookback)
    if sector_return < min_sector_return:
        return []

    candidates: list[SwingCandidate] = []
    for symbol, bars in bars_by_symbol.items():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= max(momentum_lookback, fast_lookback, 20):
            continue
        latest = prior[-1]
        slow_base = prior[-momentum_lookback - 1]
        fast_base = prior[-fast_lookback - 1]
        if slow_base.close <= 0 or fast_base.close <= 0:
            continue
        return_20d = ((latest.close / slow_base.close) - 1) * 100
        return_5d = ((latest.close / fast_base.close) - 1) * 100
        if return_5d < min_symbol_return:
            continue
        avg_volume = sum(bar.volume for bar in prior[-20:]) / 20
        volume_ratio = latest.volume / avg_volume if avg_volume > 0 else 1.0
        score = return_20d * 0.7 + return_5d * 1.5 + min(volume_ratio, 3.0) * 2.0 + sector_return * 0.5
        candidates.append(
            SwingCandidate(
                symbol=symbol,
                score=score,
                return_5d_pct=return_5d,
                return_20d_pct=return_20d,
                volume_ratio=volume_ratio,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def sector_daily_return_pct(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    lookback_days: int,
) -> float:
    returns: list[float] = []
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= lookback_days:
            continue
        latest = prior[-1]
        base = prior[-lookback_days - 1]
        if base.close > 0:
            returns.append(((latest.close / base.close) - 1) * 100)
    return sum(returns) / len(returns) if returns else 0.0


def sell_position(
    trades: list[BacktestTrade],
    per_symbol_pnl: dict[str, float],
    cash: float,
    position: BacktestPosition,
    symbol: str,
    timestamp: datetime,
    price: float,
    reason: str,
) -> float:
    qty = position.qty
    notional = qty * price
    pnl = (price - position.avg_entry) * qty
    per_symbol_pnl[symbol] = per_symbol_pnl.get(symbol, 0.0) + pnl
    cash_after = cash + notional
    trades.append(
        BacktestTrade(
            timestamp=timestamp,
            symbol=symbol,
            action="sell",
            qty=qty,
            price=price,
            notional=notional,
            cash_after=cash_after,
            reason=reason,
        )
    )
    position.qty = 0.0
    position.avg_entry = 0.0
    position.peak_price = 0.0
    return cash_after


def print_sector_swing_result(result: BacktestResult) -> None:
    print("Sector swing allocator")
    print_backtest_result(result)


def write_sector_swing_trades(path: str, trades: list[BacktestTrade]) -> None:
    write_trades_csv(path, trades)
