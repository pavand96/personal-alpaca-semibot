from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

from semibot.intraday import (
    SESSION_OPEN,
    IntradayBar,
    IntradayOpeningMomentumBacktester,
    IntradayTrade,
    calculate_vwap,
    group_bars_by_symbol_day,
    latest_bar_at_or_before,
    parse_clock,
    print_intraday_result,
    write_intraday_trades_csv,
)


@dataclass(frozen=True)
class SectorCandidate:
    symbol: str
    score: float
    entry_bar: IntradayBar
    open_price: float
    vwap: float
    relative_volume: float
    gain_from_open_pct: float
    sector_return_pct: float


class SectorMomentumAllocatorBacktester(IntradayOpeningMomentumBacktester):
    def run(self, start: date, end: date) -> Any:
        settings = self.config["sector_allocator"]
        starting_cash = float(self.config["backtest"]["initial_cash"])
        cash = starting_cash
        peak_equity = starting_cash
        max_drawdown_pct = 0.0
        trades: list[IntradayTrade] = []

        lookback_days = int(settings["average_volume_lookback_days"])
        sector_lookback_days = int(settings["sector_lookback_days"])
        fetch_start = start - timedelta(days=max(lookback_days, sector_lookback_days) * 3)
        bars_by_symbol = self.fetch_minute_bars(self.config["watchlist"], fetch_start, end)
        bars_by_symbol_day = group_bars_by_symbol_day(bars_by_symbol)

        entry_time = parse_clock(settings["entry_time"])
        exit_time = parse_clock(settings["exit_time"])
        max_symbols = int(settings["max_symbols_per_day"])
        base_notional = float(settings["base_notional"])
        max_notional = float(settings["max_notional_per_trade"])
        max_total_notional = float(settings["max_total_notional"])
        min_gain = float(settings["min_open_gain_pct"])
        max_gain = float(settings["max_open_gain_pct"])
        relative_volume_min = float(settings["relative_volume_min"])
        min_sector_return = float(settings["min_sector_lookback_return_pct"])
        stop_loss = float(settings["stop_loss_pct"]) / 100
        first_take_profit = float(settings["first_take_profit_pct"]) / 100
        final_take_profit = float(settings["final_take_profit_pct"]) / 100
        partial_exit_fraction = float(settings["partial_exit_fraction"])

        days = sorted(
            {
                trading_day
                for symbol_days in bars_by_symbol_day.values()
                for trading_day in symbol_days
                if start <= trading_day <= end
            }
        )

        for trading_day in days:
            sector_return = sector_lookback_return_pct(
                bars_by_symbol_day=bars_by_symbol_day,
                trading_day=trading_day,
                lookback_days=sector_lookback_days,
            )
            if sector_return < min_sector_return:
                continue

            candidates: list[SectorCandidate] = []
            for symbol in self.config["watchlist"]:
                bars = bars_by_symbol_day.get(symbol, {}).get(trading_day, [])
                if not bars:
                    continue

                candidate = build_sector_candidate(
                    symbol=symbol,
                    trading_day=trading_day,
                    bars=bars,
                    history=bars_by_symbol_day.get(symbol, {}),
                    entry_time=entry_time,
                    lookback_days=lookback_days,
                    min_gain_pct=min_gain,
                    max_gain_pct=max_gain,
                    relative_volume_min=relative_volume_min,
                    sector_return_pct=sector_return,
                )
                if candidate:
                    candidates.append(candidate)

            deployed = 0.0
            for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)[:max_symbols]:
                notional = min(
                    max_notional,
                    base_notional * max(1.0, min(2.0, candidate.score / 4.0)),
                    max_total_notional - deployed,
                    cash,
                )
                if notional <= 0:
                    continue
                trade = simulate_sector_trade(
                    candidate=candidate,
                    bars=bars_by_symbol_day[candidate.symbol][trading_day],
                    notional=notional,
                    exit_time=exit_time,
                    stop_loss=stop_loss,
                    first_take_profit=first_take_profit,
                    final_take_profit=final_take_profit,
                    partial_exit_fraction=partial_exit_fraction,
                )
                cash -= notional
                cash += notional + trade.pnl
                deployed += notional
                trades.append(trade)

                equity = cash
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                    max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)

        ending_equity = cash
        total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100
        from semibot.intraday import IntradayBacktestResult

        return IntradayBacktestResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
        )


def build_sector_candidate(
    symbol: str,
    trading_day: date,
    bars: list[IntradayBar],
    history: dict[date, list[IntradayBar]],
    entry_time: time,
    lookback_days: int,
    min_gain_pct: float,
    max_gain_pct: float,
    relative_volume_min: float,
    sector_return_pct: float,
) -> SectorCandidate | None:
    open_bar = next((bar for bar in bars if bar.timestamp.time() >= SESSION_OPEN), None)
    entry_bar = latest_bar_at_or_before(bars, entry_time)
    if not open_bar or not entry_bar or entry_bar.timestamp.time() < entry_time:
        return None

    open_price = open_bar.open
    current_price = entry_bar.close
    if open_price <= 0 or current_price <= 0:
        return None

    gain_from_open_pct = ((current_price / open_price) - 1) * 100
    if gain_from_open_pct < min_gain_pct or gain_from_open_pct > max_gain_pct:
        return None

    cumulative_bars = [bar for bar in bars if SESSION_OPEN <= bar.timestamp.time() <= entry_time]
    current_volume = sum(bar.volume for bar in cumulative_bars)
    average_volume = average_cumulative_volume(history, trading_day, entry_time, lookback_days)
    if average_volume <= 0:
        return None
    relative_volume = current_volume / average_volume
    if relative_volume < relative_volume_min:
        return None

    vwap = calculate_vwap(cumulative_bars)
    if vwap <= 0 or current_price <= vwap:
        return None

    vwap_strength_pct = ((current_price / vwap) - 1) * 100
    score = (
        gain_from_open_pct * 1.2
        + vwap_strength_pct * 1.5
        + min(relative_volume, 5.0) * 1.0
        + max(sector_return_pct, 0.0) * 0.25
    )
    return SectorCandidate(
        symbol=symbol,
        score=score,
        entry_bar=entry_bar,
        open_price=open_price,
        vwap=vwap,
        relative_volume=relative_volume,
        gain_from_open_pct=gain_from_open_pct,
        sector_return_pct=sector_return_pct,
    )


def simulate_sector_trade(
    candidate: SectorCandidate,
    bars: list[IntradayBar],
    notional: float,
    exit_time: time,
    stop_loss: float,
    first_take_profit: float,
    final_take_profit: float,
    partial_exit_fraction: float,
) -> IntradayTrade:
    entry_price = candidate.entry_bar.close
    qty = notional / entry_price
    remaining_qty = qty
    realized_pnl = 0.0
    stop_price = entry_price * (1 - stop_loss)
    first_target = entry_price * (1 + first_take_profit)
    final_target = entry_price * (1 + final_take_profit)
    first_target_hit = False
    exit_price = entry_price
    exit_timestamp = candidate.entry_bar.timestamp
    exit_reason = "entry_bar_close"

    for bar in bars:
        if bar.timestamp <= candidate.entry_bar.timestamp:
            continue
        if bar.timestamp.time() > exit_time:
            break

        if bar.low <= stop_price:
            realized_pnl += (stop_price - entry_price) * remaining_qty
            exit_price = stop_price
            exit_timestamp = bar.timestamp
            exit_reason = "breakeven_stop" if first_target_hit else "stop_loss"
            remaining_qty = 0.0
            break

        if not first_target_hit and bar.high >= first_target:
            exit_qty = qty * partial_exit_fraction
            realized_pnl += (first_target - entry_price) * exit_qty
            remaining_qty -= exit_qty
            first_target_hit = True
            stop_price = entry_price

        if first_target_hit and bar.high >= final_target:
            realized_pnl += (final_target - entry_price) * remaining_qty
            exit_price = final_target
            exit_timestamp = bar.timestamp
            exit_reason = "partial_then_final_take_profit"
            remaining_qty = 0.0
            break

        if bar.timestamp.time() >= exit_time:
            realized_pnl += (bar.close - entry_price) * remaining_qty
            exit_price = bar.close
            exit_timestamp = bar.timestamp
            exit_reason = "partial_then_time_exit" if first_target_hit else "time_exit"
            remaining_qty = 0.0
            break

    if remaining_qty > 0:
        eod_bar = latest_bar_at_or_before(bars, exit_time)
        if eod_bar:
            realized_pnl += (eod_bar.close - entry_price) * remaining_qty
            exit_price = eod_bar.close
            exit_timestamp = eod_bar.timestamp
            exit_reason = "partial_then_time_exit" if first_target_hit else "time_exit"

    pnl_pct = (realized_pnl / notional) * 100 if notional else 0.0
    return IntradayTrade(
        entry_time=candidate.entry_bar.timestamp,
        exit_time=exit_timestamp,
        symbol=candidate.symbol,
        qty=qty,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=realized_pnl,
        pnl_pct=pnl_pct,
        open_price=candidate.open_price,
        vwap=candidate.vwap,
        relative_volume=candidate.relative_volume,
        exit_reason=exit_reason,
    )


def sector_lookback_return_pct(
    bars_by_symbol_day: dict[str, dict[date, list[IntradayBar]]],
    trading_day: date,
    lookback_days: int,
) -> float:
    returns: list[float] = []
    for symbol_days in bars_by_symbol_day.values():
        prior_days = sorted(day for day in symbol_days if day < trading_day)
        if len(prior_days) <= lookback_days:
            continue
        start_day = prior_days[-lookback_days - 1]
        end_day = prior_days[-1]
        start_close = symbol_days[start_day][-1].close
        end_close = symbol_days[end_day][-1].close
        if start_close > 0:
            returns.append(((end_close / start_close) - 1) * 100)
    return sum(returns) / len(returns) if returns else 0.0


def average_cumulative_volume(
    history: dict[date, list[IntradayBar]],
    trading_day: date,
    entry_time: time,
    lookback_days: int,
) -> float:
    volumes: list[float] = []
    for historical_day in sorted(day for day in history if day < trading_day)[-lookback_days:]:
        cumulative = sum(
            bar.volume for bar in history[historical_day] if SESSION_OPEN <= bar.timestamp.time() <= entry_time
        )
        if cumulative > 0:
            volumes.append(cumulative)
    return sum(volumes) / len(volumes) if volumes else 0.0


def write_sector_allocator_trades(path: str | Path, trades: list[IntradayTrade]) -> None:
    write_intraday_trades_csv(path, trades)


def print_sector_allocator_result(result: Any) -> None:
    print("Sector momentum allocator")
    print_intraday_result(result)
