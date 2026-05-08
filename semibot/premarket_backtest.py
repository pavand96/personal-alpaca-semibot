from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from semibot.backtest import Backtester, BacktestResult, BacktestTrade, DailyBar, print_backtest_result


@dataclass(frozen=True)
class PremarketGapStats:
    opportunity_days: int
    opportunities: int


def run_premarket_gap_backtest(
    config: dict[str, Any],
    api_key: str,
    secret_key: str,
    start: date,
    end: date,
) -> tuple[BacktestResult, PremarketGapStats]:
    """Proxy backtest for premarket spike logic using daily open gaps.

    Alpaca's regular daily bars do not preserve the full premarket tape. This uses
    the regular-session open versus the previous regular close as a conservative
    proxy for "this symbol was already gapping before/at the open."
    """
    backtester = Backtester(config, api_key=api_key, secret_key=secret_key)
    settings = config["adaptive_semis_allocator"]
    starting_cash = float(config["backtest"]["initial_cash"])
    cash = starting_cash
    min_gap = float(settings.get("spike_min_gap_pct", 3.0))
    max_gap = float(settings.get("spike_max_gap_pct", 20.0))
    notional = float(settings.get("spike_notional_per_trade", 1000.0))
    max_orders = int(config["risk"]["max_orders_per_run"])
    slippage_bps = float(config["backtest"]["slippage_bps"])
    slip = slippage_bps / 10_000

    bars_by_symbol = backtester.fetch_daily_bars(config["watchlist"], start - timedelta(days=14), end)
    dates = sorted({bar.timestamp.date() for bars in bars_by_symbol.values() for bar in bars})
    bars_by_date = index_bars_by_date(bars_by_symbol)
    previous_by_symbol: dict[str, DailyBar] = {}
    trades: list[BacktestTrade] = []
    per_symbol_pnl = {symbol: 0.0 for symbol in config["watchlist"]}
    equity_curve: list[float] = []
    peak_equity = starting_cash
    max_drawdown_pct = 0.0
    opportunity_days = 0
    opportunities = 0

    for current_date in dates:
        today = bars_by_date.get(current_date, {})
        if current_date >= start:
            candidates: list[tuple[str, float, DailyBar, DailyBar]] = []
            for symbol, bar in today.items():
                previous = previous_by_symbol.get(symbol)
                if not previous or previous.close <= 0:
                    continue
                gap_pct = ((bar.open / previous.close) - 1) * 100
                if min_gap <= gap_pct <= max_gap:
                    candidates.append((symbol, gap_pct, bar, previous))

            candidates.sort(key=lambda item: item[1], reverse=True)
            if candidates:
                opportunity_days += 1
                opportunities += len(candidates)

            for symbol, gap_pct, bar, _previous in candidates[:max_orders]:
                if cash <= 0:
                    break
                trade_notional = min(notional, cash)
                buy_price = bar.open * (1 + slip)
                sell_price = bar.close * (1 - slip)
                if buy_price <= 0 or sell_price <= 0:
                    continue
                qty = trade_notional / buy_price
                cash_after_buy = cash - trade_notional
                trades.append(
                    BacktestTrade(
                        timestamp=bar.timestamp,
                        symbol=symbol,
                        action="buy",
                        qty=qty,
                        price=buy_price,
                        notional=trade_notional,
                        cash_after=cash_after_buy,
                        reason=f"premarket gap proxy {gap_pct:+.2f}%",
                    )
                )
                sell_notional = qty * sell_price
                pnl = sell_notional - trade_notional
                per_symbol_pnl[symbol] = per_symbol_pnl.get(symbol, 0.0) + pnl
                cash = cash_after_buy + sell_notional
                trades.append(
                    BacktestTrade(
                        timestamp=bar.timestamp,
                        symbol=symbol,
                        action="sell",
                        qty=qty,
                        price=sell_price,
                        notional=sell_notional,
                        cash_after=cash,
                        reason="same-day close exit",
                    )
                )

            peak_equity = max(peak_equity, cash)
            if peak_equity > 0:
                max_drawdown_pct = min(max_drawdown_pct, ((cash - peak_equity) / peak_equity) * 100)
            equity_curve.append(cash)

        for symbol, bar in today.items():
            previous_by_symbol[symbol] = bar

    ending_equity = equity_curve[-1] if equity_curve else cash
    return (
        BacktestResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=((ending_equity - starting_cash) / starting_cash) * 100,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            per_symbol_pnl=per_symbol_pnl,
            benchmarks=backtester.run_benchmarks(start=start, end=end, starting_cash=starting_cash),
        ),
        PremarketGapStats(opportunity_days=opportunity_days, opportunities=opportunities),
    )


def index_bars_by_date(bars_by_symbol: dict[str, list[DailyBar]]) -> dict[date, dict[str, DailyBar]]:
    bars_by_date: dict[date, dict[str, DailyBar]] = {}
    for symbol, bars in bars_by_symbol.items():
        for bar in bars:
            bars_by_date.setdefault(bar.timestamp.date(), {})[symbol] = bar
    return bars_by_date


def print_premarket_gap_result(result: BacktestResult, stats: PremarketGapStats) -> None:
    print("Premarket gap proxy backtest")
    print_backtest_result(result)
    print(f"\nGap opportunity days: {stats.opportunity_days}")
    print(f"Gap opportunities: {stats.opportunities}")
