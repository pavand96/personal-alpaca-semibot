from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any

from semibot.sector_allocator import SectorMomentumAllocatorBacktester
from semibot.swing_allocator import SectorSwingAllocatorBacktester


@dataclass(frozen=True)
class BalancedAllocatorResult:
    start: date
    end: date
    starting_cash: float
    ending_equity: float
    total_return_pct: float
    approximate_max_drawdown_pct: float
    swing_return_pct: float
    confirmed_return_pct: float
    today_like_return_pct: float | None = None


class BalancedSectorAllocatorBacktester:
    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key

    def run(self, start: date, end: date) -> BalancedAllocatorResult:
        settings = self.config["sector_balanced_allocator"]
        starting_cash = float(self.config["backtest"]["initial_cash"])
        core_cash = float(settings["core_swing_exposure"])
        confirmed_cash = float(settings["confirmed_intraday_exposure"])
        if core_cash + confirmed_cash > starting_cash:
            raise ValueError("balanced allocator exposures cannot exceed starting cash")

        swing_config = deepcopy(self.config)
        swing_config["backtest"]["initial_cash"] = core_cash
        swing_config["sector_swing_allocator"]["max_total_exposure"] = core_cash * float(
            settings["exposure_usage_pct"]
        )

        confirmed_config = deepcopy(self.config)
        confirmed_config["backtest"]["initial_cash"] = confirmed_cash
        confirmed_notional = confirmed_cash * float(settings["exposure_usage_pct"])
        confirmed_config["sector_allocator"]["base_notional"] = confirmed_notional
        confirmed_config["sector_allocator"]["max_notional_per_trade"] = confirmed_notional
        confirmed_config["sector_allocator"]["max_total_notional"] = confirmed_notional

        swing_result = SectorSwingAllocatorBacktester(
            swing_config,
            self.api_key,
            self.secret_key,
        ).run(start=start, end=end)
        confirmed_result = SectorMomentumAllocatorBacktester(
            confirmed_config,
            self.api_key,
            self.secret_key,
        ).run(start=start, end=end)

        idle_cash = starting_cash - core_cash - confirmed_cash
        ending_equity = swing_result.ending_equity + confirmed_result.ending_equity + idle_cash
        total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100
        approximate_dd = (
            swing_result.max_drawdown_pct * core_cash
            + confirmed_result.max_drawdown_pct * confirmed_cash
        ) / starting_cash

        return BalancedAllocatorResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            approximate_max_drawdown_pct=approximate_dd,
            swing_return_pct=swing_result.total_return_pct,
            confirmed_return_pct=confirmed_result.total_return_pct,
        )


def print_balanced_allocator_result(result: BalancedAllocatorResult) -> None:
    print("Balanced sector allocator")
    print(f"Backtest {result.start.isoformat()} to {result.end.isoformat()}")
    print(f"Starting cash: ${result.starting_cash:,.2f}")
    print(f"Ending equity: ${result.ending_equity:,.2f}")
    print(f"Total return: {result.total_return_pct:.2f}%")
    print(f"Approx max drawdown: {result.approximate_max_drawdown_pct:.2f}%")
    print(f"Swing sleeve return: {result.swing_return_pct:.2f}%")
    print(f"Confirmed intraday sleeve return: {result.confirmed_return_pct:.2f}%")
