from __future__ import annotations

import argparse
import os
import time
from datetime import date

from dotenv import load_dotenv

from semibot.adaptive_allocator import (
    AdaptiveSemiPortfolioBacktester,
    print_adaptive_allocator_result,
    write_adaptive_allocator_trades,
)
from semibot.backtest import Backtester, print_backtest_result, write_trades_csv
from semibot.balanced_allocator import (
    BalancedSectorAllocatorBacktester,
    print_balanced_allocator_result,
)
from semibot.bot import SemiMomentumBot
from semibot.config import load_config
from semibot.intraday import (
    IntradayOpeningMomentumBacktester,
    print_intraday_result,
    write_intraday_trades_csv,
)
from semibot.ml import (
    MLStrategy,
    print_kelly_result,
    print_ml_signals,
    print_optimization_results,
    print_training_result,
)
from semibot.sector_allocator import (
    SectorMomentumAllocatorBacktester,
    print_sector_allocator_result,
    write_sector_allocator_trades,
)
from semibot.swing_allocator import (
    SectorSwingAllocatorBacktester,
    print_sector_swing_result,
    write_sector_swing_trades,
)

_DATE_REQUIRED_COMMANDS = frozenset([
    "backtest",
    "intraday-backtest",
    "sector-allocator-backtest",
    "sector-balanced-backtest",
    "adaptive-semis-backtest",
    "sector-swing-backtest",
    "train-model",
    "ml-backtest",
    "ml-walk-forward-backtest",
    "kelly-analysis",
    "optimize-ml-params",
])

_ALL_COMMANDS = sorted(
    _DATE_REQUIRED_COMMANDS | {"monitor", "trade-once", "run", "ml-signal", "ml-trade-once", "premarket-run", "premarket-trade"}
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca semiconductor momentum monitor/trader")
    parser.add_argument("command", choices=_ALL_COMMANDS)
    parser.add_argument("--config", default="config.yml", help="Path to YAML config")
    parser.add_argument("--start", help="Backtest start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Backtest end date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit real orders to the configured Alpaca account. Without this, orders are dry-run.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or .env file.")

    config = load_config(args.config)

    start: date | None = None
    end: date | None = None
    if args.command in _DATE_REQUIRED_COMMANDS:
        if not args.start:
            raise SystemExit(f"{args.command} requires --start YYYY-MM-DD")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else date.today()
        if end <= start:
            raise SystemExit("--end must be after --start")

    def cmd_backtest() -> None:
        result = Backtester(config, api_key=api_key, secret_key=secret_key).run(start=start, end=end)
        print_backtest_result(result)
        write_trades_csv(config["backtest"]["trades_file"], result.trades)
        print(f"\nTrade history written to {config['backtest']['trades_file']}")

    def cmd_intraday_backtest() -> None:
        result = IntradayOpeningMomentumBacktester(
            config, api_key=api_key, secret_key=secret_key
        ).run(start=start, end=end)
        print_intraday_result(result)
        write_intraday_trades_csv(config["intraday"]["trades_file"], result.trades)
        print(f"\nIntraday trade history written to {config['intraday']['trades_file']}")

    def cmd_sector_allocator_backtest() -> None:
        result = SectorMomentumAllocatorBacktester(
            config, api_key=api_key, secret_key=secret_key
        ).run(start=start, end=end)
        print_sector_allocator_result(result)
        write_sector_allocator_trades(config["sector_allocator"]["trades_file"], result.trades)
        print(f"\nSector allocator trade history written to {config['sector_allocator']['trades_file']}")

    def cmd_sector_balanced_backtest() -> None:
        result = BalancedSectorAllocatorBacktester(
            config, api_key=api_key, secret_key=secret_key
        ).run(start=start, end=end)
        print_balanced_allocator_result(result)

    def cmd_adaptive_semis_backtest() -> None:
        result = AdaptiveSemiPortfolioBacktester(
            config, api_key=api_key, secret_key=secret_key
        ).run(start=start, end=end)
        print_adaptive_allocator_result(result)
        write_adaptive_allocator_trades(config["adaptive_semis_allocator"]["trades_file"], result.trades)
        print(f"\nAdaptive semi trade history written to {config['adaptive_semis_allocator']['trades_file']}")

    def cmd_sector_swing_backtest() -> None:
        result = SectorSwingAllocatorBacktester(
            config, api_key=api_key, secret_key=secret_key
        ).run(start=start, end=end)
        print_sector_swing_result(result)
        write_sector_swing_trades(config["sector_swing_allocator"]["trades_file"], result.trades)
        print(f"\nSector swing trade history written to {config['sector_swing_allocator']['trades_file']}")

    def cmd_train_model() -> None:
        result = MLStrategy(config, api_key=api_key, secret_key=secret_key).train(start=start, end=end)
        print_training_result(result)

    def cmd_ml_backtest() -> None:
        result = MLStrategy(config, api_key=api_key, secret_key=secret_key).ml_backtest(start=start, end=end)
        print("ML backtest")
        print_backtest_result(result)
        write_trades_csv(config["ml"]["ml_trades_file"], result.trades)
        print(f"\nML trade history written to {config['ml']['ml_trades_file']}")

    def cmd_ml_walk_forward_backtest() -> None:
        result = MLStrategy(config, api_key=api_key, secret_key=secret_key).ml_walk_forward_backtest(
            start=start,
            end=end,
        )
        print("ML walk-forward backtest")
        print_backtest_result(result)
        write_trades_csv(config["ml"]["ml_trades_file"], result.trades)
        print(f"\nML walk-forward trade history written to {config['ml']['ml_trades_file']}")

    def cmd_kelly_analysis() -> None:
        result, kelly = MLStrategy(config, api_key=api_key, secret_key=secret_key).kelly_analysis(
            start=start, end=end
        )
        print_kelly_result(result, kelly)

    def cmd_optimize_ml_params() -> None:
        strategy = MLStrategy(config, api_key=api_key, secret_key=secret_key)
        results = strategy.optimize_parameters(start=start, end=end)
        print_optimization_results(results, config["ml"]["optimizer_results_file"])

    def cmd_ml_signal() -> None:
        strategy = MLStrategy(config, api_key=api_key, secret_key=secret_key)
        signals = strategy.latest_signals()
        print_ml_signals(
            signals,
            buy_probability=float(config["ml"]["buy_probability"]),
            sell_probability=float(config["ml"]["sell_probability"]),
        )

    def cmd_ml_trade_once() -> None:
        MLStrategy(config, api_key=api_key, secret_key=secret_key).ml_trade_once(execute=args.execute)

    def cmd_monitor() -> None:
        SemiMomentumBot(config, api_key=api_key, secret_key=secret_key).monitor()

    def cmd_trade_once() -> None:
        AdaptiveSemiPortfolioBacktester(config, api_key=api_key, secret_key=secret_key).trade_once(execute=args.execute)

    def cmd_premarket_trade() -> None:
        AdaptiveSemiPortfolioBacktester(config, api_key=api_key, secret_key=secret_key).trade_once(execute=args.execute, premarket=True)

    def cmd_premarket_run() -> None:
        """Run pre-market gap scanner in a loop until market opens."""
        bot = AdaptiveSemiPortfolioBacktester(config, api_key=api_key, secret_key=secret_key)
        interval = int(config["runtime"].get("premarket_interval_seconds", 300))
        while True:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "pre-market gap scan")
            try:
                from semibot.bot import SemiMomentumBot
                clock_bot = SemiMomentumBot(config, api_key=api_key, secret_key=secret_key)
                clock = clock_bot.trading.get_clock()
                if getattr(clock, "is_open", False):
                    print("Market opened — pre-market loop exiting.")
                    break
                bot.trade_once(execute=args.execute, premarket=True)
            except Exception as exc:
                print(f"Pre-market scan error: {exc}")
            time.sleep(interval)

    def cmd_run() -> None:
        bot = AdaptiveSemiPortfolioBacktester(config, api_key=api_key, secret_key=secret_key)
        interval = int(config["runtime"]["interval_seconds"])
        while True:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "running trading check")
            try:
                bot.trade_once(execute=args.execute)
            except Exception as exc:
                print(f"Error during trading check, retrying next interval: {exc}")
            time.sleep(interval)

    commands: dict[str, object] = {
        "backtest": cmd_backtest,
        "intraday-backtest": cmd_intraday_backtest,
        "sector-allocator-backtest": cmd_sector_allocator_backtest,
        "sector-balanced-backtest": cmd_sector_balanced_backtest,
        "adaptive-semis-backtest": cmd_adaptive_semis_backtest,
        "sector-swing-backtest": cmd_sector_swing_backtest,
        "train-model": cmd_train_model,
        "ml-backtest": cmd_ml_backtest,
        "ml-walk-forward-backtest": cmd_ml_walk_forward_backtest,
        "kelly-analysis": cmd_kelly_analysis,
        "optimize-ml-params": cmd_optimize_ml_params,
        "ml-signal": cmd_ml_signal,
        "ml-trade-once": cmd_ml_trade_once,
        "monitor": cmd_monitor,
        "trade-once": cmd_trade_once,
        "premarket-trade": cmd_premarket_trade,
        "premarket-run": cmd_premarket_run,
        "run": cmd_run,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
