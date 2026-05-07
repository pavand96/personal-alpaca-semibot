from __future__ import annotations

import argparse
import os
import time
from datetime import date

from dotenv import load_dotenv

from semibot.backtest import Backtester, print_backtest_result, write_trades_csv
from semibot.bot import SemiMomentumBot
from semibot.config import load_config
from semibot.ml import MLStrategy, print_ml_signals, print_optimization_results, print_training_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca semiconductor momentum monitor/trader")
    parser.add_argument(
        "command",
        choices=[
            "monitor",
            "trade-once",
            "run",
            "backtest",
            "train-model",
            "ml-backtest",
            "optimize-ml-params",
            "ml-signal",
            "ml-trade-once",
        ],
    )
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

    if args.command in {"backtest", "train-model", "ml-backtest", "optimize-ml-params"}:
        if not args.start:
            raise SystemExit(f"{args.command} requires --start YYYY-MM-DD")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else date.today()
        if end <= start:
            raise SystemExit("--end must be after --start")

    if args.command == "backtest":
        result = Backtester(config, api_key=api_key, secret_key=secret_key).run(start=start, end=end)
        print_backtest_result(result)
        write_trades_csv(config["backtest"]["trades_file"], result.trades)
        print(f"\nTrade history written to {config['backtest']['trades_file']}")
        return

    if args.command == "train-model":
        result = MLStrategy(config, api_key=api_key, secret_key=secret_key).train(start=start, end=end)
        print_training_result(result)
        return

    if args.command == "ml-backtest":
        result = MLStrategy(config, api_key=api_key, secret_key=secret_key).ml_backtest(start=start, end=end)
        print("ML backtest")
        print_backtest_result(result)
        write_trades_csv(config["ml"]["ml_trades_file"], result.trades)
        print(f"\nML trade history written to {config['ml']['ml_trades_file']}")
        return

    if args.command == "optimize-ml-params":
        strategy = MLStrategy(config, api_key=api_key, secret_key=secret_key)
        results = strategy.optimize_parameters(start=start, end=end)
        print_optimization_results(results, config["ml"]["optimizer_results_file"])
        return

    if args.command == "ml-signal":
        strategy = MLStrategy(config, api_key=api_key, secret_key=secret_key)
        signals = strategy.latest_signals()
        print_ml_signals(
            signals,
            buy_probability=float(config["ml"]["buy_probability"]),
            sell_probability=float(config["ml"]["sell_probability"]),
        )
        return

    if args.command == "ml-trade-once":
        MLStrategy(config, api_key=api_key, secret_key=secret_key).ml_trade_once(execute=args.execute)
        return

    bot = SemiMomentumBot(config, api_key=api_key, secret_key=secret_key)

    if args.command == "monitor":
        bot.monitor()
        return

    if args.command == "trade-once":
        bot.trade_once(execute=args.execute)
        return

    interval = int(config["runtime"]["interval_seconds"])
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "running trading check")
        bot.trade_once(execute=args.execute)
        time.sleep(interval)


if __name__ == "__main__":
    main()
