from __future__ import annotations

import argparse
import os
import time
from datetime import date

from dotenv import load_dotenv

from semibot.backtest import Backtester, print_backtest_result, write_trades_csv
from semibot.bot import SemiMomentumBot
from semibot.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca semiconductor momentum monitor/trader")
    parser.add_argument("command", choices=["monitor", "trade-once", "run", "backtest"])
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

    if args.command == "backtest":
        if not args.start:
            raise SystemExit("Backtest requires --start YYYY-MM-DD")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else date.today()
        if end <= start:
            raise SystemExit("--end must be after --start")

        result = Backtester(config, api_key=api_key, secret_key=secret_key).run(start=start, end=end)
        print_backtest_result(result)
        write_trades_csv(config["backtest"]["trades_file"], result.trades)
        print(f"\nTrade history written to {config['backtest']['trades_file']}")
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
