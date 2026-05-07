from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv

from semibot.bot import SemiMomentumBot
from semibot.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca semiconductor momentum monitor/trader")
    parser.add_argument("command", choices=["monitor", "trade-once", "run"])
    parser.add_argument("--config", default="config.yml", help="Path to YAML config")
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
