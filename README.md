# Personal Alpaca Semiconductor Bot

This is a small paper-first Alpaca trading script for monitoring semiconductor stocks and optionally placing market orders from a simple momentum rule.

It is not a money printer, and it should not be treated as financial advice. The defaults are intentionally conservative: Alpaca paper trading is enabled, `dry_run` is enabled, and live order submission requires both a config change and the `--execute` flag.

## Setup

```bash
cd /home/pavand96/personal-alpaca-semibot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your Alpaca paper API keys.

## Monitor

```bash
python main.py monitor
```

This prints the latest price, previous daily close, percent change, and held quantity for each configured symbol. It also writes rows to `logs/semibot_events.csv`.

## Dry-Run Trading Check

```bash
python main.py trade-once
```

This evaluates the strategy and logs intended orders without sending them.

## Continuous Daily Monitoring

```bash
python main.py run
```

The bot wakes up every `runtime.interval_seconds` seconds. With the default config it will only act while Alpaca reports that the market is open.

## Enabling Paper Orders

1. Keep `alpaca.paper: true`.
2. Set `risk.dry_run: false` in `config.yml`.
3. Run:

```bash
python main.py trade-once --execute
```

Use paper trading for a while before considering real money. Paper fills and live fills can differ.

## Strategy

For each symbol:

- Buy `strategy.per_trade_notional` when latest trade price is at least `strategy.buy_threshold_pct` above the previous daily close.
- Sell the full held quantity when latest trade price is at or below `strategy.sell_threshold_pct` relative to the previous daily close.
- Respect `strategy.max_position_notional`, `strategy.max_symbols_to_buy_per_run`, and `risk.max_orders_per_run`.

Tune the watchlist and thresholds in `config.yml`.
