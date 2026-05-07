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

## Backtest

```bash
python main.py backtest --start 2025-01-01 --end 2026-05-01
```

This downloads historical daily bars from Alpaca and simulates the configured momentum rule. Signals are calculated from the daily close compared with the previous daily close, then filled at the next available open with configurable slippage. Results are printed in the terminal and trade history is written to `logs/backtest_trades.csv`.

The backtest also compares the result against semiconductor stock buy-and-hold benchmarks. By default it includes an equal-weight basket of the configured watchlist (`SEMIS_EQ`) and each watchlist stock individually.

## Machine Learning

Train a model on historical daily bars:

```bash
python main.py train-model --start 2020-01-01 --end 2025-01-01
```

The trainer builds daily momentum, volatility, moving-average, range, volume, and symbol features, then validates them with time-ordered splits. The saved model is written to `models/semibot_model.joblib`.

Backtest the saved model on later dates:

```bash
python main.py ml-backtest --start 2025-01-01 --end 2026-05-01
```

Inspect the latest model probabilities:

```bash
python main.py ml-signal
```

Run a dry-run ML trading check:

```bash
python main.py ml-trade-once
```

To allow paper orders, keep `alpaca.paper: true`, set `risk.dry_run: false`, and pass `--execute`:

```bash
python main.py ml-trade-once --execute
```

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
