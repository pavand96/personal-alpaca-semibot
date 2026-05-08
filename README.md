# Personal Alpaca Semiconductor Bot

This is a small paper-first Alpaca trading script for monitoring semiconductor stocks and optionally placing price-controlled limit orders from momentum rules.

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

## Intraday Opening Momentum

Backtest the 9:45 AM opening-momentum rule:

```bash
python main.py intraday-backtest --start 2025-11-07 --end 2026-05-07
```

The rule buys at 9:45 AM only when price is at least 1% above the open, above VWAP, relative cumulative volume is above 1.5x the prior 20-day average for the same time window, and the move from open is not already above 4%. It uses a 0.5% stop loss, 1.0% take profit, exits by 3:55 PM, and allows at most one trade per symbol per day.

## Adaptive Semi Portfolio

Backtest a higher-exposure semi rotation strategy:

```bash
python main.py adaptive-semis-backtest --start 2025-01-01 --end 2026-01-01
```

This strategy deploys most of the account into the strongest few semiconductor names when sector momentum is positive, rebalances every few days, uses hard/trailing stops, and moves to cash when sector momentum weakens. A broad-market risk filter also blocks new entries when QQQ/SPY momentum and trend are weak or the semi basket is in a deeper drawdown.

The strategy can retain existing winners through normal rebalances when the position has a profit cushion and momentum remains intact. The optional `buzz_earnings_overlay` boosts adaptive candidates with positive recent Alpaca news, blocks candidates with negative news spikes, and avoids new entries around earnings dates from `data/earnings_calendar.csv` unless an existing position already has a configured profit cushion. The earnings CSV should contain `symbol,date` or `symbol,earnings_date` columns.

## Sector Watchlists

`config.yml` includes research watchlists under `sector_watchlists` for energy, software, cybersecurity, quantum, AI infrastructure, cloud/data, and defense/space. These lists are not active trading universes by default; the active bot watchlist remains the top-level semiconductor `watchlist` until a strategy explicitly opts into another sector.

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

Backtest walk-forward retraining, currently every 15 days on a rolling 4-year training window:

```bash
python main.py ml-walk-forward-backtest --start 2025-01-01 --end 2026-01-01
```

Estimate Kelly sizing from closed ML backtest trades:

```bash
python main.py kelly-analysis --start 2025-11-07 --end 2026-05-07
```

The report prints full, half, and quarter Kelly fractions plus suggested per-trade notionals. Treat full Kelly as aggressive; fractional Kelly is usually more practical for noisy, correlated stock strategies.

Search historical parameter settings with drawdown and trade-count penalties:

```bash
python main.py optimize-ml-params --start 2025-01-01 --end 2026-05-01
```

The optimizer tests probability thresholds, position sizing, max buys, stop loss, and trailing stop settings from `config.yml`. It writes ranked results to `logs/ml_parameter_optimization.csv`.
When live trading uses percent-of-equity sizing, the optimizer disables that setting during fixed-dollar notional sweeps so the optimized notional grid is meaningful.

Inspect the latest model probabilities:

```bash
python main.py ml-signal
```

Run a dry-run ML trading check:

```bash
python main.py ml-trade-once
```

Live ML buys are filtered through the stricter intraday quality gate: at least 1% above the open, above VWAP, at least 1.5x relative volume, not already more than 4% above the open, and no existing position in that symbol. The news filter blocks configured negative keywords; for paper testing it defaults to fail-open when the news API is unavailable.

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

### Sell-only mode

Setting `strategy.max_symbols_to_buy_per_run: 0` (or `ml.max_symbols_to_buy_per_run: 0` for the ML strategy) puts the bot into sell-only mode. No new positions are opened, but all exit logic continues to run: stop loss, trailing stop, take profit, exit-before-close, and the daily-loss kill-switch flatten. This is useful for an orderly wind-down without modifying any other config values.
