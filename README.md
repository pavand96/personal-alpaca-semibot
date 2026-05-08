# Personal Alpaca Semiconductor Bot

This is a small paper-first Alpaca trading script for monitoring semiconductor stocks and optionally placing price-controlled limit orders from momentum rules.

It is not a money printer, and it should not be treated as financial advice. The current config ships with `alpaca.paper: true` and `risk.dry_run: false` — orders are submitted to your **Alpaca paper account** whenever you pass `--execute`. No real money is at risk with these defaults. To disable all order submission entirely, set `risk.dry_run: true`.

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

Run a dry-run adaptive trading check, which is the default live strategy:

```bash
python main.py trade-once
```

The default live path uses the adaptive semi allocator: it ranks the semiconductor watchlist, applies market/risk-off context, buzz/earnings awareness, stop logic, and portfolio exposure caps before submitting any order. The separate ML command remains available for signal inspection and research.

To allow paper orders, keep `alpaca.paper: true`, set `risk.dry_run: false`, and pass `--execute`:

```bash
python main.py trade-once --execute
```

## Continuous Daily Monitoring

```bash
python main.py run
```

The bot wakes up every `runtime.interval_seconds` seconds. With the default config it will only act while Alpaca reports that the market is open.

## Enabling Paper Orders

`risk.dry_run: false` is already set in `config.yml`. Keep `alpaca.paper: true` and pass `--execute` to submit orders to your Alpaca paper account:

```bash
python main.py trade-once --execute
```

Use paper trading for a while before considering real money. Paper fills and live fills can differ.

## Spike Stream (Extended-Hours Gap Capture)

The spike stream watches all 25 watchlist symbols (20 liquid semis + 5 speculative quantum/AI names) via Alpaca WebSocket in real time and fires limit orders within ~200 ms when 3+ symbols simultaneously gap above threshold.

### Manual run

```bash
# Pre-market (run at 3:45 AM ET — 15-min warmup before 4:00 AM open)
scripts/run_spike.sh premarket

# After-hours (run at 4:15 PM ET — stream exits at 7:45 PM ET)
scripts/run_spike.sh afterhours
```

The script is phase-aware: it observes only (no orders) during phase 1 (2026-05-08 – 2026-06-05) and submits paper orders during phase 2 (2026-06-08 – 2026-07-07).

### News monitor

Before each spike session the news monitor should run to pre-populate `logs/news_signals.json` with catalyst signals. The spike stream reads this file at startup to set notional boosts.

```bash
python main.py news-monitor
```

### Crontab setup

Add the following to your crontab (`crontab -e`). All times are ET (`CRON_TZ=America/New_York` must appear once at the top):

```
CRON_TZ=America/New_York

# News monitor — runs before each spike window to refresh catalyst signals
30 3  * * 1-5 cd /home/pavand96/personal-alpaca-semibot && .venv/bin/python main.py news-monitor >> logs/news_monitor_premarket_$(date +\%F).log 2>&1
0  16 * * 1-5 cd /home/pavand96/personal-alpaca-semibot && .venv/bin/python main.py news-monitor >> logs/news_monitor_afterhours_$(date +\%F).log 2>&1

# Spike stream — pre-market (3:45 AM; lock prevents re-entry if already running)
45 3  * * 1-5 /home/pavand96/personal-alpaca-semibot/scripts/run_spike.sh premarket

# Spike stream — after-hours (4:15 PM; exits automatically at 7:45 PM)
15 16 * * 1-5 /home/pavand96/personal-alpaca-semibot/scripts/run_spike.sh afterhours
```

Logs are written to `logs/spike_premarket_YYYY-MM-DD.log` and `logs/spike_afterhours_YYYY-MM-DD.log`. A flock-based lock prevents duplicate scanner processes if a run extends past the next cron tick.

## Strategy

For each symbol:

- Buy `strategy.per_trade_notional` when latest trade price is at least `strategy.buy_threshold_pct` above the previous daily close.
- Sell the full held quantity when latest trade price is at or below `strategy.sell_threshold_pct` relative to the previous daily close.
- Respect `strategy.max_position_notional`, `strategy.max_symbols_to_buy_per_run`, and `risk.max_orders_per_run`.

Tune the watchlist and thresholds in `config.yml`.

### Sell-only mode

Setting `strategy.max_symbols_to_buy_per_run: 0` (or `ml.max_symbols_to_buy_per_run: 0` for the ML strategy) puts the bot into sell-only mode. No new positions are opened, but all exit logic continues to run: stop loss, trailing stop, take profit, exit-before-close, and the daily-loss kill-switch flatten. This is useful for an orderly wind-down without modifying any other config values.
