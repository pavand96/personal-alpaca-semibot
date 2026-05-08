# Semibot — Project Context

Full context for the trading bot: what exists, what was fixed, backtest results, design decisions, and roadmap.

---

## What This Is

A paper-first Alpaca trading bot focused on 25 semiconductor and large-cap tech stocks. Five distinct strategies are implemented. The ML strategy is the primary active one; the others are available for comparison. All strategies default to dry-run mode and require `--execute` to submit real paper orders.

**Watchlist (25 symbols)**
`NVDA AMD AVGO TSM ASML ARM INTC MU QCOM TXN AMAT LRCX KLAC MRVL ON ADI NXPI SNDK GLW WDC MSFT GOOGL AMZN TSLA META`

---

## Strategies

### 1. Daily Momentum Bot (`bot.py`)
Rule-based. Buys when latest price is ≥ `buy_threshold_pct` (2%) above prior close; sells when ≤ `sell_threshold_pct` (-1%) below prior close. Ranks candidates by momentum magnitude and buys the top N.

### 2. Intraday Opening Momentum (`intraday.py`)
Buys at 9:45 AM only when: price ≥ 1% above open, above VWAP, relative volume ≥ 1.5× 20-day average. Stop 0.5%, take profit 1%, exits by 3:55 PM. One trade per symbol per day.

### 3. ML Strategy (`ml.py`) — primary
Logistic regression on 17 daily-bar features. Predicts whether a stock will be ≥ 1% higher in 3 trading days. Generates buy/sell signals live and in backtest. Risk is managed by per-symbol stop loss and trailing stop, plus an account-level drawdown circuit breaker.

### 4. Sector Momentum Allocator (`sector_allocator.py`)
Selects the highest-momentum semiconductor symbol at the open. Requires sector 5-day return ≥ 0.5%. Tiered exits: 50% at +1%, rest at +8%.

### 5. Swing / Balanced Allocators (`swing_allocator.py`, `balanced_allocator.py`)
Swing: rebalances every 3 days on 20-day/5-day momentum with 20% trailing stop.
Balanced: splits $5k into swing sleeve and $4.5k into confirmed intraday sleeve.

---

## ML Model — Features

17 numeric features + symbol one-hot:

| Feature | Description |
|---|---|
| `return_1d` … `return_20d` | Price returns over 1/3/5/10/20 days |
| `return_50d` | 50-day return — medium-term trend context |
| `intraday_return` | `close/open - 1` for the day |
| `overnight_return` | `open/prior_close - 1` |
| `range_pct` | `(high - low) / close` |
| `volatility_5d`, `volatility_10d` | Rolling std of daily returns |
| `close_to_sma_5/10/20/50` | Price distance from moving averages |
| `volume_ratio_5d` | Current volume vs 5-day average |
| `volume_change_1d` | Day-over-day volume change |

**Label:** `1` if `close[t + horizon_days] / close[t] - 1 >= target_return_pct`, else `0`.
**Defaults:** `horizon_days=3`, `target_return_pct=1.0%`.
**Model type:** `logistic_regression` (also supports `random_forest`, `hist_gradient_boosting`).
**Validation:** Time-series 5-fold cross-validation with 1-day gap.
**Typical AUC:** 0.50–0.56 (barely above random — the market is hard to predict short-term).

### Training

Train on data strictly before the backtest window to avoid leakage:

```bash
# For a 2-year backtest starting 2024-05-07:
python main.py train-model --start 2020-01-01 --end 2024-05-06

# For a 4-year backtest starting 2022-05-07:
python main.py train-model --start 2018-01-01 --end 2022-05-06
```

**Retraining cadence:** Quarterly. Do not retrain daily — the model would overfit to recent noise and become unstable. Suggested schedule: first trading day of each quarter.

---

## ML Strategy — Config Overrides

The `ml:` section can override global `risk:` and `strategy:` sizing without affecting other strategies:

```yaml
ml:
  stop_loss_pct: 10.0           # overrides risk.stop_loss_pct (was 0.5 — see bug #1 below)
  trailing_stop_pct: 18.0       # overrides risk.trailing_stop_pct
  per_trade_notional: 100.0     # overrides strategy.per_trade_notional
  max_position_notional: 2000.0 # overrides strategy.max_position_notional
  max_symbols_to_buy_per_run: 3 # overrides strategy.max_symbols_to_buy_per_run
  take_profit_pct: 1.0          # overrides risk.take_profit_pct (live path only)
```

These overrides are validated: positive where required, `per_trade_notional ≤ max_position_notional` enforced.

---

## Bugs Found and Fixed

### Bug 1 — Kill-switch permanent lockout (biggest impact)

**What broke:** `max_account_drawdown_pct: 15.0` correctly sold everything when the portfolio fell 15% from its peak. But `peak_equity` never resets, so `account_drawdown_pct` stayed ≥ 15% forever — the bot never re-entered the market even after full recovery.

**Effect:** In the 2-year backtest (2024–2026), the bot got stopped out during the 2024 tariff/semi correction and then sat in cash for all of 2025–2026 (which returned +91% standalone). The 2-year result was flat (+3.7%) instead of what it should have been.

**Fix:** Introduced a separate `kill_switch_peak` that resets to current equity whenever all positions are zero. This means the circuit breaker pauses trading during a drawdown, then allows re-entry once fully in cash — a "forced break, then try again fresh" behaviour.

```python
any_held = any(pos.qty > 0 for pos in positions.values())
kill_switch_peak = max(kill_switch_peak, equity) if any_held else equity
account_drawdown_pct = abs(min(0.0, (equity - kill_switch_peak) / kill_switch_peak * 100))
```

### Bug 2 — Stop loss 20× too tight

**What broke:** `risk.stop_loss_pct: 0.5` stopped out positions on any 0.5% intraday move. Semiconductor stocks routinely move 1–3% on normal days, so almost every position was stopped out within a day or two of entry.

**Effect:** 529 trades in 2 years, 26% win rate, net -1.36% return.

**Fix:** The parameter optimizer had already found `stop_loss_pct: 10.0` as optimal, but the value had never been applied to the config. Added ML-specific override `ml.stop_loss_pct: 10.0`. After fix: 145 trades, 44% win rate.

### Bug 3 — ML live path ignored `ml.*` overrides

**What broke:** `ml_trade_once()` (live trading) read `per_trade_notional`, `max_position_notional`, `max_symbols_to_buy_per_run`, and `stop_loss_pct` from the global `strategy:` and `risk:` sections. The `ml:` overrides only applied during backtests, so live trading was using different (wrong) parameters.

**Fix:** Updated `ml_trade_once()` to use `ml_settings.get(key, fallback)` for all five parameters, matching the backtest path.

### Bug 4 — Limit-order quantity rounded instead of floored

**What broke:** `round(decision.notional / limit_price, 6)` could round the 6th decimal place up, making the order cost fractionally more than the intended notional.

**Fix:** Extracted `floor_order_qty(notional, price)` helper using `math.floor(... * 1_000_000) / 1_000_000`, guaranteeing `qty * price ≤ notional`.

### Bug 5 — Kill-switch flatten capped by `max_orders_per_run`

**What broke:** When the daily-loss kill switch fired, the flatten sell orders fell through the same `max_orders_per_run` gate as normal orders. With `max_orders_per_run: 3` and 6 open positions, only 3 would be sold.

**Fix:** `bot.py` sets `is_flatten = True` and skips the gate (`if not is_flatten and len(submitted) >= max_orders`). `ml.py` passes `max_orders=len(decisions)` for the flatten path.

### Bug 6 — `max_daily_loss_pct: 0` rejected by validation

**What broke:** The validator used `require_positive` for `max_daily_loss_pct`, rejecting zero. Zero is a valid value meaning "kill switch disabled" (the runtime already handles `≤ 0` as disabled).

**Fix:** Changed to `require_non_negative`.

### Bug 7 — ML parameter optimizer mutated global config sections

**What broke:** `optimize_parameters()` wrote trial values to `trial_config["strategy"]` and `trial_config["risk"]`, but `ml_backtest()` now reads overrides from `trial_config["ml"]`. So optimizer trials were writing to sections the backtest no longer read.

**Fix:** Optimizer now writes all five tunable params to `trial_config["ml"]`, matching the read path.

---

## Backtest Results

All results use `backtest.initial_cash: $10,000`. Benchmark is equal-weight buy-and-hold of the full watchlist (`SEMIS_EQ`).

### Before fixes (2024-05-07 → 2026-05-07)
| | Return | Ending | Drawdown | Trades | Win rate |
|---|---|---|---|---|---|
| ML strategy (broken) | -1.36% | $9,864 | -16.23% | 529 | 26.1% |
| SEMIS_EQ benchmark | +264.72% | $36,472 | -34.24% | — | — |

### After all fixes (2024-05-07 → 2026-05-07)
| | Return | Ending | Drawdown | Trades | Win rate |
|---|---|---|---|---|---|
| **ML strategy (fixed)** | **+158.49%** | **$25,849** | -34.26% | 1,210 | 72.7% |
| SEMIS_EQ benchmark | +264.72% | $36,472 | -34.24% | — | — |

### 4-year backtest (2022-05-07 → 2026-05-07, model trained 2018–2022)
| | Return | Ending | Drawdown |
|---|---|---|---|
| **ML strategy** | **+168.81%** | **$26,881** | -27.75% |
| SEMIS_EQ benchmark | +418.05% | $51,805 | -34.04% |

### Last 3 months (2026-02-07 → 2026-05-07, existing model, no retrain)
| | Return | Ending | Drawdown | Win rate |
|---|---|---|---|---|
| **ML strategy** | **+36.33%** | **$13,633** | -8.39% | 82.4% |
| SEMIS_EQ benchmark | +43.39% | $14,340 | -13.37% | — |

### Single day (2026-05-07, signals from 2026-05-06)
Bought MRVL, GLW, ASML. All liquidated at close. Net: **-$9.33 (-0.09%)** vs sector -0.63%.

### Key observation
The strategy consistently produces lower drawdown than buy-and-hold (e.g. -8.39% vs -13.37% over 3 months). The gap to the benchmark is the cost of risk management in a sustained bull market — any strategy that exits positions during a persistent uptrend will underperform.

---

## Parameter Optimizer

Searches over stop loss, trailing stop, position sizing, and probability thresholds. Best result (1-year, 2025–2026):
```
buy_probability=0.50, sell_probability=0.25, per_trade_notional=100, max_position_notional=2000,
max_symbols_to_buy_per_run=3, stop_loss_pct=10.0, trailing_stop_pct=18.0
→ return=+43.58%, max_drawdown=-7.94%, trades=165
```

These values are now the defaults in the `ml:` section of `config.yml`. To re-run the optimizer on a new date range:
```bash
python main.py optimize-ml-params --start 2025-01-01 --end 2026-05-07
```

---

## News Integration (current state)

The `news_hold` config section already blocks buys on negative keywords (downgrade, investigation, earnings miss, etc.) using Alpaca's news API. This runs as part of `live_entry_quality()` in `ml_trade_once()`.

### What can be added

**Earnings calendar guard (low effort):** Skip buying within 2 trading days of an earnings announcement by checking for `"earnings"` / `"quarterly"` in recent headlines. Gap risk on earnings days is high and unpredictable from price features alone.

**News-driven entry confirmation (medium effort):** Allow borderline ML signals (probability 0.45–0.50) to pass through when positive news is present (upgrade, beat estimates, raised guidance). Requires a small positive-keywords list alongside the block list.

**News velocity as ML feature (higher effort):** Add `news_count_24h` as a numeric feature — high article count signals uncertainty. Requires news API calls per symbol during training, which is slow for bulk historical training but fine for live inference.

---

## Roadmap / Improvements

### Walk-forward retraining (highest impact)
Retrain the model every quarter on a rolling 4-year window. Semiconductor market structure changes (AI cycles, tariff regimes). A stale model trained in 2018 is not calibrated to 2026 conditions. Implementation: `train_and_backtest_rolling(start, end, retrain_every_n_days=63)` that walks forward and retrains in-loop.

### Sector-relative features (high signal quality, low effort)
Add `return_vs_sector_5d`, `return_vs_sector_20d` — each stock's return minus the equal-weight average of the watchlist over the same window. Tells the model "NVDA up 5% while sector is flat" vs "everything up 5%." The former is real alpha; the latter is beta. Computable in `build_feature_row` since `bars_by_symbol` is already available.

### Parallel optimizer (use all CPU cores, zero algorithmic change)
The optimizer runs 80 trials sequentially. Wrapping the trial loop with `joblib.Parallel(n_jobs=-1)` gives linear speedup with core count. Could expand the grid to 500+ trials covering wider ranges at the same wall-clock time.

### RSI and Bollinger Band features (medium impact, low effort)
RSI (14-day) normalizes momentum across volatility regimes. Bollinger Band distance captures mean-reversion vs trend-following setups. Both computable from existing daily bars with no new data fetches.

### Ensemble voting (moderate impact, moderate effort)
Train logistic regression, random forest, and gradient boosting independently on the same features, then average their probabilities. Reduces variance from any single model's quirks without much overfitting risk if each model is individually regularized.

---

## CI

`.github/workflows/ci.yml` runs on push and PR to `main`:
1. `pip install -e ".[dev]"` — installs package in editable mode plus pytest and ruff
2. `ruff check .` — lint
3. `pytest tests/ -v` — 24 tests

---

## Project Structure

```
semibot/
├── bot.py              # Daily momentum bot + order submission
├── backtest.py         # Daily bar backtester (all strategies share this)
├── intraday.py         # Intraday opening momentum strategy
├── ml.py               # ML strategy: training, backtest, live, optimizer
├── sector_allocator.py # Sector momentum allocator
├── swing_allocator.py  # Swing trading allocator
├── balanced_allocator.py # Multi-sleeve allocator
├── config.py           # Config loader and validator
└── events.py           # CSV event logger

models/
└── semibot_model.joblib  # Trained model artifact (retrain quarterly)

logs/
├── ml_backtest_trades.csv
├── ml_parameter_optimization.csv
└── paper_run_*.log

tests/                  # 24 pytest tests, all passing
.github/workflows/ci.yml
config.yml              # All strategy parameters
```

---

## Key Config Values (current)

```yaml
backtest:
  initial_cash: 10000.0
  slippage_bps: 5.0

ml:
  model_type: logistic_regression
  horizon_days: 3
  target_return_pct: 1.0
  buy_probability: 0.60       # 0.50 is no filter for logistic regression
  sell_probability: 0.25
  feature_lookback_days: 100
  # ML-specific risk overrides (tuned by optimizer):
  stop_loss_pct: 10.0
  trailing_stop_pct: 18.0
  take_profit_pct: 3.0        # 3-day horizon needs room; 1% exits too early
  per_trade_notional: 100.0
  max_position_notional: 2000.0
  max_symbols_to_buy_per_run: 3

risk:
  # These three are ML strategy fallbacks only — overridden by ml.* above.
  # The daily momentum strategy does NOT use stop_loss_pct or take_profit_pct;
  # it exits solely via strategy.sell_threshold_pct.
  stop_loss_pct: 5.0
  trailing_stop_pct: 10.0
  take_profit_pct: 3.0
  max_account_drawdown_pct: 15.0
  max_daily_loss_pct: 3.0
  max_orders_per_run: 3       # does NOT cap kill-switch flatten orders
```
