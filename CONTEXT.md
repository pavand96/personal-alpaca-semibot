# Semibot — Project Context

Full context for the trading bot: what exists, what was fixed, backtest results, design decisions, and roadmap.

---

## What This Is

A paper-first Alpaca trading bot focused on 20 semiconductor stocks. Five distinct strategies are implemented plus a real-time spike capture stream. All strategies default to dry-run mode and require `--execute` to submit real paper orders.

**Watchlist (20 symbols)**
`NVDA AMD AVGO TSM ASML ARM SMCI MU QCOM TXN AMAT LRCX KLAC MRVL ON ADI NXPI SNDK GLW WDC`

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

### 5. Adaptive Semis Allocator (`adaptive_allocator.py`) — active live strategy
Multi-signal ranking: combines momentum, trend, relative strength, breadth, volume, and gap boost. Risk filters: regime (50/200 SMA), bear block, sector drawdown, market SMA. Includes:
- **Rebound mode**: override of risk filters when sector drawdown ≥ 8% and short-term breadth/returns start recovering. +2.84pp vs baseline in the Apr 2025 selloff.
- **Spike 1-day hold exit**: `spike_tracker.json` records every spike entry; on each live run `get_symbols_to_exit()` generates sell decisions for positions held since yesterday.
- **Earnings notional boost**: 1.5× notional when an earnings catalyst signal is active for the symbol.

### 6. Spike Stream Scanner (`spike_stream.py`) — pre/after-market
Real-time WebSocket scanner running outside regular hours. Three detection layers:

**Layer 1 — Startup snapshot** (3:45 AM)
Immediately after stream launch, snapshots all 20 watchlist symbols. Any symbol already gapping ≥ `spike_min_gap_pct` (5%) is queued. Catches stocks that moved while the process was starting up and any overnight catalyst that became visible before the first trade arrives.

**Layer 2 — News WebSocket** (seconds after headline)
`NewsDataStream` subscribes to all news for watchlist symbols. On any catalyst headline (earnings/upgrade/guidance/acquisition keywords) a snapshot is fetched immediately.
- Gap ≥ `earnings_bypass_gap_pct` (8%) AND earnings keywords → order fires at once (no sector confirmation)
- High-score pre-detected signals (score ≥ 4) lower bypass threshold to 6%
- Smaller gaps → queued; fires when sector_confirm threshold is met

**Layer 3 — Trade WebSocket** (200ms after first trade)
`StockDataStream` receives every live trade for every watchlist symbol. Orders fire once `spike_sector_confirm` (3) distinct symbols have all hit the gap threshold.

**Shared state**: `_pending` dict, `_already_ordered` set, `_order_lock` (threading.Lock). Whichever layer triggers first wins; the others are no-ops.

**Bidirectional on earnings day**: `_earnings_today` (from yfinance) tracks symbols reporting today. Gap ≤ -5% on an earnings day → sell existing position (`_place_sell_order`). `_check_earnings_miss()` is called before buy logic in all three layers.

**Reference closes**:
- Pre-market: `previous_daily_bar.close` (yesterday 4 PM)
- After-hours: `daily_bar.close` (today's locked 4 PM)

---

## Cron Schedule (`cron/semibot.cron`)

```
CRON_TZ=America/New_York

# News monitor — every 15 min all day and night
*/15 * * * 1-5    python main.py news-monitor

# Pre-market spike stream (3:45 AM = 15 min before pre-market opens at 4 AM)
# Fallbacks at 6:45 AM and 8:45 AM (flock prevents duplicates)
45 3 * * 1-5      scripts/run_spike.sh premarket
45 6 * * 1-5      scripts/run_spike.sh premarket
45 8 * * 1-5      scripts/run_spike.sh premarket

# Adaptive semis (regular session: 9:45 AM – 4:10 PM, every 5 min)
45,50,55 9 * * 1-5   scripts/run_ml_trade.sh
*/5 10-15 * * 1-5    scripts/run_ml_trade.sh
0,5,10 16 * * 1-5    scripts/run_ml_trade.sh

# After-hours spike stream (4:15 PM, self-exits at 7:45 PM)
15 16 * * 1-5     scripts/run_spike.sh afterhours
```

The 15-minute head start at 3:45 AM is used to: load earnings calendar (yfinance), load news signals file, fetch reference closes, connect both WebSocket streams. Zero lag when pre-market opens at 4:00:00 AM.

---

## News Monitor (`news_monitor.py`)

Runs every 15 minutes all day and night via cron. Polls Alpaca news API for every watchlist symbol, classifies headlines, and writes `logs/news_signals.json`. The spike stream reads this file at startup so it is pre-warmed with overnight context.

**Signal classification:**
| Type | Keywords | Score |
|------|----------|-------|
| earnings | eps, beat, quarterly, revenue, results, … | 2 + matched count (max 5) |
| upgrade | upgrade, raises, price target, outperform, … | 2 |
| corporate | acquisition, merger, FDA, approval, guidance, … | 2 |
| general | (catch-all with any catalyst keyword) | 1 |

**Signal fields per symbol:** `headline`, `detected_at`, `catalyst_type`, `keywords`, `score`, `bypass_confirm`

Signals expire after `news_signal_ttl_hours` (20h). At startup the spike stream also runs a fresh 24h scan and merges it with the file.

---

## Earnings Calendar (`earnings_calendar.py`)

Wraps yfinance. Called at stream startup (3:45 AM) and by the news monitor cron.

- `fetch_earnings_calendar(symbols, lookahead_days=14)` → `dict[str, date]` — earnings dates per symbol
- `get_reporting_today(symbols)` → `list[str]` — symbols reporting today (bidirectional mode)
- `get_reporting_soon(symbols, days=7)` → `dict[str, date]` — upcoming reporters
- `enrich_signals_with_calendar(signals, symbols, days=14)` — adds `earnings_days_away` and sets `bypass_confirm: true` for today's reporters
- `print_earnings_schedule(symbols, days=14)` — formatted table printed at startup

---

## Spike Tracker (`spike_tracker.py`)

Records every spike stream buy to `logs/spike_tracker.json`. Each entry: `symbol`, `entry_date`, `gap_pct`, `entry_price`.

- `record_spike_entry(path, symbol, gap_pct, entry_price)` — writes/updates entry
- `remove_spike_entry(path, symbol)` — called after earnings-miss sell
- `get_symbols_to_exit(path, today)` → `list[str]` — symbols where `entry_date < today`

`live_decisions()` in `adaptive_allocator.py` calls `get_symbols_to_exit()` on every regular-session run and generates sell decisions for next-day exits.

---

## Rebound Mode

Triggers when: sector drawdown ≥ 8% (`rebound_trigger_drawdown_pct`) AND short-term (5-day) sector return ≥ 2% AND market return ≥ -0.5% AND breadth recovering (≥ 35% of symbols above 5-day SMA).

When active: overrides risk-off filters and bear-block, uses reduced exposure (55%), wider rebalance (2 days), and ranks candidates by `ReboundContext` — weighted combo of 3-day recovery, volume surge, and drawdown depth.

Backtested: +2.84pp over baseline during the April 2025 tariff selloff.

---

## Phased Paper Trading

- **Dry-run phase** (May 8 – Jun 5, 2026): `run_spike.sh` logs only, zero orders
- **Paper phase** (Jun 8 – Jul 7, 2026): `--execute` passed, Alpaca paper account (`paper: true`)
- Cron `flock` prevents duplicate spike stream instances from the fallback launches

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
**Model type:** `hist_gradient_boosting` (handles missing values natively, more robust to class imbalance).
**Validation:** Time-series 5-fold cross-validation with 10-day gap (prevents lookahead across the 3-day prediction horizon).
**Typical AUC:** 0.50–0.56 (barely above random — the market is hard to predict short-term).

### Training

Train on data strictly before the backtest window to avoid leakage:

```bash
# For a 2-year backtest starting 2024-05-07:
python main.py train-model --start 2020-01-01 --end 2024-05-06

# For a 4-year backtest starting 2022-05-07:
python main.py train-model --start 2018-01-01 --end 2022-05-06
```

**Retraining cadence:** Quarterly. Do not retrain daily — the model would overfit to recent noise and become unstable.

---

## ML Strategy — Config Overrides

The `ml:` section overrides global `risk:` and `strategy:` sizing without affecting other strategies:

```yaml
ml:
  stop_loss_pct: 10.0
  trailing_stop_pct: 18.0
  per_trade_notional: 100.0       # or per_trade_pct_of_equity: 0.5
  max_position_notional: 500.0
  max_symbols_to_buy_per_run: 1
  take_profit_pct: 3.0
```

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

**What broke:** `ml_trade_once()` (live trading) read `per_trade_notional`, `max_position_notional`, `max_symbols_to_buy_per_run`, and `stop_loss_pct` from the global `strategy:` and `risk:` sections. The `ml:` overrides only applied during backtests.

**Fix:** Updated `ml_trade_once()` to use `ml_settings.get(key, fallback)` for all five parameters, matching the backtest path.

### Bug 4 — Limit-order quantity rounded instead of floored

**What broke:** `round(decision.notional / limit_price, 6)` could round the 6th decimal place up, making the order cost fractionally more than the intended notional.

**Fix:** Extracted `floor_order_qty(notional, price)` helper using `math.floor(... * 1_000_000) / 1_000_000`.

### Bug 5 — Kill-switch flatten capped by `max_orders_per_run`

**What broke:** When the daily-loss kill switch fired, the flatten sell orders fell through the same `max_orders_per_run` gate as normal orders. With `max_orders_per_run: 3` and 6 open positions, only 3 would be sold.

**Fix:** `bot.py` sets `is_flatten = True` and skips the gate. `ml.py` passes `max_orders=len(decisions)` for the flatten path.

### Bug 6 — `max_daily_loss_pct: 0` rejected by validation

**Fix:** Changed validator from `require_positive` to `require_non_negative`.

### Bug 7 — ML parameter optimizer mutated wrong config sections

**What broke:** `optimize_parameters()` wrote trial values to `trial_config["strategy"]` and `trial_config["risk"]`, but `ml_backtest()` reads overrides from `trial_config["ml"]`. Optimizer trials were writing to sections the backtest no longer read.

**Fix:** Optimizer now writes all five tunable params to `trial_config["ml"]`.

---

## Known Issues / Active Bugs

### Issue 1 — Deadlock in `_startup_gap_scan` (CRITICAL)

`_startup_gap_scan` acquires `_order_lock` in a `with` block and then calls `_check_earnings_miss()` which also tries to acquire `_order_lock`. Python's `threading.Lock` is not reentrant → deadlock on the first earnings-miss-eligible symbol at startup.

**Fix needed:** Move the `_check_earnings_miss` call outside the `with self._order_lock:` block in `_startup_gap_scan`, or restructure the loop so the earnings miss check runs before entering the lock.

### Issue 2 — `_get_held_symbols()` REST call inside `_order_lock` in `_on_trade`

In `_on_trade`, `_get_held_symbols()` is called while holding `_order_lock`. This is a blocking REST call (~200ms) that blocks all other trade and news callbacks queued in the executor. Result: the trade stream falls behind in high-volume pre-market conditions.

**Fix needed:** Cache `_held_symbols` with a ~30s TTL. Refresh asynchronously outside the lock.

### Issue 3 — IEX data feed misses most pre-market trades

`data_feed: iex` is configured. IEX's free tier only covers regular NYSE/NASDAQ hours reliably. Between 4:00 AM and 9:30 AM, most pre-market trades route through ECNs (ARCA, BATS) not covered by IEX. The trade WebSocket (Layer 3) will see very few, possibly zero, pre-market ticks for many symbols.

**Impact:** Layer 3 (trade WebSocket) is nearly blind in pre-market. Only Layer 1 (startup snapshot) and Layer 2 (news WebSocket) will fire reliably.

**Fix needed:** Either upgrade to SIP data feed (paid Alpaca plan) for full pre-market coverage, or document that Layer 3 is effectively a no-op before 9:30 AM and rely on Layers 1 and 2 exclusively.

### Issue 4 — `latest_trade` in startup snapshot may be stale

At 3:45 AM, `snap.latest_trade` from the historical snapshot API may be yesterday's last trade (4:00 PM prior day). If no pre-market trade has occurred yet, the gap calculation uses a stale price and the startup scan will miss symbols that have gapped in news flow but not yet traded.

**Fix needed:** Fall back to `snap.minute_bar` or `snap.daily_bar.open` (which reflects the current pre-market price via the open field on the snapshot) when `latest_trade` is from a prior session.

### Issue 5 — Watchdog `>= 19 and >= 45` bug

`if now.hour >= 19 and now.minute >= 45:` does not correctly express "after 7:45 PM." At 20:00 (8 PM), `now.minute` is 0, so `0 >= 45` is False and the watchdog never stops. The stream would run past 8 PM.

**Fix needed:**
```python
cutoff = now.replace(hour=19, minute=45, second=0, microsecond=0)
if now >= cutoff:
```

### Issue 6 — `sector_confirm=3` blocks single-stock earnings beats

If only NVDA reports a blowout quarter while the sector is mixed, `_pending` never reaches 3 symbols and no order fires — even with a 15% gap. The news bypass helps (≥8% gap skips confirmation) but only if news arrives via the WebSocket. If the gap is 6% and the news WebSocket hasn't fired yet, the startup snapshot will queue NVDA in `_pending` and it will sit there indefinitely.

**Fix:** After a configurable timeout (e.g. 60s) flush any pending symbol in `_earnings_today` regardless of sector confirm count.

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

### Last 3 months (2026-02-07 → 2026-05-07, existing model)
| | Return | Ending | Drawdown | Win rate |
|---|---|---|---|---|
| **ML strategy** | **+36.33%** | **$13,633** | -8.39% | 82.4% |
| SEMIS_EQ benchmark | +43.39% | $14,340 | -13.37% | — |

---

## Parameter Optimizer

```
buy_probability=0.50, sell_probability=0.25, per_trade_notional=100, max_position_notional=2000,
max_symbols_to_buy_per_run=3, stop_loss_pct=10.0, trailing_stop_pct=18.0
→ return=+43.58%, max_drawdown=-7.94%, trades=165
```

To re-run:
```bash
python main.py optimize-ml-params --start 2025-01-01 --end 2026-05-07
```

---

## Project Structure

```
semibot/
├── bot.py                  # Daily momentum bot + order submission
├── backtest.py             # Daily bar backtester (shared by all strategies)
├── intraday.py             # Intraday opening momentum strategy
├── ml.py                   # ML strategy: training, backtest, live, optimizer
├── sector_allocator.py     # Sector momentum allocator
├── swing_allocator.py      # Swing trading allocator
├── balanced_allocator.py   # Multi-sleeve allocator
├── adaptive_allocator.py   # Adaptive semis allocator (active live strategy)
├── spike_stream.py         # Real-time WebSocket spike scanner (pre/after-market)
├── spike_tracker.py        # Persists spike entries → drives 1-day hold exits
├── news_monitor.py         # Background news scanner (runs via cron all day/night)
├── earnings_calendar.py    # yfinance earnings dates (today reporters + lookahead)
├── premarket_backtest.py   # Premarket gap backtest (open vs prev close proxy)
├── config.py               # Config loader and validator
└── events.py               # CSV event logger

models/
└── semibot_model.joblib     # Trained model artifact (retrain quarterly)

logs/
├── news_signals.json        # Written by news-monitor cron, read at stream startup
├── spike_tracker.json       # Records spike buys for 1-day hold exit
├── news_monitor.log         # Cron output for news monitor
├── spike_premarket.log      # Cron output for spike stream
└── semibot_events.csv       # All order events (append-only)

cron/
└── semibot.cron             # Full cron schedule (install with crontab)

scripts/
├── run_spike.sh             # Phased launch (dry-run / paper) with flock
└── run_ml_trade.sh          # Regular-session adaptive semis launcher
```

---

## Key Config Values (spike stream)

```yaml
adaptive_semis_allocator:
  spike_min_gap_pct: 5.0          # enter if gap >= 5%
  spike_max_gap_pct: 20.0         # skip if gap > 20% (reversal risk)
  spike_notional_per_trade: 1000.0
  spike_sector_confirm: 3         # require 3+ symbols gapping simultaneously
  spike_tracker_path: logs/spike_tracker.json
  earnings_notional_multiplier: 1.5
  earnings_bypass_gap_pct: 8.0    # skip sector confirm for large earnings gaps
  news_signals_file: logs/news_signals.json
  news_monitor_lookback_minutes: 30
  news_signal_ttl_hours: 20
  earnings_lookahead_days: 7
  spike_sell_gap_pct: 5.0         # sell position on earnings-day gap <= -5%

alpaca:
  paper: true
  data_feed: iex   # ← upgrade to sip for full pre-market trade coverage
```
