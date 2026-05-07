from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from semibot.backtest import (
    BacktestPosition,
    BacktestResult,
    BacktestTrade,
    Backtester,
    DailyBar,
    apply_slippage,
    market_value,
)
from semibot.bot import Decision, SemiMomentumBot, format_decision
from semibot.intraday import MARKET_TZ, SESSION_OPEN, calculate_vwap, is_regular_session


NUMERIC_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "intraday_return",
    "overnight_return",
    "range_pct",
    "volatility_5d",
    "volatility_10d",
    "close_to_sma_5",
    "close_to_sma_10",
    "close_to_sma_20",
    "volume_ratio_5d",
    "volume_change_1d",
]
CATEGORICAL_FEATURES = ["symbol"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class MLSignal:
    symbol: str
    probability: float
    price: float
    timestamp: datetime
    action: str
    reason: str


@dataclass(frozen=True)
class ValidationFold:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    rows: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    roc_auc: float | None


@dataclass(frozen=True)
class TrainingResult:
    rows: int
    positive_rate: float
    folds: list[ValidationFold]
    model_path: str


@dataclass(frozen=True)
class OptimizationResult:
    score: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    feasible: bool
    params: dict[str, Any]


@dataclass(frozen=True)
class KellyResult:
    sample_count: int
    win_rate: float
    average_win_pct: float
    average_loss_pct: float
    payoff_ratio: float
    full_kelly_fraction: float
    half_kelly_fraction: float
    quarter_kelly_fraction: float
    suggested_full_notional: float
    suggested_half_notional: float
    suggested_quarter_notional: float


@dataclass(frozen=True)
class LiveIntradayBar:
    timestamp: datetime
    high: float
    low: float
    close: float
    volume: float


class MLStrategy:
    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        self.backtester = Backtester(config, api_key=api_key, secret_key=secret_key)
        self.news = NewsClient(api_key=api_key, secret_key=secret_key)

    def train(self, start: date, end: date) -> TrainingResult:
        frame = self.load_training_frame(start=start, end=end)
        if frame.empty:
            raise RuntimeError("No ML training rows were built from historical bars.")

        estimator = build_estimator(
            model_type=self.config["ml"]["model_type"],
            random_state=int(self.config["ml"]["random_state"]),
        )
        folds = validate_time_series(
            frame=frame,
            estimator=estimator,
            n_splits=int(self.config["ml"]["validation_splits"]),
            gap=int(self.config["ml"]["validation_gap_days"]),
        )

        estimator.fit(frame[MODEL_FEATURES], frame["target"])
        artifact = {
            "estimator": estimator,
            "feature_columns": MODEL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "symbols": self.config["watchlist"],
            "horizon_days": int(self.config["ml"]["horizon_days"]),
            "target_return_pct": float(self.config["ml"]["target_return_pct"]),
            "trained_start": start.isoformat(),
            "trained_end": end.isoformat(),
            "trained_rows": len(frame),
            "positive_rate": float(frame["target"].mean()),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        model_path = Path(self.config["ml"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_path)

        return TrainingResult(
            rows=len(frame),
            positive_rate=float(frame["target"].mean()),
            folds=folds,
            model_path=str(model_path),
        )

    def ml_backtest(self, start: date, end: date) -> BacktestResult:
        artifact = load_model(self.config["ml"]["model_path"])
        settings = self.config["backtest"]
        ml_settings = self.config["ml"]
        starting_cash = float(settings["initial_cash"])
        cash = starting_cash
        positions = {symbol: BacktestPosition() for symbol in self.config["watchlist"]}
        trades: list[BacktestTrade] = []
        per_symbol_pnl = {symbol: 0.0 for symbol in self.config["watchlist"]}

        fetch_start = start - timedelta(days=int(ml_settings["feature_lookback_days"]) * 3)
        bars_by_symbol = self.backtester.fetch_daily_bars(self.config["watchlist"], fetch_start, end)
        dates = sorted({bar.timestamp.date() for bars in bars_by_symbol.values() for bar in bars})
        bars_by_date = index_bars_by_date(bars_by_symbol)

        max_orders = int(self.config["risk"]["max_orders_per_run"])
        max_buys = int(self.config["strategy"]["max_symbols_to_buy_per_run"])
        per_trade_notional = float(self.config["strategy"]["per_trade_notional"])
        max_position_notional = float(self.config["strategy"]["max_position_notional"])
        min_price = float(self.config["strategy"]["min_price"])
        max_price = float(self.config["strategy"]["max_price"])
        buy_probability = float(ml_settings["buy_probability"])
        sell_probability = float(ml_settings["sell_probability"])
        slippage_bps = float(settings["slippage_bps"])
        stop_loss_pct = float(self.config["risk"]["stop_loss_pct"])
        trailing_stop_pct = float(self.config["risk"]["trailing_stop_pct"])
        max_account_drawdown_pct = float(self.config["risk"]["max_account_drawdown_pct"])

        peak_equity = starting_cash
        max_drawdown_pct = 0.0
        pending_signals: list[dict[str, Any]] = []

        for current_date in dates:
            if current_date < start:
                continue

            today_bars = bars_by_date.get(current_date, {})
            if pending_signals:
                orders_used = 0
                for signal in pending_signals:
                    if orders_used >= max_orders:
                        break
                    bar = today_bars.get(signal["symbol"])
                    if not bar:
                        continue
                    trade = self.backtester.execute_signal(
                        signal=signal,
                        bar=bar,
                        cash=cash,
                        positions=positions,
                        per_symbol_pnl=per_symbol_pnl,
                        slippage_bps=slippage_bps,
                    )
                    if trade:
                        cash = trade.cash_after
                        trades.append(trade)
                        orders_used += 1
                pending_signals = []

            equity = cash + market_value(positions, today_bars)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
            account_drawdown_pct = abs(min(0.0, ((equity - peak_equity) / peak_equity) * 100)) if peak_equity else 0.0

            update_position_peaks(positions, today_bars)
            risk_signals = risk_exit_signals(
                positions=positions,
                bars=today_bars,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
            )
            if account_drawdown_pct >= max_account_drawdown_pct:
                risk_signals.extend(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "qty": position.qty,
                        "reason": f"account drawdown {account_drawdown_pct:.2f}% >= limit {max_account_drawdown_pct:.2f}%",
                    }
                    for symbol, position in positions.items()
                    if position.qty > 0
                )
                pending_signals = dedupe_sell_signals(risk_signals)
                continue

            feature_rows = feature_rows_for_date(bars_by_symbol, current_date)
            if not feature_rows:
                continue
            predictions = predict_rows(artifact, pd.DataFrame(feature_rows))

            signals: list[dict[str, Any]] = list(risk_signals)
            risk_sell_symbols = {signal["symbol"] for signal in risk_signals}
            buys_used = 0
            for signal in sorted(predictions, key=lambda item: item.probability, reverse=True):
                if signal.symbol in risk_sell_symbols:
                    continue
                position = positions[signal.symbol]
                held_value = position.qty * signal.price
                if signal.price < min_price or signal.price > max_price:
                    continue
                if position.qty > 0 and signal.probability <= sell_probability:
                    signals.append(
                        {
                            "symbol": signal.symbol,
                            "action": "sell",
                            "qty": position.qty,
                            "reason": signal.reason,
                        }
                    )
                    continue
                if (
                    signal.probability >= buy_probability
                    and held_value + per_trade_notional <= max_position_notional
                    and buys_used < max_buys
                ):
                    adjusted_notional = correlation_adjusted_notional(
                        symbol=signal.symbol,
                        base_notional=per_trade_notional,
                        held_symbols=[
                            symbol
                            for symbol, held_position in positions.items()
                            if held_position.qty > 0 and symbol != signal.symbol
                        ],
                        bars_by_symbol=bars_by_symbol,
                        as_of_date=current_date,
                        settings=self.config["portfolio"],
                    )
                    if adjusted_notional <= 0:
                        continue
                    buys_used += 1
                    signals.append(
                        {
                            "symbol": signal.symbol,
                            "action": "buy",
                            "notional": adjusted_notional,
                            "reason": f"{signal.reason}; correlation-adjusted notional ${adjusted_notional:.2f}",
                        }
                    )
            pending_signals = signals

        if settings.get("liquidate_at_end", True) and dates:
            final_bars = bars_by_date.get(dates[-1], {})
            for symbol, position in positions.items():
                if position.qty <= 0 or symbol not in final_bars:
                    continue
                trade = self.backtester.execute_signal(
                    signal={"symbol": symbol, "action": "sell", "qty": position.qty, "reason": "final liquidation"},
                    bar=final_bars[symbol],
                    cash=cash,
                    positions=positions,
                    per_symbol_pnl=per_symbol_pnl,
                    slippage_bps=slippage_bps,
                    use_close=True,
                )
                if trade:
                    cash = trade.cash_after
                    trades.append(trade)

        ending_equity = cash
        total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100
        benchmarks = self.backtester.run_benchmarks(start=start, end=end, starting_cash=starting_cash)
        return BacktestResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            per_symbol_pnl=per_symbol_pnl,
            benchmarks=benchmarks,
        )

    def optimize_parameters(self, start: date, end: date) -> list[OptimizationResult]:
        ml_settings = self.config["ml"]
        max_trials = int(ml_settings["optimizer_max_trials"])
        drawdown_penalty = float(ml_settings["optimizer_return_drawdown_penalty"])
        trade_penalty = float(ml_settings["optimizer_trade_penalty"])
        max_allowed_drawdown = float(self.config["risk"]["max_account_drawdown_pct"])

        grid = product(
            ml_settings["buy_probability_grid"],
            ml_settings["sell_probability_grid"],
            ml_settings["per_trade_notional_grid"],
            ml_settings["max_position_notional_grid"],
            ml_settings["max_symbols_to_buy_per_run_grid"],
            ml_settings["stop_loss_pct_grid"],
            ml_settings["trailing_stop_pct_grid"],
        )

        results: list[OptimizationResult] = []
        for index, (
            buy_probability,
            sell_probability,
            per_trade_notional,
            max_position_notional,
            max_symbols_to_buy_per_run,
            stop_loss_pct,
            trailing_stop_pct,
        ) in enumerate(grid, start=1):
            if index > max_trials:
                break
            if float(sell_probability) >= float(buy_probability):
                continue
            if float(per_trade_notional) > float(max_position_notional):
                continue

            trial_config = deepcopy(self.config)
            trial_config["ml"]["buy_probability"] = float(buy_probability)
            trial_config["ml"]["sell_probability"] = float(sell_probability)
            trial_config["strategy"]["per_trade_notional"] = float(per_trade_notional)
            trial_config["strategy"]["max_position_notional"] = float(max_position_notional)
            trial_config["strategy"]["max_symbols_to_buy_per_run"] = int(max_symbols_to_buy_per_run)
            trial_config["risk"]["stop_loss_pct"] = float(stop_loss_pct)
            trial_config["risk"]["trailing_stop_pct"] = float(trailing_stop_pct)

            result = MLStrategy(trial_config, api_key=self.api_key, secret_key=self.secret_key).ml_backtest(
                start=start,
                end=end,
            )
            feasible = abs(result.max_drawdown_pct) <= max_allowed_drawdown
            score = (
                result.total_return_pct
                + drawdown_penalty * result.max_drawdown_pct
                - trade_penalty * (len(result.trades) / 100)
            )
            if not feasible:
                score -= 1000

            results.append(
                OptimizationResult(
                    score=score,
                    total_return_pct=result.total_return_pct,
                    max_drawdown_pct=result.max_drawdown_pct,
                    trades=len(result.trades),
                    feasible=feasible,
                    params={
                        "buy_probability": float(buy_probability),
                        "sell_probability": float(sell_probability),
                        "per_trade_notional": float(per_trade_notional),
                        "max_position_notional": float(max_position_notional),
                        "max_symbols_to_buy_per_run": int(max_symbols_to_buy_per_run),
                        "stop_loss_pct": float(stop_loss_pct),
                        "trailing_stop_pct": float(trailing_stop_pct),
                    },
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        write_optimization_csv(self.config["ml"]["optimizer_results_file"], results)
        return results

    def kelly_analysis(self, start: date, end: date) -> tuple[BacktestResult, KellyResult]:
        result = self.ml_backtest(start=start, end=end)
        returns = closed_trade_returns(result.trades)
        if not returns:
            raise RuntimeError("No closed ML trades available for Kelly analysis.")

        wins = [value for value in returns if value > 0]
        losses = [-value for value in returns if value < 0]
        win_rate = len(wins) / len(returns)
        average_win = float(np.mean(wins)) if wins else 0.0
        average_loss = float(np.mean(losses)) if losses else 0.0
        payoff_ratio = average_win / average_loss if average_loss > 0 else 0.0

        full_kelly = 0.0
        if payoff_ratio > 0:
            full_kelly = win_rate - ((1 - win_rate) / payoff_ratio)
        full_kelly = max(0.0, min(full_kelly, 1.0))

        starting_cash = float(self.config["backtest"]["initial_cash"])
        return (
            result,
            KellyResult(
                sample_count=len(returns),
                win_rate=win_rate,
                average_win_pct=average_win * 100,
                average_loss_pct=average_loss * 100,
                payoff_ratio=payoff_ratio,
                full_kelly_fraction=full_kelly,
                half_kelly_fraction=full_kelly / 2,
                quarter_kelly_fraction=full_kelly / 4,
                suggested_full_notional=starting_cash * full_kelly,
                suggested_half_notional=starting_cash * full_kelly / 2,
                suggested_quarter_notional=starting_cash * full_kelly / 4,
            ),
        )

    def latest_signals(self, end: date | None = None) -> list[MLSignal]:
        artifact = load_model(self.config["ml"]["model_path"])
        end_date = end or date.today()
        fetch_start = end_date - timedelta(days=int(self.config["ml"]["feature_lookback_days"]) * 3)
        bars_by_symbol = self.backtester.fetch_daily_bars(self.config["watchlist"], fetch_start, end_date)
        latest_date = max(bar.timestamp.date() for bars in bars_by_symbol.values() for bar in bars)
        rows = feature_rows_for_date(bars_by_symbol, latest_date)
        return sorted(predict_rows(artifact, pd.DataFrame(rows)), key=lambda item: item.probability, reverse=True)

    def ml_trade_once(self, execute: bool = False) -> list[Decision]:
        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        bot.assert_account_can_trade()
        if self.config["risk"]["require_market_open"] and not bot.trading.get_clock().is_open:
            print("Market is closed. No ML orders submitted.")
            return []

        positions = bot.get_positions()
        dry_run = bool(self.config["risk"]["dry_run"]) or not execute
        max_orders = int(self.config["risk"]["max_orders_per_run"])
        per_trade_notional = float(self.config["strategy"]["per_trade_notional"])
        max_position_notional = float(self.config["strategy"]["max_position_notional"])
        max_total_position_notional = float(self.config["risk"].get("max_total_position_notional", 10000.0))
        total_position_notional = sum(
            abs(float(getattr(position, "market_value", 0.0) or 0.0))
            for position in positions.values()
        )
        max_buys = int(self.config["strategy"]["max_symbols_to_buy_per_run"])
        buy_probability = float(self.config["ml"]["buy_probability"])
        sell_probability = float(self.config["ml"]["sell_probability"])
        stop_loss_pct = float(self.config["risk"]["stop_loss_pct"])
        take_profit_pct = float(self.config["risk"].get("take_profit_pct", 1.0))
        exit_before_close = parse_hhmm_time(str(self.config["risk"].get("exit_before_close", "15:55")))
        now_et = datetime.now(MARKET_TZ).time()
        should_exit_for_close = now_et >= exit_before_close

        decisions: list[Decision] = []
        risk_sell_symbols: set[str] = set()
        for position in positions.values():
            unrealized_plpc = float(getattr(position, "unrealized_plpc", 0.0) or 0.0)
            unrealized_pct = unrealized_plpc * 100
            qty = float(position.qty)
            if unrealized_plpc <= -(stop_loss_pct / 100):
                risk_sell_symbols.add(position.symbol)
                decisions.append(
                    Decision(
                        position.symbol,
                        "sell",
                        f"live stop loss {unrealized_plpc:.2%} <= -{stop_loss_pct:.2f}%",
                        qty=qty,
                    )
                )
                continue
            if should_exit_for_close:
                risk_sell_symbols.add(position.symbol)
                decisions.append(
                    Decision(
                        position.symbol,
                        "sell",
                        f"exit before close at {exit_before_close.strftime('%H:%M')}",
                        qty=qty,
                    )
                )
                continue
            if unrealized_pct >= take_profit_pct:
                risk_sell_symbols.add(position.symbol)
                decisions.append(
                    Decision(
                        position.symbol,
                        "sell",
                        f"take profit {unrealized_pct:.2f}% >= {take_profit_pct:.2f}%",
                        qty=qty,
                    )
                )
                continue

        signals = self.latest_signals()
        quality_by_symbol = self.live_entry_quality(
            [
                signal.symbol
                for signal in signals
                if signal.probability >= buy_probability and signal.symbol not in positions
            ]
        )

        buys_used = 0
        queued_buy_notional = 0.0
        for signal in signals:
            if signal.symbol in risk_sell_symbols:
                continue
            position = positions.get(signal.symbol)
            held_qty = float(position.qty) if position else 0.0
            held_value = abs(float(position.market_value)) if position else 0.0
            if held_qty > 0 and signal.probability <= sell_probability:
                decisions.append(Decision(signal.symbol, "sell", signal.reason, qty=held_qty))
                continue
            quality_passed, quality_reason = quality_by_symbol.get(
                signal.symbol,
                (False, "blocked by live entry quality filter"),
            )
            if (
                signal.probability >= buy_probability
                and quality_passed
                and held_qty <= 0
                and held_value + per_trade_notional <= max_position_notional
                and total_position_notional + queued_buy_notional + per_trade_notional <= max_total_position_notional
                and buys_used < max_buys
            ):
                buys_used += 1
                queued_buy_notional += per_trade_notional
                decisions.append(
                    Decision(signal.symbol, "buy", f"{signal.reason}; {quality_reason}", notional=per_trade_notional)
                )
            elif signal.probability >= buy_probability and held_qty <= 0 and not quality_passed:
                decisions.append(
                    Decision(signal.symbol, "hold", quality_reason)
                )
            elif (
                signal.probability >= buy_probability
                and total_position_notional + queued_buy_notional + per_trade_notional > max_total_position_notional
            ):
                decisions.append(
                    Decision(
                        signal.symbol,
                        "hold",
                        f"portfolio cap ${max_total_position_notional:,.2f} would be exceeded",
                    )
                )

        submitted: list[Decision] = []
        for decision in decisions:
            if len(submitted) >= max_orders:
                break
            if decision.action == "hold":
                print(f"ML HOLD {format_decision(decision)}")
                bot.log_decision(decision, event="ml_hold")
                continue
            if dry_run:
                print(f"ML DRY RUN {format_decision(decision)}")
                bot.log_decision(decision, event="ml_dry_run_order")
            else:
                bot.submit_order(decision)
                bot.log_decision(decision, event="ml_submitted_order")
            submitted.append(decision)

        if not submitted:
            print("No ML trade signals crossed the configured probability thresholds.")
        return submitted

    def live_entry_quality(self, symbols: list[str]) -> dict[str, tuple[bool, str]]:
        settings = self.config.get("live_entry_filter", {})
        if not bool(settings.get("enabled", True)):
            return {symbol: (True, "entry filter disabled") for symbol in symbols}
        if not symbols:
            return {}

        now = datetime.now(MARKET_TZ)
        min_time = parse_hhmm_time(str(settings.get("min_time", "09:45")))
        if now.time() < min_time:
            return {
                symbol: (False, f"blocked before entry filter start {min_time.strftime('%H:%M')}")
                for symbol in symbols
            }

        lookback_days = int(settings.get("average_volume_lookback_days", 20))
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(now.date() - timedelta(days=lookback_days * 3), time.min, tzinfo=timezone.utc),
            end=now.astimezone(timezone.utc),
            adjustment=self.backtester.adjustment,
            feed=self.backtester.feed,
        )
        response = self.backtester.data.get_stock_bars(request)
        raw_bars = getattr(response, "data", response)

        result: dict[str, tuple[bool, str]] = {}
        for symbol in symbols:
            bars = [
                bar
                for bar in raw_bars.get(symbol, [])
                if is_regular_session(bar.timestamp.astimezone(MARKET_TZ).time())
            ]
            if not bars:
                result[symbol] = (False, "blocked: no intraday bars")
                continue

            bars_by_day: dict[date, list[Any]] = {}
            for bar in bars:
                local_date = bar.timestamp.astimezone(MARKET_TZ).date()
                bars_by_day.setdefault(local_date, []).append(bar)
            for day_bars in bars_by_day.values():
                day_bars.sort(key=lambda item: item.timestamp)

            today_bars = bars_by_day.get(now.date(), [])
            if not today_bars:
                result[symbol] = (False, "blocked: no bars for today")
                continue
            open_bar = next(
                (bar for bar in today_bars if bar.timestamp.astimezone(MARKET_TZ).time() >= SESSION_OPEN),
                None,
            )
            latest_bar = today_bars[-1]
            if not open_bar or latest_bar.timestamp.astimezone(MARKET_TZ).time() < min_time:
                result[symbol] = (False, f"blocked before entry filter start {min_time.strftime('%H:%M')}")
                continue

            open_price = float(open_bar.open)
            current_price = float(latest_bar.close)
            if open_price <= 0 or current_price <= 0:
                result[symbol] = (False, "blocked: invalid open/current price")
                continue

            cumulative_bars = [
                bar
                for bar in today_bars
                if SESSION_OPEN <= bar.timestamp.astimezone(MARKET_TZ).time() <= latest_bar.timestamp.astimezone(MARKET_TZ).time()
            ]
            vwap = calculate_vwap(
                [
                    _intraday_bar_like(
                        timestamp=bar.timestamp.astimezone(MARKET_TZ),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(getattr(bar, "volume", 0.0) or 0.0),
                    )
                    for bar in cumulative_bars
                ]
            )
            gain_from_open_pct = ((current_price / open_price) - 1) * 100
            current_volume = sum(float(getattr(bar, "volume", 0.0) or 0.0) for bar in cumulative_bars)
            average_volume = average_intraday_cumulative_volume(
                bars_by_day=bars_by_day,
                trading_day=now.date(),
                cutoff=latest_bar.timestamp.astimezone(MARKET_TZ).time(),
                lookback_days=lookback_days,
            )
            relative_volume = current_volume / average_volume if average_volume > 0 else 0.0

            if bool(settings.get("require_above_open", True)) and current_price <= open_price:
                result[symbol] = (
                    False,
                    f"blocked below open price=${current_price:.2f} open=${open_price:.2f}",
                )
                continue
            if bool(settings.get("require_above_vwap", True)) and (vwap <= 0 or current_price <= vwap):
                result[symbol] = (
                    False,
                    f"blocked below VWAP price=${current_price:.2f} vwap=${vwap:.2f}",
                )
                continue
            if gain_from_open_pct < float(settings.get("min_open_gain_pct", 0.25)):
                result[symbol] = (
                    False,
                    f"blocked weak open gain {gain_from_open_pct:.2f}% < {float(settings.get('min_open_gain_pct', 0.25)):.2f}%",
                )
                continue
            if gain_from_open_pct > float(settings.get("max_open_gain_pct", 4.0)):
                result[symbol] = (
                    False,
                    f"blocked too extended open gain {gain_from_open_pct:.2f}% > {float(settings.get('max_open_gain_pct', 4.0)):.2f}%",
                )
                continue
            if relative_volume < float(settings.get("relative_volume_min", 1.2)):
                result[symbol] = (
                    False,
                    f"blocked weak relative volume {relative_volume:.2f}x < {float(settings.get('relative_volume_min', 1.2)):.2f}x",
                )
                continue

            result[symbol] = (
                True,
                f"entry ok price>${open_price:.2f} open, vwap=${vwap:.2f}, "
                f"open_gain={gain_from_open_pct:.2f}%, rel_vol={relative_volume:.2f}x",
            )

        return result

    def load_training_frame(self, start: date, end: date) -> pd.DataFrame:
        fetch_start = start - timedelta(days=int(self.config["ml"]["feature_lookback_days"]) * 3)
        bars_by_symbol = self.backtester.fetch_daily_bars(self.config["watchlist"], fetch_start, end)
        frame = build_feature_frame(
            bars_by_symbol=bars_by_symbol,
            horizon_days=int(self.config["ml"]["horizon_days"]),
            target_return_pct=float(self.config["ml"]["target_return_pct"]),
        )
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def parse_hhmm_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _intraday_bar_like(
    timestamp: datetime,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> LiveIntradayBar:
    return LiveIntradayBar(
        timestamp=timestamp,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def average_intraday_cumulative_volume(
    bars_by_day: dict[date, list[Any]],
    trading_day: date,
    cutoff: time,
    lookback_days: int,
) -> float:
    volumes: list[float] = []
    historical_days = sorted(day for day in bars_by_day if day < trading_day)[-lookback_days:]
    for historical_day in historical_days:
        cumulative_volume = sum(
            float(getattr(bar, "volume", 0.0) or 0.0)
            for bar in bars_by_day[historical_day]
            if SESSION_OPEN <= bar.timestamp.astimezone(MARKET_TZ).time() <= cutoff
        )
        if cumulative_volume > 0:
            volumes.append(cumulative_volume)
    return sum(volumes) / len(volumes) if volumes else 0.0


def build_feature_frame(
    bars_by_symbol: dict[str, list[DailyBar]],
    horizon_days: int,
    target_return_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_return = target_return_pct / 100

    for symbol, bars in bars_by_symbol.items():
        for index in range(20, len(bars) - horizon_days):
            row = build_feature_row(symbol=symbol, bars=bars, index=index)
            future_close = bars[index + horizon_days].close
            row["target"] = int((future_close / bars[index].close - 1) >= target_return)
            row["future_return_pct"] = (future_close / bars[index].close - 1) * 100
            rows.append(row)

    return pd.DataFrame(rows)


def build_feature_row(symbol: str, bars: list[DailyBar], index: int) -> dict[str, Any]:
    current = bars[index]
    closes = np.array([bar.close for bar in bars[: index + 1]], dtype=float)
    volumes = np.array([bar.volume for bar in bars[: index + 1]], dtype=float)
    returns = pd.Series(closes).pct_change()

    def simple_return(days: int) -> float:
        return float(closes[-1] / closes[-1 - days] - 1)

    def close_to_sma(days: int) -> float:
        average = float(np.mean(closes[-days:]))
        return float(closes[-1] / average - 1) if average else 0.0

    volume_mean_5 = float(np.mean(volumes[-5:])) if np.any(volumes[-5:]) else 0.0
    previous_volume = float(volumes[-2]) if len(volumes) >= 2 else 0.0

    return {
        "date": current.timestamp.date(),
        "timestamp": current.timestamp,
        "symbol": symbol,
        "price": current.close,
        "return_1d": simple_return(1),
        "return_3d": simple_return(3),
        "return_5d": simple_return(5),
        "return_10d": simple_return(10),
        "return_20d": simple_return(20),
        "intraday_return": (current.close / current.open - 1) if current.open else 0.0,
        "overnight_return": (current.open / bars[index - 1].close - 1) if bars[index - 1].close else 0.0,
        "range_pct": ((current.high - current.low) / current.close) if current.close else 0.0,
        "volatility_5d": float(returns.tail(5).std() or 0.0),
        "volatility_10d": float(returns.tail(10).std() or 0.0),
        "close_to_sma_5": close_to_sma(5),
        "close_to_sma_10": close_to_sma(10),
        "close_to_sma_20": close_to_sma(20),
        "volume_ratio_5d": (current.volume / volume_mean_5 - 1) if volume_mean_5 else 0.0,
        "volume_change_1d": (current.volume / previous_volume - 1) if previous_volume else 0.0,
    }


def feature_rows_for_date(bars_by_symbol: dict[str, list[DailyBar]], signal_date: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, bars in bars_by_symbol.items():
        index = next((idx for idx, bar in enumerate(bars) if bar.timestamp.date() == signal_date), None)
        if index is None or index < 20:
            continue
        rows.append(build_feature_row(symbol=symbol, bars=bars, index=index))
    return rows


def predict_rows(artifact: dict[str, Any], rows: pd.DataFrame) -> list[MLSignal]:
    if rows.empty:
        return []
    probabilities = artifact["estimator"].predict_proba(rows[MODEL_FEATURES])[:, 1]
    signals: list[MLSignal] = []
    for (_, row), probability in zip(rows.iterrows(), probabilities):
        action = "buy" if probability >= 0.5 else "avoid"
        signals.append(
            MLSignal(
                symbol=str(row["symbol"]),
                probability=float(probability),
                price=float(row["price"]),
                timestamp=row["timestamp"],
                action=action,
                reason=f"ML probability {probability:.2%}",
            )
        )
    return signals


def build_estimator(model_type: str, random_state: int) -> Pipeline:
    model_name = str(model_type).strip().lower()
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=random_state,
        )
    else:
        raise ValueError(
            "Unsupported ml.model_type. Choose one of: logistic_regression, random_forest, hist_gradient_boosting"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("symbol", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def validate_time_series(frame: pd.DataFrame, estimator: Pipeline, n_splits: int, gap: int) -> list[ValidationFold]:
    unique_dates = np.array(sorted(frame["date"].unique()))
    if len(unique_dates) < n_splits + 2:
        raise RuntimeError("Not enough unique dates for the configured ML validation_splits.")

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    folds: list[ValidationFold] = []
    for fold_number, (train_date_indexes, test_date_indexes) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = set(unique_dates[train_date_indexes])
        test_dates = set(unique_dates[test_date_indexes])
        train = frame[frame["date"].isin(train_dates)]
        test = frame[frame["date"].isin(test_dates)]

        fold_estimator = clone(estimator)
        fold_estimator.fit(train[MODEL_FEATURES], train["target"])
        predictions = fold_estimator.predict(test[MODEL_FEATURES])
        probabilities = fold_estimator.predict_proba(test[MODEL_FEATURES])[:, 1]
        roc_auc = None
        if test["target"].nunique() == 2:
            roc_auc = float(roc_auc_score(test["target"], probabilities))

        folds.append(
            ValidationFold(
                fold=fold_number,
                train_start=min(train_dates),
                train_end=max(train_dates),
                test_start=min(test_dates),
                test_end=max(test_dates),
                rows=len(test),
                positive_rate=float(test["target"].mean()),
                accuracy=float(accuracy_score(test["target"], predictions)),
                precision=float(precision_score(test["target"], predictions, zero_division=0)),
                recall=float(recall_score(test["target"], predictions, zero_division=0)),
                roc_auc=roc_auc,
            )
        )
    return folds


def index_bars_by_date(bars_by_symbol: dict[str, list[DailyBar]]) -> dict[date, dict[str, DailyBar]]:
    bars_by_date: dict[date, dict[str, DailyBar]] = {}
    for symbol, bars in bars_by_symbol.items():
        for bar in bars:
            bars_by_date.setdefault(bar.timestamp.date(), {})[symbol] = bar
    return bars_by_date


def update_position_peaks(positions: dict[str, BacktestPosition], bars: dict[str, DailyBar]) -> None:
    for symbol, position in positions.items():
        if position.qty <= 0:
            continue
        bar = bars.get(symbol)
        if bar:
            position.peak_price = max(position.peak_price, bar.close)


def risk_exit_signals(
    positions: dict[str, BacktestPosition],
    bars: dict[str, DailyBar],
    stop_loss_pct: float,
    trailing_stop_pct: float,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for symbol, position in positions.items():
        if position.qty <= 0:
            continue
        bar = bars.get(symbol)
        if not bar:
            continue

        if position.avg_entry > 0:
            loss_pct = ((bar.close - position.avg_entry) / position.avg_entry) * 100
            if loss_pct <= -stop_loss_pct:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "qty": position.qty,
                        "reason": f"stop loss {loss_pct:.2f}% <= -{stop_loss_pct:.2f}%",
                    }
                )
                continue

        if position.peak_price > 0:
            pullback_pct = ((bar.close - position.peak_price) / position.peak_price) * 100
            if pullback_pct <= -trailing_stop_pct:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "qty": position.qty,
                        "reason": f"trailing stop {pullback_pct:.2f}% <= -{trailing_stop_pct:.2f}%",
                    }
                )
    return signals


def dedupe_sell_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for signal in signals:
        symbol = signal["symbol"]
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(signal)
    return deduped


def closed_trade_returns(trades: list[BacktestTrade]) -> list[float]:
    lots: dict[str, list[dict[str, float]]] = {}
    returns: list[float] = []

    for trade in trades:
        if trade.action == "buy":
            lots.setdefault(trade.symbol, []).append({"qty": trade.qty, "price": trade.price})
            continue
        if trade.action != "sell":
            continue

        remaining = trade.qty
        symbol_lots = lots.setdefault(trade.symbol, [])
        while remaining > 1e-9 and symbol_lots:
            lot = symbol_lots[0]
            closed_qty = min(remaining, lot["qty"])
            if lot["price"] > 0:
                returns.append((trade.price - lot["price"]) / lot["price"])
            lot["qty"] -= closed_qty
            remaining -= closed_qty
            if lot["qty"] <= 1e-9:
                symbol_lots.pop(0)

    return returns


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"ML model not found at {model_path}. Run train-model first.")
    return joblib.load(model_path)


def print_training_result(result: TrainingResult) -> None:
    print(f"Training rows: {result.rows}")
    print(f"Positive target rate: {result.positive_rate:.2%}")
    print("\nTime-series validation")
    for fold in result.folds:
        auc = f"{fold.roc_auc:.3f}" if fold.roc_auc is not None else "n/a"
        print(
            f"Fold {fold.fold}: test={fold.test_start}..{fold.test_end} rows={fold.rows} "
            f"pos={fold.positive_rate:.2%} acc={fold.accuracy:.3f} "
            f"precision={fold.precision:.3f} recall={fold.recall:.3f} auc={auc}"
        )
    print(f"\nModel written to {result.model_path}")


def print_ml_signals(signals: list[MLSignal], buy_probability: float, sell_probability: float) -> None:
    print(f"ML signals: buy >= {buy_probability:.0%}, sell/avoid <= {sell_probability:.0%}")
    for signal in signals:
        if signal.probability >= buy_probability:
            label = "BUY"
        elif signal.probability <= sell_probability:
            label = "SELL/AVOID"
        else:
            label = "HOLD"
        print(
            f"{signal.symbol:5} probability={signal.probability:7.2%} "
            f"price=${signal.price:9.2f} asof={signal.timestamp.date()} {label}"
        )


def write_optimization_csv(path: str | Path, results: list[OptimizationResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "score",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "feasible",
        "buy_probability",
        "sell_probability",
        "per_trade_notional",
        "max_position_notional",
        "max_symbols_to_buy_per_run",
        "stop_loss_pct",
        "trailing_stop_pct",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "score": round(result.score, 4),
                    "total_return_pct": round(result.total_return_pct, 4),
                    "max_drawdown_pct": round(result.max_drawdown_pct, 4),
                    "trades": result.trades,
                    "feasible": result.feasible,
                    **result.params,
                }
            )


def print_optimization_results(results: list[OptimizationResult], output_path: str, top_n: int = 10) -> None:
    print(f"Optimization trials: {len(results)}")
    print(f"Results written to {output_path}")
    print("\nTop parameter sets")
    for rank, result in enumerate(results[:top_n], start=1):
        params = result.params
        print(
            f"{rank:2}. score={result.score:7.2f} return={result.total_return_pct:7.2f}% "
            f"max_dd={result.max_drawdown_pct:7.2f}% trades={result.trades:4} "
            f"feasible={result.feasible} buy={params['buy_probability']:.2f} "
            f"sell={params['sell_probability']:.2f} trade=${params['per_trade_notional']:.0f} "
            f"max_pos=${params['max_position_notional']:.0f} max_buys={params['max_symbols_to_buy_per_run']} "
            f"stop={params['stop_loss_pct']:.1f}% trail={params['trailing_stop_pct']:.1f}%"
        )


def print_kelly_result(backtest: BacktestResult, kelly: KellyResult) -> None:
    print("Kelly analysis from ML backtest closed trades")
    print(f"Backtest return: {backtest.total_return_pct:.2f}%")
    print(f"Max drawdown: {backtest.max_drawdown_pct:.2f}%")
    print(f"Closed samples: {kelly.sample_count}")
    print(f"Win rate: {kelly.win_rate:.2%}")
    print(f"Average win: {kelly.average_win_pct:.2f}%")
    print(f"Average loss: {kelly.average_loss_pct:.2f}%")
    print(f"Payoff ratio: {kelly.payoff_ratio:.2f}")
    print(f"Full Kelly: {kelly.full_kelly_fraction:.2%}")
    print(f"Half Kelly: {kelly.half_kelly_fraction:.2%}")
    print(f"Quarter Kelly: {kelly.quarter_kelly_fraction:.2%}")
    print("\nSuggested per-trade notional on starting cash")
    print(f"Full Kelly: ${kelly.suggested_full_notional:,.2f}")
    print(f"Half Kelly: ${kelly.suggested_half_notional:,.2f}")
    print(f"Quarter Kelly: ${kelly.suggested_quarter_notional:,.2f}")
