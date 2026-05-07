from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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


class MLStrategy:
    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        self.backtester = Backtester(config, api_key=api_key, secret_key=secret_key)

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

            feature_rows = feature_rows_for_date(bars_by_symbol, current_date)
            if not feature_rows:
                continue
            predictions = predict_rows(artifact, pd.DataFrame(feature_rows))

            signals: list[dict[str, Any]] = []
            buys_used = 0
            for signal in sorted(predictions, key=lambda item: item.probability, reverse=True):
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
                    buys_used += 1
                    signals.append(
                        {
                            "symbol": signal.symbol,
                            "action": "buy",
                            "notional": per_trade_notional,
                            "reason": signal.reason,
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
        max_buys = int(self.config["strategy"]["max_symbols_to_buy_per_run"])
        buy_probability = float(self.config["ml"]["buy_probability"])
        sell_probability = float(self.config["ml"]["sell_probability"])

        decisions: list[Decision] = []
        buys_used = 0
        for signal in self.latest_signals():
            position = positions.get(signal.symbol)
            held_qty = float(position.qty) if position else 0.0
            held_value = abs(float(position.market_value)) if position else 0.0
            if held_qty > 0 and signal.probability <= sell_probability:
                decisions.append(Decision(signal.symbol, "sell", signal.reason, qty=held_qty))
                continue
            if (
                signal.probability >= buy_probability
                and held_value + per_trade_notional <= max_position_notional
                and buys_used < max_buys
            ):
                buys_used += 1
                decisions.append(
                    Decision(signal.symbol, "buy", signal.reason, notional=per_trade_notional)
                )

        submitted: list[Decision] = []
        for decision in decisions:
            if len(submitted) >= max_orders:
                break
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
