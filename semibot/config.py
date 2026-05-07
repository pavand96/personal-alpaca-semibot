from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "alpaca": {
        "paper": True,
        "data_feed": "iex",
    },
    "watchlist": [
        "NVDA",
        "AMD",
        "AVGO",
        "TSM",
        "ASML",
        "ARM",
        "INTC",
        "MU",
        "QCOM",
        "TXN",
        "AMAT",
        "LRCX",
        "KLAC",
        "MRVL",
        "ON",
        "ADI",
        "NXPI",
        "SNDK",
        "GLW",
        "WDC",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
    ],
    "strategy": {
        "buy_threshold_pct": 2.0,
        "sell_threshold_pct": -1.0,
        "per_trade_notional": 100.0,
        "max_position_notional": 500.0,
        "max_symbols_to_buy_per_run": 2,
        "min_price": 1.0,
        "max_price": 10000.0,
    },
    "risk": {
        "dry_run": True,
        "require_market_open": True,
        "max_orders_per_run": 3,
        "stop_loss_pct": 0.5,
        "trailing_stop_pct": 15.0,
        "max_account_drawdown_pct": 15.0,
        "max_daily_loss_pct": 3.0,
        "flatten_on_daily_loss": True,
        "max_total_position_notional": 10000.0,
        "take_profit_pct": 1.0,
        "exit_before_close": "15:55",
    },
    "live_entry_filter": {
        "enabled": True,
        "min_time": "09:45",
        "require_above_open": True,
        "require_above_vwap": True,
        "min_open_gain_pct": 0.25,
        "max_open_gain_pct": 4.0,
        "relative_volume_min": 1.2,
        "average_volume_lookback_days": 20,
    },
    "portfolio": {
        "correlation_sizing_enabled": True,
        "correlation_lookback_days": 60,
        "high_correlation_threshold": 0.75,
        "high_correlation_notional_multiplier": 0.5,
        "very_high_correlation_threshold": 0.90,
        "very_high_correlation_notional_multiplier": 0.25,
        "min_correlation_adjusted_notional": 50.0,
    },
    "news_hold": {
        "enabled": True,
        "lookback_hours": 24,
        "limit_per_symbol": 5,
        "hold_on_any_recent_news": False,
        "fail_closed": False,
        "block_keywords": [
            "accounting probe",
            "bankruptcy",
            "class action",
            "cuts guidance",
            "data breach",
            "delisting",
            "downgrade",
            "fraud",
            "guidance cut",
            "investigation",
            "lawsuit",
            "misses estimates",
            "probe",
            "recall",
            "resigns",
            "sec investigation",
            "short report",
            "slashes forecast",
            "subpoena",
        ],
    },
    "runtime": {
        "interval_seconds": 300,
        "log_file": "logs/semibot_events.csv",
    },
    "backtest": {
        "initial_cash": 10000.0,
        "benchmark_symbols": [],
        "benchmark_include_watchlist": True,
        "benchmark_equal_weight_watchlist": True,
        "bar_adjustment": "split",
        "slippage_bps": 5.0,
        "liquidate_at_end": True,
        "trades_file": "logs/backtest_trades.csv",
    },
    "ml": {
        "model_path": "models/semibot_model.joblib",
        "model_type": "logistic_regression",
        "horizon_days": 3,
        "target_return_pct": 1.0,
        "buy_probability": 0.60,
        "sell_probability": 0.40,
        "feature_lookback_days": 30,
        "validation_splits": 5,
        "validation_gap_days": 1,
        "random_state": 42,
        "ml_trades_file": "logs/ml_backtest_trades.csv",
        "optimizer_results_file": "logs/ml_parameter_optimization.csv",
        "optimizer_return_drawdown_penalty": 1.5,
        "optimizer_trade_penalty": 0.05,
        "optimizer_max_trials": 80,
        "buy_probability_grid": [0.50, 0.55, 0.60, 0.65],
        "sell_probability_grid": [0.25, 0.30, 0.35, 0.40],
        "per_trade_notional_grid": [100.0, 250.0],
        "max_position_notional_grid": [500.0, 1000.0, 2000.0],
        "max_symbols_to_buy_per_run_grid": [2, 3],
        "stop_loss_pct_grid": [6.0, 8.0, 10.0],
        "trailing_stop_pct_grid": [12.0, 18.0, 25.0],
    },
    "intraday": {
        "entry_time": "09:45",
        "exit_time": "15:55",
        "per_trade_notional": 250.0,
        "min_open_gain_pct": 1.0,
        "max_open_gain_pct": 4.0,
        "relative_volume_min": 1.5,
        "average_volume_lookback_days": 20,
        "stop_loss_pct": 0.5,
        "take_profit_pct": 1.0,
        "trades_file": "logs/intraday_backtest_trades.csv",
    },
    "sector_allocator": {
        "entry_time": "10:00",
        "exit_time": "15:55",
        "sector_lookback_days": 5,
        "min_sector_lookback_return_pct": 0.5,
        "max_symbols_per_day": 1,
        "base_notional": 9500.0,
        "max_notional_per_trade": 9500.0,
        "max_total_notional": 9500.0,
        "min_open_gain_pct": 0.25,
        "max_open_gain_pct": 5.0,
        "relative_volume_min": 1.2,
        "average_volume_lookback_days": 20,
        "stop_loss_pct": 3.0,
        "first_take_profit_pct": 1.0,
        "final_take_profit_pct": 8.0,
        "partial_exit_fraction": 0.5,
        "trades_file": "logs/sector_allocator_trades.csv",
    },
    "sector_swing_allocator": {
        "sector_lookback_days": 5,
        "min_sector_return_pct": 0.5,
        "momentum_lookback_days": 20,
        "fast_momentum_days": 5,
        "min_symbol_momentum_pct": 0.0,
        "max_symbols": 1,
        "max_total_exposure": 9500.0,
        "rebalance_days": 3,
        "trailing_stop_pct": 20.0,
        "trades_file": "logs/sector_swing_trades.csv",
    },
    "sector_balanced_allocator": {
        "core_swing_exposure": 5000.0,
        "confirmed_intraday_exposure": 4500.0,
        "exposure_usage_pct": 0.95,
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = deepcopy(DEFAULT_CONFIG)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            user_config = yaml.safe_load(file) or {}
        deep_merge(config, user_config)

    config["watchlist"] = normalize_symbols(config.get("watchlist", []))
    benchmark_symbols = config["backtest"].get("benchmark_symbols", [])
    config["backtest"]["benchmark_symbols"] = normalize_symbols(benchmark_symbols, allow_empty=True)
    validate_config(config)
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def normalize_symbols(symbols: list[Any], allow_empty: bool = False) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for symbol in symbols:
        if not isinstance(symbol, str):
            raise ValueError(
                f"watchlist symbols must be strings; got {symbol!r}. "
                'Quote YAML-sensitive tickers such as "ON".'
            )
        cleaned = symbol.strip().upper()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)

    if not normalized and not allow_empty:
        raise ValueError("watchlist must contain at least one symbol")

    return normalized


def validate_config(config: dict[str, Any]) -> None:
    if not bool(config["alpaca"].get("paper", True)) and not bool(config["risk"].get("dry_run", True)):
        raise ValueError("refusing to load config with alpaca.paper=false and risk.dry_run=false")

    require_positive(config["strategy"], "per_trade_notional")
    require_positive(config["strategy"], "max_position_notional")
    if float(config["strategy"]["per_trade_notional"]) > float(config["strategy"]["max_position_notional"]):
        raise ValueError("strategy.per_trade_notional cannot exceed strategy.max_position_notional")
    require_non_negative(config["strategy"], "max_symbols_to_buy_per_run")
    require_positive(config["strategy"], "min_price")
    require_positive(config["strategy"], "max_price")
    if float(config["strategy"]["min_price"]) > float(config["strategy"]["max_price"]):
        raise ValueError("strategy.min_price cannot exceed strategy.max_price")

    for key in [
        "max_orders_per_run",
        "stop_loss_pct",
        "trailing_stop_pct",
        "max_account_drawdown_pct",
        "max_daily_loss_pct",
        "max_total_position_notional",
        "take_profit_pct",
    ]:
        require_positive(config["risk"], key)
    validate_hhmm(config["risk"], "exit_before_close")

    validate_hhmm(config["live_entry_filter"], "min_time")
    require_non_negative(config["live_entry_filter"], "min_open_gain_pct")
    require_positive(config["live_entry_filter"], "max_open_gain_pct")
    if float(config["live_entry_filter"]["min_open_gain_pct"]) > float(config["live_entry_filter"]["max_open_gain_pct"]):
        raise ValueError("live_entry_filter.min_open_gain_pct cannot exceed max_open_gain_pct")
    require_positive(config["live_entry_filter"], "relative_volume_min")
    require_positive(config["live_entry_filter"], "average_volume_lookback_days")

    require_positive(config["backtest"], "initial_cash")
    require_non_negative(config["backtest"], "slippage_bps")

    for section_name in ["intraday", "sector_allocator"]:
        section = config[section_name]
        validate_hhmm(section, "entry_time")
        validate_hhmm(section, "exit_time")
        if str(section["entry_time"]) >= str(section["exit_time"]):
            raise ValueError(f"{section_name}.entry_time must be before exit_time")
        require_positive(section, "relative_volume_min")
        require_positive(section, "average_volume_lookback_days")
        require_non_negative(section, "min_open_gain_pct")
        require_positive(section, "max_open_gain_pct")
        if float(section["min_open_gain_pct"]) > float(section["max_open_gain_pct"]):
            raise ValueError(f"{section_name}.min_open_gain_pct cannot exceed max_open_gain_pct")
        require_positive(section, "stop_loss_pct")
        require_positive(section, "take_profit_pct" if section_name == "intraday" else "first_take_profit_pct")

    balanced = config["sector_balanced_allocator"]
    require_non_negative(balanced, "core_swing_exposure")
    require_non_negative(balanced, "confirmed_intraday_exposure")
    require_positive(balanced, "exposure_usage_pct")
    if float(balanced["exposure_usage_pct"]) > 1.0:
        raise ValueError("sector_balanced_allocator.exposure_usage_pct cannot exceed 1.0")
    total_exposure = float(balanced["core_swing_exposure"]) + float(balanced["confirmed_intraday_exposure"])
    if total_exposure > float(config["backtest"]["initial_cash"]):
        raise ValueError("sector_balanced_allocator exposures cannot exceed backtest.initial_cash")


def require_positive(section: dict[str, Any], key: str) -> None:
    if float(section[key]) <= 0:
        raise ValueError(f"{key} must be positive")


def require_non_negative(section: dict[str, Any], key: str) -> None:
    if float(section[key]) < 0:
        raise ValueError(f"{key} cannot be negative")


def validate_hhmm(section: dict[str, Any], key: str) -> None:
    value = str(section[key])
    try:
        hour, minute = value.split(":", maxsplit=1)
        hour_int = int(hour)
        minute_int = int(minute)
    except ValueError as error:
        raise ValueError(f"{key} must be HH:MM") from error
    if not 0 <= hour_int <= 23 or not 0 <= minute_int <= 59:
        raise ValueError(f"{key} must be HH:MM")
