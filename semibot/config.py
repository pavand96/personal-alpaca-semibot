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
