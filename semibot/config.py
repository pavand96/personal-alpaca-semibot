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
    "watchlist": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "ARM", "INTC", "MU"],
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
        "slippage_bps": 5.0,
        "liquidate_at_end": True,
        "trades_file": "logs/backtest_trades.csv",
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
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def normalize_symbols(symbols: list[Any]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for symbol in symbols:
        cleaned = str(symbol).strip().upper()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)

    if not normalized:
        raise ValueError("watchlist must contain at least one symbol")

    return normalized
