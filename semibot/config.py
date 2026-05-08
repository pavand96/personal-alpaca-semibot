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
    "sector_watchlists": {
        "energy": [
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "WMB",
            "EOG",
            "VLO",
            "MPC",
            "PSX",
            "KMI",
            "BKR",
            "OKE",
            "TRGP",
            "OXY",
            "FANG",
            "EQT",
            "HAL",
            "DVN",
            "CTRA",
            "TPL",
            "EXE",
            "APA",
        ],
        "software": [
            "MSFT",
            "ORCL",
            "CRM",
            "ADBE",
            "INTU",
            "NOW",
            "PLTR",
            "APP",
            "WDAY",
            "ADSK",
            "TEAM",
            "DOCU",
            "ZM",
            "HUBS",
            "BILL",
            "PAYC",
            "DAY",
            "PCOR",
            "MANH",
            "TYL",
            "PTC",
            "FICO",
        ],
        "cybersecurity": [
            "PANW",
            "CRWD",
            "FTNT",
            "ZS",
            "OKTA",
            "CYBR",
            "TENB",
            "S",
            "NET",
            "CHKP",
            "QLYS",
            "VRNS",
            "GEN",
            "RBRK",
            "RDWR",
            "BB",
        ],
        "quantum": [
            "IONQ",
            "RGTI",
            "QBTS",
            "QUBT",
            "ARQQ",
            "IBM",
            "GOOGL",
            "MSFT",
            "AMZN",
            "NVDA",
            "INTC",
            "HON",
            "NOK",
            "ERIC",
            "TER",
            "COHR",
            "MKSI",
            "TSEM",
        ],
        "ai_infrastructure": [
            "NVDA",
            "AVGO",
            "AMD",
            "TSM",
            "ASML",
            "ARM",
            "MU",
            "INTC",
            "LRCX",
            "AMAT",
            "KLAC",
            "MRVL",
            "ANET",
            "SMCI",
            "DELL",
            "HPE",
            "VRT",
            "ETN",
        ],
        "cloud_data": [
            "MSFT",
            "AMZN",
            "GOOGL",
            "ORCL",
            "SNOW",
            "MDB",
            "DDOG",
            "NET",
            "CRWD",
            "PLTR",
            "ESTC",
            "CFLT",
            "GTLB",
            "PATH",
            "AI",
        ],
        "defense_space": [
            "LMT",
            "RTX",
            "NOC",
            "GD",
            "LHX",
            "BA",
            "HWM",
            "TDG",
            "RKLB",
            "PL",
        ],
    },
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
        "stop_loss_pct": 5.0,
        "trailing_stop_pct": 10.0,
        "max_account_drawdown_pct": 15.0,
        "max_daily_loss_pct": 3.0,
        "flatten_on_daily_loss": True,
        "max_total_position_notional": 1500.0,
        "take_profit_pct": 3.0,
        "exit_before_close": "15:55",
    },
    "orders": {
        "type": "limit",
        "limit_price_offset_bps": 10.0,
    },
    "live_entry_filter": {
        "enabled": True,
        "min_time": "09:45",
        "require_above_open": True,
        "require_above_vwap": True,
        "min_open_gain_pct": 1.0,
        "max_open_gain_pct": 4.0,
        "relative_volume_min": 1.5,
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
    "buzz_earnings_overlay": {
        "enabled": True,
        "fail_open": True,
        "news_enabled": True,
        "news_lookback_days": 7,
        "news_fetch_chunk_days": 30,
        "news_limit_per_request": 50,
        "article_count_cap": 5,
        "article_weight": 0.4,
        "positive_keyword_weight": 1.0,
        "negative_keyword_weight": 2.0,
        "score_weight": 2.0,
        "max_score_boost": 20.0,
        "negative_score_block_threshold": -3.0,
        "earnings_calendar_file": "data/earnings_calendar.csv",
        "block_new_entries_near_earnings": True,
        "earnings_avoid_days_before": 2,
        "earnings_avoid_days_after": 1,
        "allow_earnings_if_profit_cushion_pct": 5.0,
        "positive_keywords": [
            "accelerates",
            "ai demand",
            "analyst upgrade",
            "beat",
            "beats",
            "bullish",
            "contract win",
            "demand",
            "earnings beat",
            "guidance raised",
            "raises forecast",
            "record revenue",
            "strong demand",
            "upgrade",
        ],
        "negative_keywords": [
            "accounting probe",
            "bankruptcy",
            "class action",
            "cuts guidance",
            "data breach",
            "delisting",
            "downgrade",
            "earnings miss",
            "fraud",
            "guidance cut",
            "investigation",
            "lawsuit",
            "miss",
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
        "model_type": "hist_gradient_boosting",
        "horizon_days": 3,
        "target_return_pct": 1.0,
        "buy_probability": 0.60,
        "sell_probability": 0.40,
        "per_trade_pct_of_equity": 0.5,
        "per_trade_notional": 100.0,
        "max_position_notional": 500.0,
        "max_symbols_to_buy_per_run": 1,
        "feature_lookback_days": 30,
        "validation_splits": 5,
        "validation_gap_days": 10,
        "random_state": 42,
        "ml_trades_file": "logs/ml_backtest_trades.csv",
        "optimizer_results_file": "logs/ml_parameter_optimization.csv",
        "optimizer_return_drawdown_penalty": 1.5,
        "optimizer_trade_penalty": 0.05,
        "optimizer_max_trials": 80,
        "walk_forward_retrain_days": 15,
        "walk_forward_training_lookback_days": 1460,
        "buy_probability_grid": [0.55, 0.60, 0.65, 0.70],
        "sell_probability_grid": [0.25, 0.30, 0.35, 0.40],
        "per_trade_notional_grid": [50.0, 100.0],
        "max_position_notional_grid": [250.0, 500.0, 1000.0],
        "max_symbols_to_buy_per_run_grid": [1, 2],
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
        "base_notional": 4500.0,
        "max_notional_per_trade": 4500.0,
        "max_total_notional": 4500.0,
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
        "max_total_exposure": 4500.0,
        "rebalance_days": 3,
        "trailing_stop_pct": 20.0,
        "trades_file": "logs/sector_swing_trades.csv",
    },
    "sector_balanced_allocator": {
        "core_swing_exposure": 5000.0,
        "confirmed_intraday_exposure": 4500.0,
        "exposure_usage_pct": 0.95,
    },
    "adaptive_semis_allocator": {
        "max_total_exposure": 4500.0,
        "max_symbols": 3,
        "rebalance_days": 5,
        "sector_lookback_days": 20,
        "sector_slow_lookback_days": 63,
        "min_sector_return_pct": 0.0,
        "min_sector_slow_return_pct": -2.0,
        "risk_off_sector_return_pct": -5.0,
        "sector_drawdown_lookback_days": 63,
        "max_sector_drawdown_pct": -12.0,
        "risk_filter_symbols": ["QQQ", "SPY"],
        "market_momentum_lookback_days": 20,
        "market_sma_days": 50,
        "min_market_momentum_pct": -2.0,
        "require_market_above_sma": True,
        "momentum_lookback_days": 63,
        "fast_momentum_days": 10,
        "min_symbol_momentum_pct": 0.0,
        "min_trade_notional": 100.0,
        "retain_winners": True,
        "retain_min_profit_pct": 7.5,
        "retain_min_fast_momentum_pct": -3.0,
        "retain_min_slow_momentum_pct": 5.0,
        "trailing_stop_pct": 13.0,
        "trailing_stop_pct_bull": 17.0,
        "trailing_wide_min_profit_pct": 12.0,
        "hard_stop_pct": 12.0,
        "regime_filter_enabled": True,
        "regime_sma_fast_days": 50,
        "regime_sma_slow_days": 200,
        "neutral_sizing_pct": 0.5,
        "bear_block_entries": True,
        "symbol_sma_filter_days": 50,
        "use_sma_alignment": True,
        "use_relative_strength": True,
        "use_trend_consistency": True,
        "score_proportional_sizing": True,
        "bull_exposure_pct": 0.80,
        "bull_max_symbols": 5,
        "rebalance_days_bull": 3,
        "atr_sizing_enabled": True,
        "atr_lookback_days": 14,
        "trades_file": "logs/adaptive_semis_trades.csv",
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
    normalize_sector_watchlists(config)
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


def normalize_sector_watchlists(config: dict[str, Any]) -> None:
    raw_watchlists = config.get("sector_watchlists", {})
    if not isinstance(raw_watchlists, dict):
        raise ValueError("sector_watchlists must be a mapping of sector name to symbol list")

    normalized: dict[str, list[str]] = {}
    for sector, symbols in raw_watchlists.items():
        if not isinstance(sector, str) or not sector.strip():
            raise ValueError("sector_watchlists sector names must be non-empty strings")
        if not isinstance(symbols, list):
            raise ValueError(f"sector_watchlists.{sector} must be a list of symbols")
        sector_name = sector.strip().lower().replace("-", "_").replace(" ", "_")
        normalized[sector_name] = normalize_symbols(symbols, allow_empty=True)
    config["sector_watchlists"] = normalized


def validate_config(config: dict[str, Any]) -> None:
    if not bool(config["alpaca"].get("paper", True)) and not bool(config["risk"].get("dry_run", True)):
        raise ValueError("refusing to load config with alpaca.paper=false and risk.dry_run=false")

    require_positive(config["strategy"], "per_trade_notional")
    require_positive(config["strategy"], "max_position_notional")
    if float(config["strategy"]["per_trade_notional"]) > float(config["strategy"]["max_position_notional"]):
        raise ValueError("strategy.per_trade_notional cannot exceed strategy.max_position_notional")
    # 0 is valid: means sell-only mode (existing positions can be sold, no new buys)
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
        "max_total_position_notional",
        "take_profit_pct",
    ]:
        require_positive(config["risk"], key)
    require_non_negative(config["risk"], "max_daily_loss_pct")
    validate_hhmm(config["risk"], "exit_before_close")

    ml = config.get("ml", {})
    for key in ("stop_loss_pct", "trailing_stop_pct", "take_profit_pct", "per_trade_notional", "max_position_notional"):
        if key in ml:
            require_positive(ml, key)
    if "max_symbols_to_buy_per_run" in ml:
        # 0 is valid: means sell-only mode (existing positions can be sold, no new buys)
        require_non_negative(ml, "max_symbols_to_buy_per_run")
    if "per_trade_notional" in ml and "max_position_notional" in ml:
        if float(ml["per_trade_notional"]) > float(ml["max_position_notional"]):
            raise ValueError("ml.per_trade_notional cannot exceed ml.max_position_notional")
    if "per_trade_pct_of_equity" in ml:
        val = float(ml["per_trade_pct_of_equity"])
        if not 0.0 < val <= 100.0:
            raise ValueError(f"ml.per_trade_pct_of_equity must be between 0 and 100 (exclusive), got {val}")
    for key in ("buy_probability", "sell_probability"):
        if key in ml:
            val = float(ml[key])
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"ml.{key} must be between 0 and 1, got {val}")
    if "buy_probability" in ml and "sell_probability" in ml:
        # Equal is intentionally allowed: it creates a single-threshold mode where the bot buys
        # when flat (probability >= threshold) and sells when held (probability <= threshold).
        # The held/not-held guard makes the two branches mutually exclusive within one run.
        if float(ml["buy_probability"]) < float(ml["sell_probability"]):
            raise ValueError(
                "ml.buy_probability must be >= ml.sell_probability "
                f"(got buy={ml['buy_probability']}, sell={ml['sell_probability']})"
            )
    for key in ("walk_forward_retrain_days", "walk_forward_training_lookback_days"):
        if key in ml:
            require_positive(ml, key)

    order_type = str(config["orders"].get("type", "market")).lower()
    if order_type not in {"market", "limit"}:
        raise ValueError("orders.type must be market or limit")
    require_non_negative(config["orders"], "limit_price_offset_bps")

    validate_hhmm(config["live_entry_filter"], "min_time")
    require_non_negative(config["live_entry_filter"], "min_open_gain_pct")
    require_positive(config["live_entry_filter"], "max_open_gain_pct")
    if float(config["live_entry_filter"]["min_open_gain_pct"]) > float(config["live_entry_filter"]["max_open_gain_pct"]):
        raise ValueError("live_entry_filter.min_open_gain_pct cannot exceed max_open_gain_pct")
    require_positive(config["live_entry_filter"], "relative_volume_min")
    require_positive(config["live_entry_filter"], "average_volume_lookback_days")

    overlay = config["buzz_earnings_overlay"]
    for key in (
        "news_lookback_days",
        "news_fetch_chunk_days",
        "news_limit_per_request",
        "article_count_cap",
        "negative_keyword_weight",
        "score_weight",
        "max_score_boost",
        "allow_earnings_if_profit_cushion_pct",
    ):
        require_positive(overlay, key)
    require_non_negative(overlay, "earnings_avoid_days_before")
    require_non_negative(overlay, "earnings_avoid_days_after")
    require_non_negative(overlay, "article_weight")
    require_non_negative(overlay, "positive_keyword_weight")

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

    adaptive = config["adaptive_semis_allocator"]
    for key in (
        "max_total_exposure",
        "max_symbols",
        "rebalance_days",
        "sector_lookback_days",
        "sector_slow_lookback_days",
        "sector_drawdown_lookback_days",
        "market_momentum_lookback_days",
        "market_sma_days",
        "momentum_lookback_days",
        "fast_momentum_days",
        "min_trade_notional",
        "retain_min_profit_pct",
        "trailing_stop_pct",
        "hard_stop_pct",
    ):
        require_positive(adaptive, key)
    require_non_negative(adaptive, "min_symbol_momentum_pct")
    require_non_negative(adaptive, "retain_min_slow_momentum_pct")
    adaptive["risk_filter_symbols"] = normalize_symbols(
        adaptive.get("risk_filter_symbols", []),
        allow_empty=True,
    )
    if float(adaptive["max_total_exposure"]) > float(config["backtest"]["initial_cash"]):
        raise ValueError("adaptive_semis_allocator.max_total_exposure cannot exceed backtest.initial_cash")


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
