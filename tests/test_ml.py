from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from semibot.ml import MLSignal, MLStrategy


def make_ml_strategy(ml_overrides: dict | None = None) -> MLStrategy:
    strategy = MLStrategy.__new__(MLStrategy)
    strategy.config = {
        "alpaca": {"paper": True},
        "watchlist": ["NVDA", "AMD"],
        "risk": {
            "dry_run": True,
            "require_market_open": False,
            "max_orders_per_run": 5,
            "stop_loss_pct": 0.5,
            "trailing_stop_pct": 25.0,
            "take_profit_pct": 1.0,
            "max_daily_loss_pct": 3.0,
            "flatten_on_daily_loss": True,
            "max_total_position_notional": 10000.0,
            "exit_before_close": "15:55",
        },
        "strategy": {
            "per_trade_notional": 250.0,
            "max_position_notional": 1000.0,
            "max_symbols_to_buy_per_run": 2,
            "min_price": 1.0,
            "max_price": 10000.0,
        },
        "ml": {
            "buy_probability": 0.50,
            "sell_probability": 0.25,
            "stop_loss_pct": 10.0,
            "per_trade_notional": 100.0,
            "max_position_notional": 2000.0,
            "max_symbols_to_buy_per_run": 3,
            **(ml_overrides or {}),
        },
        "portfolio": {
            "correlation_sizing_enabled": False,
            "correlation_lookback_days": 60,
            "high_correlation_threshold": 0.75,
            "high_correlation_notional_multiplier": 0.5,
            "very_high_correlation_threshold": 0.90,
            "very_high_correlation_notional_multiplier": 0.25,
            "min_correlation_adjusted_notional": 50.0,
        },
        "live_entry_filter": {
            "enabled": False,
            "min_time": "09:45",
            "require_above_open": False,
            "require_above_vwap": False,
            "min_open_gain_pct": 0.0,
            "max_open_gain_pct": 5.0,
            "relative_volume_min": 1.0,
            "average_volume_lookback_days": 20,
        },
    }
    strategy.api_key = "fake"
    strategy.secret_key = "fake"
    return strategy


def _fake_signal(symbol: str, probability: float = 0.80) -> MLSignal:
    return MLSignal(
        symbol=symbol,
        probability=probability,
        price=100.0,
        timestamp=datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc),
        action="buy",
        reason=f"ML probability {probability:.0%}",
    )


def test_ml_live_uses_ml_override_per_trade_notional() -> None:
    strategy = make_ml_strategy()

    captured_decisions: list = []

    def fake_submit(bot, decisions, dry_run, max_orders):
        captured_decisions.extend(decisions)
        return decisions

    fake_bot = MagicMock(unsafe=True)
    fake_bot.daily_loss_kill_switch_triggered.return_value = False
    fake_bot.get_positions.return_value = {}
    fake_bot.trading.get_clock.return_value = SimpleNamespace(is_open=True)

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_fake_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (True, "ok")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=fake_submit),
    ):
        strategy.ml_trade_once(execute=False)

    buy_decisions = [d for d in captured_decisions if d.action == "buy"]
    assert len(buy_decisions) == 1
    # ml.per_trade_notional (100.0) must be used, not strategy.per_trade_notional (250.0)
    assert buy_decisions[0].notional == 100.0


def test_ml_live_uses_ml_override_stop_loss() -> None:
    strategy = make_ml_strategy()

    captured_stop: list[float] = []

    fake_bot = MagicMock(unsafe=True)
    fake_bot.daily_loss_kill_switch_triggered.return_value = False
    fake_bot.trading.get_clock.return_value = SimpleNamespace(is_open=True)
    # position with -6% unrealized P&L: above global 0.5% stop, below ml 10% stop → should NOT sell
    fake_position = SimpleNamespace(
        symbol="NVDA",
        qty="1.0",
        market_value="94.0",
        unrealized_plpc="-0.06",  # -6% loss
    )
    fake_bot.get_positions.return_value = {"NVDA": fake_position}

    def fake_submit(bot, decisions, dry_run, max_orders):
        return decisions

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_fake_signal("NVDA", probability=0.55)]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=fake_submit),
    ):
        decisions = strategy.ml_trade_once(execute=False)

    sell_decisions = [d for d in decisions if d.action == "sell" and "stop loss" in d.reason]
    # ml stop_loss_pct=10.0 so a -6% position should NOT trigger the stop
    assert len(sell_decisions) == 0
