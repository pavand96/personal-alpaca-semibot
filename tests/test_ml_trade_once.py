"""Tests for MLStrategy.ml_trade_once() — the live entry point for the ML strategy."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from semibot.ml import MLSignal, MLStrategy

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_strategy(
    risk: dict | None = None,
    ml: dict | None = None,
) -> MLStrategy:
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
            "max_total_position_notional": 10_000.0,
            "exit_before_close": "15:55",
            **(risk or {}),
        },
        "strategy": {
            "per_trade_notional": 250.0,
            "max_position_notional": 1_000.0,
            "max_symbols_to_buy_per_run": 2,
            "min_price": 1.0,
            "max_price": 10_000.0,
        },
        "ml": {
            "buy_probability": 0.50,
            "sell_probability": 0.25,
            "stop_loss_pct": 10.0,
            "per_trade_notional": 100.0,
            "max_position_notional": 2_000.0,
            "max_symbols_to_buy_per_run": 3,
            "take_profit_pct": 2.0,
            **(ml or {}),
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


def _signal(symbol: str, probability: float = 0.80, action: str = "buy") -> MLSignal:
    return MLSignal(
        symbol=symbol,
        probability=probability,
        price=100.0,
        timestamp=datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc),
        action=action,
        reason=f"ML probability {probability:.0%}",
    )


def _position(symbol: str, qty: float = 1.0, unrealized_plpc: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=str(qty),
        market_value=str(qty * 100.0),
        unrealized_plpc=str(unrealized_plpc),
    )


def _make_fake_bot(
    kill_switch: bool = False,
    positions: dict | None = None,
    is_open: bool = True,
    trading_blocked: bool = False,
) -> MagicMock:
    fake_bot = MagicMock(unsafe=True)
    fake_bot.daily_loss_kill_switch_triggered.return_value = kill_switch
    fake_bot.get_positions.return_value = positions or {}
    fake_bot.trading.get_clock.return_value = SimpleNamespace(is_open=is_open)
    fake_bot.trading.get_account.return_value = SimpleNamespace(
        equity="10000",
        last_equity="10000",
        trading_blocked=trading_blocked,
        account_blocked=False,
    )
    return fake_bot


def _passthrough_submit(bot, decisions, dry_run, max_orders):
    """Replacement for submit_ml_decisions that just returns what was passed."""
    return [d for d in decisions if d.action != "hold"][:max_orders]


# ---------------------------------------------------------------------------
# Market-closed gate
# ---------------------------------------------------------------------------

def test_ml_trade_once_returns_empty_when_market_closed() -> None:
    strategy = make_strategy(risk={"require_market_open": True})
    fake_bot = _make_fake_bot(is_open=False)

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
    ):
        result = strategy.ml_trade_once()

    assert result == []


def test_ml_trade_once_proceeds_when_require_market_open_false() -> None:
    strategy = make_strategy(risk={"require_market_open": False})
    fake_bot = _make_fake_bot(is_open=False)

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=_passthrough_submit),
    ):
        result = strategy.ml_trade_once()

    assert result == []  # no signals → nothing submitted, but no early-return


# ---------------------------------------------------------------------------
# Dry-run vs execute
# ---------------------------------------------------------------------------

def test_ml_trade_once_dry_run_never_submits_real_order() -> None:
    strategy = make_strategy(risk={"dry_run": True})
    fake_bot = _make_fake_bot()
    captured: list[dict] = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.append({"dry_run": dry_run, "decisions": decisions})
        return decisions

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (True, "ok")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once(execute=True)

    assert captured[0]["dry_run"] is True


def test_ml_trade_once_execute_true_with_dry_run_false_submits_live() -> None:
    strategy = make_strategy(risk={"dry_run": False})
    fake_bot = _make_fake_bot()
    captured: list[dict] = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.append({"dry_run": dry_run, "decisions": decisions})
        return decisions

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (True, "ok")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once(execute=True)

    assert captured[0]["dry_run"] is False


def test_ml_trade_once_no_execute_flag_forces_dry_run() -> None:
    strategy = make_strategy(risk={"dry_run": False})
    fake_bot = _make_fake_bot()
    captured: list[dict] = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.append({"dry_run": dry_run})
        return decisions

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (True, "ok")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once()  # execute defaults to False

    assert captured[0]["dry_run"] is True


# ---------------------------------------------------------------------------
# Kill-switch paths
# ---------------------------------------------------------------------------

def test_ml_trade_once_kill_switch_flattens_all_positions_bypassing_max_orders() -> None:
    strategy = make_strategy(risk={"max_orders_per_run": 1, "flatten_on_daily_loss": True})
    held = {s: _position(s) for s in ["NVDA", "AMD", "AVGO", "TSM", "INTC"]}
    fake_bot = _make_fake_bot(kill_switch=True, positions=held)
    captured: list[dict] = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.append({"decisions": decisions, "max_orders": max_orders})
        return decisions

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        result = strategy.ml_trade_once()

    assert len(result) == 5
    assert all(d.action == "sell" for d in result)
    # max_orders must equal the number of positions so the cap does not fire
    assert captured[0]["max_orders"] == 5


def test_ml_trade_once_kill_switch_with_flatten_false_returns_empty() -> None:
    strategy = make_strategy(risk={"flatten_on_daily_loss": False})
    held = {"NVDA": _position("NVDA")}
    fake_bot = _make_fake_bot(kill_switch=True, positions=held)

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch("semibot.ml.submit_ml_decisions") as mock_submit,
    ):
        result = strategy.ml_trade_once()

    assert result == []
    mock_submit.assert_not_called()


def test_ml_trade_once_kill_switch_sell_reason_contains_kill_switch() -> None:
    strategy = make_strategy(risk={"flatten_on_daily_loss": True})
    held = {"NVDA": _position("NVDA")}
    fake_bot = _make_fake_bot(kill_switch=True, positions=held)

    def capturing_submit(bot, decisions, dry_run, max_orders):
        return decisions

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        result = strategy.ml_trade_once()

    assert all("kill switch" in d.reason for d in result)


# ---------------------------------------------------------------------------
# Stop loss and take profit
# ---------------------------------------------------------------------------

def test_ml_trade_once_stop_loss_sells_when_loss_exceeds_ml_threshold() -> None:
    # -12% loss > ml.stop_loss_pct=10% → sell
    strategy = make_strategy()
    pos = _position("NVDA", qty=1.0, unrealized_plpc=-0.12)
    fake_bot = _make_fake_bot(positions={"NVDA": pos})

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=_passthrough_submit),
    ):
        result = strategy.ml_trade_once()

    sell_decisions = [d for d in result if d.action == "sell" and "stop loss" in d.reason]
    assert len(sell_decisions) == 1
    assert sell_decisions[0].symbol == "NVDA"


def test_ml_trade_once_stop_loss_does_not_fire_below_ml_threshold() -> None:
    # -6% loss < ml.stop_loss_pct=10% → no stop sell
    strategy = make_strategy()
    pos = _position("NVDA", qty=1.0, unrealized_plpc=-0.06)
    fake_bot = _make_fake_bot(positions={"NVDA": pos})

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA", probability=0.55)]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=_passthrough_submit),
    ):
        result = strategy.ml_trade_once()

    stop_sells = [d for d in result if "stop loss" in d.reason]
    assert len(stop_sells) == 0


def test_ml_trade_once_take_profit_sells_when_gain_exceeds_threshold() -> None:
    # +3% gain > ml.take_profit_pct=2% → sell; exit_before_close set to 23:59 to prevent interference
    strategy = make_strategy(risk={"exit_before_close": "23:59"})
    pos = _position("NVDA", qty=1.0, unrealized_plpc=0.03)
    fake_bot = _make_fake_bot(positions={"NVDA": pos})

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=_passthrough_submit),
    ):
        result = strategy.ml_trade_once()

    tp_sells = [d for d in result if "take profit" in d.reason]
    assert len(tp_sells) == 1
    assert tp_sells[0].symbol == "NVDA"


# ---------------------------------------------------------------------------
# Exit before close
# ---------------------------------------------------------------------------

def test_ml_trade_once_exit_before_close_sells_all_positions() -> None:
    strategy = make_strategy(risk={"exit_before_close": "09:00"})  # always past threshold
    held = {"NVDA": _position("NVDA"), "AMD": _position("AMD")}
    fake_bot = _make_fake_bot(positions=held)

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=_passthrough_submit),
    ):
        result = strategy.ml_trade_once()

    close_sells = [d for d in result if "exit before close" in d.reason]
    assert len(close_sells) == 2


# ---------------------------------------------------------------------------
# ML signal → buy / sell decisions
# ---------------------------------------------------------------------------

def test_ml_trade_once_buy_uses_ml_per_trade_notional() -> None:
    # ml.per_trade_notional=100.0 must be used, not strategy.per_trade_notional=250.0
    strategy = make_strategy()
    fake_bot = _make_fake_bot()
    captured: list = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.extend(decisions)
        return [d for d in decisions if d.action != "hold"]

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (True, "ok")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once()

    buy_decisions = [d for d in captured if d.action == "buy"]
    assert len(buy_decisions) == 1
    assert buy_decisions[0].notional == 100.0


def test_ml_trade_once_ml_sell_signal_on_held_position() -> None:
    # held NVDA with probability=0.20 ≤ sell_probability=0.25 → sell
    strategy = make_strategy()
    pos = _position("NVDA", qty=1.0, unrealized_plpc=0.0)
    fake_bot = _make_fake_bot(positions={"NVDA": pos})
    captured: list = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.extend(decisions)
        return [d for d in decisions if d.action != "hold"]

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA", probability=0.20)]),
        patch.object(strategy, "live_entry_quality", return_value={}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once()

    sell_decisions = [d for d in captured if d.action == "sell"]
    assert len(sell_decisions) == 1
    assert sell_decisions[0].symbol == "NVDA"


def test_ml_trade_once_respects_max_buys_from_ml_config() -> None:
    # ml.max_symbols_to_buy_per_run=1 → only 1 buy despite 3 qualifying signals
    strategy = make_strategy(ml={"max_symbols_to_buy_per_run": 1})
    fake_bot = _make_fake_bot()
    captured: list = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.extend(decisions)
        return [d for d in decisions if d.action != "hold"]

    signals = [_signal("NVDA"), _signal("AMD"), _signal("AVGO")]
    quality = {"NVDA": (True, "ok"), "AMD": (True, "ok"), "AVGO": (True, "ok")}

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=signals),
        patch.object(strategy, "live_entry_quality", return_value=quality),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once()

    buy_decisions = [d for d in captured if d.action == "buy"]
    assert len(buy_decisions) == 1


def test_ml_trade_once_quality_filter_failure_produces_hold() -> None:
    # quality filter fails → hold, not buy
    strategy = make_strategy()
    fake_bot = _make_fake_bot()
    captured: list = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.extend(decisions)
        return [d for d in decisions if d.action != "hold"]

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (False, "below VWAP")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once()

    hold_decisions = [d for d in captured if d.action == "hold"]
    buy_decisions = [d for d in captured if d.action == "buy"]
    assert len(hold_decisions) == 1
    assert len(buy_decisions) == 0


def test_ml_trade_once_portfolio_cap_blocks_buy() -> None:
    # total position notional near cap → new buy blocked with hold
    strategy = make_strategy(risk={"max_total_position_notional": 150.0})
    # one position already using $100; adding $100 more would exceed $150 cap
    pos = _position("AMD", qty=1.0, unrealized_plpc=0.0)
    fake_bot = _make_fake_bot(positions={"AMD": pos})
    captured: list = []

    def capturing_submit(bot, decisions, dry_run, max_orders):
        captured.extend(decisions)
        return [d for d in decisions if d.action != "hold"]

    with (
        patch("semibot.ml.SemiMomentumBot", return_value=fake_bot),
        patch.object(strategy, "latest_signals", return_value=[_signal("NVDA")]),
        patch.object(strategy, "live_entry_quality", return_value={"NVDA": (True, "ok")}),
        patch("semibot.ml.submit_ml_decisions", side_effect=capturing_submit),
    ):
        strategy.ml_trade_once()

    hold_decisions = [d for d in captured if d.action == "hold" and "cap" in d.reason]
    assert len(hold_decisions) == 1
