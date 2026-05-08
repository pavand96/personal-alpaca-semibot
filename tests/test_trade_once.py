"""Tests for SemiMomentumBot.trade_once() — the live entry point for the momentum strategy."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from semibot.bot import Decision, MarketView, SemiMomentumBot


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_bot(risk: dict | None = None, strategy: dict | None = None) -> SemiMomentumBot:
    bot = SemiMomentumBot.__new__(SemiMomentumBot)
    bot.config = {
        "risk": {
            "dry_run": True,
            "require_market_open": False,
            "max_orders_per_run": 3,
            "max_daily_loss_pct": 3.0,
            "flatten_on_daily_loss": True,
            **(risk or {}),
        },
        "strategy": {
            "buy_threshold_pct": 2.0,
            "sell_threshold_pct": -1.0,
            "per_trade_notional": 100.0,
            "max_position_notional": 500.0,
            "max_symbols_to_buy_per_run": 5,
            "min_price": 1.0,
            "max_price": 10000.0,
            **(strategy or {}),
        },
        "orders": {"type": "market", "limit_price_offset_bps": 10.0},
    }
    bot.trading = MagicMock(unsafe=True)
    bot.trading.get_account.return_value = SimpleNamespace(
        equity="10000",
        last_equity="10000",
        trading_blocked=False,
        account_blocked=False,
    )
    bot.trading.get_clock.return_value = SimpleNamespace(is_open=True)
    bot.trading.get_all_positions.return_value = []
    return bot


def _buy(symbol: str, notional: float = 100.0) -> Decision:
    return Decision(symbol, "buy", "momentum", notional=notional)


def _sell(symbol: str, qty: float = 1.0) -> Decision:
    return Decision(symbol, "sell", "momentum", qty=qty)


def _hold(symbol: str) -> Decision:
    return Decision(symbol, "hold", "at cap")


# ---------------------------------------------------------------------------
# Market-closed gate
# ---------------------------------------------------------------------------

def test_trade_once_returns_empty_when_market_is_closed() -> None:
    bot = make_bot(risk={"require_market_open": True})
    bot.trading.get_clock.return_value = SimpleNamespace(is_open=False)

    with patch.object(bot, "log_decision"):
        result = bot.trade_once()

    assert result == []


def test_trade_once_proceeds_when_require_market_open_is_false() -> None:
    bot = make_bot(risk={"require_market_open": False})
    bot.trading.get_clock.return_value = SimpleNamespace(is_open=False)

    with (
        patch.object(bot, "decide", return_value=[]),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        result = bot.trade_once()

    # no market-closed early-return; empty decisions → empty submitted list
    assert result == []


# ---------------------------------------------------------------------------
# Dry-run vs execute
# ---------------------------------------------------------------------------

def test_trade_once_dry_run_never_calls_submit_order() -> None:
    bot = make_bot(risk={"dry_run": True})

    with (
        patch.object(bot, "decide", return_value=[_buy("NVDA")]),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        result = bot.trade_once(execute=True)

    bot.trading.submit_order.assert_not_called()
    assert len(result) == 1


def test_trade_once_execute_calls_submit_order_when_dry_run_false() -> None:
    bot = make_bot(risk={"dry_run": False})
    bot.trading.submit_order.return_value = SimpleNamespace(id="order-123")

    with (
        patch.object(bot, "decide", return_value=[_buy("NVDA")]),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        result = bot.trade_once(execute=True)

    bot.trading.submit_order.assert_called_once()
    assert len(result) == 1


def test_trade_once_config_dry_run_wins_even_when_execute_is_true() -> None:
    # dry_run=True in config should override execute=True at the call site
    bot = make_bot(risk={"dry_run": True})

    with (
        patch.object(bot, "decide", return_value=[_buy("NVDA")]),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        bot.trade_once(execute=True)

    bot.trading.submit_order.assert_not_called()


def test_trade_once_no_execute_flag_is_dry_run_regardless_of_config() -> None:
    # even with dry_run=False in config, omitting execute keeps it dry
    bot = make_bot(risk={"dry_run": False})

    with (
        patch.object(bot, "decide", return_value=[_buy("NVDA")]),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        bot.trade_once()  # execute defaults to False

    bot.trading.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# max_orders_per_run cap (normal path)
# ---------------------------------------------------------------------------

def test_trade_once_caps_submitted_orders_at_max_orders() -> None:
    bot = make_bot(risk={"max_orders_per_run": 2})
    decisions = [_buy("NVDA"), _buy("AMD"), _buy("AVGO"), _buy("TSM")]

    with (
        patch.object(bot, "decide", return_value=decisions),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        result = bot.trade_once()

    assert len(result) == 2
    assert [d.symbol for d in result] == ["NVDA", "AMD"]


def test_trade_once_hold_decisions_do_not_count_toward_max_orders() -> None:
    bot = make_bot(risk={"max_orders_per_run": 2})
    decisions = [_hold("NVDA"), _hold("AMD"), _buy("AVGO"), _buy("TSM"), _buy("INTC")]

    with (
        patch.object(bot, "decide", return_value=decisions),
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        result = bot.trade_once()

    assert len(result) == 2
    assert all(d.action == "buy" for d in result)


# ---------------------------------------------------------------------------
# Kill-switch paths
# ---------------------------------------------------------------------------

def test_trade_once_kill_switch_flattens_all_positions_bypassing_max_orders() -> None:
    bot = make_bot(risk={"max_daily_loss_pct": 3.0, "max_orders_per_run": 1, "flatten_on_daily_loss": True})
    bot.trading.get_account.return_value = SimpleNamespace(
        equity="9400", last_equity="10000", trading_blocked=False, account_blocked=False
    )
    held = [
        SimpleNamespace(symbol=s, qty="1.0")
        for s in ["NVDA", "AMD", "AVGO", "TSM", "INTC"]
    ]
    bot.trading.get_all_positions.return_value = held

    with patch.object(bot, "log_decision"):
        result = bot.trade_once()

    # all 5 positions sold even though max_orders_per_run=1
    assert len(result) == 5
    assert all(d.action == "sell" for d in result)
    assert all("kill switch" in d.reason for d in result)


def test_trade_once_kill_switch_with_flatten_false_returns_empty() -> None:
    bot = make_bot(risk={"max_daily_loss_pct": 3.0, "flatten_on_daily_loss": False})
    bot.trading.get_account.return_value = SimpleNamespace(
        equity="9400", last_equity="10000", trading_blocked=False, account_blocked=False
    )

    with patch.object(bot, "log_decision"):
        result = bot.trade_once()

    assert result == []
    bot.trading.submit_order.assert_not_called()


def test_trade_once_kill_switch_not_triggered_below_threshold() -> None:
    bot = make_bot(risk={"max_daily_loss_pct": 3.0})
    # only -1.5% loss, below 3% threshold → normal decide() path
    bot.trading.get_account.return_value = SimpleNamespace(
        equity="9850", last_equity="10000", trading_blocked=False, account_blocked=False
    )

    with (
        patch.object(bot, "decide", return_value=[]) as mock_decide,
        patch.object(bot, "get_market_views", return_value=[]),
        patch.object(bot, "log_decision"),
    ):
        bot.trade_once()

    mock_decide.assert_called_once()


# ---------------------------------------------------------------------------
# account_blocked / trading_blocked gates
# ---------------------------------------------------------------------------

def test_trade_once_raises_when_trading_blocked() -> None:
    bot = make_bot()
    bot.trading.get_account.return_value = SimpleNamespace(
        equity="10000", last_equity="10000", trading_blocked=True, account_blocked=False
    )

    with pytest.raises(RuntimeError, match="trading blocked"):
        bot.trade_once()
