from types import SimpleNamespace

import pytest

from semibot.bot import MarketView, SemiMomentumBot, account_daily_return_pct, limit_price_for_side


def make_bot() -> SemiMomentumBot:
    bot = SemiMomentumBot.__new__(SemiMomentumBot)
    bot.config = {
        "strategy": {
            "buy_threshold_pct": 2.0,
            "sell_threshold_pct": -1.0,
            "per_trade_notional": 100.0,
            "max_position_notional": 250.0,
            "max_symbols_to_buy_per_run": 1,
            "min_price": 1.0,
            "max_price": 500.0,
        }
    }
    return bot


def test_decide_buys_top_momentum_name_only() -> None:
    bot = make_bot()

    decisions = bot.decide(
        [
            MarketView("SLOW", 100.0, 99.0, 1.0, 0.0, 0.0),
            MarketView("FAST", 103.0, 100.0, 3.0, 0.0, 0.0),
            MarketView("NEXT", 104.0, 100.0, 4.0, 0.0, 0.0),
        ]
    )

    buys = [decision for decision in decisions if decision.action == "buy"]
    assert len(buys) == 1
    assert buys[0].symbol == "NEXT"
    assert buys[0].notional == 100.0


def test_decide_sells_when_position_crosses_sell_threshold() -> None:
    bot = make_bot()

    decisions = bot.decide([MarketView("AMD", 98.0, 100.0, -2.0, 1.5, 147.0)])

    assert decisions[0].action == "sell"
    assert decisions[0].qty == 1.5


def test_decide_holds_when_position_cap_reached() -> None:
    bot = make_bot()

    decisions = bot.decide([MarketView("NVDA", 105.0, 100.0, 5.0, 1.0, 200.0)])

    assert decisions[0].action == "hold"
    assert "max position" in decisions[0].reason


def test_account_daily_return_pct_uses_last_equity() -> None:
    account = SimpleNamespace(equity="9700", last_equity="10000")

    assert account_daily_return_pct(account) == pytest.approx(-3.0)


def test_limit_price_for_side_offsets_buy_and_sell() -> None:
    assert limit_price_for_side(100.0, "buy", 10.0) == 100.10
    assert limit_price_for_side(100.0, "sell", 10.0) == 99.90
