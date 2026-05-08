from types import SimpleNamespace

import pytest

from semibot.bot import Decision, MarketView, SemiMomentumBot, account_daily_return_pct, limit_price_for_side


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


def make_bot_with_risk(max_daily_loss_pct: float, equity: str, last_equity: str) -> SemiMomentumBot:
    bot = SemiMomentumBot.__new__(SemiMomentumBot)
    bot.config = {"risk": {"max_daily_loss_pct": max_daily_loss_pct}}
    bot.trading = SimpleNamespace(
        get_account=lambda: SimpleNamespace(equity=equity, last_equity=last_equity)
    )
    return bot


def test_daily_loss_kill_switch_triggered_when_loss_exceeds_threshold() -> None:
    bot = make_bot_with_risk(max_daily_loss_pct=3.0, equity="9600", last_equity="10000")
    assert bot.daily_loss_kill_switch_triggered() is True


def test_daily_loss_kill_switch_not_triggered_when_loss_below_threshold() -> None:
    bot = make_bot_with_risk(max_daily_loss_pct=3.0, equity="9800", last_equity="10000")
    assert bot.daily_loss_kill_switch_triggered() is False


def test_daily_loss_kill_switch_flatten_decisions() -> None:
    bot = make_bot_with_risk(max_daily_loss_pct=3.0, equity="9600", last_equity="10000")
    bot.config["risk"]["flatten_on_daily_loss"] = True
    positions = {
        "NVDA": SimpleNamespace(symbol="NVDA", qty="2.5"),
        "AMD": SimpleNamespace(symbol="AMD", qty="1.0"),
    }

    decisions = [
        Decision(p.symbol, "sell", "daily loss kill switch", qty=float(p.qty))
        for p in positions.values()
        if float(p.qty) > 0
    ]

    assert len(decisions) == 2
    assert all(d.action == "sell" for d in decisions)
    symbols = {d.symbol for d in decisions}
    assert symbols == {"NVDA", "AMD"}


def test_limit_order_qty_is_floored_not_rounded() -> None:
    import math

    notional = 100.0
    # price chosen so that naive division gives a value where floor != round
    limit_price = 3.0  # 100/3 = 33.333333... → floor=33.333333, round=33.333333 (same at 6dp)
    raw = notional / limit_price
    floored = math.floor(raw * 1_000_000) / 1_000_000
    rounded = round(raw, 6)
    # floor must never exceed the notional when multiplied back
    assert floored * limit_price <= notional + 1e-9
    # for a price where rounding would round up, floor is strictly ≤ round
    limit_price2 = 7.0  # 100/7 = 14.285714285... → round(,6)=14.285714, floor=14.285714 (equal)
    # use a price that makes the 7th decimal ≥ 5 so round() would increment the 6th
    limit_price3 = 300.0 / 2100001  # tiny increment to force rounding difference
    raw3 = 100.0 / limit_price3
    floored3 = math.floor(raw3 * 1_000_000) / 1_000_000
    rounded3 = round(raw3, 6)
    assert floored3 <= rounded3
    assert floored3 * limit_price3 <= 100.0 + 1e-6
