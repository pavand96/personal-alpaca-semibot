from semibot.backtest import apply_slippage


def test_apply_slippage_makes_buys_more_expensive() -> None:
    assert apply_slippage(100.0, "buy", 5.0) == 100.05


def test_apply_slippage_makes_sells_less_expensive() -> None:
    assert apply_slippage(100.0, "sell", 5.0) == 99.95
