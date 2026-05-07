from datetime import date, datetime, time

from semibot.intraday import IntradayBar, build_entry_setup

TRADING_DAY = date(2026, 5, 7)


def bar(clock: time, open_: float, high: float, low: float, close: float, volume: float = 1000) -> IntradayBar:
    return IntradayBar(
        symbol="NVDA",
        timestamp=datetime.combine(TRADING_DAY, clock),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def history() -> dict[date, list[IntradayBar]]:
    prior = date(2026, 5, 6)
    return {
        prior: [
            IntradayBar("NVDA", datetime.combine(prior, time(9, 30)), 100, 101, 99, 100, 1000),
            IntradayBar("NVDA", datetime.combine(prior, time(9, 45)), 100, 101, 99, 100, 1000),
        ]
    }


def test_build_entry_setup_exits_on_stop_loss_before_target() -> None:
    bars = [
        bar(time(9, 30), 100, 100.5, 99.8, 100, 2000),
        bar(time(9, 45), 101, 102, 100.8, 101, 2000),
        bar(time(9, 46), 101, 101.2, 99.0, 100),
    ]

    trade = build_entry_setup(
        symbol="NVDA",
        trading_day=TRADING_DAY,
        bars=bars,
        history=history(),
        entry_time=time(9, 45),
        exit_time=time(15, 55),
        lookback_days=1,
        min_gain=0.005,
        max_gain=0.04,
        relative_volume_min=1.0,
        per_trade_notional=100.0,
        stop_loss=0.01,
        take_profit=0.02,
    )

    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == 99.99


def test_build_entry_setup_exits_on_take_profit() -> None:
    bars = [
        bar(time(9, 30), 100, 100.5, 99.8, 100, 2000),
        bar(time(9, 45), 101, 102, 100.8, 101, 2000),
        bar(time(9, 46), 101, 104, 100.9, 103),
    ]

    trade = build_entry_setup(
        symbol="NVDA",
        trading_day=TRADING_DAY,
        bars=bars,
        history=history(),
        entry_time=time(9, 45),
        exit_time=time(15, 55),
        lookback_days=1,
        min_gain=0.005,
        max_gain=0.04,
        relative_volume_min=1.0,
        per_trade_notional=100.0,
        stop_loss=0.01,
        take_profit=0.02,
    )

    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == 103.02
