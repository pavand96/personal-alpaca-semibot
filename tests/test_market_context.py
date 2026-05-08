from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from semibot.backtest import DailyBar
from semibot.market_context import build_market_context_from_bars


def _settings() -> dict:
    return {
        "sector_lookback_days": 3,
        "sector_slow_lookback_days": 5,
        "sector_drawdown_lookback_days": 5,
        "market_momentum_lookback_days": 3,
        "market_sma_days": 3,
        "min_market_momentum_pct": 0.0,
        "require_market_above_sma": True,
        "risk_off_sector_return_pct": -5.0,
        "min_sector_slow_return_pct": -1.0,
        "max_sector_drawdown_pct": -10.0,
    }


def _bars(symbol: str, closes: list[float], start: date = date(2026, 1, 1)) -> list[DailyBar]:
    result: list[DailyBar] = []
    for index, close in enumerate(closes):
        timestamp = datetime.combine(start + timedelta(days=index), datetime.min.time(), tzinfo=UTC)
        result.append(
            DailyBar(
                symbol=symbol,
                timestamp=timestamp,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1_000,
            )
        )
    return result


def test_market_context_allows_entries_when_sector_and_market_are_healthy() -> None:
    current_date = date(2026, 1, 8)
    context = build_market_context_from_bars(
        bars_by_symbol={
            "NVDA": _bars("NVDA", [100, 101, 102, 103, 104, 105, 106]),
            "AMD": _bars("AMD", [50, 51, 52, 53, 54, 55, 56]),
        },
        risk_bars_by_symbol={"QQQ": _bars("QQQ", [100, 101, 102, 103, 104, 105, 106])},
        settings=_settings(),
        current_date=current_date,
    )

    assert context.risk_off is False
    assert context.market_ok is True
    assert context.reasons == ("market context allows entries",)


def test_market_context_blocks_when_market_proxy_trend_fails() -> None:
    current_date = date(2026, 1, 8)
    context = build_market_context_from_bars(
        bars_by_symbol={
            "NVDA": _bars("NVDA", [100, 101, 102, 103, 104, 105, 106]),
            "AMD": _bars("AMD", [50, 51, 52, 53, 54, 55, 56]),
        },
        risk_bars_by_symbol={"QQQ": _bars("QQQ", [106, 105, 104, 103, 102, 101, 100])},
        settings=_settings(),
        current_date=current_date,
    )

    assert context.risk_off is True
    assert context.market_ok is False
    assert "market proxy trend blocked entries" in context.reasons


def test_market_context_blocks_when_sector_drawdown_is_too_deep() -> None:
    current_date = date(2026, 1, 8)
    context = build_market_context_from_bars(
        bars_by_symbol={"NVDA": _bars("NVDA", [100, 120, 121, 122, 90, 88, 87])},
        risk_bars_by_symbol={"QQQ": _bars("QQQ", [100, 101, 102, 103, 104, 105, 106])},
        settings=_settings(),
        current_date=current_date,
    )

    assert context.risk_off is True
    assert any("sector drawdown" in reason for reason in context.reasons)
