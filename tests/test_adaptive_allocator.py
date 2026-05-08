from datetime import UTC, date, datetime, timedelta

from semibot.adaptive_allocator import (
    AdaptiveCandidate,
    apply_buzz_earnings_overlay,
    detect_rebound_context,
    rank_rebound_candidates,
    should_retain_winner,
)
from semibot.backtest import BacktestPosition, DailyBar


def _bar(symbol: str, close: float = 100.0) -> DailyBar:
    return DailyBar(
        symbol=symbol,
        timestamp=datetime(2026, 1, 5, tzinfo=UTC),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000,
    )


def _series(symbol: str, closes: list[float]) -> list[DailyBar]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        DailyBar(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1_000_000 + index,
        )
        for index, close in enumerate(closes)
    ]


def _candidate(symbol: str, score: float) -> AdaptiveCandidate:
    return AdaptiveCandidate(
        symbol=symbol,
        score=score,
        fast_return_pct=5.0,
        slow_return_pct=10.0,
        volume_ratio=1.5,
    )


def _settings(**overrides):
    return {
        "enabled": True,
        "news_lookback_days": 7,
        "article_count_cap": 5,
        "article_weight": 0.4,
        "positive_keyword_weight": 1.0,
        "negative_keyword_weight": 2.0,
        "score_weight": 2.0,
        "max_score_boost": 20.0,
        "negative_score_block_threshold": -3.0,
        "block_new_entries_near_earnings": True,
        "earnings_avoid_days_before": 2,
        "earnings_avoid_days_after": 1,
        "allow_earnings_if_profit_cushion_pct": 5.0,
        "positive_keywords": ["upgrade", "strong demand"],
        "negative_keywords": ["downgrade", "investigation"],
        **overrides,
    }


def test_buzz_overlay_boosts_positive_news_candidate() -> None:
    candidates = [_candidate("NVDA", 10.0), _candidate("AMD", 12.0)]
    positions = {"NVDA": BacktestPosition(), "AMD": BacktestPosition()}
    bars = {"NVDA": _bar("NVDA"), "AMD": _bar("AMD")}
    news = {
        "NVDA": [(date(2026, 1, 3), "analyst upgrade after strong demand")],
        "AMD": [],
    }

    adjusted = apply_buzz_earnings_overlay(
        candidates=candidates,
        current_date=date(2026, 1, 5),
        positions=positions,
        today_bars=bars,
        news_by_symbol=news,
        earnings_by_symbol={},
        settings=_settings(),
    )

    assert adjusted[0].symbol == "NVDA"
    assert adjusted[0].score > 12.0


def test_buzz_overlay_blocks_negative_news_candidate() -> None:
    candidates = [_candidate("NVDA", 10.0)]
    positions = {"NVDA": BacktestPosition()}
    bars = {"NVDA": _bar("NVDA")}
    news = {"NVDA": [(date(2026, 1, 3), "downgrade and investigation announced")]}

    adjusted = apply_buzz_earnings_overlay(
        candidates=candidates,
        current_date=date(2026, 1, 5),
        positions=positions,
        today_bars=bars,
        news_by_symbol=news,
        earnings_by_symbol={},
        settings=_settings(),
    )

    assert adjusted == []


def test_earnings_overlay_allows_held_position_with_profit_cushion() -> None:
    candidates = [_candidate("NVDA", 10.0)]
    positions = {"NVDA": BacktestPosition(qty=1.0, avg_entry=90.0)}
    bars = {"NVDA": _bar("NVDA", close=100.0)}

    adjusted = apply_buzz_earnings_overlay(
        candidates=candidates,
        current_date=date(2026, 1, 5),
        positions=positions,
        today_bars=bars,
        news_by_symbol={},
        earnings_by_symbol={"NVDA": {date(2026, 1, 6)}},
        settings=_settings(),
    )

    assert [candidate.symbol for candidate in adjusted] == ["NVDA"]


def test_earnings_overlay_blocks_new_entry_near_earnings() -> None:
    candidates = [_candidate("NVDA", 10.0)]
    positions = {"NVDA": BacktestPosition()}
    bars = {"NVDA": _bar("NVDA", close=100.0)}

    adjusted = apply_buzz_earnings_overlay(
        candidates=candidates,
        current_date=date(2026, 1, 5),
        positions=positions,
        today_bars=bars,
        news_by_symbol={},
        earnings_by_symbol={"NVDA": {date(2026, 1, 6)}},
        settings=_settings(),
    )

    assert adjusted == []


def test_should_retain_winner_when_profit_and_momentum_hold() -> None:
    bars = {
        "NVDA": [
            _bar("NVDA", close=90.0),
            _bar("NVDA", close=96.0),
            _bar("NVDA", close=100.0),
            _bar("NVDA", close=108.0),
        ]
    }
    bars["NVDA"][0] = DailyBar("NVDA", datetime(2026, 1, 1, tzinfo=UTC), 90.0, 90.0, 90.0, 90.0, 1)
    bars["NVDA"][1] = DailyBar("NVDA", datetime(2026, 1, 2, tzinfo=UTC), 96.0, 96.0, 96.0, 96.0, 1)
    bars["NVDA"][2] = DailyBar("NVDA", datetime(2026, 1, 3, tzinfo=UTC), 100.0, 100.0, 100.0, 100.0, 1)
    bars["NVDA"][3] = DailyBar("NVDA", datetime(2026, 1, 4, tzinfo=UTC), 108.0, 108.0, 108.0, 108.0, 1)

    assert should_retain_winner(
        symbol="NVDA",
        position=BacktestPosition(qty=1.0, avg_entry=90.0),
        current_price=108.0,
        bars_by_symbol=bars,
        current_date=date(2026, 1, 5),
        fast_lookback=1,
        slow_lookback=3,
        min_profit_pct=7.5,
        min_fast_momentum_pct=-3.0,
        min_slow_momentum_pct=5.0,
    )


def test_rebound_context_activates_after_deep_drawdown_and_repair() -> None:
    closes = [100.0] * 20 + [120.0] * 30 + [90.0, 88.0, 87.0, 89.0, 92.0, 95.0]
    settings = {
        "rebound_mode_enabled": True,
        "sector_drawdown_lookback_days": 30,
        "rebound_fast_lookback_days": 3,
        "rebound_trigger_drawdown_pct": -15.0,
        "rebound_min_sector_fast_return_pct": 3.0,
        "rebound_min_market_fast_return_pct": 0.5,
        "rebound_breadth_sma_days": 5,
        "rebound_min_breadth_pct": 0.5,
    }

    context = detect_rebound_context(
        bars_by_symbol={"NVDA": _series("NVDA", closes), "AMD": _series("AMD", closes)},
        risk_bars_by_symbol={"QQQ": _series("QQQ", [100.0] * 53 + [101.0, 102.0, 103.0])},
        current_date=date(2026, 2, 26),
        settings=settings,
    )

    assert context.active is True
    assert "rebound active" in context.reason


def test_rebound_candidates_rank_short_term_velocity() -> None:
    weak = [100.0] * 65 + [90.0, 91.0, 92.0, 93.0]
    strong = [100.0] * 65 + [80.0, 85.0, 90.0, 95.0]
    settings = {
        "rebound_symbol_lookback_days": 3,
        "rebound_min_symbol_return_pct": 2.0,
        "rebound_volume_weight": 2.0,
        "rebound_drawdown_weight": 0.25,
    }

    candidates = rank_rebound_candidates(
        bars_by_symbol={"WEAK": _series("WEAK", weak), "STRONG": _series("STRONG", strong)},
        current_date=date(2026, 3, 11),
        settings=settings,
    )

    assert [candidate.symbol for candidate in candidates] == ["STRONG", "WEAK"]
