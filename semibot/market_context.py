from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from semibot.backtest import Backtester, DailyBar
from semibot.intraday import MARKET_TZ


@dataclass(frozen=True)
class MarketContext:
    as_of_date: date
    risk_off: bool
    market_ok: bool
    sector_return_pct: float
    sector_slow_return_pct: float
    sector_drawdown_pct: float
    reasons: tuple[str, ...]

    @classmethod
    def neutral(cls, as_of_date: date, reason: str) -> MarketContext:
        return cls(
            as_of_date=as_of_date,
            risk_off=False,
            market_ok=True,
            sector_return_pct=0.0,
            sector_slow_return_pct=0.0,
            sector_drawdown_pct=0.0,
            reasons=(reason,),
        )

    def summary(self) -> str:
        state = "risk-off" if self.risk_off else "risk-on"
        reason_text = "; ".join(self.reasons) if self.reasons else "no blocking market context"
        return (
            f"{state} as of {self.as_of_date.isoformat()} "
            f"(sector={self.sector_return_pct:.2f}%, slow={self.sector_slow_return_pct:.2f}%, "
            f"dd={self.sector_drawdown_pct:.2f}%, market_ok={self.market_ok}): {reason_text}"
        )


def build_market_context(
    config: dict[str, Any],
    api_key: str,
    secret_key: str,
    as_of_date: date | None = None,
) -> MarketContext:
    current_date = as_of_date or datetime.now(MARKET_TZ).date()
    settings = config.get("adaptive_semis_allocator")
    if not settings:
        return MarketContext.neutral(current_date, "adaptive market context config unavailable; fail open")

    risk_filter_symbols = [str(symbol).upper() for symbol in settings.get("risk_filter_symbols", [])]
    max_lookback = max(
        int(settings["sector_lookback_days"]),
        int(settings["sector_slow_lookback_days"]),
        int(settings["sector_drawdown_lookback_days"]),
        int(settings["market_momentum_lookback_days"]),
        int(settings["market_sma_days"]),
        30,
    )
    fetch_start = current_date - timedelta(days=max_lookback * 3)
    backtester = Backtester(config, api_key=api_key, secret_key=secret_key)
    bars_by_symbol = backtester.fetch_daily_bars(config["watchlist"], fetch_start, current_date)
    risk_bars_by_symbol = (
        backtester.fetch_daily_bars(risk_filter_symbols, fetch_start, current_date)
        if risk_filter_symbols
        else {}
    )
    return build_market_context_from_bars(
        bars_by_symbol=bars_by_symbol,
        risk_bars_by_symbol=risk_bars_by_symbol,
        settings=settings,
        current_date=current_date,
    )


def build_market_context_from_bars(
    bars_by_symbol: dict[str, list[DailyBar]],
    risk_bars_by_symbol: dict[str, list[DailyBar]],
    settings: dict[str, Any],
    current_date: date,
) -> MarketContext:
    sector_return = sector_return_pct(
        bars_by_symbol,
        current_date,
        int(settings["sector_lookback_days"]),
    )
    sector_slow_return = sector_return_pct(
        bars_by_symbol,
        current_date,
        int(settings["sector_slow_lookback_days"]),
    )
    sector_drawdown = sector_drawdown_pct(
        bars_by_symbol,
        current_date,
        int(settings["sector_drawdown_lookback_days"]),
    )
    market_ok = market_filter_allows_entries(
        bars_by_symbol=risk_bars_by_symbol,
        current_date=current_date,
        momentum_lookback=int(settings["market_momentum_lookback_days"]),
        sma_days=int(settings["market_sma_days"]),
        min_momentum_pct=float(settings["min_market_momentum_pct"]),
        require_above_sma=bool(settings["require_market_above_sma"]),
    )

    min_sector_slow_return = float(settings["min_sector_slow_return_pct"])
    risk_off_sector_return = float(settings["risk_off_sector_return_pct"])
    max_sector_drawdown = float(settings["max_sector_drawdown_pct"])

    reasons: list[str] = []
    if sector_return <= risk_off_sector_return:
        reasons.append(f"sector {sector_return:.2f}% <= risk-off {risk_off_sector_return:.2f}%")
    if sector_slow_return < min_sector_slow_return:
        reasons.append(f"slow sector {sector_slow_return:.2f}% < minimum {min_sector_slow_return:.2f}%")
    if sector_drawdown <= max_sector_drawdown:
        reasons.append(f"sector drawdown {sector_drawdown:.2f}% <= maximum {max_sector_drawdown:.2f}%")
    if not market_ok:
        reasons.append("market proxy trend blocked entries")

    return MarketContext(
        as_of_date=current_date,
        risk_off=bool(reasons),
        market_ok=market_ok,
        sector_return_pct=sector_return,
        sector_slow_return_pct=sector_slow_return,
        sector_drawdown_pct=sector_drawdown,
        reasons=tuple(reasons) or ("market context allows entries",),
    )


def sector_return_pct(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    lookback_days: int,
) -> float:
    returns: list[float] = []
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= lookback_days:
            continue
        latest = prior[-1]
        base = prior[-lookback_days - 1]
        if base.close > 0:
            returns.append(((latest.close / base.close) - 1) * 100)
    return sum(returns) / len(returns) if returns else 0.0


def sector_drawdown_pct(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    lookback_days: int,
) -> float:
    drawdowns: list[float] = []
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) < lookback_days:
            continue
        window = prior[-lookback_days:]
        peak = max(bar.close for bar in window)
        latest = window[-1].close
        if peak > 0:
            drawdowns.append(((latest - peak) / peak) * 100)
    return sum(drawdowns) / len(drawdowns) if drawdowns else 0.0


def market_filter_allows_entries(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    momentum_lookback: int,
    sma_days: int,
    min_momentum_pct: float,
    require_above_sma: bool,
) -> bool:
    if not bars_by_symbol:
        return True

    usable_symbols = 0
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= max(momentum_lookback, sma_days):
            continue
        usable_symbols += 1
        latest = prior[-1]
        momentum_base = prior[-momentum_lookback - 1]
        if momentum_base.close <= 0:
            return False
        momentum_pct = ((latest.close / momentum_base.close) - 1) * 100
        if momentum_pct < min_momentum_pct:
            return False
        if require_above_sma:
            sma = sum(bar.close for bar in prior[-sma_days:]) / sma_days
            if latest.close < sma:
                return False

    return usable_symbols > 0
