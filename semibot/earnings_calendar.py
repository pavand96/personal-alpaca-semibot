"""
Earnings calendar — fetches upcoming report dates via yfinance.

Used by the spike stream to know which symbols are reporting today so it can
act on BOTH directions:
  - Positive gap (earnings beat)  → buy immediately
  - Negative gap (earnings miss)  → sell existing position immediately

Also used by the news monitor cron to surface upcoming earnings in the
daily signal log so there are no surprises at 4 AM.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

log = logging.getLogger(__name__)


def fetch_earnings_calendar(
    symbols: list[str],
    lookahead_days: int = 14,
) -> dict[str, date]:
    """Return the next earnings date for each symbol within lookahead_days.

    Uses yfinance.  Silently skips symbols where the fetch fails.
    Returns only symbols with an earnings date in [today, today+lookahead_days].
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — earnings calendar unavailable. Run: pip install yfinance")
        return {}

    today = date.today()
    cutoff = today + timedelta(days=lookahead_days)
    result: dict[str, date] = {}

    for symbol in symbols:
        try:
            cal = yf.Ticker(symbol).calendar
            if not cal:
                continue
            raw_dates = cal.get("Earnings Date", [])
            if not raw_dates:
                continue
            if isinstance(raw_dates, date):
                raw_dates = [raw_dates]
            upcoming = [d for d in raw_dates if today <= d <= cutoff]
            if upcoming:
                result[symbol] = min(upcoming)
        except Exception as exc:
            log.debug("Earnings fetch failed for %s: %s", symbol, exc)

    return result


def get_reporting_today(symbols: list[str]) -> list[str]:
    """Return symbols with earnings scheduled today."""
    cal = fetch_earnings_calendar(symbols, lookahead_days=0)
    today = date.today()
    return [s for s, d in cal.items() if d == today]


def get_reporting_soon(symbols: list[str], days: int = 7) -> dict[str, date]:
    """Return all symbols reporting within the next `days` days."""
    return fetch_earnings_calendar(symbols, lookahead_days=days)


def print_earnings_schedule(symbols: list[str], days: int = 14) -> None:
    """Print a formatted upcoming earnings schedule for all watchlist symbols."""
    cal = fetch_earnings_calendar(symbols, lookahead_days=days)
    today = date.today()
    if not cal:
        print(f"  No earnings scheduled in next {days} days for watched symbols")
        return
    print(f"  Upcoming earnings (next {days} days):")
    for sym, d in sorted(cal.items(), key=lambda x: x[1]):
        days_away = (d - today).days
        urgency = "TODAY" if days_away == 0 else f"in {days_away}d"
        print(f"    {sym:6}  {d}  ({urgency})")


def enrich_signals_with_calendar(
    signals: dict[str, dict],
    symbols: list[str],
    days: int = 14,
) -> dict[str, dict]:
    """Add or update earnings_date field in the signals dict from the calendar."""
    cal = fetch_earnings_calendar(symbols, lookahead_days=days)
    today = date.today()
    enriched = dict(signals)
    for sym, d in cal.items():
        days_away = (d - today).days
        entry = enriched.setdefault(sym, {
            "headline": f"Earnings scheduled {d}",
            "detected_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
            "catalyst_type": "earnings_scheduled",
            "keywords": ["earnings"],
            "score": 3 if days_away == 0 else 2,
            "bypass_confirm": days_away == 0,
        })
        entry["earnings_date"] = str(d)
        entry["earnings_days_away"] = days_away
        # Bump score if reporting today
        if days_away == 0 and entry.get("score", 0) < 3:
            entry["score"] = 3
            entry["bypass_confirm"] = True
    return enriched
