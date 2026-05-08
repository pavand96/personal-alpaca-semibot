"""
Background news monitor — runs every 15 min all day/night via cron.

Polls Alpaca news for every watchlist symbol, scores catalyst signals, and
writes them to a JSON signals file (logs/news_signals.json).  The spike
stream reads this file at startup so it's already pre-warmed with context
from hours before the session begins.

Signal schema (one entry per symbol):
  {
    "NVDA": {
      "headline":      "NVIDIA beats Q1 EPS by 12%, raises guidance",
      "detected_at":   "2026-05-07T17:32:14",
      "catalyst_type": "earnings",          # earnings | upgrade | corporate | general
      "keywords":      ["earnings", "beat", "guidance"],
      "score":         3,                   # 1–5; higher → more confidence
      "bypass_confirm": true                # pre-flag for spike stream
    }
  }

Signals older than news_signal_ttl_hours (default 20h) are removed on each
run so the file only contains actionable context for the next session.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

log = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")

# Catalyst classification — order matters: more specific first
_EARNINGS_KEYWORDS = frozenset({
    "earnings", "eps", "quarterly", "quarter",
    "q1", "q2", "q3", "q4", "beats", "beat", "misses", "missed",
    "revenue", "results", "profit",
})
_UPGRADE_KEYWORDS = frozenset({
    "upgrade", "upgraded", "raises", "raised", "price target",
    "overweight", "outperform", "buy rating", "initiates",
})
_CORPORATE_KEYWORDS = frozenset({
    "acquisition", "merger", "buyout", "contract", "partnership",
    "fda", "approval", "approved", "breakthrough", "guidance", "outlook",
})
_ALL_CATALYST_KEYWORDS = _EARNINGS_KEYWORDS | _UPGRADE_KEYWORDS | _CORPORATE_KEYWORDS


def _classify(text: str) -> tuple[str, list[str], int]:
    """Return (catalyst_type, matched_keywords, score)."""
    matched = [kw for kw in _ALL_CATALYST_KEYWORDS if kw in text]
    if any(kw in text for kw in _EARNINGS_KEYWORDS):
        score = min(5, 2 + len([k for k in matched if k in _EARNINGS_KEYWORDS]))
        return "earnings", matched, score
    if any(kw in text for kw in _UPGRADE_KEYWORDS):
        return "upgrade", matched, 2
    if any(kw in text for kw in _CORPORATE_KEYWORDS):
        return "corporate", matched, 2
    return "general", matched, 1


def _load_signals(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_signals(path: str, signals: dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(signals, f, indent=2, default=str)


def _purge_stale(signals: dict[str, dict], ttl_hours: int) -> dict[str, dict]:
    """Remove signals older than ttl_hours."""
    cutoff = datetime.now(UTC) - timedelta(hours=ttl_hours)
    fresh = {}
    for sym, sig in signals.items():
        try:
            detected = datetime.fromisoformat(sig["detected_at"])
            if detected.tzinfo is None:
                detected = detected.replace(tzinfo=UTC)
            if detected >= cutoff:
                fresh[sym] = sig
        except (KeyError, ValueError):
            pass
    return fresh


def scan_news(
    config: dict[str, Any],
    api_key: str,
    secret_key: str,
    lookback_minutes: int = 30,
) -> dict[str, dict]:
    """Fetch recent news for all watchlist symbols and return new signals detected."""
    now = datetime.now(UTC)
    since = now - timedelta(minutes=lookback_minutes)
    client = NewsClient(api_key=api_key, secret_key=secret_key)
    new_signals: dict[str, dict] = {}

    for symbol in config["watchlist"]:
        try:
            req = NewsRequest(
                symbols=symbol,
                start=since,
                end=now,
                limit=10,
                include_content=False,
            )
            resp = client.get_news(req)
            articles = getattr(resp, "news", getattr(resp, "data", []))
            if isinstance(articles, dict):
                articles = [
                    a for vals in articles.values()
                    for a in (vals if isinstance(vals, list) else [vals])
                ]
            for article in (articles or []):
                headline = (getattr(article, "headline", "") or "").lower()
                summary = (getattr(article, "summary", "") or "").lower()
                text = headline + " " + summary
                if not any(kw in text for kw in _ALL_CATALYST_KEYWORDS):
                    continue
                catalyst_type, keywords, score = _classify(text)
                bypass = catalyst_type == "earnings" and score >= 3
                new_signals[symbol] = {
                    "headline": getattr(article, "headline", ""),
                    "detected_at": now.isoformat(),
                    "catalyst_type": catalyst_type,
                    "keywords": keywords[:6],
                    "score": score,
                    "bypass_confirm": bypass,
                }
                break  # one signal per symbol per scan is enough
        except Exception as exc:
            log.warning("News fetch failed for %s: %s", symbol, exc)

    return new_signals


def run_once(config: dict[str, Any], api_key: str, secret_key: str) -> None:
    """Main entry point for cron: scan news, merge calendar, purge, save, print summary."""
    from semibot.earnings_calendar import enrich_signals_with_calendar, print_earnings_schedule

    settings = config.get("adaptive_semis_allocator", {})
    signals_path = str(settings.get("news_signals_file", "logs/news_signals.json"))
    lookback = int(settings.get("news_monitor_lookback_minutes", 30))
    ttl_hours = int(settings.get("news_signal_ttl_hours", 20))
    bypass_gap = float(settings.get("earnings_bypass_gap_pct", 8.0))
    lookahead = int(settings.get("earnings_lookahead_days", 7))

    now_str = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{now_str}] News monitor scan (lookback={lookback}min, ttl={ttl_hours}h)")

    # 1. News-driven signals
    existing = _load_signals(signals_path)
    new_signals = scan_news(config, api_key, secret_key, lookback_minutes=lookback)
    merged = {**existing, **new_signals}
    merged = _purge_stale(merged, ttl_hours)

    # 2. Enrich with earnings calendar (adds/updates earnings_date field)
    merged = enrich_signals_with_calendar(merged, config["watchlist"], days=lookahead)

    _save_signals(signals_path, merged)

    if new_signals:
        print(f"  New news signals: {len(new_signals)}")
        for sym, sig in sorted(new_signals.items(), key=lambda x: -x[1]["score"]):
            bypass_note = f"  [BYPASS ≥{bypass_gap:.0f}%]" if sig["bypass_confirm"] else ""
            print(f"  {sym:6}  score={sig['score']}  type={sig['catalyst_type']:10}{bypass_note}  {sig['headline'][:70]}")
    else:
        print(f"  No new news signals  (active in file: {len(merged)})")

    # 3. Print upcoming earnings schedule
    print(f"  Earnings calendar (next {lookahead} days):")
    print_earnings_schedule(config["watchlist"], days=lookahead)

    if merged:
        today_signals = [s for s, sig in merged.items() if sig.get("earnings_days_away") == 0]
        if today_signals:
            print(f"  *** REPORTING TODAY: {', '.join(sorted(today_signals))} — stream will be bidirectional ***")


def load_signals_file(path: str) -> dict[str, dict]:
    """Read the signals file for use by the spike stream at startup."""
    return _load_signals(path)
