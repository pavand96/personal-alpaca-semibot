"""Persistent tracker for spike positions so live_decisions can exit them after N days."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def _load(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: str, data: dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def record_spike_entry(path: str, symbol: str, gap_pct: float, entry_price: float) -> None:
    data = _load(path)
    data[symbol] = {
        "entry_date": str(date.today()),
        "gap_pct": round(gap_pct, 2),
        "entry_price": round(entry_price, 4),
    }
    _save(path, data)
    print(f"  [spike_tracker] recorded {symbol} entry_date={date.today()} gap={gap_pct:+.1f}%")


def remove_spike_entry(path: str, symbol: str) -> None:
    data = _load(path)
    if symbol in data:
        data.pop(symbol)
        _save(path, data)


def get_symbols_to_exit(path: str, today: date) -> list[str]:
    """Return symbols whose spike entry was before today — hold time >= 1 session."""
    data = _load(path)
    expired = []
    for symbol, info in data.items():
        entry = date.fromisoformat(info["entry_date"])
        if entry < today:
            expired.append(symbol)
    return expired


def load_all(path: str) -> dict[str, dict]:
    return _load(path)
