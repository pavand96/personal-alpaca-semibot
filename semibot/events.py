from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "timestamp",
    "event",
    "symbol",
    "price",
    "previous_close",
    "change_pct",
    "action",
    "quantity",
    "notional",
    "reason",
    "decision",
    "skip_reason",
    "source",
    "bucket",
    "gap_pct",
    "spread_pct",
    "max_spread_pct",
    "dollar_volume",
    "min_dollar_volume",
    "trade_count",
    "min_trade_count",
    "single_trade_value",
    "confirmation_count",
    "confirmation_required",
    "limit_price",
    "qty",
    "strategy",
]


def _fieldnames_for_path(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return list(FIELDNAMES)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        existing = next(reader, [])

    fieldnames = list(FIELDNAMES)
    fieldnames.extend(field for field in existing if field and field not in fieldnames)
    if existing != fieldnames:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    return fieldnames


def append_event(log_file: str | Path, event: dict[str, Any]) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = _fieldnames_for_path(path)

    row = {field: "" for field in fieldnames}
    row.update(event)
    row["timestamp"] = row.get("timestamp") or datetime.utcnow().isoformat(timespec="seconds")

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
