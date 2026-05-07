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
]


def append_event(log_file: str | Path, event: dict[str, Any]) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    row = {field: "" for field in FIELDNAMES}
    row.update(event)
    row["timestamp"] = row.get("timestamp") or datetime.utcnow().isoformat(timespec="seconds")

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
