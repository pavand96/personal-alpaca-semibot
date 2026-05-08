"""
Performance ledger — persistent SQLite record of fills, equity, and benchmark.

All timestamps are ISO-8601 in Eastern Time.

Tables
------
fills              every submitted (or dry-run) order with reason
equity_snapshots   periodic account-value snapshots
benchmark_prices   reference prices at session start/stop
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
DEFAULT_DB = "logs/ledger.db"
_W = 78  # display width

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    qty         REAL    NOT NULL,
    fill_price  REAL    NOT NULL,
    notional    REAL    NOT NULL,
    reason      TEXT    NOT NULL,
    strategy    TEXT    NOT NULL DEFAULT 'spike',
    order_id    TEXT,
    dry_run     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    TEXT NOT NULL,
    cash                  REAL,
    position_market_value REAL,
    total_equity          REAL,
    unrealized_pnl        REAL
);
CREATE TABLE IF NOT EXISTS benchmark_prices (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT    NOT NULL,
    symbol TEXT    NOT NULL,
    price  REAL    NOT NULL
);
"""


# ── DB helpers ────────────────────────────────────────────────────────────────


def _open_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now_et() -> str:
    return datetime.now(_ET).isoformat(timespec="seconds")


# ── Public write API ──────────────────────────────────────────────────────────


def record_fill(
    db_path: str,
    symbol: str,
    side: str,
    qty: float,
    fill_price: float,
    notional: float,
    reason: str,
    strategy: str = "spike",
    order_id: str | None = None,
    dry_run: bool = True,
) -> None:
    """Record one fill (buy or sell). Thread-safe; opens its own connection."""
    try:
        conn = _open_db(db_path)
        conn.execute(
            "INSERT INTO fills "
            "(ts,symbol,side,qty,fill_price,notional,reason,strategy,order_id,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                _now_et(), symbol, side.lower(), qty, fill_price,
                notional, reason, strategy, order_id, int(dry_run),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("ledger.record_fill failed: %s", exc)


def record_equity(
    db_path: str,
    cash: float | None,
    position_market_value: float | None,
    total_equity: float | None,
    unrealized_pnl: float | None,
) -> None:
    """Record an account-equity snapshot."""
    try:
        conn = _open_db(db_path)
        conn.execute(
            "INSERT INTO equity_snapshots "
            "(ts,cash,position_market_value,total_equity,unrealized_pnl) "
            "VALUES (?,?,?,?,?)",
            (_now_et(), cash, position_market_value, total_equity, unrealized_pnl),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("ledger.record_equity failed: %s", exc)


def record_benchmark(db_path: str, prices: dict[str, float]) -> None:
    """Record a price snapshot for benchmark symbols (QQQ, watchlist closes, etc.)."""
    if not prices:
        return
    try:
        conn = _open_db(db_path)
        ts = _now_et()
        conn.executemany(
            "INSERT INTO benchmark_prices (ts,symbol,price) VALUES (?,?,?)",
            [(ts, sym, price) for sym, price in prices.items()],
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("ledger.record_benchmark failed: %s", exc)


# ── P&L computation ───────────────────────────────────────────────────────────


def _fifo_pnl(
    fills: list[sqlite3.Row],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """FIFO-match buys to sells.

    Returns:
        closed_trades — list of realized-P&L dicts
        open_lots     — symbol → list of remaining unmatched buy lots
    """
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    closed: list[dict[str, Any]] = []

    for f in fills:
        sym = f["symbol"]
        side = f["side"].lower()
        qty = float(f["qty"])
        price = float(f["fill_price"])

        if side == "buy":
            lots[sym].append({
                "qty": qty,
                "price": price,
                "ts": f["ts"],
                "reason": f["reason"],
                "strategy": f["strategy"],
                "dry_run": bool(f["dry_run"]),
            })
        elif side == "sell":
            remaining = qty
            realized = 0.0
            total_cost = 0.0
            total_matched = 0.0
            entry_ts: str | None = None

            while remaining > 0.0001 and lots[sym]:
                lot = lots[sym][0]
                matched = min(remaining, lot["qty"])
                realized += matched * (price - lot["price"])
                total_cost += matched * lot["price"]
                total_matched += matched
                if entry_ts is None:
                    entry_ts = lot["ts"]
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] < 0.0001:
                    lots[sym].popleft()

            avg_cost = total_cost / total_matched if total_matched > 0 else 0.0
            pct = (price / avg_cost - 1) * 100 if avg_cost > 0 else 0.0
            closed.append({
                "symbol": sym,
                "exit_ts": f["ts"],
                "entry_ts": entry_ts,
                "qty": qty,
                "entry_price": avg_cost,
                "exit_price": price,
                "realized_pnl": realized,
                "realized_pnl_pct": pct,
                "exit_reason": f["reason"],
                "strategy": f["strategy"],
                "dry_run": bool(f["dry_run"]),
            })

    open_lots: dict[str, list[dict[str, Any]]] = {
        sym: list(q) for sym, q in lots.items() if q
    }
    return closed, open_lots


def _resolve_prices(
    symbols: list[str],
    bench_rows: list[sqlite3.Row],
    api_key: str | None,
    secret_key: str | None,
    config: dict[str, Any] | None,
) -> dict[str, float]:
    """Current prices for open positions: Alpaca API first, benchmark table fallback."""
    prices: dict[str, float] = {}
    if not symbols:
        return prices

    if api_key and secret_key:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockSnapshotRequest

            feed_raw = ((config or {}).get("alpaca", {}).get("data_feed", "sip")).upper()
            feed = DataFeed[feed_raw] if feed_raw in DataFeed.__members__ else DataFeed.SIP
            client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
            snaps = client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=symbols, feed=feed)
            )
            for sym, snap in snaps.items():
                lt = getattr(snap, "latest_trade", None)
                if lt:
                    prices[sym] = float(lt.price)
        except Exception as exc:
            log.debug("ledger: live price fetch failed (%s) — falling back to benchmark table", exc)

    # Fallback: latest benchmark_prices entry per symbol
    latest_bench: dict[str, float] = {}
    for row in bench_rows:
        latest_bench[row["symbol"]] = float(row["price"])
    for sym in symbols:
        if sym not in prices and sym in latest_bench:
            prices[sym] = latest_bench[sym]

    return prices


# ── Display helpers ───────────────────────────────────────────────────────────


def _sep(title: str) -> None:
    pad = max(0, _W - len(title) - 1)
    print(f"\n{title} {'─' * pad}")


def _pnl_str(pnl: float, pct: float | None = None) -> str:
    sign = "+" if pnl >= 0 else ""
    s = f"{sign}${pnl:,.2f}"
    if pct is not None:
        sp = "+" if pct >= 0 else ""
        s += f"  ({sp}{pct:.1f}%)"
    return s


def _ts_short(ts: str | None) -> str:
    """'2026-05-08T04:02:15-04:00' → '05-08 04:02'"""
    if not ts:
        return "—"
    try:
        return ts[5:16].replace("T", " ")
    except Exception:
        return ts[:16]


# ── Main report ───────────────────────────────────────────────────────────────


def print_ledger(
    db_path: str,
    api_key: str | None = None,
    secret_key: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Print a formatted performance report from the ledger database."""
    if not Path(db_path).exists():
        print(
            f"No ledger found at {db_path}.\n"
            "It is created automatically on the first fill or session start."
        )
        return

    conn = _open_db(db_path)
    fills = conn.execute("SELECT * FROM fills ORDER BY ts, id").fetchall()
    equity_rows = conn.execute("SELECT * FROM equity_snapshots ORDER BY ts").fetchall()
    bench_rows = conn.execute("SELECT * FROM benchmark_prices ORDER BY ts").fetchall()
    conn.close()

    # Compute P&L
    closed_trades, open_lots = _fifo_pnl(fills)
    curr_prices = _resolve_prices(
        list(open_lots.keys()), bench_rows, api_key, secret_key, config
    )

    # Aggregate unrealized
    total_unrealized = 0.0
    unrealized_complete = True
    open_detail: list[dict[str, Any]] = []
    for sym, lots in sorted(open_lots.items()):
        total_qty = sum(lot["qty"] for lot in lots)
        total_cost = sum(lot["qty"] * lot["price"] for lot in lots)
        avg_cost = total_cost / total_qty if total_qty > 0 else 0.0
        curr = curr_prices.get(sym)
        if curr is not None:
            unrlzd = total_qty * (curr - avg_cost)
            unrlzd_pct: float | None = (curr / avg_cost - 1) * 100 if avg_cost > 0 else 0.0
            total_unrealized += unrlzd
        else:
            unrlzd = None
            unrlzd_pct = None
            unrealized_complete = False
        open_detail.append({
            "symbol": sym,
            "qty": total_qty,
            "avg_cost": avg_cost,
            "notional": total_cost,
            "curr": curr,
            "unrealized_pnl": unrlzd,
            "unrealized_pnl_pct": unrlzd_pct,
            "entry_ts": lots[0]["ts"] if lots else None,
            "reason": lots[0]["reason"] if lots else "",
        })

    total_realized = sum(t["realized_pnl"] for t in closed_trades)

    # ── Header ────────────────────────────────────────────────────────────────
    all_rows = list(fills) + list(equity_rows)
    ts_start = fills[0]["ts"] if fills else (equity_rows[0]["ts"] if equity_rows else "")
    ts_end = fills[-1]["ts"] if fills else (equity_rows[-1]["ts"] if equity_rows else "")
    ts_range = f"{_ts_short(ts_start)} → {_ts_short(ts_end)}" if ts_start else ""

    print("═" * _W)
    print(f"  PERFORMANCE LEDGER   {db_path}")
    if ts_range:
        print(f"  {ts_range}")
    print("═" * _W)

    # ── Fills ─────────────────────────────────────────────────────────────────
    buys = [f for f in fills if f["side"] == "buy"]
    sells = [f for f in fills if f["side"] == "sell"]
    _sep(f"FILLS  ({len(fills)} total  ·  {len(buys)} buys  ·  {len(sells)} sells)")

    if fills:
        print(
            f"  {'#':>3}  {'Time':11}  {'Sym':5}  {'Side':4}  "
            f"{'Qty':>7}  {'Price':>10}  {'Notional':>10}  Reason"
        )
        for i, f in enumerate(fills, 1):
            dr = " [DR]" if f["dry_run"] else ""
            print(
                f"  {i:>3}  {_ts_short(f['ts']):11}  {f['symbol']:5}  {f['side'].upper():4}  "
                f"{f['qty']:>7.3f}  ${f['fill_price']:>9,.2f}  ${f['notional']:>9,.2f}  "
                f"{f['reason']}{dr}"
            )
    else:
        print("  (no fills recorded yet)")

    # ── Closed trades ─────────────────────────────────────────────────────────
    _sep(f"CLOSED TRADES  (realized P&L · {len(closed_trades)} trades)")

    if closed_trades:
        print(
            f"  {'Sym':5}  {'Entry':11}  {'Exit':11}  "
            f"{'Qty':>6}  {'Entry$':>9}  {'Exit$':>9}  P&L"
        )
        for t in closed_trades:
            dr = " [DR]" if t["dry_run"] else ""
            print(
                f"  {t['symbol']:5}  {_ts_short(t['entry_ts']):11}  {_ts_short(t['exit_ts']):11}  "
                f"{t['qty']:>6.3f}  ${t['entry_price']:>8,.2f}  ${t['exit_price']:>8,.2f}  "
                f"{_pnl_str(t['realized_pnl'], t['realized_pnl_pct'])}{dr}"
            )
            print(f"        exit reason: {t['exit_reason']}")
        print(f"\n  Total realized: {_pnl_str(total_realized)}")
    else:
        print("  (no closed trades yet)")

    # ── Open positions ────────────────────────────────────────────────────────
    _sep(f"OPEN POSITIONS  ({len(open_detail)} symbols)")

    if open_detail:
        print(
            f"  {'Sym':5}  {'Qty':>7}  {'Avg Cost':>9}  {'Notional':>10}  "
            f"{'Curr$':>9}  Unrlzd P&L  Entry / Reason"
        )
        for pos in open_detail:
            curr_s = f"${pos['curr']:>8,.2f}" if pos["curr"] is not None else "n/a      "
            unrlzd_s = (
                _pnl_str(pos["unrealized_pnl"], pos["unrealized_pnl_pct"])
                if pos["unrealized_pnl"] is not None
                else "n/a"
            )
            print(
                f"  {pos['symbol']:5}  {pos['qty']:>7.3f}  ${pos['avg_cost']:>8,.2f}  "
                f"${pos['notional']:>9,.2f}  {curr_s}  {unrlzd_s}"
            )
            print(f"        entered {_ts_short(pos['entry_ts'])}  reason: {pos['reason']}")
        note = _pnl_str(total_unrealized) if unrealized_complete else f"partial (≥{_pnl_str(total_unrealized)})"
        print(f"\n  Unrealized total: {note}")
    else:
        print("  (no open positions)")

    # ── Equity snapshots ──────────────────────────────────────────────────────
    _sep(f"ACCOUNT EQUITY  ({len(equity_rows)} snapshots)")

    if equity_rows:
        print(
            f"  {'Time':19}  {'Cash':>12}  {'Positions':>12}  {'Total':>12}  Unrlzd P&L"
        )
        for row in equity_rows:
            def _m(v: Any, fmt: str = ",.2f") -> str:
                return f"${float(v):{fmt}}" if v is not None else "n/a"
            print(
                f"  {_ts_short(row['ts']):19}  {_m(row['cash']):>12}  "
                f"{_m(row['position_market_value']):>12}  {_m(row['total_equity']):>12}  "
                f"{_m(row['unrealized_pnl'])}"
            )
        first_eq = equity_rows[0]["total_equity"]
        last_eq = equity_rows[-1]["total_equity"]
        if first_eq and last_eq:
            delta = float(last_eq) - float(first_eq)
            pct = delta / float(first_eq) * 100 if float(first_eq) > 0 else 0.0
            print(
                f"\n  Account P&L: {_pnl_str(delta, pct)}"
                f"  (initial ${float(first_eq):,.2f} → current ${float(last_eq):,.2f})"
            )
    else:
        print("  (no equity snapshots yet)")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if bench_rows:
        first_prices: dict[str, float] = {}
        latest_prices_b: dict[str, float] = {}
        for row in bench_rows:
            sym = row["symbol"]
            p = float(row["price"])
            first_prices.setdefault(sym, p)
            latest_prices_b[sym] = p

        _sep(f"BENCHMARK  ({len(first_prices)} symbols · since {_ts_short(bench_rows[0]['ts'])})")
        print(f"  {'Symbol':6}  {'Start ($)':>12}  {'Latest ($)':>12}  Return")
        for sym in sorted(first_prices):
            start_p = first_prices[sym]
            latest_p = latest_prices_b[sym]
            ret = (latest_p / start_p - 1) * 100 if start_p > 0 else 0.0
            sign = "+" if ret >= 0 else ""
            print(f"  {sym:6}  ${start_p:>11,.2f}  ${latest_p:>11,.2f}  {sign}{ret:.2f}%")

    # ── P&L summary ───────────────────────────────────────────────────────────
    _sep("P&L SUMMARY")

    n_closed = len(closed_trades)
    n_open = len(open_detail)
    print(
        f"  Realized:    {_pnl_str(total_realized):<28}"
        f"  ({n_closed} closed {'trade' if n_closed == 1 else 'trades'})"
    )
    if unrealized_complete or not open_detail:
        print(
            f"  Unrealized:  {_pnl_str(total_unrealized):<28}"
            f"  ({n_open} open {'position' if n_open == 1 else 'positions'})"
        )
    else:
        print(
            f"  Unrealized:  partial ≥{_pnl_str(total_unrealized):<20}"
            f"  ({n_open} open positions, some prices unavailable)"
        )

    initial_eq = float(equity_rows[0]["total_equity"]) if equity_rows and equity_rows[0]["total_equity"] else None
    if initial_eq and (unrealized_complete or not open_detail):
        net = total_realized + total_unrealized
        net_pct = net / initial_eq * 100 if initial_eq > 0 else 0.0
        print(
            f"  Net:         {_pnl_str(net, net_pct):<28}"
            f"  (on ${initial_eq:,.2f} initial equity)"
        )

    print()
