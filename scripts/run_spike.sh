#!/usr/bin/env bash
# Real-time extended-hours spike detector.
#
# Usage: run_spike.sh [premarket|afterhours]
#   premarket  — started at 4:00 AM ET; stream exits when market opens at 9:30 AM
#   afterhours — started at 4:15 PM ET; stream exits at 7:45 PM ET
#
# Phase 1 — dry-run observation:  2026-05-08 → 2026-06-05  (20 trading days)
# Phase 2 — paper execution:      2026-06-08 → 2026-07-07  (20 trading days)
#
# Uses Alpaca WebSocket trade stream (spike-stream command).
# On every incoming trade, gap = (trade.price / last_regular_close) - 1.
# Fires limit order within ~200ms when gap >= spike_min_gap_pct AND
# spike_sector_confirm distinct symbols are simultaneously gapping.

set -euo pipefail

WINDOW="${1:-unknown}"
PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

DRY_RUN_START="2026-05-08"
DRY_RUN_END="2026-06-05"
PAPER_START="2026-06-08"
PAPER_END="2026-07-07"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/spike_${WINDOW}_${TODAY}.log"
LOCK_FILE="/tmp/semibot_spike_${WINDOW}_${TODAY}.lock"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" | tee -a "$LOG_FILE"; }

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "INFO  spike-stream ($WINDOW) already running — exiting"
    exit 0
fi

if [[ "$TODAY" < "$DRY_RUN_START" ]]; then
    log "INFO  not yet started (dry-run begins $DRY_RUN_START)"
    exit 0
fi
if [[ "$TODAY" > "$PAPER_END" ]]; then
    log "INFO  scheduled window ended ($PAPER_END)"
    exit 0
fi
if [[ "$TODAY" > "$DRY_RUN_END" && "$TODAY" < "$PAPER_START" ]]; then
    log "INFO  between phases ($DRY_RUN_END -> $PAPER_START)"
    exit 0
fi

if [[ ! "$TODAY" > "$DRY_RUN_END" ]]; then
    PHASE="DRY-RUN spike stream (phase 1/2, ends $DRY_RUN_END)"
    EXECUTE_FLAG=""
else
    PHASE="PAPER EXECUTION spike stream (phase 2/2, ends $PAPER_END)"
    EXECUTE_FLAG="--execute"
fi

log "INFO  $PHASE (window=$WINDOW)"

# shellcheck disable=SC2086
.venv/bin/python main.py spike-stream $EXECUTE_FLAG >> "$LOG_FILE" 2>&1

log "INFO  spike-stream stopped (window=$WINDOW)"
