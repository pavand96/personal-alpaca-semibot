#!/usr/bin/env bash
# Daily pre-market spike runner.
#
# Starts during pre-market and lets main.py premarket-run scan until the market opens.
# Phase 1 is dry-run; phase 2 passes --execute for paper execution.

set -euo pipefail

PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

DRY_RUN_START="2026-05-08"
DRY_RUN_END="2026-06-05"
PAPER_START="2026-06-08"
PAPER_END="2026-07-07"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/premarket_trade_${TODAY}.log"
LOCK_FILE="/tmp/semibot_premarket_${TODAY}.lock"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" >> "$LOG_FILE"; }

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
    PHASE="DRY-RUN premarket spike scan (phase 1/2, ends $DRY_RUN_END)"
    EXECUTE_FLAG=""
else
    PHASE="PAPER EXECUTION premarket spike scan (phase 2/2, ends $PAPER_END)"
    EXECUTE_FLAG="--execute"
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "INFO  premarket scanner already running"
    exit 0
fi

log "INFO  $PHASE"

# shellcheck disable=SC2086
.venv/bin/python main.py premarket-run $EXECUTE_FLAG >> "$LOG_FILE" 2>&1

log "INFO  done"
