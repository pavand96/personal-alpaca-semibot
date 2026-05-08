#!/usr/bin/env bash
# Daily market-aware ML trade runner.
#
# Phase 1 — dry-run observation:  2026-05-08 → 2026-06-05  (20 trading days)
# Phase 2 — paper execution:      2026-06-08 → 2026-07-07  (20 trading days)
#
# Run via cron every 5 min during market hours (Mon–Fri, ET).
# The bot's own require_market_open guard stops it acting outside NYSE hours.

set -euo pipefail

PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

DRY_RUN_START="2026-05-08"
DRY_RUN_END="2026-06-05"
PAPER_START="2026-06-08"
PAPER_END="2026-07-07"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/ml_trade_${TODAY}.log"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" >> "$LOG_FILE"; }

# Outside both phases: do nothing.
if [[ "$TODAY" < "$DRY_RUN_START" ]]; then
    log "INFO  not yet started (dry-run begins $DRY_RUN_START)"
    exit 0
fi
if [[ "$TODAY" > "$PAPER_END" ]]; then
    log "INFO  scheduled window ended ($PAPER_END)"
    exit 0
fi

# Between phases (the weekend after dry-run ends): skip silently.
if [[ "$TODAY" > "$DRY_RUN_END" && "$TODAY" < "$PAPER_START" ]]; then
    log "INFO  between phases ($DRY_RUN_END → $PAPER_START)"
    exit 0
fi

# Determine execute flag.
# TODAY <= DRY_RUN_END  ≡  ! (TODAY > DRY_RUN_END)
if [[ ! "$TODAY" > "$DRY_RUN_END" ]]; then
    PHASE="DRY-RUN (phase 1/2, ends $DRY_RUN_END)"
    EXECUTE_FLAG=""
else
    PHASE="PAPER EXECUTION (phase 2/2, ends $PAPER_END)"
    EXECUTE_FLAG="--execute"
fi

log "INFO  $PHASE"

# Print current signals first so the log shows what the model sees.
.venv/bin/python main.py ml-signal >> "$LOG_FILE" 2>&1

# Run the strategy. ml-trade-once now blocks fresh buys during risk-off market context.
# shellcheck disable=SC2086
.venv/bin/python main.py ml-trade-once $EXECUTE_FLAG >> "$LOG_FILE" 2>&1

log "INFO  done"
