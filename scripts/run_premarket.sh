#!/usr/bin/env bash
# Pre-market gap scanner for semiconductor watchlist.
#
# Runs every 5 minutes starting at 8:00 AM ET (via cron).
# Detects stocks gapping up >= premarket_min_gap_pct vs prior close and places
# extended-hours limit orders. The loop exits automatically when market opens.
#
# To activate real orders: add --execute flag (currently dry-run only).

set -euo pipefail

PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/premarket_${TODAY}.log"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" >> "$LOG_FILE"; }

log "INFO  pre-market gap scan started"

# Dry-run: remove --execute once you have validated behavior over several days.
.venv/bin/python main.py premarket-run >> "$LOG_FILE" 2>&1

log "INFO  pre-market scan loop finished (market opened)"
