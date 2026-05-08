#!/usr/bin/env bash
# Real-time extended-hours spike detector.
#
# Usage: run_spike.sh [premarket|afterhours]
#   premarket  — started at 4:00 AM ET; stream exits when market opens at 9:30 AM
#   afterhours — started at 4:15 PM ET; stream exits at 7:45 PM ET
#
# Uses Alpaca WebSocket trade stream (spike-stream command).
# On every incoming trade, gap = (trade.price / last_regular_close) - 1.
# If gap >= spike_min_gap_pct, an extended-hours limit order fires immediately (~200ms latency).
# Each symbol is only ordered once per session.
#
# IMPORTANT: Currently DRY-RUN only. Add --execute once you have watched
# several sessions of dry-run output and confirmed the thresholds are correct.

set -euo pipefail

WINDOW="${1:-unknown}"
PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/spike_${WINDOW}_${TODAY}.log"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" | tee -a "$LOG_FILE"; }

log "INFO  spike-stream starting (window=$WINDOW)"

# Add --execute below once dry-run is validated.
.venv/bin/python main.py spike-stream >> "$LOG_FILE" 2>&1

log "INFO  spike-stream stopped (window=$WINDOW)"
