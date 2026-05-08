#!/usr/bin/env bash
# Stop an active spike stream for a scheduled handoff.
#
# Usage: stop_spike.sh [premarket|afterhours]

set -euo pipefail

WINDOW="${1:-unknown}"
PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/spike_${WINDOW}_${TODAY}.log"

mkdir -p "${PROJECT_DIR}/logs"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" >> "$LOG_FILE"; }

log "INFO  requested spike-stream stop (window=$WINDOW)"

if pkill -TERM -f "main.py spike-stream"; then
    log "INFO  sent TERM to spike-stream process(es)"
else
    log "INFO  no active spike-stream process found"
fi
