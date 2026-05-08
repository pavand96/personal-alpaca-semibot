#!/usr/bin/env bash
# Extended-hours spike scanner — covers pre-market (4am–9:30am) and after-hours (4:15pm–7:45pm ET).
#
# Usage: run_spike.sh [premarket|afterhours]
#   premarket  — started at 4:00 AM ET; loop exits when market opens
#   afterhours — started at 4:15 PM ET; loop exits at 7:45 PM ET
#
# Both windows use the same spike-run command; the time-aware reference logic inside
# fetch_extended_hours_gaps() picks the right close price automatically.
#
# Scans every spike_interval_seconds (default 60s).
# Symbols already ordered this session are not re-entered.
#
# IMPORTANT: Currently DRY-RUN. Add --execute once you have observed a few sessions
# of dry-run output and confirmed the thresholds and notional are appropriate.

set -euo pipefail

WINDOW="${1:-unknown}"
PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/spike_${WINDOW}_${TODAY}.log"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" | tee -a "$LOG_FILE"; }

log "INFO  spike scanner started (window=$WINDOW)"

# Remove --dry-run flag (or add --execute) once thresholds are validated.
.venv/bin/python main.py spike-run >> "$LOG_FILE" 2>&1

log "INFO  spike scanner finished (window=$WINDOW)"
