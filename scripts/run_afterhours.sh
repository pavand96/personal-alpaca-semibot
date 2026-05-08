#!/usr/bin/env bash
# After-hours spike scanner for semiconductor watchlist.
#
# Triggers at 4:15 PM ET (via cron). Scans every 5 minutes for symbols
# with an after-hours move >= afterhours_min_gap_pct vs today's 4pm close.
# Places extended-hours limit orders immediately when a spike is detected.
# Loop exits automatically at 7:45pm ET.
#
# IMPORTANT: Currently DRY-RUN only. Add --execute once you have validated
# that the gap thresholds and notional sizes are appropriate.

set -euo pipefail

PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
export TZ="America/New_York"

TODAY="$(date +%F)"
LOG_FILE="${PROJECT_DIR}/logs/afterhours_${TODAY}.log"

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" >> "$LOG_FILE"; }

log "INFO  after-hours spike scanner started"

# Dry-run: add --execute once gap thresholds are validated over several sessions.
.venv/bin/python main.py afterhours-run >> "$LOG_FILE" 2>&1

log "INFO  after-hours scan loop finished"
