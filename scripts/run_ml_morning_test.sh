#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/pavand96/personal-alpaca-semibot"
TARGET_DATE="2026-05-08"
export TZ="America/New_York"

RUN_DATE="${SEMIBOT_TEST_DATE:-$(date +%F)}"
INTERVAL_SECONDS="${SEMIBOT_INTERVAL_SECONDS:-60}"
END_TIME="${SEMIBOT_END_TIME:-16:15}"
RUN_ONCE="${SEMIBOT_RUN_ONCE:-false}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "==== semibot day-long ML dry-run test $(date -Is) ===="

if [ "$RUN_DATE" != "$TARGET_DATE" ]; then
  echo "Skipping: run date $RUN_DATE does not match target date $TARGET_DATE"
  exit 0
fi

while true; do
  now_time="$(date +%H:%M)"
  echo "-- cycle $(date -Is) --"

  echo "-- ml-signal --"
  .venv/bin/python main.py ml-signal

  echo "-- ml-trade-once dry-run --"
  .venv/bin/python main.py ml-trade-once

  if [ "$RUN_ONCE" = "true" ]; then
    break
  fi

  if [[ "$now_time" > "$END_TIME" || "$now_time" == "$END_TIME" ]]; then
    echo "Reached stop time $END_TIME"
    break
  fi

  sleep "$INTERVAL_SECONDS"
done

echo "==== done $(date -Is) ===="
