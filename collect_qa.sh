#!/bin/bash
set -euo pipefail

INTERVAL=${1:-60}

echo "=== QA Collector ==="
echo "Interval: ${INTERVAL}s"

while true; do
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running QA..."

  if python daily_qa.py; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] QA completed successfully"

    # commit if there are changes to qa.json files
    if git diff --quiet -- '*.json' 2>/dev/null; then
      echo "No QA changes to commit"
    else
      git add arxiv/*/qa.json
      git commit -m "auto: update QA data"
      echo "Committed QA changes"
    fi
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] QA failed, retrying next cycle"
  fi

  echo "Sleeping ${INTERVAL}s..."
  sleep "$INTERVAL"
done
