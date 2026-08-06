#!/usr/bin/env bash
# Refresh a seed-mining run's partial summary until every recorded lane exits.
set -euo pipefail

RUN_DIR=${1:?usage: watch_seedmine_summary.sh RUN_DIR [interval_s]}
INTERVAL_S=${2:-30}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
TEXT_OUT="$RUN_DIR/combined_summary.txt"
TMP_OUT="$RUN_DIR/.combined_summary.txt.$$"

cleanup() {
  rm -f "$TMP_OUT"
}
trap cleanup EXIT INT TERM

while true; do
  "$PYTHON_BIN" "$CODE_ROOT/scripts/rl/summarize_seedmine.py" \
    "$RUN_DIR" --write > "$TMP_OUT"
  mv -f "$TMP_OUT" "$TEXT_OUT"

  alive=0
  for pid_file in "$RUN_DIR"/*.pid; do
    [ -e "$pid_file" ] || continue
    [ "$(basename "$pid_file")" = "summary_watch.pid" ] && continue
    pid=$(tr -dc '0-9' < "$pid_file")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [ "$alive" -eq 1 ] || break
  sleep "$INTERVAL_S"
done
