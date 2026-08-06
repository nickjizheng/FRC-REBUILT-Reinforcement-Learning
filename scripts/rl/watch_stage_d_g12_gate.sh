#!/bin/bash
# Enforce the first-40 deterministic promotion/stop gate for Gen-12.
set -u
OUT=${1:?run directory}
RESULT="$OUT/gate_40det.json"

/root/venv/bin/python /root/watch_stage_d_g12_gate.py \
  --run-dir "$OUT" --result "$RESULT" --poll-seconds 120
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "GEN12_GATE_PASS $(date -Is)"
  exit 0
fi

echo "GEN12_GATE_FAIL $(date -Is)"
mkdir -p /root/preserved
if [ -s "$OUT/latest.pt" ]; then
  cp -p "$OUT/latest.pt" "/root/preserved/stageD_g12_gate_failed_$(date +%Y%m%d_%H%M%S).pt"
fi
pkill -f '[t]rain_watchdog_v2.sh' 2>/dev/null || true
touch /root/STOP_STAGE_BLUE2
exit "$rc"
