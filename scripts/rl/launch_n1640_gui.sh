#!/usr/bin/env bash
# Retry-until-boot GUI launcher for the v12 PAYLOAD policy (best-ceiling model:
# det max 113 / sampled 126, four 100+ episodes). Prefix + suffix, v9 revision
# (collect-until-preferred). Isaac's RTX renderer cold-render access-violates
# intermittently on the laptop; each boot is a dice roll, so retry until one
# reaches PLAY_APP_READY. On success the GUI is kept alive (we wait on it).
set -u
PY="/c/il/venv/Scripts/python.exe"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CKPT="runs/preserved/stageC_n1640_champion.pt"
PREFIX="runs/preserved/stageC_highest_1163753.pt"
TEMPLATE="assets/rl/env_template_200.usd"   # v12's TRAINING/eval template (96 is OOD -> caps score, 0 cycles)
LOG="runs/n1640_gui_live.log"
MAX_TRIES=8

cd "$ROOT" || exit 1

for try in $(seq 1 $MAX_TRIES); do
  echo "========== ATTEMPT $try/$MAX_TRIES ==========" | tee -a "$LOG"
  TRYLOG="$LOG.try$try"
  : > "$TRYLOG"
  "$PY" scripts/rl/play_policy.py \
    --checkpoint "$CKPT" \
    --stagec-v2 \
    --stagec-v2-prefix-checkpoint "$PREFIX" \
    --dump-on-press \
    --spawn-under-trench \
    --stagec-v2-reset-mode full \
    --episode-len-s 90 \
    --template "$TEMPLATE" \
    > "$TRYLOG" 2>&1 &
  PID=$!
  echo "launched pid=$PID -> $TRYLOG"

  booted=0
  for w in $(seq 1 90); do
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
      if grep -qaiE "access violation|VK_ERROR|Failed to create.*device" "$TRYLOG" 2>/dev/null; then
        echo "ATTEMPT $try CRASHED (RTX cold-render) -- retrying" | tee -a "$LOG"
      else
        echo "ATTEMPT $try EXITED (non-render). tail:" | tee -a "$LOG"
        tail -12 "$TRYLOG" | tee -a "$LOG"
      fi
      break
    fi
    if grep -qa "PLAY_APP_READY True" "$TRYLOG" 2>/dev/null; then
      echo "ATTEMPT $try BOOTED -> PLAY_APP_READY. pid=$PID (GUI open, keeping alive)" | tee -a "$LOG"
      booted=1
      break
    fi
  done

  if [ "$booted" = "1" ]; then
    echo "SUCCESS_BOOTED pid=$PID log=$TRYLOG" | tee -a "$LOG"
    wait "$PID"
    echo "GUI_CLOSED exit=$? (pid=$PID)" | tee -a "$LOG"
    exit 0
  fi
  if kill -0 "$PID" 2>/dev/null; then
    echo "ATTEMPT $try alive at 90s, no marker yet -> treating as booting, keeping." | tee -a "$LOG"
    echo "SUCCESS_ALIVE pid=$PID log=$TRYLOG" | tee -a "$LOG"
    wait "$PID"
    echo "GUI_CLOSED exit=$? (pid=$PID)" | tee -a "$LOG"
    exit 0
  fi
done

echo "ALL $MAX_TRIES ATTEMPTS CRASHED -- RTX renderer unstable this session." | tee -a "$LOG"
exit 2
