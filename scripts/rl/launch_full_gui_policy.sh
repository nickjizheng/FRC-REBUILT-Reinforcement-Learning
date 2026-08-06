#!/usr/bin/env bash
# Retry-until-boot launcher for the full match GUI with the RL policy hook.
# Isaac's RTX renderer cold-render access-violates ~intermittently on the RTX 5080
# Laptop (Blackwell). Each attempt is a dice roll; retry until one boots past the
# SimulationContext init render into the scene + POLICY_READY.
set -u
PY="/c/il/venv/Scripts/python.exe"
CKPT="runs/peak_936k.pt"
LOG="runs/full_gui_policy_live.log"
MAX_TRIES=6

cd "$(dirname "$0")/../.." || exit 1

for try in $(seq 1 $MAX_TRIES); do
  echo "========== ATTEMPT $try/$MAX_TRIES ==========" | tee -a "$LOG"
  : > "$LOG.try$try"
  "$PY" run_sim.py --policy "$CKPT" --mask-illegal-fire \
    --gui-intake-substeps 2 --gui-camera-views 3 \
    --gui-render-hz 60 --render-width 1600 --render-height 900 \
    > "$LOG.try$try" 2>&1 &
  PID=$!
  echo "launched pid=$PID -> $LOG.try$try"

  # Watch up to ~40s for either a crash (access violation) or a successful boot
  # past the render-init crash point (FRC_REBUILT_SCENE = scene built + rendered).
  booted=0
  for w in $(seq 1 40); do
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
      # process died
      if grep -qa "access violation" "$LOG.try$try" 2>/dev/null; then
        echo "ATTEMPT $try CRASHED (RTX access violation on cold render) -- retrying" | tee -a "$LOG"
      else
        echo "ATTEMPT $try EXITED (non-render). tail:" | tee -a "$LOG"
        tail -5 "$LOG.try$try" | tee -a "$LOG"
      fi
      break
    fi
    # survived past the render-init crash window if the scene printed
    if grep -qa "POLICY_READY" "$LOG.try$try" 2>/dev/null; then
      echo "ATTEMPT $try BOOTED PAST RENDER -> POLICY_READY. pid=$PID keeping alive." | tee -a "$LOG"
      booted=1
      break
    fi
  done

  if [ "$booted" = "1" ]; then
    echo "SUCCESS_BOOTED pid=$PID log=$LOG.try$try" | tee -a "$LOG"
    exit 0
  fi
  # if still alive but no POLICY_READY yet, give it more time before deciding
  if kill -0 "$PID" 2>/dev/null; then
    echo "ATTEMPT $try still alive at 40s, no POLICY_READY yet -> treating as booting, keeping." | tee -a "$LOG"
    echo "SUCCESS_ALIVE pid=$PID log=$LOG.try$try" | tee -a "$LOG"
    exit 0
  fi
done

echo "ALL $MAX_TRIES ATTEMPTS CRASHED -- RTX renderer unstable this session." | tee -a "$LOG"
exit 2
