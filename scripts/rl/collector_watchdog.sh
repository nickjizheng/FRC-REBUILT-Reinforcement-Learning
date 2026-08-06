#!/bin/bash
# Collector watchdog for the reward-first / phase-PBRS distributed run.
# The Isaac RTX renderer crashes silently ("Simulation App Shutting Down", no
# Python traceback) after ~tens of minutes, so unattended collectors starve the
# learner. This tracks each collector by the PID it launched (NO pgrep -f, which
# self-matches "collector-id N" in the watchdog's own cmdline) and relaunches any
# that die, until the learner exits or /root/STOP_WATCHDOG appears. On exit it
# reaps its collectors so none orphan and fill /dev/shm.
#
# Usage: collector_watchdog.sh [gpu_list] [pbrs_weight] [minutes]
#   gpu_list: space-separated GPU/collector ids, e.g. "0 1 2" (default) or "1 2".
#   collector id N always runs on GPU N (seed 400+N) to match the launcher.
set -u
GPUS=${1:-"0 1 2"}
PBRS_WEIGHT=${2:-1.0}
MINUTES=${3:-600}
ROOT=/dev/shm/frc_dist_ft
OUT=/root/autodl-tmp/runs/drqv2_rewardfirst
TEMPLATE=/root/frc-rl/assets/rl/env_template_200.usd
STOP=/root/STOP_WATCHDOG

cd /root/frc-rl && source /root/venv/bin/activate && source setup_render_env.sh >/dev/null 2>&1
declare -A CPID

launch() {
  local c=$1
  # wait for the target GPU to be free (previous instance's VRAM released)
  for _ in $(seq 1 30); do
    local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c" | tr -d " ")
    [ "${u:-9999}" -lt 500 ] && break
    sleep 3
  done
  CUDA_VISIBLE_DEVICES=$c setsid python scripts/rl/collector.py \
    --collector-id "$c" --root "$ROOT" --num-envs 2 --stage C \
    --template "$TEMPLATE" --episode-len-s 90 --preload-prob 0.0 \
    --spawn-under-trench --mask-illegal-fire \
    --collect-weight 0.3 --stddev-end 0.2 \
    --phase-pbrs --pbrs-weight "$PBRS_WEIGHT" --pbrs-gamma 0.999 \
    --seed $((400 + c)) --minutes "$MINUTES" >> "$OUT.collector$c.log.wd" 2>&1 &
  CPID[$c]=$!
  echo "$(date '+%F %T') WATCHDOG launch collector$c -> GPU$c pid ${CPID[$c]}"
}

cleanup() {
  echo "$(date '+%F %T') WATCHDOG exiting, reaping collectors"
  for c in "${!CPID[@]}"; do kill "${CPID[$c]}" 2>/dev/null; done
  exit 0
}
trap cleanup TERM INT

# initial fan-out (stagger so cameras don't warm-race)
for c in $GPUS; do launch "$c"; sleep 25; done
echo "$(date '+%F %T') WATCHDOG monitoring collectors: $GPUS"

while true; do
  [ -f "$STOP" ] && { echo "$(date '+%F %T') STOP file seen"; cleanup; }
  # learner gone -> whole run is over, reap and exit
  if ! pgrep -f learner_finetune.py >/dev/null 2>&1; then
    echo "$(date '+%F %T') learner not running -> shutting down watchdog"; cleanup
  fi
  for c in $GPUS; do
    if ! kill -0 "${CPID[$c]}" 2>/dev/null; then
      echo "$(date '+%F %T') collector$c (pid ${CPID[$c]}) DEAD -> relaunch"
      launch "$c"
    fi
  done
  sleep 30
done
