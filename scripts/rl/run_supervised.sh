#!/bin/bash
# Supervised DrQ-v2 training: restart the trainer on ANY non-zero exit - PhysX
# segfault (139), watchdog stall (3), SimulationUnstable (4), OOM, etc. - always
# resuming from the run's OWN latest.pt so a physics blow-up becomes recoverable
# instead of fatal. The very first launch resumes from RESUME_INIT.
#
# Usage:
#   run_supervised.sh <GPU> <OUT_DIR> <RESUME_INIT> [MAX_RESTARTS] -- <train args...>
# Example:
#   scripts/rl/run_supervised.sh 0 /root/runs/sb_g0 /root/runs/.../ckpt_000030000.pt 50 -- \
#     --stage B --num-envs 4 --minutes 480 --seed 500 \
#     --template assets/rl/env_template_96.usd \
#     --collect-weight-start 0.3 --collect-weight-end 0.3
set -u

GPU=${1:?usage: run_supervised.sh <gpu> <out> <resume_init> [max_restarts] -- <args>}
OUT=${2:?out dir}
RESUME_INIT=${3:?initial checkpoint (used only until latest.pt exists)}
MAXR=${4:-50}
shift 4
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

cd /root/frc-rl && source /root/venv/bin/activate && source setup_render_env.sh
mkdir -p "$OUT"

restart=0
fastfail=0
while : ; do
  if [ -f "$OUT/latest.pt" ]; then RESUME="$OUT/latest.pt"; else RESUME="$RESUME_INIT"; fi
  echo "SUPERVISOR launch #$restart gpu=$GPU resume=$RESUME $(date -Is)"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/rl/train_drqv2.py \
    --out "$OUT" --resume "$RESUME" "${EXTRA[@]}"
  code=$?
  dur=$(( $(date +%s) - start ))
  echo "SUPERVISOR trainer exit=$code after ${dur}s (restart $restart/$MAXR) $(date -Is)"

  # clean finish (reached --minutes) -> done
  [ "$code" -eq 0 ] && { echo "SUPERVISOR clean exit; stopping"; break; }

  restart=$((restart + 1))
  [ "$restart" -gt "$MAXR" ] && { echo "SUPERVISOR hit MAX_RESTARTS=$MAXR; stopping"; break; }

  # crash-loop guard: a <45s life means it died on boot/checkpoint, not mid-run
  if [ "$dur" -lt 45 ]; then
    fastfail=$((fastfail + 1))
    echo "SUPERVISOR fast failure #$fastfail (${dur}s)"
    [ "$fastfail" -ge 5 ] && { echo "SUPERVISOR 5 consecutive fast failures; aborting"; break; }
    sleep $((15 * fastfail))
  else
    fastfail=0
    sleep 10
  fi
done
echo "SUPERVISOR done gpu=$GPU out=$OUT restarts=$restart $(date -Is)"
