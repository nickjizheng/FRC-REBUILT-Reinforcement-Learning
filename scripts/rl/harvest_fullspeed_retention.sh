#!/bin/bash
# Deterministic, exact-source harvest for the full-speed retention experiment.
set -eu

CHECKPOINT=${1:?source checkpoint}
CAPTURE_ROOT=${2:?capture root}
OUT=${3:?harvest output directory}
EPISODES_PER_ARM=${EPISODES_PER_ARM:-5}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
SEED_BASE=${SEED_BASE:-810000}
STAGE_D_AUX_MODE=${STAGE_D_AUX_MODE:-legacy}
STAGE_D_PREFIX_RESCUE_S=${STAGE_D_PREFIX_RESCUE_S:-0}
ALLOW_VOLATILE_CAPTURE=${ALLOW_VOLATILE_CAPTURE:-0}

CODE_ROOT=/root/autodl-tmp/frc_staged_v1_20260722
PREFIX=/root/preserved/stageC_highest_1163753.pt
TEMPLATE=$CODE_ROOT/assets/rl/env_template_456.usd

test -s "$CHECKPOINT"
test -s "$PREFIX"
test -s "$TEMPLATE"
case "$STAGE_D_AUX_MODE" in
  legacy|none) ;;
  *) echo "STAGE_D_AUX_MODE must be legacy or none" >&2; exit 2 ;;
esac
case "$SEED_BASE" in
  ''|*[!0-9]*) echo "SEED_BASE must be a non-negative integer" >&2; exit 2 ;;
esac
CAPTURE_ROOT_RESOLVED=$(readlink -m "$CAPTURE_ROOT")
case "$CAPTURE_ROOT_RESOLVED" in
  /dev/shm|/dev/shm/*)
    if [ "$ALLOW_VOLATILE_CAPTURE" != "1" ]; then
      echo "refusing volatile teacher captures under /dev/shm; set ALLOW_VOLATILE_CAPTURE=1 only for disposable evaluation" >&2
      exit 2
    fi
    ;;
esac
mkdir -p "$CAPTURE_ROOT" "$OUT"
echo "HARVEST_START $(date -Is) checkpoint=$CHECKPOINT capture_root=$CAPTURE_ROOT_RESOLVED volatile_capture=$ALLOW_VOLATILE_CAPTURE aux=$STAGE_D_AUX_MODE rescue_s=$STAGE_D_PREFIX_RESCUE_S workers_per_gpu=$WORKERS_PER_GPU seed_base=$SEED_BASE" > "$OUT/progress.log"

run_arm() {
  local gpu=$1 slot=$2 seed=$3
  local name="g${gpu}_s${slot}_${seed}"
  local capture_dir="$CAPTURE_ROOT/$name"
  local rc=0
  local -a stage_d_flags=(
    --stage-d --stage-d-first-inactive blue
    --stage-d-prefix-rescue-s "$STAGE_D_PREFIX_RESCUE_S"
  )
  if [ "$STAGE_D_AUX_MODE" = "legacy" ]; then
    stage_d_flags+=(
      --stage-d-ferry --stage-d-ferry-dump-on-press
      --stage-d-ferry-entitled-only --stage-d-ferry-blackout-only
      --stage-d-ferry-min-load 10 --stage-d-return-when-live
      --stage-d-live-return-load 26 --stage-d-return-lead-s 8
      --stage-d-owncourt-loop --stage-d-owncourt-min-balls 2
      --stage-d-owncourt-rearm --stage-d-owncourt-blackout-intake
    )
  fi
  mkdir -p "$capture_dir"
  (
    cd "$CODE_ROOT"
    env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$gpu" FRC_POLICY_SPEED_SCALE=1.0 \
      /root/venv/bin/python -u scripts/rl/eval_stagec_seedmine.py \
        --checkpoint "$CHECKPOINT" --prefix-checkpoint "$PREFIX" \
        --template "$TEMPLATE" --mode full \
        --episodes "$EPISODES_PER_ARM" --num-envs 2 --episode-len-s 160 \
        --env-seed "$seed" --action-seed "$seed" \
        --action-mode deterministic "${stage_d_flags[@]}" \
        --capture-dir "$capture_dir" \
        --out "$OUT/$name.jsonl" --summary-out "$OUT/$name.json" \
        --overwrite > "$OUT/$name.log" 2>&1
  ) || rc=$?
  local rows=0
  if [ -f "$OUT/$name.jsonl" ]; then
    rows=$(wc -l < "$OUT/$name.jsonl")
  fi
  if [ "$rc" -ne 0 ] || [ "$rows" -ne "$EPISODES_PER_ARM" ] || \
      [ ! -s "$OUT/$name.json" ] || \
      ! grep -q 'SEED_EVAL_DONE' "$OUT/$name.log"; then
    echo "ARM_FAILED gpu=$gpu slot=$slot seed=$seed rc=$rc rows=$rows $(date -Is)" \
      >> "$OUT/progress.log"
    return 1
  fi
  echo "ARM_DONE gpu=$gpu slot=$slot seed=$seed rc=0 rows=$rows $(date -Is)" \
    >> "$OUT/progress.log"
}

for slot in $(seq 0 $((WORKERS_PER_GPU - 1))); do
  for gpu in 0 1 2 3; do
    seed=$((SEED_BASE + gpu * 1000 + slot * 100 + 1))
    run_arm "$gpu" "$slot" "$seed" &
  done
  sleep 12
done
wait || true
expected_arms=$((4 * WORKERS_PER_GPU))
completed_arms=$(grep -c '^ARM_DONE ' "$OUT/progress.log" || true)
if [ "$completed_arms" -ne "$expected_arms" ]; then
  echo "HARVEST_FAILED completed_arms=$completed_arms expected_arms=$expected_arms $(date -Is)" \
    >> "$OUT/progress.log"
  exit 3
fi
echo "HARVEST_DONE $(date -Is)" >> "$OUT/progress.log"
