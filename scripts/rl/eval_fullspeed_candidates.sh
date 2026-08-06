#!/bin/bash
# Frozen, paired full-match screen for candidate Stage-D checkpoints.
#
# Each checkpoint owns one GPU and sees the same deterministic seed.  The
# evaluator loads the checkpoint once, so unlike live collector telemetry an
# episode cannot mix several learner publications.
set -eu

OUT=${1:?output directory}
EPISODES=${EPISODES:-20}
NUM_ENVS=${NUM_ENVS:-2}
SEED=${SEED:-830001}
CODE_ROOT=${CODE_ROOT:-/root/autodl-tmp/frc_staged_v1_20260722}
PREFIX=${PREFIX:-/root/preserved/stageC_highest_1163753.pt}
TEMPLATE=${TEMPLATE:-$CODE_ROOT/assets/rl/env_template_456.usd}

shift
if [ "$#" -ne 4 ]; then
  echo "usage: $0 OUT name=checkpoint name=checkpoint name=checkpoint name=checkpoint" >&2
  exit 2
fi
test "$(tr -d '[:space:]' < /root/policy_speed_scale.txt)" = "1.0"
case "$NUM_ENVS" in
  ''|*[!0-9]*) echo "NUM_ENVS must be a positive integer" >&2; exit 2 ;;
esac
test "$NUM_ENVS" -gt 0
test -s "$PREFIX"
test -s "$TEMPLATE"
mkdir -p "$OUT"

run_candidate() {
  local gpu=$1 spec=$2
  local name=${spec%%=*}
  local checkpoint=${spec#*=}
  test "$name" != "$checkpoint"
  test -s "$checkpoint"
  (
    cd "$CODE_ROOT"
    env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$gpu" FRC_POLICY_SPEED_SCALE=1.0 \
      /root/venv/bin/python -u scripts/rl/eval_stagec_seedmine.py \
        --checkpoint "$checkpoint" --prefix-checkpoint "$PREFIX" \
        --template "$TEMPLATE" --mode full \
        --episodes "$EPISODES" --num-envs "$NUM_ENVS" --episode-len-s 160 \
        --env-seed "$SEED" --action-seed "$SEED" \
        --action-mode deterministic --stage-d \
        --stage-d-first-inactive blue --stage-d-ferry \
        --stage-d-ferry-dump-on-press --stage-d-ferry-entitled-only \
        --stage-d-ferry-blackout-only --stage-d-ferry-min-load 10 \
        --stage-d-return-when-live --stage-d-live-return-load 26 \
        --stage-d-return-lead-s 8 --stage-d-owncourt-loop \
        --stage-d-owncourt-min-balls 2 --stage-d-owncourt-rearm \
        --stage-d-owncourt-blackout-intake \
        --out "$OUT/$name.jsonl" --summary-out "$OUT/$name.summary.json" \
        --overwrite > "$OUT/$name.log" 2>&1
  )
  echo "$name gpu=$gpu rc=$? $(date -Is)" >> "$OUT/progress.log"
}

printf 'START %s episodes=%s num_envs=%s seed=%s\n' \
  "$(date -Is)" "$EPISODES" "$NUM_ENVS" "$SEED" > "$OUT/progress.log"
gpu=0
for spec in "$@"; do
  run_candidate "$gpu" "$spec" &
  gpu=$((gpu + 1))
done
wait
printf 'DONE %s\n' "$(date -Is)" >> "$OUT/progress.log"
