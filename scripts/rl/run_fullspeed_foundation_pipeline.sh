#!/bin/bash
# Exact-contract no-ferry/no-own-court harvest -> curation -> Foundation-B run.
set -eu

CHECKPOINT=${1:?frozen source checkpoint}
RID=${2:-$(date +%Y%m%d_%H%M%S)}
SCRIPT_ROOT=${SCRIPT_ROOT:-/root/retention_scripts}
HARVEST=/root/autodl-tmp/fs_foundation_harvest_$RID
CAPTURE=/dev/shm/fs_foundation_harvest_$RID
TEACHERS=/root/autodl-tmp/elite_anchor_fs_foundation_$RID
REPORT=$HARVEST/teacher_report.json
LOG=$HARVEST/pipeline.log

test -s "$CHECKPOINT"
test "$(tr -d '[:space:]' < /root/policy_speed_scale.txt)" = "1.0"
if pgrep -f '[l]earner_cycle_v2.py|[c]ollector_cycle_v2.py|[e]val_stagec_seedmine.py' >/dev/null; then
  echo "refusing duplicate foundation pipeline: training/evaluation is active" >&2
  exit 4
fi
mkdir -p "$HARVEST"
echo "PIPELINE_START $(date -Is) source=$CHECKPOINT" > "$LOG"

env STAGE_D_AUX_MODE=none STAGE_D_PREFIX_RESCUE_S=35 \
  WORKERS_PER_GPU=4 EPISODES_PER_ARM=4 \
  "$SCRIPT_ROOT/harvest_fullspeed_retention.sh" \
    "$CHECKPOINT" "$CAPTURE" "$HARVEST" >> "$LOG" 2>&1

/root/venv/bin/python "$SCRIPT_ROOT/select_fullspeed_retention_teachers.py" \
  --source-checkpoint "$CHECKPOINT" --harvest-dir "$HARVEST" \
  --output-dir "$TEACHERS" --report "$REPORT" --count 5 \
  --min-opener-score 50 --min-live1-score 25 \
  --min-live2-score 10 --min-endgame-score 10 \
  --min-unique-seeds 4 --max-per-seed 2 \
  --require-no-ferry-owncourt >> "$LOG" 2>&1

"$SCRIPT_ROOT/launch_fullspeed_foundation_b.sh" \
  "$CHECKPOINT" "$TEACHERS" "$REPORT" "fsfoundation_$RID" \
  >> "$LOG" 2>&1
echo "PIPELINE_LAUNCHED $(date -Is)" >> "$LOG"
