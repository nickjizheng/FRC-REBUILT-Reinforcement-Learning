#!/bin/bash
# Fresh peak harvest -> exact-source curation -> schedule-repaired Arm A.
set -eu

CHECKPOINT=${1:-/root/preserved/auto_peaks_fs/peak_106.0_3279578_1515.pt}
RID=${2:-$(date +%Y%m%d_%H%M%S)}
SCRIPT_ROOT=/root/retention_scripts
HARVEST=/root/autodl-tmp/fs_peak106_harvest_$RID
CAPTURE=/dev/shm/fs_peak106_harvest_$RID
TEACHERS=/root/autodl-tmp/elite_anchor_fs_peak106_$RID
REPORT=$HARVEST/teacher_report.json
LOG=$HARVEST/pipeline.log

mkdir -p "$HARVEST"
echo "PIPELINE_START $(date -Is)" > "$LOG"
"$SCRIPT_ROOT/harvest_fullspeed_retention.sh" \
  "$CHECKPOINT" "$CAPTURE" "$HARVEST" >> "$LOG" 2>&1
/root/venv/bin/python "$SCRIPT_ROOT/select_fullspeed_retention_teachers.py" \
  --source-checkpoint "$CHECKPOINT" --harvest-dir "$HARVEST" \
  --output-dir "$TEACHERS" --report "$REPORT" --count 5 \
  >> "$LOG" 2>&1
"$SCRIPT_ROOT/launch_fullspeed_retention_a.sh" \
  "$CHECKPOINT" "$TEACHERS" "$REPORT" "fsretA_$RID" \
  >> "$LOG" 2>&1
echo "PIPELINE_LAUNCHED $(date -Is)" >> "$LOG"
