#!/bin/bash
# Adaptive four-GPU deterministic harvest, strict curation, then gated Gen-12.
set -u

CR=/root/autodl-tmp/frc_staged_v1_20260722
PREFIX=/root/preserved/stageC_highest_1163753.pt
CHECKPOINT=/root/preserved/auto_peaks/peak_119.2_2857060_2103.pt
TEMPLATE=$CR/assets/rl/env_template_456.usd
RID=$(date +%Y%m%d_%H%M%S)
OUT=/root/autodl-tmp/elite_g12_harvest
CAP=/dev/shm/elite_g12_harvest_${RID}
TEACHERS=/root/autodl-tmp/elite_anchor_g12
REPORT=$OUT/teacher_filter.json
MIN_TEACHERS=${MIN_TEACHERS:-5}
# eval_stagec_seedmine intentionally caps one process at two camera envs.
# Two processes per GPU is the stable operating point on this server after
# repeated Vulkan-context contention at four and five processes per GPU.
# Six episodes each yields 48 episodes per adaptive wave.
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
EPISODES_PER_ARM=${EPISODES_PER_ARM:-6}
MAX_WAVES=${MAX_WAVES:-3}
HARVEST_LOCK=/root/G12_HARVEST_ACTIVE

mkdir -p "$OUT" "$CAP"
touch "$HARVEST_LOCK"
echo "$CAP" > "$OUT/capture_root.txt"
echo "HARVEST_START $(date -Is) rid=$RID" > "$OUT/progress.log"

if pgrep -f '[r]un_stagec_v2_cycle3_efficiency|[l]earner_cycle_v2|[c]ollector_cycle_v2.*frc_stage_blue2' >/dev/null; then
  echo "ABORT active training stack" | tee -a "$OUT/progress.log"
  exit 4
fi

run_arm () {
  local wave=$1
  local arm=$2
  local seed=$3
  local gpu=$4
  local name="w${wave}_g${gpu}_s${seed}"
  local capdir="$CAP/$name"
  mkdir -p "$capdir"
  (
    cd "$CR" || exit 2
    env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" \
      /root/venv/bin/python -u scripts/rl/eval_stagec_seedmine.py \
        --checkpoint "$CHECKPOINT" --prefix-checkpoint "$PREFIX" \
        --template "$TEMPLATE" --mode full \
        --episodes "$EPISODES_PER_ARM" --num-envs 2 --episode-len-s 160 \
        --env-seed "$seed" --action-seed "$seed" --action-mode deterministic \
        --stage-d --stage-d-first-inactive blue --stage-d-ferry \
        --stage-d-ferry-dump-on-press --stage-d-ferry-entitled-only \
        --stage-d-ferry-blackout-only --stage-d-ferry-min-load 10 \
        --stage-d-return-when-live --stage-d-live-return-load 26 \
        --stage-d-return-lead-s 8 --stage-d-owncourt-loop \
        --stage-d-owncourt-min-balls 2 --stage-d-owncourt-rearm \
        --stage-d-owncourt-blackout-intake \
        --capture-dir "$capdir" --out "$OUT/$name.jsonl" \
        --summary-out "$OUT/$name.summary.json" --overwrite \
        > "$OUT/$name.log" 2>&1
  )
  echo "ARM_DONE wave=$wave arm=$arm gpu=$gpu seed=$seed rc=$? $(date -Is)" \
    >> "$OUT/progress.log"
}

selected=0
for wave in $(seq 1 "$MAX_WAVES"); do
  base=$((880000 + wave * 10000))
  echo "WAVE_START wave=$wave $(date -Is)" >> "$OUT/progress.log"
  # Start one process per GPU in each cohort.  Grouping all processes for one
  # GPU first causes Vulkan-context initialization to serialize or stall.
  for slot in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    for gpu in 0 1 2 3; do
      seed=$((base + gpu * 1000 + slot * 100 + 1))
      run_arm "$wave" "g${gpu}s${slot}" "$seed" "$gpu" &
    done
    sleep 15
  done
  wait

  /root/venv/bin/python /root/filter_stage_d_g12_teachers.py \
    --source-checkpoint "$CHECKPOINT" \
    --input-dir /root/autodl-tmp/elite_anchor_g11 \
    --input-dir "$CAP" \
    --output-dir "$TEACHERS" --report "$REPORT" \
    >> "$OUT/filter.log" 2>&1
  selected=$(/root/venv/bin/python - "$REPORT" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))["selected_count"]))
PY
)
  echo "WAVE_FILTER wave=$wave selected=$selected $(date -Is)" \
    >> "$OUT/progress.log"
  [ "$selected" -ge "$MIN_TEACHERS" ] && break
done

if [ "$selected" -lt "$MIN_TEACHERS" ]; then
  echo "NO_LAUNCH selected=$selected required=$MIN_TEACHERS $(date -Is)" \
    | tee -a "$OUT/progress.log"
  touch "$OUT/NO_LAUNCH"
  exit 5
fi

echo "CURATION_PASS selected=$selected $(date -Is)" >> "$OUT/progress.log"
rm -f "$HARVEST_LOCK"
/root/launch_stage_d_g12.sh "$TEACHERS" "$REPORT" \
  >> "$OUT/launch_g12.log" 2>&1
rc=$?
echo "GEN12_LAUNCH_DONE rc=$rc $(date -Is)" >> "$OUT/progress.log"
exit "$rc"
