#!/bin/bash
# Stage C v2: one stable two-env collector per GPU 0..2, learner on GPU 3.
# Existing Stage C launchers and output directories are never touched.
#
# Usage: run_stagec_v2.sh <champion.pt> <anchor_dir> [minutes] [template] [out]
set -u

CHAMPION=${1:?usage: run_stagec_v2.sh <champion.pt> <anchor_dir> [minutes] [template] [out]}
ANCHOR_DIR=${2:?champion anchor directory}
MINUTES=${3:-600}
TEMPLATE=${4:-/root/frc-rl/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${5:-/root/autodl-tmp/runs/stagec_v2_${RUN_ID}}
ROOT=${STAGEC_V2_ROOT:-/dev/shm/frc_stagec_v2_${RUN_ID}}
STOP=${STAGEC_V2_STOP:-/root/STOP_STAGEC_V2}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DASHBOARD_RUNS=${STAGEC_V2_DASHBOARD_RUNS:-$CODE_ROOT/runs}

TARGET_LOAD=15
# Used only as the physical RETURN preload source. FULL, postdump, and collect
# keep all 200 native template balls in place.
RESERVE_COUNT=18
RESERVE_BATCHES=1
MAX_DUMP_TICKS=180
STDDEV_START=1.0
STDDEV_END=0.30
STDDEV_STEPS=150000
CYCLE_SCORE_FRACTION=0.75
CYCLE_SCORE_FLOOR=6
COLLECT_WEIGHT=0.30
PROGRESS_PER_M=5.0
PROGRESS_STEP_CAP=0.75
RAMP_BONUS=6.0
LEAVE_GRACE_STEPS=5
LEAVE_PENALTY_PER_STEP=0.03
LEAVE_PENALTY_CAP=5.0
RETURN_GRACE_STEPS=10
RETURN_PENALTY_PER_STEP=0.02
RETURN_PENALTY_CAP=5.0
SHOOT_GRACE_S=2.0
SHOOT_PENALTY_PER_STEP=0.05
SHOOT_PENALTY_CAP=5.0
DUMP_LOST_AIM_GRACE_TICKS=15
PARTIAL_DUMP_PENALTY_PER_BALL=0.5
PARTIAL_DUMP_PENALTY_CAP=15.0

cd "$CODE_ROOT" || exit 1
source /root/venv/bin/activate
source setup_render_env.sh >/dev/null 2>&1
mkdir -p "$ROOT" "$OUT"
rm -rf "$ROOT"/collector_* "$ROOT"/weights 2>/dev/null || true
rm -f "$STOP"

cat > "$OUT/launcher_config.json" <<EOF
{"run_id":"$RUN_ID","code_root":"$CODE_ROOT","root":"$ROOT","out":"$OUT","dashboard_runs":"$DASHBOARD_RUNS","champion":"$CHAMPION","anchor_dir":"$ANCHOR_DIR","template":"$TEMPLATE","minutes":$MINUTES,"target_load":$TARGET_LOAD,"return_skill_preload":8,"chamber_capacity":60,"cycle_score_fraction":$CYCLE_SCORE_FRACTION,"cycle_score_floor":$CYCLE_SCORE_FLOOR,"collect_weight":$COLLECT_WEIGHT,"progress_per_m":$PROGRESS_PER_M,"progress_step_cap":$PROGRESS_STEP_CAP,"ramp_bonus":$RAMP_BONUS,"leave_grace_steps":$LEAVE_GRACE_STEPS,"leave_penalty_per_step":$LEAVE_PENALTY_PER_STEP,"leave_penalty_cap":$LEAVE_PENALTY_CAP,"return_grace_steps":$RETURN_GRACE_STEPS,"return_penalty_per_step":$RETURN_PENALTY_PER_STEP,"return_penalty_cap":$RETURN_PENALTY_CAP,"shoot_grace_s":$SHOOT_GRACE_S,"shoot_penalty_per_step":$SHOOT_PENALTY_PER_STEP,"shoot_penalty_cap":$SHOOT_PENALTY_CAP,"dump_lost_aim_grace_ticks":$DUMP_LOST_AIM_GRACE_TICKS,"partial_dump_penalty_per_ball":$PARTIAL_DUMP_PENALTY_PER_BALL,"partial_dump_penalty_cap":$PARTIAL_DUMP_PENALTY_CAP}
EOF

echo "$(date '+%F %T') Stage C v2 run=$RUN_ID code=$CODE_ROOT out=$OUT"
echo "$(date '+%F %T') champion=$CHAMPION anchors=$ANCHOR_DIR"

LPID=""
LSTART=0
LFAIL=0
declare -A CPID
declare -A CSTART
declare -A CFAIL
for cid in 0 1 2; do CFAIL[$cid]=0; CSTART[$cid]=0; done

launch_learner() {
  local resume="$CHAMPION"
  if [ -s "$OUT/latest.pt" ]; then resume="$OUT/latest.pt"; fi
  CUDA_VISIBLE_DEVICES=3 setsid python scripts/rl/learner_cycle_v2.py \
    --root "$ROOT" --num-collectors 3 --collector-envs 2 \
    --stream-groups full,full,postdump,collect,return,return \
    --group-weights full=.30,postdump=.25,collect=.10,return=.35 \
    --resume "$resume" --prefix-checkpoint "$CHAMPION" \
    --anchor-dir "$ANCHOR_DIR" --out "$OUT" \
    --minutes "$MINUTES" --batch-size 256 --learning-rate 5e-5 \
    --gamma 0.999 --n-step 3 --critic-only-updates 5000 \
    --anchor-beta-start .30 --anchor-beta-floor .08 --anchor-decay-updates 60000 \
    --stddev-start "$STDDEV_START" --stddev-end "$STDDEV_END" \
    --stddev-steps "$STDDEV_STEPS" --initial-stddev .50 \
    --suffix-alpha 1.0 \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" \
    --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" \
    --collect-weight "$COLLECT_WEIGHT" \
    --progress-per-m "$PROGRESS_PER_M" \
    --progress-step-cap "$PROGRESS_STEP_CAP" \
    --ramp-bonus "$RAMP_BONUS" \
    --leave-grace-steps "$LEAVE_GRACE_STEPS" \
    --leave-penalty-per-step "$LEAVE_PENALTY_PER_STEP" \
    --leave-penalty-cap "$LEAVE_PENALTY_CAP" \
    --return-grace-steps "$RETURN_GRACE_STEPS" \
    --return-penalty-per-step "$RETURN_PENALTY_PER_STEP" \
    --return-penalty-cap "$RETURN_PENALTY_CAP" \
    --shoot-grace-s "$SHOOT_GRACE_S" \
    --shoot-penalty-per-step "$SHOOT_PENALTY_PER_STEP" \
    --shoot-penalty-cap "$SHOOT_PENALTY_CAP" \
    --dump-lost-aim-grace-ticks "$DUMP_LOST_AIM_GRACE_TICKS" \
    --partial-dump-penalty-per-ball "$PARTIAL_DUMP_PENALTY_PER_BALL" \
    --partial-dump-penalty-cap "$PARTIAL_DUMP_PENALTY_CAP" \
    >> "$OUT/learner.log" 2>&1 &
  LPID=$!
  LSTART=$(date +%s)
  echo "$(date '+%F %T') learner GPU3 pid=$LPID resume=$resume"
}

launch_collector() {
  local cid=$1
  local modes episode
  case "$cid" in
    0) modes="full,full"; episode=120 ;;
    1) modes="postdump,collect"; episode=75 ;;
    2) modes="return,return"; episode=75 ;;
    *) echo "bad collector id $cid"; return 2 ;;
  esac
  sleep 5
  CUDA_VISIBLE_DEVICES=$cid setsid python scripts/rl/collector_cycle_v2.py \
    --collector-id "$cid" --root "$ROOT" --num-envs 2 \
    --stagec-v2-prefix-checkpoint "$CHAMPION" \
    --reset-modes "$modes" --template "$TEMPLATE" --episode-len-s "$episode" \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" --collect-weight "$COLLECT_WEIGHT" \
    --dump-on-press --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" \
    --progress-per-m "$PROGRESS_PER_M" \
    --progress-step-cap "$PROGRESS_STEP_CAP" \
    --ramp-bonus "$RAMP_BONUS" \
    --leave-grace-steps "$LEAVE_GRACE_STEPS" \
    --leave-penalty-per-step "$LEAVE_PENALTY_PER_STEP" \
    --leave-penalty-cap "$LEAVE_PENALTY_CAP" \
    --return-grace-steps "$RETURN_GRACE_STEPS" \
    --return-penalty-per-step "$RETURN_PENALTY_PER_STEP" \
    --return-penalty-cap "$RETURN_PENALTY_CAP" \
    --shoot-grace-s "$SHOOT_GRACE_S" \
    --shoot-penalty-per-step "$SHOOT_PENALTY_PER_STEP" \
    --shoot-penalty-cap "$SHOOT_PENALTY_CAP" \
    --dump-lost-aim-grace-ticks "$DUMP_LOST_AIM_GRACE_TICKS" \
    --partial-dump-penalty-per-ball "$PARTIAL_DUMP_PENALTY_PER_BALL" \
    --partial-dump-penalty-cap "$PARTIAL_DUMP_PENALTY_CAP" \
    --stddev-start "$STDDEV_START" --stddev-end "$STDDEV_END" \
    --stddev-steps "$STDDEV_STEPS" --seed $((7000 + cid)) \
    --minutes "$MINUTES" --telemetry "$OUT/cycle_telemetry.jsonl" \
    >> "$OUT/collector${cid}.log" 2>&1 &
  CPID[$cid]=$!
  CSTART[$cid]=$(date +%s)
  echo "$(date '+%F %T') collector$cid GPU$cid modes=$modes pid=${CPID[$cid]}"
}

cleanup() {
  echo "$(date '+%F %T') Stage C v2 cleanup"
  [ -n "${LPID:-}" ] && kill "$LPID" 2>/dev/null || true
  for cid in "${!CPID[@]}"; do kill "${CPID[$cid]}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

launch_learner
sleep 20
for cid in 0 1 2; do launch_collector "$cid"; sleep 20; done

mkdir -p "$DASHBOARD_RUNS"
ln -sfn "$OUT" "$DASHBOARD_RUNS/stagec_v2_latest"
echo "$(date '+%F %T') all Stage C v2 processes launched; watchdog active"

END=$(( $(date +%s) + ${MINUTES%.*} * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if [ -f "$STOP" ]; then
    echo "$(date '+%F %T') stop file detected"
    exit 0
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    echo "$(date '+%F %T') learner pid=$LPID died; restarting from latest"
    now=$(date +%s)
    if [ $((now - LSTART)) -lt 600 ]; then LFAIL=$((LFAIL + 1)); else LFAIL=1; fi
    if [ "$LFAIL" -ge 3 ]; then
      echo "$(date '+%F %T') learner failed 3 times inside ten-minute windows; aborting" | tee "$OUT/LAUNCH_FAILED"
      exit 1
    fi
    sleep 5
    launch_learner
  fi
  for cid in 0 1 2; do
    if ! kill -0 "${CPID[$cid]}" 2>/dev/null; then
      echo "$(date '+%F %T') collector$cid pid=${CPID[$cid]} died; restarting"
      now=$(date +%s)
      if [ $((now - CSTART[$cid])) -lt 600 ]; then
        CFAIL[$cid]=$((CFAIL[$cid] + 1))
      else
        CFAIL[$cid]=1
      fi
      if [ "${CFAIL[$cid]}" -ge 3 ]; then
        echo "$(date '+%F %T') collector$cid failed 3 times inside ten-minute windows; aborting" | tee "$OUT/LAUNCH_FAILED"
        exit 1
      fi
      launch_collector "$cid"
    fi
  done
  sleep 20
done

echo "$(date '+%F %T') Stage C v2 duration complete"
