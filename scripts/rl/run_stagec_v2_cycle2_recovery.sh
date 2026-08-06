#!/bin/bash
# Cycle-2 scoring consolidation from a known-good Stage-C v2 checkpoint.
#
# Four collectors keep the learner/data ratio bounded while still using all
# four GPUs: GPU0 runs two FULL collectors, GPU1 runs POSTDUMP+COLLECT, GPU2
# runs RETURN, and GPU3 learns.  Successful FULL/RETURN suffixes are archived
# and replayed so later failures cannot immediately erase the working cycle.
set -u

CHAMPION=${1:?immutable first-cycle prefix checkpoint}
RESUME=${2:?known-good Stage-C v2 checkpoint}
ANCHOR_DIR=${3:?champion anchor directory}
MINUTES=${4:-600}
TEMPLATE=${5:-/root/frc-rl/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${6:-/root/autodl-tmp/runs/stagec_v2_cycle2_recovery_${RUN_ID}}
ROOT=${STAGEC_V2_ROOT:-/dev/shm/frc_stagec_v2_cycle2_recovery_${RUN_ID}}
STOP=${STAGEC_V2_STOP:-/root/STOP_STAGEC_V2_CYCLE2_RECOVERY}
ELITE_SEED_DIR=${STAGEC_V2_ELITE_SEED_DIR:-}
RESET_SCHEDULES=${STAGEC_V2_RESET_SCHEDULES:-1}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DASHBOARD_RUNS=${STAGEC_V2_DASHBOARD_RUNS:-$CODE_ROOT/runs}

NCOLL=4
STREAM_GROUPS=full,full,full,full,postdump,collect,return,return
GROUP_WEIGHTS=full=.45,postdump=.10,collect=.10,return=.35

TARGET_LOAD=15
RESERVE_COUNT=18
RESERVE_BATCHES=1
MAX_DUMP_TICKS=180
STDDEV_START=1.0
# The preserved 350k checkpoint is contract-bound to 0.30.  Keep that exact
# schedule instead of silently rewriting checkpoint metadata on resume.
STDDEV_END=0.30
STDDEV_STEPS=150000
INITIAL_STDDEV=0.35
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

if [ ! -s "$CHAMPION" ] || [ ! -s "$RESUME" ] || [ ! -d "$ANCHOR_DIR" ]; then
  echo "missing champion, resume checkpoint, or anchor directory" >&2
  exit 2
fi
if [ "$RESET_SCHEDULES" != "0" ] && [ "$RESET_SCHEDULES" != "1" ]; then
  echo "STAGEC_V2_RESET_SCHEDULES must be 0 or 1" >&2
  exit 2
fi
if [ -d "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "refusing to overwrite non-empty output directory: $OUT" >&2
  exit 2
fi

mkdir -p "$ROOT" "$OUT" "$OUT/elite_episodes"
if [ -n "$ELITE_SEED_DIR" ]; then
  if [ ! -d "$ELITE_SEED_DIR" ]; then
    echo "elite seed directory does not exist: $ELITE_SEED_DIR" >&2
    exit 2
  fi
  cp "$ELITE_SEED_DIR"/elite_*.npz "$OUT/elite_episodes/"
fi
rm -rf "$ROOT"/collector_* "$ROOT"/weights 2>/dev/null || true
rm -f "$STOP"

cat > "$OUT/launcher_manifest.json" <<EOF
{"run_id":"$RUN_ID","code_root":"$CODE_ROOT","root":"$ROOT","out":"$OUT","champion":"$CHAMPION","resume":"$RESUME","collectors":$NCOLL,"stream_groups":"$STREAM_GROUPS","group_weights":"$GROUP_WEIGHTS","learning_rate":0.00002,"elite_replay_fraction":0.25,"elite_archive_max_files":48,"reset_schedules":$RESET_SCHEDULES,"critic_only_updates":5000,"initial_stddev":$INITIAL_STDDEV}
EOF
sha256sum "$CHAMPION" "$RESUME" > "$OUT/input_sha256.txt"

mode_of() {
  case "$1" in
    0|1) echo "full,full" ;;
    2) echo "postdump,collect" ;;
    3) echo "return,return" ;;
  esac
}
gpu_of() {
  case "$1" in
    0|1) echo 0 ;;
    2) echo 1 ;;
    3) echo 2 ;;
  esac
}
episode_of() {
  case "$1" in
    0|1) echo 120 ;;
    *) echo 75 ;;
  esac
}

LPID=""
LSTART=0
LFAIL=0
declare -A CPID CSTART
for cid in $(seq 0 $((NCOLL - 1))); do CSTART[$cid]=0; done

launch_learner() {
  local resume="$RESUME"
  local reset_flag=""
  if [ "$RESET_SCHEDULES" = "1" ]; then
    reset_flag="--reset-schedules-on-resume"
  fi
  if [ -s "$OUT/latest.pt" ]; then
    resume="$OUT/latest.pt"
    reset_flag=""
  fi
  CUDA_VISIBLE_DEVICES=3 setsid python scripts/rl/learner_cycle_v2.py \
    --root "$ROOT" --num-collectors "$NCOLL" --collector-envs 2 \
    --stream-groups "$STREAM_GROUPS" --group-weights "$GROUP_WEIGHTS" \
    --resume "$resume" --prefix-checkpoint "$CHAMPION" \
    --anchor-dir "$ANCHOR_DIR" --out "$OUT" --minutes "$MINUTES" \
    --batch-size 256 --learning-rate 2e-5 --gamma 0.999 --n-step 3 \
    --critic-only-updates 5000 --anchor-beta-start .30 --anchor-beta-floor .08 \
    --anchor-decay-updates 60000 --stddev-start "$STDDEV_START" \
    --stddev-end "$STDDEV_END" --stddev-steps "$STDDEV_STEPS" \
    --initial-stddev "$INITIAL_STDDEV" $reset_flag --suffix-alpha 1.0 \
    --elite-dir "$OUT/elite_episodes" --elite-replay-fraction .25 \
    --elite-replay-capacity 50000 --elite-consolidation-updates 60000 \
    --elite-archive-max-files 48 \
    --elite-behavior-weight 2.0 --elite-behavior-batch-size 32 \
    --elite-behavior-score-capacity 1800 --elite-behavior-trigger-capacity 200 \
    --elite-behavior-trigger-fraction .25 \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" --collect-weight "$COLLECT_WEIGHT" \
    --progress-per-m "$PROGRESS_PER_M" --progress-step-cap "$PROGRESS_STEP_CAP" \
    --ramp-bonus "$RAMP_BONUS" --leave-grace-steps "$LEAVE_GRACE_STEPS" \
    --leave-penalty-per-step "$LEAVE_PENALTY_PER_STEP" \
    --leave-penalty-cap "$LEAVE_PENALTY_CAP" \
    --return-grace-steps "$RETURN_GRACE_STEPS" \
    --return-penalty-per-step "$RETURN_PENALTY_PER_STEP" \
    --return-penalty-cap "$RETURN_PENALTY_CAP" --shoot-grace-s "$SHOOT_GRACE_S" \
    --shoot-penalty-per-step "$SHOOT_PENALTY_PER_STEP" \
    --shoot-penalty-cap "$SHOOT_PENALTY_CAP" \
    --dump-lost-aim-grace-ticks "$DUMP_LOST_AIM_GRACE_TICKS" \
    --partial-dump-penalty-per-ball "$PARTIAL_DUMP_PENALTY_PER_BALL" \
    --partial-dump-penalty-cap "$PARTIAL_DUMP_PENALTY_CAP" \
    >> "$OUT/learner.log" 2>&1 &
  LPID=$!
  LSTART=$(date +%s)
  echo "$(date '+%F %T') learner GPU3 pid=$LPID resume=$resume reset=${reset_flag:-no}"
}

launch_collector() {
  local cid=$1
  local gpu modes episode
  gpu=$(gpu_of "$cid")
  modes=$(mode_of "$cid")
  episode=$(episode_of "$cid")
  CUDA_VISIBLE_DEVICES=$gpu setsid python scripts/rl/collector_cycle_v2.py \
    --collector-id "$cid" --root "$ROOT" --num-envs 2 \
    --stagec-v2-prefix-checkpoint "$CHAMPION" --reset-modes "$modes" \
    --template "$TEMPLATE" --episode-len-s "$episode" \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" --collect-weight "$COLLECT_WEIGHT" \
    --dump-on-press --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" --progress-per-m "$PROGRESS_PER_M" \
    --progress-step-cap "$PROGRESS_STEP_CAP" --ramp-bonus "$RAMP_BONUS" \
    --leave-grace-steps "$LEAVE_GRACE_STEPS" \
    --leave-penalty-per-step "$LEAVE_PENALTY_PER_STEP" \
    --leave-penalty-cap "$LEAVE_PENALTY_CAP" \
    --return-grace-steps "$RETURN_GRACE_STEPS" \
    --return-penalty-per-step "$RETURN_PENALTY_PER_STEP" \
    --return-penalty-cap "$RETURN_PENALTY_CAP" --shoot-grace-s "$SHOOT_GRACE_S" \
    --shoot-penalty-per-step "$SHOOT_PENALTY_PER_STEP" \
    --shoot-penalty-cap "$SHOOT_PENALTY_CAP" \
    --dump-lost-aim-grace-ticks "$DUMP_LOST_AIM_GRACE_TICKS" \
    --partial-dump-penalty-per-ball "$PARTIAL_DUMP_PENALTY_PER_BALL" \
    --partial-dump-penalty-cap "$PARTIAL_DUMP_PENALTY_CAP" \
    --stddev-start "$STDDEV_START" --stddev-end "$STDDEV_END" \
    --stddev-steps "$STDDEV_STEPS" --seed $((8100 + cid)) --minutes "$MINUTES" \
    --telemetry "$OUT/cycle_telemetry.jsonl" >> "$OUT/collector${cid}.log" 2>&1 &
  CPID[$cid]=$!
  CSTART[$cid]=$(date +%s)
  echo "$(date '+%F %T') collector$cid GPU$gpu modes=$modes pid=${CPID[$cid]}"
}

cleanup() {
  echo "$(date '+%F %T') cleanup"
  [ -n "${LPID:-}" ] && kill "$LPID" 2>/dev/null || true
  for cid in "${!CPID[@]}"; do kill "${CPID[$cid]}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

echo "$(date '+%F %T') cycle-2 recovery run=$RUN_ID out=$OUT"
launch_learner
sleep 20
for cid in $(seq 0 $((NCOLL - 1))); do
  launch_collector "$cid"
  sleep 10
done

mkdir -p "$DASHBOARD_RUNS"
ln -sfn "$OUT" "$DASHBOARD_RUNS/stagec_v2_latest"
echo "$OUT" > /root/stagec_v2_cycle3.outdir
echo "$(date '+%F %T') all collectors launched; watchdog active"

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
      echo "$(date '+%F %T') learner failed 3x; aborting" | tee "$OUT/LAUNCH_FAILED"
      exit 1
    fi
    sleep 5
    launch_learner
  fi
  for cid in $(seq 0 $((NCOLL - 1))); do
    if ! kill -0 "${CPID[$cid]:-0}" 2>/dev/null; then
      echo "$(date '+%F %T') collector$cid died; respawning"
      launch_collector "$cid"
    fi
  done
  sleep 20
done

echo "$(date '+%F %T') cycle-2 recovery duration complete"
