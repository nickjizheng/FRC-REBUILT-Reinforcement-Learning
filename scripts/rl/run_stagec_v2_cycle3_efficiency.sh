#!/bin/bash
# Cycle-3 route-efficiency branch from a preserved Stage C v2 champion.
#
# The first-cycle prefix remains immutable.  Six FULL environments train the
# real cycle-2/cycle-3 sequence, while two homogeneous RETURN environments keep
# return-and-shoot sharp.  Historical elite archives are intentionally not
# seeded because their stored rewards predate the outer-rail revision.
set -u

CHAMPION=${1:?immutable first-cycle prefix checkpoint}
RESUME=${2:?known-good Stage-C v2 checkpoint}
ANCHOR_DIR=${3:?champion anchor directory}
MINUTES=${4:-600}
TEMPLATE=${5:-/root/frc-rl/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${6:-/root/autodl-tmp/runs/stagec_v2_cycle3_efficiency_${RUN_ID}}
ROOT=${STAGEC_V2_ROOT:-/dev/shm/frc_stagec_v2_cycle3_efficiency_${RUN_ID}}
STOP=${STAGEC_V2_STOP:-/root/STOP_STAGEC_V2_CYCLE3_EFFICIENCY}
RESET_SCHEDULES=${STAGEC_V2_RESET_SCHEDULES:-1}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DASHBOARD_RUNS=${STAGEC_V2_DASHBOARD_RUNS:-$CODE_ROOT/runs}

NCOLL=4
STREAM_GROUPS=full,full,full,full,full,full,return,return
GROUP_WEIGHTS=full=.75,return=.25
NCOLL=${STAGEC_V2_NCOLL:-$NCOLL}
STREAM_GROUPS=${STAGEC_V2_STREAM_GROUPS:-$STREAM_GROUPS}
GROUP_WEIGHTS=${STAGEC_V2_GROUP_WEIGHTS:-$GROUP_WEIGHTS}

TARGET_LOAD=${STAGEC_V2_TARGET_LOAD:-15}
RESERVE_COUNT=${STAGEC_V2_RESERVE_COUNT:-18}
RESERVE_BATCHES=${STAGEC_V2_RESERVE_BATCHES:-1}
MAX_DUMP_TICKS=180
STDDEV_START=${STAGEC_V2_STDDEV_START:-1.0}
STDDEV_END=${STAGEC_V2_STDDEV_END:-0.30}
STDDEV_STEPS=${STAGEC_V2_STDDEV_STEPS:-150000}
INITIAL_STDDEV=${STAGEC_V2_INITIAL_STDDEV:-0.32}
LEARNING_RATE=${STAGEC_V2_LEARNING_RATE:-0.00001}
CRITIC_ONLY_UPDATES=${STAGEC_V2_CRITIC_ONLY_UPDATES:-5000}
ACTOR_UPDATE_INTERVAL=${STAGEC_V2_ACTOR_UPDATE_INTERVAL:-1}
ACTOR_PHASES=${STAGEC_V2_ACTOR_PHASES:-leave,collect,return,score}
RESET_OPTIMIZER_STATE=${STAGEC_V2_RESET_OPTIMIZER_STATE:-1}
SUFFIX_ALPHA=${STAGEC_V2_SUFFIX_ALPHA:-1.0}
ELITE_BEHAVIOR_WEIGHT=${STAGEC_V2_ELITE_BEHAVIOR_WEIGHT:-1.0}
ELITE_BEHAVIOR_BATCH_SIZE=${STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE:-32}
ELITE_BEHAVIOR_SCORE_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY:-2400}
ELITE_BEHAVIOR_TRIGGER_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY:-300}
ELITE_BEHAVIOR_TRIGGER_FRACTION=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_FRACTION:-.25}
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

# Route-efficiency v3.  V2 still let the policy accept a small capped fine and
# remain on the positive outer rail for most of the episode.  V3 moves the
# boundary inward, allows only 0.2 s for a clean crossing, charges immediately
# at full scale, escalates rapidly, and makes prolonged camping cost up to the
# equivalent of 15 scored balls per cycle.
RAMP_SIDE_DEADBAND_X=${STAGEC_V2_RAMP_SIDE_DEADBAND_X:-0.25}
REQUIRE_RAMP_OUT=${STAGEC_V2_REQUIRE_RAMP_OUT:-0}
RAMP_OUT_HALF_WIDTH=${STAGEC_V2_RAMP_OUT_HALF_WIDTH:-0.90}
RAMP_OUT_BONUS=${STAGEC_V2_RAMP_OUT_BONUS:-0.0}
OFF_RAMP_EXIT_PENALTY=${STAGEC_V2_OFF_RAMP_EXIT_PENALTY:-0.0}
OUTER_RAIL_ENTER_X=${STAGEC_V2_OUTER_RAIL_ENTER_X:-2.55}
OUTER_RAIL_EXIT_X=${STAGEC_V2_OUTER_RAIL_EXIT_X:-2.20}
OUTER_RAIL_MAX_X=${STAGEC_V2_OUTER_RAIL_MAX_X:-3.60}
OUTER_RAIL_GRACE_STEPS=${STAGEC_V2_OUTER_RAIL_GRACE_STEPS:-2}
OUTER_RAIL_PENALTY_PER_STEP=${STAGEC_V2_OUTER_RAIL_PENALTY_PER_STEP:-0.30}
OUTER_RAIL_PENALTY_CAP=${STAGEC_V2_OUTER_RAIL_PENALTY_CAP:-150.0}
OUTER_RAIL_MIN_SCALE=${STAGEC_V2_OUTER_RAIL_MIN_SCALE:-1.0}
OUTER_RAIL_ESCALATION_STEPS=${STAGEC_V2_OUTER_RAIL_ESCALATION_STEPS:-10}
OUTER_RAIL_MAX_MULTIPLIER=${STAGEC_V2_OUTER_RAIL_MAX_MULTIPLIER:-5.0}
INTAKE_SUBSTEPS=${STAGEC_V2_INTAKE_SUBSTEPS:-2}
INTAKE_DURING_RETURN=${STAGEC_V2_INTAKE_DURING_RETURN:-1}
POSTDUMP_REQUIRE_TARGET_LOAD=${STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD:-0}
POSTDUMP_COMPLETE_CYCLE=${STAGEC_V2_POSTDUMP_COMPLETE_CYCLE:-0}
POSTDUMP_DEPLETED_COUNT=${STAGEC_V2_POSTDUMP_DEPLETED_COUNT:-0}
POSTDUMP_DEPLETED_PROB=${STAGEC_V2_POSTDUMP_DEPLETED_PROB:-0.0}
PREFERRED_REPEAT_LOAD=${STAGEC_V2_PREFERRED_REPEAT_LOAD:-0}
COLLECT_STALL_STEPS=${STAGEC_V2_COLLECT_STALL_STEPS:-0}
RETURN_TIME_GUARD=${STAGEC_V2_RETURN_TIME_GUARD:-0.0}
REPEAT_LOAD_RETURN_BONUS=${STAGEC_V2_REPEAT_LOAD_RETURN_BONUS:-0.0}
REPEAT_LOAD_SCORE_BONUS=${STAGEC_V2_REPEAT_LOAD_SCORE_BONUS:-0.0}
REWARD_REVISION=${STAGEC_V2_REWARD_REVISION:-score_efficiency_v10_return_intake}
WEIGHT_PUBLISH_UPDATES=${STAGEC_V2_WEIGHT_PUBLISH_UPDATES:-400}
EVAL_SNAPSHOT_UPDATES=${STAGEC_V2_EVAL_SNAPSHOT_UPDATES:-5000}
FREEZE_COLLECTOR_WEIGHTS=${STAGEC_V2_FREEZE_COLLECTOR_WEIGHTS:-0}
ELITE_BEHAVIOR_SEEDMINE_ONLY=${STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY:-0}
SEEDMINE_ELITE_DIR=${STAGEC_V2_SEEDMINE_ELITE_DIR:-}
SEEDMINE_SOURCE_CHECKPOINT=${STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT:-}
SEEDMINE_FLAGS=""
if [ -n "$SEEDMINE_ELITE_DIR" ] || [ -n "$SEEDMINE_SOURCE_CHECKPOINT" ]; then
  if [ -z "$SEEDMINE_ELITE_DIR" ] || [ -z "$SEEDMINE_SOURCE_CHECKPOINT" ]; then
    echo "seed-mine elite directory and source checkpoint must be set together" >&2
    exit 2
  fi
  SEEDMINE_FLAGS="--seedmine-elite-dir $SEEDMINE_ELITE_DIR --seedmine-source-checkpoint $SEEDMINE_SOURCE_CHECKPOINT"
fi
RAMP_OUT_FLAG=""
if [ "$REQUIRE_RAMP_OUT" = "1" ]; then
  RAMP_OUT_FLAG="--require-ramp-out"
elif [ "$REQUIRE_RAMP_OUT" != "0" ]; then
  echo "STAGEC_V2_REQUIRE_RAMP_OUT must be 0 or 1" >&2
  exit 2
fi
POSTDUMP_COMPLETE_FLAG=""
if [ "$POSTDUMP_COMPLETE_CYCLE" = "1" ]; then
  if [ "$POSTDUMP_REQUIRE_TARGET_LOAD" != "1" ] || [ "$REQUIRE_RAMP_OUT" != "1" ]; then
    echo "STAGEC_V2_POSTDUMP_COMPLETE_CYCLE requires target-load and ramp-out modes" >&2
    exit 2
  fi
  POSTDUMP_COMPLETE_FLAG="--postdump-complete-cycle"
elif [ "$POSTDUMP_COMPLETE_CYCLE" != "0" ]; then
  echo "STAGEC_V2_POSTDUMP_COMPLETE_CYCLE must be 0 or 1" >&2
  exit 2
fi
POSTDUMP_TARGET_FLAG=""
if [ "$POSTDUMP_REQUIRE_TARGET_LOAD" = "1" ]; then
  POSTDUMP_TARGET_FLAG="--postdump-require-target-load"
elif [ "$POSTDUMP_REQUIRE_TARGET_LOAD" != "0" ]; then
  echo "STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD must be 0 or 1" >&2
  exit 2
fi
RESET_OPTIMIZER_FLAG=""
if [ "$RESET_OPTIMIZER_STATE" = "1" ]; then
  RESET_OPTIMIZER_FLAG="--reset-optimizer-state-on-resume"
elif [ "$RESET_OPTIMIZER_STATE" != "0" ]; then
  echo "STAGEC_V2_RESET_OPTIMIZER_STATE must be 0 or 1" >&2
  exit 2
fi
SEEDMINE_BEHAVIOR_FLAG=""
if [ "$ELITE_BEHAVIOR_SEEDMINE_ONLY" = "1" ]; then
  SEEDMINE_BEHAVIOR_FLAG="--elite-behavior-seedmine-only"
elif [ "$ELITE_BEHAVIOR_SEEDMINE_ONLY" != "0" ]; then
  echo "STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY must be 0 or 1" >&2
  exit 2
fi
RETURN_INTAKE_FLAG=""
if [ "$INTAKE_DURING_RETURN" = "1" ]; then
  RETURN_INTAKE_FLAG="--intake-during-return"
elif [ "$INTAKE_DURING_RETURN" != "0" ]; then
  echo "STAGEC_V2_INTAKE_DURING_RETURN must be 0 or 1" >&2
  exit 2
fi
FREEZE_COLLECTOR_FLAG=""
if [ "$FREEZE_COLLECTOR_WEIGHTS" = "1" ]; then
  FREEZE_COLLECTOR_FLAG="--freeze-collector-weights"
elif [ "$FREEZE_COLLECTOR_WEIGHTS" != "0" ]; then
  echo "STAGEC_V2_FREEZE_COLLECTOR_WEIGHTS must be 0 or 1" >&2
  exit 2
fi

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
if [ "$REWARD_REVISION" != "score_efficiency_v10_return_intake" ]; then
  echo "this code revision only launches score_efficiency_v10_return_intake, got: $REWARD_REVISION" >&2
  exit 2
fi
if [ "$INTAKE_DURING_RETURN" != "1" ]; then
  echo "score_efficiency_v10_return_intake requires STAGEC_V2_INTAKE_DURING_RETURN=1" >&2
  exit 2
fi
if [ -d "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "refusing to overwrite non-empty output directory: $OUT" >&2
  exit 2
fi

mkdir -p "$ROOT" "$OUT" "$OUT/elite_episodes"
rm -rf "$ROOT"/collector_* "$ROOT"/weights 2>/dev/null || true
rm -f "$STOP"

cat > "$OUT/launcher_manifest.json" <<EOF
{"run_id":"$RUN_ID","code_root":"$CODE_ROOT","root":"$ROOT","out":"$OUT","champion":"$CHAMPION","resume":"$RESUME","collectors":$NCOLL,"stream_groups":"$STREAM_GROUPS","group_weights":"$GROUP_WEIGHTS","collector_action_modes":"${STAGEC_V2_COLLECTOR_ACTION_MODES:-explore}","learning_rate":$LEARNING_RATE,"elite_replay_fraction":0.25,"elite_behavior_weight":$ELITE_BEHAVIOR_WEIGHT,"elite_behavior_batch_size":$ELITE_BEHAVIOR_BATCH_SIZE,"elite_behavior_score_capacity":$ELITE_BEHAVIOR_SCORE_CAPACITY,"elite_behavior_trigger_capacity":$ELITE_BEHAVIOR_TRIGGER_CAPACITY,"elite_behavior_trigger_fraction":$ELITE_BEHAVIOR_TRIGGER_FRACTION,"elite_behavior_seedmine_only":$ELITE_BEHAVIOR_SEEDMINE_ONLY,"elite_archive_max_files":96,"reset_schedules":$RESET_SCHEDULES,"reset_optimizer_state":$RESET_OPTIMIZER_STATE,"critic_only_updates":$CRITIC_ONLY_UPDATES,"actor_update_interval":$ACTOR_UPDATE_INTERVAL,"actor_phases":"$ACTOR_PHASES","suffix_alpha":$SUFFIX_ALPHA,"initial_stddev":$INITIAL_STDDEV,"weight_publish_updates":$WEIGHT_PUBLISH_UPDATES,"freeze_collector_weights":$FREEZE_COLLECTOR_WEIGHTS,"eval_snapshot_updates":$EVAL_SNAPSHOT_UPDATES,"seedmine_elite_dir":"$SEEDMINE_ELITE_DIR","seedmine_source_checkpoint":"$SEEDMINE_SOURCE_CHECKPOINT","target_load":$TARGET_LOAD,"preferred_repeat_load":$PREFERRED_REPEAT_LOAD,"collect_stall_steps":$COLLECT_STALL_STEPS,"return_time_guard":$RETURN_TIME_GUARD,"intake_during_return":$INTAKE_DURING_RETURN,"repeat_load_return_bonus":$REPEAT_LOAD_RETURN_BONUS,"repeat_load_score_bonus":$REPEAT_LOAD_SCORE_BONUS,"full_episode_s":${STAGEC_V2_FULL_EPISODE_S:-120},"postdump_episode_s":${STAGEC_V2_POSTDUMP_EPISODE_S:-45},"collect_episode_s":${STAGEC_V2_COLLECT_EPISODE_S:-60},"return_episode_s":${STAGEC_V2_RETURN_EPISODE_S:-75},"reward_revision":"$REWARD_REVISION","refresh_ramp_side_on_dump":true,"require_ramp_out":$REQUIRE_RAMP_OUT,"ramp_out_half_width":$RAMP_OUT_HALF_WIDTH,"ramp_out_bonus":$RAMP_OUT_BONUS,"off_ramp_exit_penalty":$OFF_RAMP_EXIT_PENALTY,"postdump_require_target_load":$POSTDUMP_REQUIRE_TARGET_LOAD,"postdump_complete_cycle":$POSTDUMP_COMPLETE_CYCLE,"postdump_depleted_count":$POSTDUMP_DEPLETED_COUNT,"postdump_depleted_prob":$POSTDUMP_DEPLETED_PROB,"outer_rail_enter_x":$OUTER_RAIL_ENTER_X,"outer_rail_exit_x":$OUTER_RAIL_EXIT_X,"outer_rail_max_x":$OUTER_RAIL_MAX_X,"outer_rail_grace_steps":$OUTER_RAIL_GRACE_STEPS,"outer_rail_penalty_per_step":$OUTER_RAIL_PENALTY_PER_STEP,"outer_rail_penalty_cap":$OUTER_RAIL_PENALTY_CAP,"outer_rail_min_scale":$OUTER_RAIL_MIN_SCALE,"outer_rail_escalation_steps":$OUTER_RAIL_ESCALATION_STEPS,"outer_rail_max_multiplier":$OUTER_RAIL_MAX_MULTIPLIER,"intake_substeps":$INTAKE_SUBSTEPS}
EOF
sha256sum "$CHAMPION" "$RESUME" > "$OUT/input_sha256.txt"

mode_of() {
  case "$1" in
    0|1|2) echo "full,full" ;;
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
    0|1|2) echo 120 ;;
    3) echo 75 ;;
  esac
}

# A semicolon-separated override supplies one homogeneous mode pair per
# collector.  This keeps the proven watchdog launcher while allowing a focused
# route curriculum without replacing the historical script.
if [ -n "${STAGEC_V2_COLLECTOR_MODES:-}" ]; then
  IFS=';' read -r -a STAGEC_V2_MODE_ARRAY <<< "$STAGEC_V2_COLLECTOR_MODES"
  if [ "${#STAGEC_V2_MODE_ARRAY[@]}" -ne "$NCOLL" ]; then
    echo "STAGEC_V2_COLLECTOR_MODES must contain exactly $NCOLL entries" >&2
    exit 2
  fi
  mode_of() { echo "${STAGEC_V2_MODE_ARRAY[$1]}"; }
  gpu_of() { echo $(( $1 / ${STAGEC_V2_COLLECTORS_PER_GPU:-4} )); }
  episode_of() {
    case "$(mode_of "$1")" in
      full,full) echo "${STAGEC_V2_FULL_EPISODE_S:-120}" ;;
      postdump,postdump) echo "${STAGEC_V2_POSTDUMP_EPISODE_S:-45}" ;;
      collect,collect) echo "${STAGEC_V2_COLLECT_EPISODE_S:-60}" ;;
      return,return) echo "${STAGEC_V2_RETURN_EPISODE_S:-75}" ;;
      *) echo 75 ;;
    esac
  }
fi

action_flag_of() { echo ""; }
action_name_of() { echo "explore"; }
if [ -n "${STAGEC_V2_COLLECTOR_ACTION_MODES:-}" ]; then
  IFS=';' read -r -a STAGEC_V2_ACTION_ARRAY <<< "$STAGEC_V2_COLLECTOR_ACTION_MODES"
  if [ "${#STAGEC_V2_ACTION_ARRAY[@]}" -ne "$NCOLL" ]; then
    echo "STAGEC_V2_COLLECTOR_ACTION_MODES must contain exactly $NCOLL entries" >&2
    exit 2
  fi
  action_flag_of() {
    case "${STAGEC_V2_ACTION_ARRAY[$1]}" in
      mean) echo "--deterministic-suffix" ;;
      explore) echo "" ;;
      *) echo "invalid collector action mode: ${STAGEC_V2_ACTION_ARRAY[$1]}" >&2; return 2 ;;
    esac
  }
  action_name_of() { echo "${STAGEC_V2_ACTION_ARRAY[$1]}"; }
fi

LPID=""
LSTART=0
LFAIL=0
declare -A CPID CSTART CFAIL
for cid in $(seq 0 $((NCOLL - 1))); do
  CSTART[$cid]=0
  CFAIL[$cid]=0
done

launch_learner() {
  local resume="$RESUME"
  local reset_flag=""
  local migration_flag="--allow-route-efficiency-revision-migration"
  local target_migration_flag=""
  local suffix_alpha_migration_flag=""
  local optimizer_reset_flag="$RESET_OPTIMIZER_FLAG"
  if [ "${STAGEC_V2_ALLOW_TARGET_LOAD_MIGRATION:-0}" = "1" ]; then
    target_migration_flag="--allow-target-load-migration"
  fi
  if [ "${STAGEC_V2_ALLOW_SUFFIX_ALPHA_MIGRATION:-0}" = "1" ]; then
    suffix_alpha_migration_flag="--allow-suffix-alpha-migration"
  fi
  if [ "$RESET_SCHEDULES" = "1" ]; then
    reset_flag="--reset-schedules-on-resume"
  fi
  if [ "$FREEZE_COLLECTOR_WEIGHTS" != "1" ] && [ -s "$OUT/latest.pt" ]; then
    resume="$OUT/latest.pt"
    reset_flag=""
    migration_flag=""
    target_migration_flag=""
    suffix_alpha_migration_flag=""
    optimizer_reset_flag=""
  fi
  CUDA_VISIBLE_DEVICES=3 setsid python scripts/rl/learner_cycle_v2.py \
    --root "$ROOT" --num-collectors "$NCOLL" --collector-envs 2 \
    --stream-groups "$STREAM_GROUPS" --group-weights "$GROUP_WEIGHTS" \
    --resume "$resume" --prefix-checkpoint "$CHAMPION" \
    --anchor-dir "$ANCHOR_DIR" --out "$OUT" --minutes "$MINUTES" \
    --full-episode-s "${STAGEC_V2_FULL_EPISODE_S:-120}" \
    --batch-size 256 --learning-rate "$LEARNING_RATE" --gamma 0.999 --n-step 3 \
    --critic-only-updates "$CRITIC_ONLY_UPDATES" --anchor-beta-start .30 --anchor-beta-floor .12 \
    --anchor-decay-updates 80000 --stddev-start "$STDDEV_START" \
    --stddev-end "$STDDEV_END" --stddev-steps "$STDDEV_STEPS" \
    --initial-stddev "$INITIAL_STDDEV" \
    --actor-update-interval "$ACTOR_UPDATE_INTERVAL" \
    --actor-phases "$ACTOR_PHASES" \
    --weight-publish-updates "$WEIGHT_PUBLISH_UPDATES" \
    $FREEZE_COLLECTOR_FLAG \
    --eval-snapshot-updates "$EVAL_SNAPSHOT_UPDATES" \
    $reset_flag $migration_flag $optimizer_reset_flag \
    $target_migration_flag $suffix_alpha_migration_flag \
    --suffix-alpha "$SUFFIX_ALPHA" \
    --elite-dir "$OUT/elite_episodes" --elite-replay-fraction .25 \
    --elite-replay-capacity 50000 --elite-consolidation-updates 80000 \
    --elite-archive-max-files 96 \
    --elite-behavior-weight "$ELITE_BEHAVIOR_WEIGHT" \
    --elite-behavior-batch-size "$ELITE_BEHAVIOR_BATCH_SIZE" \
    --elite-behavior-score-capacity "$ELITE_BEHAVIOR_SCORE_CAPACITY" \
    --elite-behavior-trigger-capacity "$ELITE_BEHAVIOR_TRIGGER_CAPACITY" \
    --elite-behavior-trigger-fraction "$ELITE_BEHAVIOR_TRIGGER_FRACTION" \
    $SEEDMINE_BEHAVIOR_FLAG $SEEDMINE_FLAGS \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" --collect-weight "$COLLECT_WEIGHT" \
    --progress-per-m "$PROGRESS_PER_M" --progress-step-cap "$PROGRESS_STEP_CAP" \
    --ramp-bonus "$RAMP_BONUS" \
    --route-efficiency-revision --refresh-ramp-side-on-dump \
    --ramp-side-deadband-x "$RAMP_SIDE_DEADBAND_X" \
    $RAMP_OUT_FLAG --ramp-out-half-width "$RAMP_OUT_HALF_WIDTH" \
    --ramp-out-bonus "$RAMP_OUT_BONUS" \
    --off-ramp-exit-penalty "$OFF_RAMP_EXIT_PENALTY" \
    $POSTDUMP_TARGET_FLAG $POSTDUMP_COMPLETE_FLAG \
    --postdump-depleted-count "$POSTDUMP_DEPLETED_COUNT" \
    --postdump-depleted-prob "$POSTDUMP_DEPLETED_PROB" \
    --preferred-repeat-load "$PREFERRED_REPEAT_LOAD" \
    --collect-stall-steps "$COLLECT_STALL_STEPS" \
    --return-time-guard "$RETURN_TIME_GUARD" \
    $RETURN_INTAKE_FLAG \
    --repeat-load-return-bonus "$REPEAT_LOAD_RETURN_BONUS" \
    --repeat-load-score-bonus "$REPEAT_LOAD_SCORE_BONUS" \
    --outer-rail-enter-x "$OUTER_RAIL_ENTER_X" \
    --outer-rail-exit-x "$OUTER_RAIL_EXIT_X" \
    --outer-rail-max-x "$OUTER_RAIL_MAX_X" \
    --outer-rail-grace-steps "$OUTER_RAIL_GRACE_STEPS" \
    --outer-rail-penalty-per-step "$OUTER_RAIL_PENALTY_PER_STEP" \
    --outer-rail-penalty-cap "$OUTER_RAIL_PENALTY_CAP" \
    --outer-rail-min-scale "$OUTER_RAIL_MIN_SCALE" \
    --outer-rail-escalation-steps "$OUTER_RAIL_ESCALATION_STEPS" \
    --outer-rail-max-multiplier "$OUTER_RAIL_MAX_MULTIPLIER" \
    --intake-substeps "$INTAKE_SUBSTEPS" \
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
    >> "$OUT/learner.log" 2>&1 &
  LPID=$!
  LSTART=$(date +%s)
  echo "$(date '+%F %T') learner GPU3 pid=$LPID resume=$resume reset=${reset_flag:-no} migration=${migration_flag:-no}"
}

launch_collector() {
  local cid=$1
  local gpu modes episode action_flag
  gpu=$(gpu_of "$cid")
  modes=$(mode_of "$cid")
  episode=$(episode_of "$cid")
  action_flag=$(action_flag_of "$cid") || return 2
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
    --route-efficiency-revision --refresh-ramp-side-on-dump \
    --ramp-side-deadband-x "$RAMP_SIDE_DEADBAND_X" \
    $RAMP_OUT_FLAG --ramp-out-half-width "$RAMP_OUT_HALF_WIDTH" \
    --ramp-out-bonus "$RAMP_OUT_BONUS" \
    --off-ramp-exit-penalty "$OFF_RAMP_EXIT_PENALTY" \
    $POSTDUMP_TARGET_FLAG $POSTDUMP_COMPLETE_FLAG $action_flag \
    --postdump-depleted-count "$POSTDUMP_DEPLETED_COUNT" \
    --postdump-depleted-prob "$POSTDUMP_DEPLETED_PROB" \
    --preferred-repeat-load "$PREFERRED_REPEAT_LOAD" \
    --collect-stall-steps "$COLLECT_STALL_STEPS" \
    --return-time-guard "$RETURN_TIME_GUARD" \
    $RETURN_INTAKE_FLAG \
    --repeat-load-return-bonus "$REPEAT_LOAD_RETURN_BONUS" \
    --repeat-load-score-bonus "$REPEAT_LOAD_SCORE_BONUS" \
    --outer-rail-enter-x "$OUTER_RAIL_ENTER_X" \
    --outer-rail-exit-x "$OUTER_RAIL_EXIT_X" \
    --outer-rail-max-x "$OUTER_RAIL_MAX_X" \
    --outer-rail-grace-steps "$OUTER_RAIL_GRACE_STEPS" \
    --outer-rail-penalty-per-step "$OUTER_RAIL_PENALTY_PER_STEP" \
    --outer-rail-penalty-cap "$OUTER_RAIL_PENALTY_CAP" \
    --outer-rail-min-scale "$OUTER_RAIL_MIN_SCALE" \
    --outer-rail-escalation-steps "$OUTER_RAIL_ESCALATION_STEPS" \
    --outer-rail-max-multiplier "$OUTER_RAIL_MAX_MULTIPLIER" \
    --intake-substeps "$INTAKE_SUBSTEPS" \
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
    --stddev-steps "$STDDEV_STEPS" --seed $((9100 + cid)) --minutes "$MINUTES" \
    --telemetry "$OUT/cycle_telemetry.jsonl" >> "$OUT/collector${cid}.log" 2>&1 &
  CPID[$cid]=$!
  CSTART[$cid]=$(date +%s)
  echo "$(date '+%F %T') collector$cid GPU$gpu modes=$modes action=$(action_name_of "$cid") pid=${CPID[$cid]}"
}

cleanup() {
  echo "$(date '+%F %T') cleanup"
  [ -n "${LPID:-}" ] && kill "$LPID" 2>/dev/null || true
  for cid in "${!CPID[@]}"; do kill "${CPID[$cid]}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
on_signal() {
  # A TERM/INT trap returns to the interrupted for-loop unless it explicitly
  # exits.  Returning here previously let a stopped supervisor launch more
  # collectors immediately after cleanup.
  exit 0
}
trap on_signal TERM INT
trap cleanup EXIT

echo "$(date '+%F %T') cycle-3 efficiency run=$RUN_ID out=$OUT"
END=$(( $(date +%s) + ${MINUTES%.*} * 60 ))
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

while [ "$(date +%s)" -lt "$END" ]; do
  if [ -f "$STOP" ]; then
    echo "$(date '+%F %T') stop file detected"
    exit 0
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    learner_status=0
    wait "$LPID" || learner_status=$?
    if [ "$FREEZE_COLLECTOR_WEIGHTS" = "1" ]; then
      if [ "$learner_status" -eq 0 ] && [ -s "$OUT/final.pt" ]; then
        echo "$(date '+%F %T') guarded learner completed cleanly; stopping collectors without publishing candidate"
        exit 0
      fi
      echo "$(date '+%F %T') guarded learner failed exit=$learner_status; aborting without candidate promotion" | tee "$OUT/LAUNCH_FAILED"
      exit 1
    fi
    echo "$(date '+%F %T') learner pid=$LPID died exit=$learner_status; restarting from latest"
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
      collector_status=0
      wait "${CPID[$cid]}" || collector_status=$?
      now=$(date +%s)
      if [ $((now - ${CSTART[$cid]})) -lt 180 ]; then
        CFAIL[$cid]=$(( ${CFAIL[$cid]} + 1 ))
      else
        CFAIL[$cid]=1
      fi
      if [ "${CFAIL[$cid]}" -ge 5 ]; then
        echo "$(date '+%F %T') collector$cid failed 5x rapidly; aborting" | tee "$OUT/LAUNCH_FAILED"
        exit 1
      fi
      echo "$(date '+%F %T') collector$cid died exit=$collector_status; respawning failure=${CFAIL[$cid]}"
      launch_collector "$cid"
    fi
  done
  sleep 20
done

echo "$(date '+%F %T') cycle-3 efficiency duration complete"
