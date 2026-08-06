#!/bin/bash
# Guarded V9 recovery from the immutable V4 champion.
#
# V9 fixes the unreachable repeat-load objective: 15 balls marks a useful load,
# but COLLECT remains active until 20 balls, 3.0 s without a new pickup, or the
# last 25% of the match.  Candidate weights are not published back to
# collectors during this pilot; deterministic snapshots must pass evaluation
# before any policy can replace the champion-derived data generator.
set -eu

CHAMPION=${1:?immutable first-cycle prefix checkpoint}
RESUME=${2:?exact wrapped V9 champion checkpoint}
ANCHOR_DIR=${3:?champion anchor directory}
MINUTES=${4:-30}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TEMPLATE=${5:-$CODE_ROOT/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${6:-/root/autodl-tmp/runs/stagec_v2_score_efficiency_v9_pilot_${RUN_ID}}
: "${STAGEC_V2_SEEDMINE_ELITE_DIR:?validated deterministic V9 captures are required}"
: "${STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT:?exact V9 capture source is required}"

export STAGEC_V2_NCOLL=12
export STAGEC_V2_STREAM_GROUPS=full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full
export STAGEC_V2_GROUP_WEIGHTS=full=1.0
export STAGEC_V2_COLLECTOR_MODES='full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full'
export STAGEC_V2_COLLECTOR_ACTION_MODES='mean;mean;mean;mean;mean;mean;mean;mean;mean;explore;explore;explore'
export STAGEC_V2_COLLECTORS_PER_GPU=4

export STAGEC_V2_FULL_EPISODE_S=90
export STAGEC_V2_TARGET_LOAD=15
export STAGEC_V2_PREFERRED_REPEAT_LOAD=${STAGEC_V2_PREFERRED_REPEAT_LOAD:-20}
export STAGEC_V2_COLLECT_STALL_STEPS=${STAGEC_V2_COLLECT_STALL_STEPS:-30}
export STAGEC_V2_RETURN_TIME_GUARD=${STAGEC_V2_RETURN_TIME_GUARD:-0.25}
export STAGEC_V2_REPEAT_LOAD_RETURN_BONUS=${STAGEC_V2_REPEAT_LOAD_RETURN_BONUS:-8.0}
export STAGEC_V2_REPEAT_LOAD_SCORE_BONUS=${STAGEC_V2_REPEAT_LOAD_SCORE_BONUS:-12.0}
export STAGEC_V2_RESERVE_COUNT=18
export STAGEC_V2_RESERVE_BATCHES=1
export STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD=0
export STAGEC_V2_POSTDUMP_COMPLETE_CYCLE=0
export STAGEC_V2_POSTDUMP_DEPLETED_COUNT=0
export STAGEC_V2_POSTDUMP_DEPLETED_PROB=0.0
export STAGEC_V2_REWARD_REVISION=score_efficiency_v9
export STAGEC_V2_ALLOW_TARGET_LOAD_MIGRATION=0
export STAGEC_V2_ALLOW_SUFFIX_ALPHA_MIGRATION=0

# Slow, collect-only improvement with a freshly initialized optimizer.
export STAGEC_V2_LEARNING_RATE=${STAGEC_V2_LEARNING_RATE:-0.0000003}
export STAGEC_V2_CRITIC_ONLY_UPDATES=${STAGEC_V2_CRITIC_ONLY_UPDATES:-10000}
export STAGEC_V2_ACTOR_UPDATE_INTERVAL=${STAGEC_V2_ACTOR_UPDATE_INTERVAL:-20}
export STAGEC_V2_ACTOR_PHASES=${STAGEC_V2_ACTOR_PHASES:-collect}
export STAGEC_V2_RESET_OPTIMIZER_STATE=${STAGEC_V2_RESET_OPTIMIZER_STATE:-0}
export STAGEC_V2_SUFFIX_ALPHA=${STAGEC_V2_SUFFIX_ALPHA:-0.15}
export STAGEC_V2_ELITE_BEHAVIOR_WEIGHT=${STAGEC_V2_ELITE_BEHAVIOR_WEIGHT:-8.0}
export STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE=${STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE:-64}
export STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY:-12000}
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY:-600}
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_FRACTION=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_FRACTION:-0.15}
export STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY=1
export STAGEC_V2_INITIAL_STDDEV=${STAGEC_V2_INITIAL_STDDEV:-0.30}

# Freeze collector policy at the validated V9 wrapper.  Candidate snapshots
# are emitted every 400 critic steps for external deterministic promotion.
export STAGEC_V2_WEIGHT_PUBLISH_UPDATES=${STAGEC_V2_WEIGHT_PUBLISH_UPDATES:-1000000000}
export STAGEC_V2_EVAL_SNAPSHOT_UPDATES=${STAGEC_V2_EVAL_SNAPSHOT_UPDATES:-400}

# Preserve the V4 route geometry and fast-intake mechanics exactly.
export STAGEC_V2_REQUIRE_RAMP_OUT=1
export STAGEC_V2_RAMP_OUT_HALF_WIDTH=0.90
export STAGEC_V2_RAMP_OUT_BONUS=24.0
export STAGEC_V2_OFF_RAMP_EXIT_PENALTY=20.0
export STAGEC_V2_OUTER_RAIL_ENTER_X=2.55
export STAGEC_V2_OUTER_RAIL_EXIT_X=2.20
export STAGEC_V2_OUTER_RAIL_MAX_X=3.60
export STAGEC_V2_OUTER_RAIL_GRACE_STEPS=10
export STAGEC_V2_OUTER_RAIL_PENALTY_PER_STEP=0.22
export STAGEC_V2_OUTER_RAIL_PENALTY_CAP=110.0
export STAGEC_V2_OUTER_RAIL_MIN_SCALE=0.35
export STAGEC_V2_OUTER_RAIL_ESCALATION_STEPS=120
export STAGEC_V2_OUTER_RAIL_MAX_MULTIPLIER=3.0
export STAGEC_V2_INTAKE_SUBSTEPS=2

export STAGEC_V2_ROOT=${STAGEC_V2_ROOT:-/dev/shm/frc_stagec_v2_score_efficiency_v9_pilot_${RUN_ID}}
export STAGEC_V2_STOP=${STAGEC_V2_STOP:-/root/STOP_STAGEC_V2_SCORE_EFFICIENCY_V9_PILOT}
export STAGEC_V2_RESET_SCHEDULES=${STAGEC_V2_RESET_SCHEDULES:-1}

exec "$CODE_ROOT/scripts/rl/run_stagec_v2_cycle3_efficiency.sh" \
  "$CHAMPION" "$RESUME" "$ANCHOR_DIR" "$MINUTES" "$TEMPLATE" "$OUT"
