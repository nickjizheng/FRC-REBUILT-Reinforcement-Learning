#!/bin/bash
# Guarded V10 return-intake pilot from the immutable V4 route champion.
#
# V10 deliberately restores the champion's immediate COLLECT->RETURN switch at
# 15 balls.  The only mechanics change is that intake remains forced on during
# RETURN, allowing the unchanged route to pick up incidental balls on its way
# home.  Collectors stay on the validated wrapper; candidate weights require an
# external deterministic promotion gate before they can replace it.
set -eu

CHAMPION=${1:?immutable first-cycle prefix checkpoint}
RESUME=${2:?exact wrapped V10 champion checkpoint}
ANCHOR_DIR=${3:?champion anchor directory}
MINUTES=${4:-30}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TEMPLATE=${5:-$CODE_ROOT/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${6:-/root/autodl-tmp/runs/stagec_v2_score_efficiency_v10_pilot_${RUN_ID}}
: "${STAGEC_V2_SEEDMINE_ELITE_DIR:?validated deterministic V10 captures are required}"
: "${STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT:?exact V10 capture source is required}"
if [ ! -s "$STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT" ]; then
  echo "V10 seed-mine source checkpoint does not exist: $STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT" >&2
  exit 2
fi
if [ "$(sha256sum "$RESUME" | awk '{print $1}')" != \
     "$(sha256sum "$STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT" | awk '{print $1}')" ]; then
  echo "V10 seed-mine source must be the exact initial resume checkpoint" >&2
  exit 2
fi

export STAGEC_V2_NCOLL=12
export STAGEC_V2_STREAM_GROUPS=full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full
export STAGEC_V2_GROUP_WEIGHTS=full=1.0
export STAGEC_V2_COLLECTOR_MODES='full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full'
export STAGEC_V2_COLLECTOR_ACTION_MODES='mean;mean;mean;mean;mean;mean;mean;mean;mean;explore;explore;explore'
export STAGEC_V2_COLLECTORS_PER_GPU=4

export STAGEC_V2_FULL_EPISODE_S=90
export STAGEC_V2_TARGET_LOAD=15
# This remains a soft return/score shaping target. It no longer delays RETURN.
export STAGEC_V2_PREFERRED_REPEAT_LOAD=${STAGEC_V2_PREFERRED_REPEAT_LOAD:-20}
export STAGEC_V2_COLLECT_STALL_STEPS=0
export STAGEC_V2_RETURN_TIME_GUARD=0.0
export STAGEC_V2_INTAKE_DURING_RETURN=1
export STAGEC_V2_REPEAT_LOAD_RETURN_BONUS=${STAGEC_V2_REPEAT_LOAD_RETURN_BONUS:-8.0}
export STAGEC_V2_REPEAT_LOAD_SCORE_BONUS=${STAGEC_V2_REPEAT_LOAD_SCORE_BONUS:-12.0}
export STAGEC_V2_RESERVE_COUNT=18
export STAGEC_V2_RESERVE_BATCHES=1
export STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD=0
export STAGEC_V2_POSTDUMP_COMPLETE_CYCLE=0
export STAGEC_V2_POSTDUMP_DEPLETED_COUNT=0
export STAGEC_V2_POSTDUMP_DEPLETED_PROB=0.0
export STAGEC_V2_REWARD_REVISION=score_efficiency_v10_return_intake
export STAGEC_V2_ALLOW_TARGET_LOAD_MIGRATION=0
export STAGEC_V2_ALLOW_SUFFIX_ALPHA_MIGRATION=0

# Sparse RETURN-only actor fitting with a fresh optimizer. Critic warm-up sees
# legal suffix actions only; deterministic captures own actor-BC custody.
export STAGEC_V2_LEARNING_RATE=${STAGEC_V2_LEARNING_RATE:-0.0000003}
export STAGEC_V2_CRITIC_ONLY_UPDATES=${STAGEC_V2_CRITIC_ONLY_UPDATES:-10000}
export STAGEC_V2_ACTOR_UPDATE_INTERVAL=${STAGEC_V2_ACTOR_UPDATE_INTERVAL:-20}
export STAGEC_V2_ACTOR_PHASES=${STAGEC_V2_ACTOR_PHASES:-return}
export STAGEC_V2_RESET_OPTIMIZER_STATE=1
export STAGEC_V2_SUFFIX_ALPHA=${STAGEC_V2_SUFFIX_ALPHA:-0.15}
export STAGEC_V2_ELITE_BEHAVIOR_WEIGHT=${STAGEC_V2_ELITE_BEHAVIOR_WEIGHT:-8.0}
export STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE=${STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE:-64}
export STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY:-12000}
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY:-600}
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_FRACTION=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_FRACTION:-0.15}
export STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY=1
export STAGEC_V2_INITIAL_STDDEV=${STAGEC_V2_INITIAL_STDDEV:-0.30}

export STAGEC_V2_WEIGHT_PUBLISH_UPDATES=${STAGEC_V2_WEIGHT_PUBLISH_UPDATES:-1000000000}
export STAGEC_V2_FREEZE_COLLECTOR_WEIGHTS=1
export STAGEC_V2_EVAL_SNAPSHOT_UPDATES=${STAGEC_V2_EVAL_SNAPSHOT_UPDATES:-400}

# Preserve V4 route shaping and fast-intake physics exactly.
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

export STAGEC_V2_ROOT=${STAGEC_V2_ROOT:-/dev/shm/frc_stagec_v2_score_efficiency_v10_pilot_${RUN_ID}}
export STAGEC_V2_STOP=${STAGEC_V2_STOP:-/root/STOP_STAGEC_V2_SCORE_EFFICIENCY_V10_PILOT}
export STAGEC_V2_RESET_SCHEDULES=${STAGEC_V2_RESET_SCHEDULES:-1}

exec "$CODE_ROOT/scripts/rl/run_stagec_v2_cycle3_efficiency.sh" \
  "$CHAMPION" "$RESUME" "$ANCHOR_DIR" "$MINUTES" "$TEMPLATE" "$OUT"
