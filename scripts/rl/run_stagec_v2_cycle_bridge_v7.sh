#!/bin/bash
# Source-aware elite-custody continuation of the V6 cycle bridge.
#
# The environment, reward revision, 45-ball target, route constraints, and
# collector mix remain identical to V6.  V7 names a fresh run/replay boundary
# for the learner-side elite fix: retain integrated POSTDUMP/FULL successes in
# their own pools instead of allowing the solved RETURN drill to evict them.
set -eu

CHAMPION=${1:?immutable first-cycle prefix checkpoint}
RESUME=${2:?preserved Stage-C V6 bridge checkpoint}
ANCHOR_DIR=${3:?champion anchor directory}
MINUTES=${4:-600}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TEMPLATE=${5:-$CODE_ROOT/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${6:-/root/autodl-tmp/runs/stagec_v2_cycle_bridge_v7_${RUN_ID}}

export STAGEC_V2_NCOLL=12
export STAGEC_V2_STREAM_GROUPS=full,full,full,full,full,full,full,full,postdump,postdump,postdump,postdump,postdump,postdump,postdump,postdump,postdump,postdump,postdump,postdump,collect,collect,return,return
export STAGEC_V2_GROUP_WEIGHTS=full=.45,postdump=.40,collect=.05,return=.10
export STAGEC_V2_COLLECTOR_MODES='full,full;full,full;full,full;full,full;postdump,postdump;postdump,postdump;postdump,postdump;postdump,postdump;postdump,postdump;postdump,postdump;collect,collect;return,return'
export STAGEC_V2_COLLECTOR_ACTION_MODES='mean;mean;explore;explore;mean;mean;explore;explore;explore;explore;explore;explore'
export STAGEC_V2_COLLECTORS_PER_GPU=4

export STAGEC_V2_FULL_EPISODE_S=90
export STAGEC_V2_POSTDUMP_EPISODE_S=75
export STAGEC_V2_COLLECT_EPISODE_S=45
export STAGEC_V2_RETURN_EPISODE_S=45
export STAGEC_V2_TARGET_LOAD=45
export STAGEC_V2_RESERVE_COUNT=18
export STAGEC_V2_RESERVE_BATCHES=1
export STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD=1
export STAGEC_V2_POSTDUMP_COMPLETE_CYCLE=1
export STAGEC_V2_POSTDUMP_DEPLETED_COUNT=60
export STAGEC_V2_POSTDUMP_DEPLETED_PROB=0.25
export STAGEC_V2_REWARD_REVISION=cycle_bridge_v6
export STAGEC_V2_ALLOW_TARGET_LOAD_MIGRATION=0

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

export STAGEC_V2_ROOT=${STAGEC_V2_ROOT:-/dev/shm/frc_stagec_v2_cycle_bridge_v7_${RUN_ID}}
export STAGEC_V2_STOP=${STAGEC_V2_STOP:-/root/STOP_STAGEC_V2_CYCLE_BRIDGE_V7}
export STAGEC_V2_RESET_SCHEDULES=${STAGEC_V2_RESET_SCHEDULES:-1}

exec "$CODE_ROOT/scripts/rl/run_stagec_v2_cycle3_efficiency.sh" \
  "$CHAMPION" "$RESUME" "$ANCHOR_DIR" "$MINUTES" "$TEMPLATE" "$OUT"
