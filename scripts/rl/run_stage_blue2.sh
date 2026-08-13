#!/bin/bash
# Stage D "blue2" = the stage_d_v1 FERRY-FIRST relaunch (blue-first parity
# specialist).  Root-cause package for the two measured Stage-D failures
# (2026-07-24 telemetry, deterministic suffix episodes):
#
#   SYMPTOM 1 -- no blackout ferrying: ~0.2-0.6 legal ferry presses/ep inside
#   the two 25 s blackouts vs 250-380 MASKED presses while the hub was live;
#   own-court stockpile ~5 balls, own-court loop entries ~0.  The ferry loop
#   paid 1.0/ball against 10/ball for direct scoring, the own-court loop was
#   dead during blackouts (no pre-loading for the reactivation edge), and
#   return-when-live yanked COLLECT home at the 15-ball floor.
#
#   SYMPTOM 2 -- no scoring after t~80 s: s4 (105-130) 0.1-1.1 balls/ep and
#   endgame (130-160, BOTH hubs live) 0.2-0.9 balls/ep, with 30-56% of
#   deterministic episodes ending the match still parked in LEAVE.  Camping was
#   literally the cheapest action: LEAVE penalty capped at 5 (and switched OFF
#   by one ball in the magazine) while a botched crossing cost up to 110+20.
#
#   FIX PACKAGE (revision stage_d_v1, one-time migration from
#   score_efficiency_v11_rampfree; STAGE_D_RELAXED_KEYS + delay-key pops):
#     * ferry pays 5.0/entitled ball (blackout-only, custody-once) -- the
#       blackout collect+ferry loop now nets 6.5/ball immediately and keeps the
#       downstream +10, strictly dominating hold-and-wait;
#     * own-court loop armed during blackouts (--stage-d-owncourt-blackout-intake):
#       pre-load the ferried stockpile, hold at the dark hub (fire stays
#       masked, SCORE delay paused), dump on the reactivation edge;
#     * own-court intake pays 1.5/ball (parity with qualified collect);
#     * live windows return at 26 qualified (--stage-d-live-return-load) not 15,
#       and blackout collection runs to preferred 44 (chamber 60);
#     * LEAVE penalty 0.06/step cap 40 and magazine-independent (vec_env fix);
#       RETURN penalty 0.04/step cap 20; outer-rail cap 110->50, off-ramp
#       exit 20->12 -- an imperfect crossing is now always cheaper than a camp;
#     * return_time_guard 0.20->0.11 (~18 s remaining on 160 s);
#     * anchor beta 0.15 -> floor 0.0 over 30k updates: stop dragging the
#       policy toward the one-cycle 90 s Stage-C prior for the whole run.
#
# POSITIONAL ARGS:
#   $1 CHAMPION : frozen first-cycle prefix (/root/preserved/stageC_highest_1163753.pt)
#   $2 RESUME   : checkpoint to resume (stageC_d1i_generalist_final.pt)
#   $3 ANCHOR   : champion anchor directory (same as D0/D1)
#   $4 MINUTES  : wall-clock budget (default 720)
#   $5 TEMPLATE : env template (default env_template_200.usd)
set -eu
CHAMPION=${1:?prefix}; RESUME=${2:?resume ckpt}; ANCHOR=${3:?anchor dir}
MINUTES=${4:-720}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TEMPLATE=${5:-$CODE_ROOT/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}

# --- collectors / streams: INTEGRATION mix (2026-07-25 audit).  Yesterday ran
# 55% full / 45% micro-skill replay, which taught pieces of the bank behavior
# while eroding full-match movement and scoring (det mean regressed 85.5 -> 68).
# Integration runs must be dominated by the real task: 8 full collectors
# (16 envs, 75% replay), 2 bank (4 envs, 15%) to keep the blackout slice alive,
# 1 collect + 1 return (2 envs each, 5% each) to retain those skills. ---
export STAGEC_V2_NCOLL=${STAGEC_V2_NCOLL:-16}
export STAGEC_V2_STREAM_GROUPS=${STAGEC_V2_STREAM_GROUPS:-full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,postdump,postdump,postdump,postdump}
export STAGEC_V2_GROUP_WEIGHTS=${STAGEC_V2_GROUP_WEIGHTS:-full=.92,postdump=.08}
export STAGEC_V2_COLLECTOR_MODES=${STAGEC_V2_COLLECTOR_MODES:-'full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;postdump,postdump;postdump,postdump'}
export STAGEC_V2_COLLECTOR_ACTION_MODES=${STAGEC_V2_COLLECTOR_ACTION_MODES:-'mean;explore;mean;explore;mean;explore;mean;explore;mean;explore;mean;explore;mean;explore;mean;explore'}
export STAGEC_V2_COLLECTORS_PER_GPU=${STAGEC_V2_COLLECTORS_PER_GPU:-4}
# GUI inspection 2026-07-24: deterministic mid-field collection is the gating
# weakness (stall-exit at ~15 balls -> nothing to ferry -> no stockpile ->
# no nearby-shoot).  The dedicated `collect` skill lane (neutral spawn facing
# the cluster) drills exactly that, one det + one explore collector.
export STAGEC_V2_COLLECT_EPISODE_S=60

# --- bank lane slice knobs (v2: mid-blackout COLLECT warm start) ---
export STAGEC_V2_STAGE_D_BANK_CLOCK_A=${STAGEC_V2_STAGE_D_BANK_CLOCK_A:-34}
export STAGEC_V2_STAGE_D_BANK_CLOCK_B=${STAGEC_V2_STAGE_D_BANK_CLOCK_B:-84}
export STAGEC_V2_STAGE_D_BANK_SPAN=${STAGEC_V2_STAGE_D_BANK_SPAN:-52}
export STAGEC_V2_STAGE_D_BANK_STOCKPILE=${STAGEC_V2_STAGE_D_BANK_STOCKPILE:-12}
# Bank lane now SUCCEEDS on conversion (N entitled own-court balls scored),
# not on the span timer -- the fix for "ferries but never converts".
export STAGEC_V2_STAGE_D_BANK_SUCCESS_CONVERSIONS=${STAGEC_V2_STAGE_D_BANK_SUCCESS_CONVERSIONS:-6}
# --- SECTION TRAINING knobs (user plan 2026-07-26) ---------------------------
# Blackout lane: starts at the real shift edges (30 / 80) and ends when the
# blackout ends; messy edge/corner fuel; succeeds on coming home CHAMBERED.
export STAGEC_V2_STAGE_D_BANK_CLOCK_A=${STAGEC_V2_STAGE_D_BANK_CLOCK_A:-30}
export STAGEC_V2_STAGE_D_BANK_CLOCK_B=${STAGEC_V2_STAGE_D_BANK_CLOCK_B:-80}
export STAGEC_V2_STAGE_D_BANK_SUCCESS_CHAMBER=${STAGEC_V2_STAGE_D_BANK_SUCCESS_CHAMBER:-10}
export STAGEC_V2_STAGE_D_BANK_FUEL_SCATTER=${STAGEC_V2_STAGE_D_BANK_FUEL_SCATTER:-0.6}
export STAGEC_V2_STAGE_D_BANK_FUEL_CLUMPS=${STAGEC_V2_STAGE_D_BANK_FUEL_CLUMPS:-6}
export STAGEC_V2_STAGE_D_BANK_FUEL_JITTER_M=${STAGEC_V2_STAGE_D_BANK_FUEL_JITTER_M:-0.30}
# Opener lane: real match start, cut at 30 s, succeeds on 2 complete cycles.
export STAGEC_V2_STAGE_D_OPENER_SPAN_S=${STAGEC_V2_STAGE_D_OPENER_SPAN_S:-30}
export STAGEC_V2_STAGE_D_OPENER_SUCCESS_CYCLES=${STAGEC_V2_STAGE_D_OPENER_SUCCESS_CYCLES:-2}
# Live lane: the two 25 s reactivations plus the 30 s endgame, opening with a
# ~30 ball chamber and 60 on the own-court floor.
export STAGEC_V2_STAGE_D_LIVE_CLOCK_A=${STAGEC_V2_STAGE_D_LIVE_CLOCK_A:-55}
export STAGEC_V2_STAGE_D_LIVE_CLOCK_B=${STAGEC_V2_STAGE_D_LIVE_CLOCK_B:-105}
export STAGEC_V2_STAGE_D_LIVE_CLOCK_C=${STAGEC_V2_STAGE_D_LIVE_CLOCK_C:-130}
export STAGEC_V2_STAGE_D_LIVE_STOCKPILE=${STAGEC_V2_STAGE_D_LIVE_STOCKPILE:-60}
export STAGEC_V2_STAGE_D_LIVE_CHAMBER=${STAGEC_V2_STAGE_D_LIVE_CHAMBER:-30}
export STAGEC_V2_STAGE_D_LIVE_SUCCESS_CONVERSIONS=${STAGEC_V2_STAGE_D_LIVE_SUCCESS_CONVERSIONS:-6}
# Penalties you asked for: heavy charge past the red ramp, and standing still.
export STAGEC_V2_STAGE_D_DEEP_RED_PENALTY=${STAGEC_V2_STAGE_D_DEEP_RED_PENALTY:-0.6}
export STAGEC_V2_STAGE_D_IDLE_PENALTY=${STAGEC_V2_STAGE_D_IDLE_PENALTY:-0.25}
# Section 3 could not discover the dump press (160/1138 eps): pay for the press itself.
export STAGEC_V2_STAGE_D_LIVE_DUMP_REWARD=${STAGEC_V2_STAGE_D_LIVE_DUMP_REWARD:-15}
export STAGEC_V2_STAGE_D_LIVE_DUMP_MIN_LOAD=${STAGEC_V2_STAGE_D_LIVE_DUMP_MIN_LOAD:-8}
# Section 1 time pressure (user request): every opener step costs.
export STAGEC_V2_STAGE_D_OPENER_TIME_PENALTY=${STAGEC_V2_STAGE_D_OPENER_TIME_PENALTY:-0.08}
# Section 2 objective, in-phase: pay for arriving home loaded during a blackout.
export STAGEC_V2_STAGE_D_HOME_ARRIVAL_REWARD=${STAGEC_V2_STAGE_D_HOME_ARRIVAL_REWARD:-1.5}
export STAGEC_V2_STAGE_D_HOME_ARRIVAL_MIN_LOAD=${STAGEC_V2_STAGE_D_HOME_ARRIVAL_MIN_LOAD:-6}
# DOUBLE-DUMP variants: start at home just before each reactivation holding a
# chamber load with a bank on the floor -> hold, dump at the edge, intake the
# bank, dump again.  c = first reactivation (55 s), d = second (105 s) + endgame.
# These make success reachable in seconds so the lane can actually teach the
# conversion half, which the create-the-bank variants never reached (5%).
export STAGEC_V2_STAGE_D_BANK_CLOCK_C=${STAGEC_V2_STAGE_D_BANK_CLOCK_C:-50}
export STAGEC_V2_STAGE_D_BANK_CLOCK_D=${STAGEC_V2_STAGE_D_BANK_CLOCK_D:-100}
export STAGEC_V2_STAGE_D_BANK_PRELOAD=${STAGEC_V2_STAGE_D_BANK_PRELOAD:-8}
export STAGEC_V2_STAGE_D_BANK_DD_SPAN=${STAGEC_V2_STAGE_D_BANK_DD_SPAN:-34}

# --- episode: official 160 s match horizon ---
export STAGEC_V2_FULL_EPISODE_S=160
export STAGEC_V2_POSTDUMP_EPISODE_S=160

# --- stage D mechanics: blue-first parity, ferry-first economics ---
export STAGEC_V2_STAGE_D=1
export STAGEC_V2_STAGE_D_FIRST_INACTIVE=${STAGEC_V2_STAGE_D_FIRST_INACTIVE:-blue}
export STAGEC_V2_STAGE_D_SYNTHETIC_RED_AUTO_MAX=0
export STAGEC_V2_STAGE_D_FERRY=${STAGEC_V2_STAGE_D_FERRY:-1}
# 2026-07-25: scripted commands REVERTED (no action substitution), so ferrying
# must be learned again -- restore the 5.0/entitled-ball bootstrap that the
# resumed 145155 lineage was trained with.
export STAGEC_V2_STAGE_D_FERRY_REWARD=${STAGEC_V2_STAGE_D_FERRY_REWARD:-2.5}
export STAGEC_V2_STAGE_D_AUTO_FERRY_LOAD=0   # REVERTED: no scripted action substitution
export STAGEC_V2_STAGE_D_AUTO_FERRY_HOLD_S=${STAGEC_V2_STAGE_D_AUTO_FERRY_HOLD_S:-12}
export STAGEC_V2_STAGE_D_AUTO_OC_INTAKE=0   # REVERTED
export STAGEC_V2_STAGE_D_AUTO_SCORE_PRESS=0   # REVERTED
# wave-6: ferry volleys land on the ramp/hub approach corridor so every return
# drives through the bank with auto-intake on (was far-corner row 4.35).
export STAGEC_V2_STAGE_D_FERRY_TARGET_Y=0   # REVERTED: solver untouched
export STAGEC_V2_STAGE_D_FERRY_LANE_X=0   # REVERTED
export STAGEC_V2_STAGE_D_ACTIVE_FERRY_PENALTY=${STAGEC_V2_STAGE_D_ACTIVE_FERRY_PENALTY:-0.3}
export STAGEC_V2_STAGE_D_FERRY_DUMP_ON_PRESS=1
export STAGEC_V2_STAGE_D_FERRY_ENTITLED_ONLY=1
export STAGEC_V2_STAGE_D_FERRY_BLACKOUT_ONLY=1
export STAGEC_V2_STAGE_D_FERRY_MIN_LOAD=${STAGEC_V2_STAGE_D_FERRY_MIN_LOAD:-10}
export STAGEC_V2_STAGE_D_RETURN_WHEN_LIVE=1
export STAGEC_V2_STAGE_D_LIVE_RETURN_LOAD=${STAGEC_V2_STAGE_D_LIVE_RETURN_LOAD:-26}
export STAGEC_V2_STAGE_D_RETURN_LEAD_S=${STAGEC_V2_STAGE_D_RETURN_LEAD_S:-8}
export STAGEC_V2_STAGE_D_OWNCOURT_LOOP=${STAGEC_V2_STAGE_D_OWNCOURT_LOOP:-1}
export STAGEC_V2_STAGE_D_OWNCOURT_MIN_BALLS=${STAGEC_V2_STAGE_D_OWNCOURT_MIN_BALLS:-2}
export STAGEC_V2_STAGE_D_OWNCOURT_INTAKE_REWARD=${STAGEC_V2_STAGE_D_OWNCOURT_INTAKE_REWARD:-2.5}
export STAGEC_V2_STAGE_D_OWNCOURT_REARM=1
export STAGEC_V2_STAGE_D_OWNCOURT_BLACKOUT_INTAKE=1

# --- reward revision: stage_d_v1 one-time migration (fresh elite dir via new OUT) ---
export STAGEC_V2_REWARD_REVISION=stage_d_v1
# Plain SAME-revision resume of 145155 (its stored contract is matched exactly
# below), so no migration and NO optimizer reset -- optimizer resets are what
# produced the ~5-point dip after every cutover yesterday.
export STAGEC_V2_RESET_OPTIMIZER_STATE=0
export STAGEC_V2_ALLOW_STDDEV_SCHEDULE_MIGRATION=1
export STAGEC_V2_ALLOW_TARGET_LOAD_MIGRATION=0
export STAGEC_V2_ALLOW_SUFFIX_ALPHA_MIGRATION=0

# --- cycle economics ---
export STAGEC_V2_TARGET_LOAD=15            # hard floor -- DO NOT raise (deadlock trap)
export STAGEC_V2_PREFERRED_REPEAT_LOAD=${STAGEC_V2_PREFERRED_REPEAT_LOAD:-44}
# 45 = the value stored in the 145155 checkpoint we resume (verified by reading
# its metadata), so this is a plain resume.  The 90 experiment measurably did
# NOT deepen collection (p90 load stayed ~30) and cost a migration.
export STAGEC_V2_COLLECT_STALL_STEPS=${STAGEC_V2_COLLECT_STALL_STEPS:-45}
export STAGEC_V2_ALLOW_COLLECT_STALL_MIGRATION=0
export STAGEC_V2_RETURN_TIME_GUARD=${STAGEC_V2_RETURN_TIME_GUARD:-0.11}
export STAGEC_V2_INTAKE_DURING_RETURN=0
export STAGEC_V2_REPEAT_LOAD_RETURN_BONUS=8.0
export STAGEC_V2_REPEAT_LOAD_SCORE_BONUS=12.0
export STAGEC_V2_RESERVE_COUNT=18
export STAGEC_V2_RESERVE_BATCHES=1

# --- delay-penalty rebalance (leave/return keys relaxed across the migration) ---
export LEAVE_PENALTY_PER_STEP=${LEAVE_PENALTY_PER_STEP:-0.10}
export LEAVE_PENALTY_CAP=${LEAVE_PENALTY_CAP:-40.0}
export RETURN_PENALTY_PER_STEP=${RETURN_PENALTY_PER_STEP:-0.04}
export RETURN_PENALTY_CAP=${RETURN_PENALTY_CAP:-20.0}

# --- route shaping: soften the "never try" traps, keep the geometry ---
export STAGEC_V2_REQUIRE_RAMP_OUT=0
export STAGEC_V2_RAMP_OUT_HALF_WIDTH=0.90
export STAGEC_V2_RAMP_OUT_BONUS=24.0
export STAGEC_V2_OFF_RAMP_EXIT_PENALTY=${STAGEC_V2_OFF_RAMP_EXIT_PENALTY:-12.0}
export STAGEC_V2_OUTER_RAIL_ENTER_X=2.55
export STAGEC_V2_OUTER_RAIL_EXIT_X=2.20
export STAGEC_V2_OUTER_RAIL_MAX_X=3.60
export STAGEC_V2_OUTER_RAIL_GRACE_STEPS=10
export STAGEC_V2_OUTER_RAIL_PENALTY_PER_STEP=0.22
export STAGEC_V2_OUTER_RAIL_PENALTY_CAP=${STAGEC_V2_OUTER_RAIL_PENALTY_CAP:-50.0}
export STAGEC_V2_OUTER_RAIL_MIN_SCALE=0.35
export STAGEC_V2_OUTER_RAIL_ESCALATION_STEPS=120
export STAGEC_V2_OUTER_RAIL_MAX_MULTIPLIER=3.0
export STAGEC_V2_INTAKE_SUBSTEPS=2
export STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD=${STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD:-0}
export STAGEC_V2_POSTDUMP_COMPLETE_CYCLE=${STAGEC_V2_POSTDUMP_COMPLETE_CYCLE:-0}
export STAGEC_V2_POSTDUMP_DEPLETED_COUNT=0
export STAGEC_V2_POSTDUMP_DEPLETED_PROB=0.0

# --- learner core ---
# 2026-07-25: 1e-5 was aggressive for an integration run resuming a strong
# checkpoint (it let skill-lane gradients erode full-match behavior).  5e-6.
export STAGEC_V2_LEARNING_RATE=${STAGEC_V2_LEARNING_RATE:-0.000005}
export STAGEC_V2_GAMMA=0.9997
export STAGEC_V2_CRITIC_ONLY_UPDATES=${STAGEC_V2_CRITIC_ONLY_UPDATES:-8000}
export STAGEC_V2_ACTOR_UPDATE_INTERVAL=${STAGEC_V2_ACTOR_UPDATE_INTERVAL:-1}
export STAGEC_V2_ACTOR_Q_CENTER_FRACTION=${STAGEC_V2_ACTOR_Q_CENTER_FRACTION:-0.0}
export STAGEC_V2_ALLOW_ACTOR_Q_CENTER_MIGRATION=${STAGEC_V2_ALLOW_ACTOR_Q_CENTER_MIGRATION:-0}
export STAGEC_V2_ACTOR_PHASES=leave,collect,return,score
export STAGEC_V2_SUFFIX_ALPHA=1.0
export STAGEC_V2_INITIAL_STDDEV=${STAGEC_V2_INITIAL_STDDEV:-0.40}
export STAGEC_V2_STDDEV_START=1.0
export STAGEC_V2_STDDEV_END=${STAGEC_V2_STDDEV_END:-0.30}
export STAGEC_V2_STDDEV_STEPS=150000
# 2026-07-25: floor 0.0 removed ALL drift protection late in a run (my wave-1
# change).  Keep a small permanent anchor so the policy cannot wander off the
# champion prior while skill lanes push on it.
export STAGEC_V2_ANCHOR_BETA_START=${STAGEC_V2_ANCHOR_BETA_START:-0.15}
export STAGEC_V2_ANCHOR_BETA_FLOOR=${STAGEC_V2_ANCHOR_BETA_FLOOR:-0.06}
export STAGEC_V2_ANCHOR_DECAY_UPDATES=${STAGEC_V2_ANCHOR_DECAY_UPDATES:-80000}
export STAGEC_V2_ELITE_BEHAVIOR_WEIGHT=${STAGEC_V2_ELITE_BEHAVIOR_WEIGHT:-0.0}
export STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY=${STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY:-0}
export STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE=${STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE:-32}
export STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY:-2400}
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY=${STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY:-300}

# --- weight publishing ---
export STAGEC_V2_WEIGHT_PUBLISH_UPDATES=400
export STAGEC_V2_FREEZE_COLLECTOR_WEIGHTS=0
export STAGEC_V2_EVAL_SNAPSHOT_UPDATES=5000

# --- run identity ---
export STAGEC_V2_ROOT=/dev/shm/frc_stage_blue2_${RUN_ID}
export STAGEC_V2_STOP=/root/STOP_STAGE_BLUE2
export STAGEC_V2_RESET_SCHEDULES=${STAGEC_V2_RESET_SCHEDULES:-0}   # BUGFIX: hard 1 overrode resumes and reset stddev to 1.0
export STAGEC_V2_RUN_ID=$RUN_ID
OUT=/root/autodl-tmp/runs/stage_blue2_${RUN_ID}

exec "$CODE_ROOT/scripts/rl/run_stagec_v2_cycle3_efficiency.sh" \
  "$CHAMPION" "$RESUME" "$ANCHOR" "$MINUTES" "$TEMPLATE" "$OUT"
