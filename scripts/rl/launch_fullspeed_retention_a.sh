#!/bin/bash
# Arm A: repair run-relative schedules while retaining the checkpoint's 0.30
# exploration contract.  A later Arm B may migrate 0.30 -> 0.21 after this
# establishes whether the missing warm-up/anchor reset caused the decay.
set -eu

RESUME=${1:?banked full-speed checkpoint}
TEACHERS=${2:?exact-source teacher directory}
REPORT=${3:?teacher report}
RID=${4:-fsretA_$(date +%Y%m%d_%H%M%S)}

CODE_ROOT=/root/autodl-tmp/frc_staged_v1_20260722
PREFIX=/root/preserved/stageC_highest_1163753.pt
ANCHORS=/root/autodl-tmp/archives/stagec_v23_ft165k_20260714_1932_final_20260714_214228/inputs/anchors
TEMPLATE=$CODE_ROOT/assets/rl/env_template_456.usd
OUT=/root/autodl-tmp/runs/stage_blue2_${RID}

test -s "$RESUME"
test -d "$TEACHERS"
test "$(find "$TEACHERS" -maxdepth 1 -type f -name '*.npz' | wc -l)" -ge 5
test -s "$REPORT"
test "$(tr -d '[:space:]' < /root/policy_speed_scale.txt)" = "1.0"
if pgrep -f '[r]un_stagec_v2_cycle3_efficiency|[l]earner_cycle_v2|[c]ollector_cycle_v2.*frc_stage_blue2' >/dev/null; then
  echo "refusing launch: another Stage-D training stack is active" >&2
  exit 4
fi

mkdir -p /root/preserved/configs
if test -f /root/blue2_env.sh; then
  cp -p /root/blue2_env.sh "/root/preserved/configs/blue2_env_pre_${RID}.sh"
fi
rm -f /root/STOP_STAGE_BLUE2

cat > /root/blue2_env.sh <<EOF
export FRC_POLICY_SPEED_FILE=/root/policy_speed_scale.txt
export STAGEC_V2_RESET_SCHEDULES=1
export STAGEC_V2_RESET_OPTIMIZER_STATE=0
export STAGEC_V2_CRITIC_ONLY_UPDATES=8000
export STAGEC_V2_STDDEV_START=1.0
export STAGEC_V2_STDDEV_END=0.30
export STAGEC_V2_INITIAL_STDDEV=0.30
export STAGEC_V2_ANCHOR_BETA_START=0.15
export STAGEC_V2_ANCHOR_BETA_FLOOR=0.06
export STAGEC_V2_ANCHOR_DECAY_UPDATES=80000
export STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD=1
export STAGEC_V2_STAGE_D_POSTDUMP_CLOCK_A=110
export STAGEC_V2_STAGE_D_POSTDUMP_CLOCK_B=133
export STAGEC_V2_SEEDMINE_ELITE_DIR=$TEACHERS
export STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT=$RESUME
export STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY=1
export STAGEC_V2_ELITE_BEHAVIOR_WEIGHT=0.2
export STAGEC_V2_STAGE_D_LIVE_DUMP_REWARD=0
export STAGEC_V2_STAGE_D_HOME_ARRIVAL_REWARD=0
export STAGEC_V2_STAGE_D_OPENER_TIME_PENALTY=0
export STAGEC_V2_STAGE_D_DEEP_RED_PENALTY=0
export STAGEC_V2_STAGE_D_IDLE_PENALTY=0
export STAGEC_V2_STAGE_D_PREFIX_RESCUE_S=0
BLUE2_TEMPLATE=$TEMPLATE
BLUE2_CHAMP=$PREFIX
BLUE2_ANCH=$ANCHORS
EOF

cp -p /root/blue2_env.sh "/root/preserved/configs/blue2_env_${RID}.sh"
cp -p "$REPORT" "$OUT.teacher_report.json"
echo "$OUT" > /root/blue2_current_out.txt
echo "$OUT" > /root/stage_blue2.outdir
echo "$OUT" > /root/stagec_v2_cycle3.outdir

cd "$CODE_ROOT"
. /root/blue2_env.sh
export STAGEC_V2_RUN_ID="$RID"
nohup setsid scripts/rl/run_stage_blue2.sh \
  "$PREFIX" "$RESUME" "$ANCHORS" 720 "$TEMPLATE" \
  </dev/null >> "/root/autodl-tmp/runs/sup_${RID}.log" 2>&1 &
echo $! > /root/stage_d_retention_supervisor.pid
echo "RETENTION_A_LAUNCHED out=$OUT supervisor=$(cat /root/stage_d_retention_supervisor.pid)"
