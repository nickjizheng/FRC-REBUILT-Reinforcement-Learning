#!/bin/bash
# Launch Gen-12 only after strict teacher curation has passed.
set -eu

CR=/root/autodl-tmp/frc_staged_v1_20260722
PREFIX=/root/preserved/stageC_highest_1163753.pt
RESUME=/root/preserved/auto_peaks/peak_119.2_2857060_2103.pt
ANCHOR=/root/autodl-tmp/archives/stagec_v23_ft165k_20260714_1932_final_20260714_214228/inputs/anchors
TEACHERS=${1:-/root/autodl-tmp/elite_anchor_g12}
REPORT=${2:-/root/autodl-tmp/elite_g12_harvest/teacher_filter.json}
MIN_TEACHERS=${MIN_TEACHERS:-5}
TEMPLATE=$CR/assets/rl/env_template_456.usd

count=$(/root/venv/bin/python - "$REPORT" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))["selected_count"]))
PY
)
[ "$count" -ge "$MIN_TEACHERS" ] || {
  echo "REFUSING_GEN12 selected_teachers=$count required=$MIN_TEACHERS"
  exit 3
}
[ "$(find "$TEACHERS" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq "$count" ] || {
  echo "REFUSING_GEN12 teacher directory/report mismatch"
  exit 3
}

if pgrep -f '[r]un_stagec_v2_cycle3_efficiency|[l]earner_cycle_v2|[c]ollector_cycle_v2.*frc_stage_blue2' >/dev/null; then
  echo "REFUSING_GEN12 another training stack is active"
  exit 4
fi

rm -f /root/G12_HARVEST_ACTIVE
mkdir -p /root/preserved/configs
[ ! -f /root/blue2_env.sh ] || cp -p /root/blue2_env.sh \
  "/root/preserved/configs/blue2_env_pre_g12_$(date +%Y%m%d_%H%M%S).sh"
pkill -f '[t]rain_watchdog_v2.sh' 2>/dev/null || true
rm -f /root/STOP_STAGE_BLUE2

RID=$(date +%Y%m%d_%H%M%S)
OUT=/root/autodl-tmp/runs/stage_blue2_g12_${RID}
cat > /root/blue2_env.sh <<EOF
export STAGEC_V2_RESET_SCHEDULES=0
export STAGEC_V2_CRITIC_ONLY_UPDATES=2000
export STAGEC_V2_STDDEV_END=0.30
export STAGEC_V2_INITIAL_STDDEV=0.32
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
BLUE2_ANCH=$ANCHOR
EOF

cd "$CR"
. /root/blue2_env.sh
export STAGEC_V2_RUN_ID=g12_${RID}
echo "$OUT" > /root/blue2_current_out.txt
echo "$OUT" > /root/stage_blue2.outdir
cp "$REPORT" "$OUT.teacher_filter.json"
nohup setsid scripts/rl/run_stage_blue2.sh "$PREFIX" "$RESUME" "$ANCHOR" 720 "$TEMPLATE" \
  </dev/null >> "/root/autodl-tmp/runs/sup_g12_${RID}.log" 2>&1 &
echo $! > /root/stage_d_g12_supervisor.pid

sleep 8
nohup setsid /root/train_watchdog_v2.sh </dev/null >> /root/train_watchdog.log 2>&1 &
echo $! > /root/stage_d_g12_watchdog.pid

nohup setsid /root/watch_stage_d_g12_gate.sh "$OUT" \
  </dev/null >> "$OUT.gate_watch.log" 2>&1 &
echo $! > /root/stage_d_g12_gate.pid
echo "GEN12_LAUNCHED out=$OUT teachers=$count supervisor=$(cat /root/stage_d_g12_supervisor.pid)"
