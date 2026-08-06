#!/bin/bash
# Relaunch 4 HARDENED Stage-B runs, each under the crash-restart supervisor
# (run_supervised.sh) so PhysX blow-ups self-heal. Three build on the g0
# champion; one is a clean control from the Stage-A champion (uncontaminated by
# g0's buggy-era weights) to tell us whether g0 is salvageable.
#
# Usage:
#   relaunch_stageB_hardened.sh <CHAMP_CKPT> [A7_CKPT] [TEMPLATE] [MINUTES]
# CHAMP_CKPT: g0's best SCORING checkpoint - pick with eval_checkpoint.py first
#   (peak-scorer era was ~ckpt_000030000.pt).
set -u
CHAMP=${1:?champion checkpoint, e.g. /root/runs/drqv2_stageB_g0/ckpt_000030000.pt}
A7=${2:-/root/autodl-fs/stageA_champ_A7.pt}
TMPL=${3:-assets/rl/env_template_96.usd}
MIN=${4:-480}
BASE=/root/runs
SUP=scripts/rl/run_supervised.sh

cd /root/frc-rl && source /root/venv/bin/activate && source setup_render_env.sh
mkdir -p "$BASE"

# stationary reward (constant collect weight - NO anneal, which is only for
# teaching collection to a fresh agent and poisons a resumed replay), plus the
# no-progress watchdog.
COMMON=(--stage B --num-envs 4 --minutes "$MIN" --template "$TMPL"
        --collect-weight-start 0.3 --collect-weight-end 0.3 --watchdog-stall-s 180)

launch () {  # gpu name resume seed extra...
  local gpu=$1 name=$2 res=$3 seed=$4; shift 4
  nohup bash "$SUP" "$gpu" "$BASE/$name" "$res" 50 -- \
    "${COMMON[@]}" --seed "$seed" "$@" > "$BASE/$name.sup.log" 2>&1 < /dev/null &
  ln -sfn "$BASE/$name" /root/frc-rl/runs/"$name"
  echo "GPU$gpu -> $name (seed $seed, init $(basename "$res")) sup-pid $!"
  sleep 20
}

launch 0 sbh_g0cont0  "$CHAMP" 600                    # straight continuation
launch 1 sbh_g0cont1  "$CHAMP" 601                    # continuation, other seed
launch 2 sbh_g0explore "$CHAMP" 602 --explore-restart # champion + re-warmed exploration
launch 3 sbh_a7clean  "$A7"    603                    # clean control from Stage-A champ

# checkpoint backup loop (survives host migration)
pgrep -f "rsync -a /root/runs/" >/dev/null || \
  nohup bash -c 'while true; do rsync -a /root/runs/ /root/autodl-fs/runs/ 2>/dev/null; sleep 600; done' \
    > /root/runs/_backup.log 2>&1 < /dev/null &

echo "launched 4 supervised hardened runs; dashboard on :6006 (autodl 自定义服务)"
