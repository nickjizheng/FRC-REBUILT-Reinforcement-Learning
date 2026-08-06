#!/bin/bash
# Aggressive multi-cycle fine-tune (user directive 2026-07-13; +Term-D time-decay 2026-07-14)
# with an INTEGRATED collector watchdog. Warm-starts the weights from arg-1 <RESUME> (e.g. the
# high-ceiling ft_000165000), critic-only re-fit -> annealed BC-anchor unlock toward the
# IMMUTABLE champion anchor (arg-2); collectors add the DIRECT aggressive reward (full-clear
# bonus + gated abandon + escalating ordered-cycle bonus) PLUS Term D (idle-since-empty ramp,
# later-leg ramps, ramp-lane preference) to push the 2nd cycle, with custody rho=0.2, PBRS OFF. NCOLL collectors are packed round-robin over NGPU GPUs (>=1 per
# GPU); the learner takes GPU NGPU. The Isaac RTX renderer crashes silently after a while, so
# this relaunches any dead collector (tracked by launched-PID, NOT pgrep) until the learner
# exits or /root/STOP_AGGR appears.
#
# Usage: run_aggressive.sh <champion.pt> <anchor_dir> [ncoll] [minutes] [stddev_end] [ngpu] [template]
set -u
RESUME=${1:?usage: run_aggressive.sh <champion.pt> <anchor_dir> [ncoll] [minutes] [stddev_end] [ngpu] [template]}
ANCHOR_DIR=${2:?anchor dir (champion anchor npz dump)}
NCOLL=${3:-3}
MINUTES=${4:-600}
STDDEV_END=${5:-0.30}
NGPU=${6:-3}                 # collectors packed over GPU 0..NGPU-1 (round-robin); learner on GPU NGPU
TEMPLATE=${7:-/root/frc-rl/assets/rl/env_template_200.usd}
SKIP_LEARNER=${SKIP_LEARNER:-0}   # 1 = reuse the already-running learner (restart collectors only,
                                  # e.g. after a reward-param change) so its progress is preserved

ROOT=/dev/shm/frc_dist_ft
OUT=/root/autodl-tmp/runs/drqv2_aggressive
LEARNER_GPU=$NGPU
STOP=/root/STOP_AGGR
NENVS=2

cd /root/frc-rl && source /root/venv/bin/activate && source setup_render_env.sh >/dev/null 2>&1
mkdir -p "$ROOT" "$OUT"
rm -rf "$ROOT"/collector_* 2>/dev/null || true
[ "$SKIP_LEARNER" = "1" ] || rm -rf "$ROOT"/weights 2>/dev/null || true   # keep weights if reusing learner
rm -f "$STOP"

echo "$(date '+%F %T') === aggressive: $NCOLL collectors over GPU 0..$((NGPU-1)) + learner (GPU $LEARNER_GPU) ==="
echo "    champ=$RESUME  anchors=$ANCHOR_DIR  minutes=$MINUTES  stddev_end=$STDDEV_END  ngpu=$NGPU"

# learner (num_envs=2 per collector; 4 races the intake/shooter camera warm-up -> black)
if [ "$SKIP_LEARNER" = "1" ]; then
  LPID=$(pgrep -f learner_finetune.py | head -1)
  echo "$(date '+%F %T') SKIP_LEARNER=1 -> reusing running learner pid ${LPID:-NONE} (progress preserved)"
  [ -z "$LPID" ] && { echo "no running learner to reuse; aborting"; exit 1; }
else
  CUDA_VISIBLE_DEVICES=$LEARNER_GPU setsid python scripts/rl/learner_finetune.py \
    --root "$ROOT" --num-collectors "$NCOLL" --collector-envs "$NENVS" \
    --resume "$RESUME" --anchor-dir "$ANCHOR_DIR" --minutes "$MINUTES" \
    --batch-size 256 --gamma 0.999 --learning-rate 5e-5 \
    --critic-only-updates 3000 --explore-warm-steps 62500 \
    --anchor-beta-start 0.3 --anchor-beta-end-updates 23000 --anchor-batch 128 \
    --eval-snapshot-updates 5000 --out "$OUT" > "$OUT.learner.log" 2>&1 &
  LPID=$!
  echo "$(date '+%F %T') learner_ft -> GPU $LEARNER_GPU pid $LPID"
  sleep 20
fi

declare -A CPID
launch() {
  local c=$1
  local gpu=$(( c % NGPU ))
  sleep 3   # let a crashed instance's resources settle (32 GB VRAM has room to overlap)
  CUDA_VISIBLE_DEVICES=$gpu setsid python scripts/rl/collector.py \
    --collector-id "$c" --root "$ROOT" --num-envs "$NENVS" --stage C \
    --template "$TEMPLATE" --episode-len-s 120 --preload-prob 0.0 \
    --spawn-under-trench --mask-illegal-fire \
    --collect-weight 0.3 --rho-score 0.2 --rho-collect 0.2 --empty-own-court-penalty 0.0 \
    --aggressive-cycle --score-floor 25 --atonce-weight 1.0 --atonce-cap 50 \
    --atonce-min-load 4 --abandon-load 8 --abandon-weight 0.3 \
    --linger-penalty 0 --linger-grace 5 \
    --leave-bonus 5 --collect-bonus 5 --return-bonus 5 \
    --mc-per-ball 2.0 --mc-cap 40 --mc-escalation 1.2 --mc-episode-cap 150 \
    --neutral-deep-y 0.0 --mc-min-cycle-score 2 \
    --core-slope 0.010 --core-step-cap 0.50 --core-grace-steps 5 \
    --core-freeze-confirm 15 --core-freeze-cap 30 --arm-deadline-steps 300 \
    --rampB-slope 0.0125 --rampB-step-cap 0.25 --rampB-grace 30 --rampB-budget 10 \
    --rampC-slope 0.010 --rampC-step-cap 0.20 --rampC-grace 40 --rampC-budget 8 \
    --ramp-episode-cap 60 --ramp-pref 6.0 --ramp-center 1.55 --ramp-tol 0.9 \
    --neutral-refill-count 12 --neutral-refill-prob 0.6 \
    --neutral-loaded-prob 0.4 \
    --stddev-end "$STDDEV_END" --seed $((400 + c)) --minutes "$MINUTES" \
    >> "$OUT.collector$c.log" 2>&1 &
  CPID[$c]=$!
  echo "$(date '+%F %T') collector$c -> GPU$gpu pid ${CPID[$c]}"
}
# stagger starts so per-GPU camera warm-ups don't overlap (esp. multiple collectors/GPU)
for c in $(seq 0 $((NCOLL-1))); do launch "$c"; sleep 15; done
ln -sfn "$OUT" /root/frc-rl/runs/drqv2_aggressive
echo "$(date '+%F %T') all launched; watchdog active. watch: tail -f $OUT/metrics.jsonl (recent_cycles_mean, cycle2_rate)"

# integrated watchdog: relaunch dead collectors; exit + reap when the learner is gone
while true; do
  if [ -f "$STOP" ]; then
    echo "$(date '+%F %T') STOP file -> reap + exit"
    for c in "${!CPID[@]}"; do kill "${CPID[$c]}" 2>/dev/null; done; exit 0
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    echo "$(date '+%F %T') learner gone -> reap collectors + exit"
    for c in "${!CPID[@]}"; do kill "${CPID[$c]}" 2>/dev/null; done; exit 0
  fi
  for c in $(seq 0 $((NCOLL-1))); do
    if ! kill -0 "${CPID[$c]}" 2>/dev/null; then
      echo "$(date '+%F %T') collector$c (pid ${CPID[$c]}) DEAD -> relaunch"
      launch "$c"
    fi
  done
  sleep 30
done
