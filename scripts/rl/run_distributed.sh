#!/bin/bash
# Distributed DrQ-v2: one learner (GPU N) + N collectors (GPU 0..N-1) feeding ONE
# policy through a tmpfs transport. Collectors render on separate GPUs in
# parallel; the learner drains all of them into one replay and republishes
# weights. ~N x the transitions/sec of a single run -> ~N x faster convergence.
#
# Usage:
#   scripts/rl/run_distributed.sh <resume_best.pt> [num_collectors] [minutes] [template] [stage]
#
# Defaults: 3 collectors (GPU0,1,2) + learner (GPU3), 240 min, 96-ball template, stage C.
set -e

RESUME=${1:?usage: run_distributed.sh <resume_best.pt> [num_collectors] [minutes] [template] [stage]}
NCOLL=${2:-3}
MINUTES=${3:-240}
TEMPLATE=${4:-/root/frc-rl/assets/rl/env_template_200.usd}
STAGE=${5:-C}
EPLEN=${6:-90}
# stage-C flags (trench start + fire-mask); pass "" to disable for A/B scale-up
SCFLAGS=${7:---spawn-under-trench --mask-illegal-fire}
# Stage-C collect weight: constant 0.3 to match the resumed champion (the 1.5
# default is a fresh-agent anneal that poisons a resumed replay; audit).
COLLECT_WEIGHT=${8:-0.3}
STDDEV_END=${9:-0.2}   # Stage C exploration / target-smoothing floor (contract)

ROOT=/dev/shm/frc_dist
OUT=/root/autodl-tmp/runs/drqv2_${STAGE}_dist
LEARNER_GPU=$NCOLL   # e.g. 3 collectors -> learner on GPU 3

cd /root/frc-rl && source /root/venv/bin/activate && source setup_render_env.sh
mkdir -p "$ROOT" "$OUT"
# clean any stale transport state from a previous run
rm -rf "$ROOT"/collector_* "$ROOT"/weights 2>/dev/null || true

echo "=== distributed: $NCOLL collectors (GPU 0..$((NCOLL-1))) + learner (GPU $LEARNER_GPU) ==="
echo "    resume=$RESUME  stage=$STAGE  template=$(basename "$TEMPLATE")  minutes=$MINUTES  collect_weight=$COLLECT_WEIGHT"

# 1) learner first — publishes initial weights so collectors can start
CUDA_VISIBLE_DEVICES=$LEARNER_GPU nohup python scripts/rl/learner.py \
  --root "$ROOT" --num-collectors "$NCOLL" --collector-envs 4 \
  --resume "$RESUME" --minutes "$MINUTES" --batch-size 256 --updates-per-tx 1.0 \
  --replay-capacity 400000 --gamma 0.999 --collect-weight "$COLLECT_WEIGHT" \
  --stddev-end "$STDDEV_END" --explore-restart \
  --out "$OUT" > "$OUT.learner.log" 2>&1 &
echo "learner -> GPU $LEARNER_GPU (pid $!)"
sleep 15

# 2) collectors — one per GPU
for c in $(seq 0 $((NCOLL-1))); do
  CUDA_VISIBLE_DEVICES=$c nohup python scripts/rl/collector.py \
    --collector-id "$c" --root "$ROOT" --num-envs 4 --stage "$STAGE" \
    --template "$TEMPLATE" --episode-len-s "$EPLEN" --preload-prob 0.0 $SCFLAGS \
    --collect-weight "$COLLECT_WEIGHT" --stddev-end "$STDDEV_END" \
    --seed $((400 + c)) --minutes "$MINUTES" > "$OUT.collector$c.log" 2>&1 &
  echo "collector $c -> GPU $c (pid $!)"
  sleep 20
done

# discovery symlink so the dashboard finds the run
ln -sfn "$OUT" /root/frc-rl/runs/drqv2_${STAGE}_dist
echo ""
echo "launched. watch:  tail -f $OUT/metrics.jsonl"
echo "learner log:      tail -f $OUT.learner.log"
echo "collector0 log:   tail -f $OUT.collector0.log"
