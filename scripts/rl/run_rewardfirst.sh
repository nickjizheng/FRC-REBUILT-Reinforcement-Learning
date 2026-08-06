#!/bin/bash
# Reward-first warm-start fine-tune: custody-weighted reward
# in the collectors + learner_finetune (critic-only re-fit -> encoder+actor unlock with an
# annealed champion BC anchor). 3 collectors (GPU 0..2) + learner (GPU 3), one policy.
#
# Usage: run_rewardfirst.sh <champion.pt> <anchor_dir> [num_collectors] [minutes] [pbrs_weight] [template]
# Second-cycle phase-PBRS reward (design note): base = raw score; collectors add F=g*Phi'-Phi
# over the leave->collect->return->unload cycle. No custody (rho=1.0).
set -e
RESUME=${1:?usage: run_rewardfirst.sh <champion.pt> <anchor_dir> [ncoll] [minutes] [pbrs_weight] [template]}
ANCHOR_DIR=${2:?anchor dir (champion anchor npz dump; holdout auto-excluded)}
NCOLL=${3:-3}
MINUTES=${4:-600}
PBRS_WEIGHT=${5:-1.0}
TEMPLATE=${6:-/root/frc-rl/assets/rl/env_template_200.usd}

ROOT=/dev/shm/frc_dist_ft
OUT=/root/autodl-tmp/runs/drqv2_rewardfirst
LEARNER_GPU=$NCOLL

cd /root/frc-rl && source /root/venv/bin/activate && source setup_render_env.sh
mkdir -p "$ROOT" "$OUT"
rm -rf "$ROOT"/collector_* "$ROOT"/weights 2>/dev/null || true

echo "=== reward-first: $NCOLL collectors (GPU 0..$((NCOLL-1))) + learner_ft (GPU $LEARNER_GPU) pbrs_weight=$PBRS_WEIGHT ==="
echo "    champion=$RESUME  anchors=$ANCHOR_DIR  minutes=$MINUTES"

# num_envs=2 per collector: 4 renders the intake/shooter cameras with a warm-up RACE
# (envs go black -> the collector guard refuses to publish), so keep it at 2 (reliable).
NENVS=2
# 1) learner first — publishes initial (champion) weights so collectors can start
CUDA_VISIBLE_DEVICES=$LEARNER_GPU nohup python scripts/rl/learner_finetune.py \
  --root "$ROOT" --num-collectors "$NCOLL" --collector-envs "$NENVS" \
  --resume "$RESUME" --anchor-dir "$ANCHOR_DIR" --minutes "$MINUTES" \
  --batch-size 256 --gamma 0.999 --learning-rate 5e-5 \
  --critic-only-updates 3000 --explore-warm-steps 62500 \
  --anchor-beta-start 0.3 --anchor-beta-end-updates 23000 --anchor-batch 128 \
  --eval-snapshot-updates 5000 --out "$OUT" > "$OUT.learner.log" 2>&1 &
echo "learner_ft -> GPU $LEARNER_GPU (pid $!)"
sleep 20

# 2) collectors — one per GPU, second-cycle phase-PBRS shaping applied in the env
for c in $(seq 0 $((NCOLL-1))); do
  CUDA_VISIBLE_DEVICES=$c nohup python scripts/rl/collector.py \
    --collector-id "$c" --root "$ROOT" --num-envs "$NENVS" --stage C \
    --template "$TEMPLATE" --episode-len-s 90 --preload-prob 0.0 \
    --spawn-under-trench --mask-illegal-fire \
    --collect-weight 0.3 --stddev-end 0.2 \
    --phase-pbrs --pbrs-weight "$PBRS_WEIGHT" --pbrs-gamma 0.999 \
    --seed $((400 + c)) --minutes "$MINUTES" > "$OUT.collector$c.log" 2>&1 &
  echo "collector $c -> GPU $c (pid $!)"
  sleep 20
done

ln -sfn "$OUT" /root/frc-rl/runs/drqv2_rewardfirst
echo ""
echo "launched. watch:  tail -f $OUT/metrics.jsonl   (TRAIN_FT lines: phase, beta, recycled_share)"
