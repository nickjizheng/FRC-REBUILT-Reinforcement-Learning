# Server training guide (Linux)

Everything needed to run Stage-A/B DrQ-v2 training away from the dev laptop.
The code is path-relative; no Windows paths are required anywhere.

## 1. Hardware guidance

**Isaac Sim's renderer requires RT cores. A100/H100 will NOT render the
cameras — do not rent them for this workload.** Suitable GPUs:

| GPU | VRAM | Verdict |
|---|---|---|
| RTX 5090 | 32 GB | Best value. One training run uses <8 GB today. |
| RTX 6000 Ada / RTX PRO 6000 Blackwell | 48-96 GB | Headroom for more envs/cameras later, ECC, server cooling. |
| L40S | 48 GB | The data-center RTX option if renting a proper server. |

Two facts that shape the buy:

1. **One run = one process = one GPU.** The stack does not split a single run
   across GPUs. A bigger GPU does not make physics much faster either —
   throughput is FUEL-count-bound (measured: 456 FUEL ≈ 4 policy-tx/s, 32
   FUEL ≈ 40-50 tx/s regardless of GPU headroom).
2. **Multiple GPUs = parallel experiments**, which is exactly what the plan
   needs (3 seeds per promoted stage, later the 3-method bake-off). 2-4x
   RTX 5090 gives the most experiments per dollar; choose the workstation
   cards only for chassis/ECC reasons.

System per concurrent run: >=6 CPU cores, >=32 GB RAM (recommend 64+ GB
total for parallel runs; replay is ~8 GB/run), NVMe >=1 TB, Ubuntu 22.04/24.04,
NVIDIA driver >= 570.

## 2. Install

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -U pip

# PyTorch (CUDA 12.8 wheels - matches the pinned 2.7.0+cu128)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim 5.1 via pip (NVIDIA index). If this exact command drifts, follow
# NVIDIA's current "Install Isaac Sim using pip" page for 5.1.
pip install "isaacsim[all,extscache]==5.1.0.0" --extra-index-url https://pypi.nvidia.com

pip install -r requirements-server.txt
export OMNI_KIT_ACCEPT_EULA=YES   # put it in ~/.bashrc for cron/tmux runs
```

Unzip this bundle anywhere; run everything from the repo root.

## 3. One-time preparation + sanity

```bash
# regenerate the (gitignored) cloneable USD templates
python scripts/rl/export_env_template.py --max-fuel 32 --out assets/rl/env_template_32.usd
python scripts/rl/export_env_template.py --max-fuel 96 --out assets/rl/env_template_96.usd

pytest -q                          # 148 tests, CPU-only, fast
python scripts/rl/check_stack.py   # boots Isaac headless, prints versions

# measure THIS machine (the numbers below are from the RTX 5080 laptop)
python scripts/rl/vec_throughput.py --num-envs 4 --template assets/rl/env_template_32.usd
python scripts/rl/vec_env_smoke.py --num-envs 2
```

Expect `vec_env_smoke` >= ~14 policy-tx/s at N=2 with cameras; scale
`--num-envs` up until aggregate tx/s stops improving (physics saturates on
total FUEL count, so gains flatten quickly).

## 4. Train

```bash
# Stage A - collection (fresh):
python scripts/rl/train_drqv2.py --num-envs 2 --minutes 240 \
    --out runs/drqv2_stageA

# Stage B - acquire-and-score (36 s episodes, half the episodes start
# preloaded at a shooting pose, collection reward annealed 1.5 -> 0.3):
python scripts/rl/train_drqv2.py --stage B --num-envs 2 --minutes 120 \
    --template assets/rl/env_template_96.usd \
    --resume runs/drqv2_stageA/latest.pt \
    --out runs/drqv2_stageB
```

Checkpoint hygiene (automatic): `latest.pt` every minute and `best.pt` (by
recent return) are only written when all weights are finite; numbered
`ckpt_<transitions>.pt` every 10k transitions (`--checkpoint-every-tx`);
optimizer state is saved, so `--resume` continues training exactly.
`rejected_transitions` and `skipped` in the metrics line count non-finite
inputs/updates - they should stay ~0.

Monitor: `python scripts/rl/training_dashboard.py` then open
`http://127.0.0.1:8765` (over SSH: `ssh -L 8765:127.0.0.1:8765 user@server`).

Parallel runs on a multi-GPU box (one GPU per run):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rl/train_drqv2.py ... --out runs/seed0 &
CUDA_VISIBLE_DEVICES=1 python scripts/rl/train_drqv2.py ... --out runs/seed1 &
```

## 5. Evaluate + Stage-B promotion gate

```bash
python scripts/rl/eval_checkpoint.py \
    --checkpoint runs/drqv2_stageB/best.pt \
    --episodes 12 --episode-len-s 36 \
    --template assets/rl/env_template_96.usd \
    --out runs/eval_stageB.json
```

Promote Stage B only when, on the deterministic eval:

- no NaNs anywhere (`rejected_transitions`/`skipped` ~0 during training);
- `mean_collected >= 10`;
- `mean_scored >= 3` per episode;
- `pct_episodes_scored >= 70`;
- clearly beats both the `random` baseline and the Stage-A checkpoint.

## 6. Files in this bundle

- `src/`, `scripts/`, `tests/`, `tools/`, `docs/` - code, tooling, plans
- `assets/fresh_xrc/` - extracted field data (required at runtime)
- `assets/robot_runtime/` - robot mesh/mechanism data (required at runtime)
- `run_sim.py` - interactive GUI entry (works on the server too, with a display)
- `RL_BRAINSTORM.md` - the governing decisions record

Not included (regenerated or local-only): `assets/rl/*.usd` templates,
`runs/`, checkpoints, the compiled Windows launcher.
