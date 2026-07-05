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

## 1b. Compute-only containers (AutoDL etc.): make Vulkan rendering work

Many China GPU hosts (AutoDL) ship **compute-only** images: CUDA works, but
Isaac's camera rendering (Vulkan) fails with `ERROR_INCOMPATIBLE_DRIVER` /
`Failed to create any GPU devices`. The GPUs DO support graphics (check
`ls /dev/dri` — render nodes present); the image just stripped the userspace
graphics stack. Symptoms and the exact, verified fix (driver 580.105.08 open
module, 4×5090 AutoDL, 2026-07):

1. **Do NOT `apt install libnvidia-gl-580`** to get the Vulkan ICD — apt only
   has a *different* point release (e.g. 580.159.03) and it drags in
   `libnvidia-compute-580`, whose libs mismatch the kernel module and break
   CUDA too (`NVML_ERROR_LIB_RM_VERSION_MISMATCH`, `CUDA error 804`). If you
   already did this, move the mismatched libs aside and let ldconfig repoint to
   the host-mounted matched version:
   ```bash
   cd /usr/lib/x86_64-linux-gnu
   for f in *.580.159.03; do b=${f%.580.159.03}; [ -e "$b.580.105.08" ] && mv "$f" /root/nv159_backup/; done
   ldconfig   # symlinks now point at the kernel-matched 580.105.08
   ```
2. **The real missing piece is GLVND + GL/X utility libs** (vendor-neutral,
   safe, version-independent). This alone is what makes NVIDIA Vulkan init:
   ```bash
   apt-get install -y --no-install-recommends \
     libglvnd0 libegl1 libgl1 libopengl0 \
     libglu1-mesa libxt6 libsm6 libice6 libxrandr2 libxinerama1 \
     libxcursor1 libxi6 libxrender1 libxfixes3
   ```
3. **A correct Vulkan ICD** pointing at the kernel-matched driver:
   ```bash
   printf '{\n  "file_format_version":"1.0.0",\n  "ICD":{"library_path":"/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.105.08","api_version":"1.3.277"}\n}\n' > /root/nvidia_icd_105.json
   export VK_ICD_FILENAMES=/root/nvidia_icd_105.json
   ```
4. **Device nodes** — a multi-GPU slice often exposes only `/dev/nvidia3..6`
   while `/proc/driver/nvidia/gpus/` lists all host GPUs; create the missing
   minors so Vulkan enumeration doesn't abort (cgroup still blocks real access):
   ```bash
   for m in 0 1 2 7; do [ -e /dev/nvidia$m ] || mknod -m 666 /dev/nvidia$m c 195 $m; done
   ```
5. Verify: `vulkaninfo --summary` must list `NVIDIA GeForce RTX 5090`. Then the
   **first** Isaac render compiles shaders (cameras black, `std=0.0`); the
   **second** run reads the warm cache and delivers real frames (`std>1`).

The system Vulkan loader version is NOT the issue (a from-source 1.4.309 loader
failed identically until GLVND was installed). `/dev` nodes + env vars reset on
container restart — put steps 3-4 in a `setup_render_env.sh` sourced from
`~/.bashrc`; the apt packages and lib moves persist on disk.

## 2. Install

**Disk space first:** `isaacsim[all,extscache]` needs ~35-45 GB installed plus
transient pip space. Rented GPU boxes often have a small root disk and a big
data mount (`/workspace`, `/data`). Put the project, the venv, pip's
TMPDIR/cache, AND Isaac's runtime caches on the big disk:

```bash
df -h                                  # find the big mount; substitute below
cd /workspace && mkdir -p tmp pip-cache ov/cache ov/data
export TMPDIR=/workspace/tmp PIP_CACHE_DIR=/workspace/pip-cache
ln -s /workspace/ov/cache ~/.cache/ov
ln -s /workspace/ov/data  ~/.local/share/ov
```

If space is still tight (<60 GB free), install `isaacsim[all]` (without
`,extscache`, ~15-20 GB smaller); extensions then stream on first boot
(needs internet, first launch is slower).

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

## 4b. Distributed collection (Stage C+): N GPUs -> one policy

A single vision env is render-bound at ~0.2x real time, so one policy can't use
more than ~20% of a GPU. The distributed collector fixes that: **N collector
processes render on separate GPUs in parallel and feed ONE learner** through a
RAM-backed (`/dev/shm`) transport, giving ~Nx the transitions/sec to a single
policy. Files: `src/xrc_rebuilt/rl/distributed.py` (transport, unit-tested),
`scripts/rl/collector.py`, `scripts/rl/learner.py`, `scripts/rl/run_distributed.sh`.

Validate first with a short smoke (2 collectors + learner, ~4 min):
```bash
chmod +x scripts/rl/run_distributed.sh
scripts/rl/run_distributed.sh runs/drqv2_seedA6/best.pt 2 4
sleep 200
tail -5 /root/autodl-tmp/runs/drqv2_C_dist/metrics.jsonl        # transitions + updates climbing
grep -E "READY|initial weights|Traceback" /root/autodl-tmp/runs/drqv2_C_dist.learner.log \
     /root/autodl-tmp/runs/drqv2_C_dist.collector*.log
ls -la /dev/shm/xrc_dist/weights/                              # weights being republished
```
Healthy = learner "published initial weights", each collector "READY", and
`transitions`/`updates` rising in metrics.jsonl. If a collector shows a
Traceback, its GPU or the template path is the culprit; the others keep running.

Full run (3 collectors on GPU0-2 + learner on GPU3, from the Stage-B champion):
```bash
scripts/rl/run_distributed.sh runs/<stageB_champion>/best.pt 3 240
```
Tuning knobs live in the two scripts: `--updates-per-tx` (learner UTD),
`--preload-prob` / `--episode-len-s` (collector curriculum), `--chunk-steps`
(transport latency vs overhead). The dashboard finds it as `drqv2_C_dist`.

Note the transport dirs (`/dev/shm/xrc_dist`) are RAM — they vanish on reboot
and the run cleans them on launch. Stage-C specifics (200-ball template, hub
active/inactive timing) are env-config tweaks layered on once the machinery is
validated.

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
