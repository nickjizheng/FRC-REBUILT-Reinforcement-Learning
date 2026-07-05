# FRC REBUILT Reinforcement Learning

This repository reconstructs the xRC REBUILT FRC game in NVIDIA Isaac Sim and
provides a detailed competition robot for training an autonomous
match-playing reinforcement-learning policy.

The goal is a match-legal blue-alliance agent, trained **directly from the
robot's on-board cameras on the full physics engine** (no privileged-state
actor, no teacher-student distillation), that drives, collects FUEL, manages
the compact and extended configurations, and maximizes legal score under the
full REBUILT match rules (AUTO + SHIFT schedule with inactive-HUB windows).
Automatic flywheel control and aiming remain inside the robot controller; RL
learns strategic driving, collection, positioning, timing, and shoot or ferry
decisions. The method selection and its rationale are recorded turn-by-turn
in [`RL_BRAINSTORM.md`](RL_BRAINSTORM.md).

## Current status

- The field includes both HUBS, BUMPS, TRENCHES, towers, sources, walls,
  AprilTag plates, dynamic FUEL, HUB lighting, scoring eligibility, match
  timing, and randomized four-lane HUB return.
- The field starts with 456 total FUEL: 448 on the field and eight physical
  robot preloads.
- The supplied detailed CAD drives the robot visual. Moving intake, hopper,
  net, rollers, turret, swerve modules, collision proxies, and compact TRENCH
  mode are represented.
- The continuous soft bumper is blue and lies outside the rigid frame.
- The initial robot is compact beneath the blue TRENCH and faces the neutral
  zone. AUTO grants normal driver control but performs no scripted movement
  or shooting.
- The controller can target only the blue HUB. Internal requests for any other
  target are resolved to blue, and automatic target switching is disabled.
- Hold-to-shoot uses distance-based calibration, rear-turret geometry,
  moving-shot compensation, and one-to-three-ball feeder release.
- Viewport selection and its context menu are disabled during interactive
  simulation, while camera orbit, pan, and zoom remain available.
- Three chassis-mounted 640x360 cameras (intake, shooter, navigation) render
  the policy's viewpoint; the GUI shows them live without stealing keyboard
  control.
- A vectorized full-physics RL environment clones N fields on one GPU scene;
  throughput is FUEL-count-bound (curriculum stages clear the 8 policy-tx/s
  gate; full-456 runs ~4 tx/s and is reserved for evaluation).
- The current verification suite passes **145/145 tests**.

## Architecture

```text
xRC assets and rules
        |
        +--> Isaac field, PhysX FUEL, and HUB router
        |
        +--> detailed CAD and calibrated robot controller
                    |
                    +--> interactive GUI and manual controls
                    |
                    +--> vectorized full-physics RL environment
                              |
                              +--> DrQ-v2 pixel baseline
                              |    (asymmetric privileged critic)
                              |
                              +--> bake-off arms: reconstruction-free
                                   recurrent world model, TD-MPC2
```

The high-detail model is used for interactive inspection and final
evaluation. Training will use the same collision geometry and dynamics with a
lightweight visual configuration so many environments can run in parallel on
the GPU.

## Launch

Prerequisites on the development machine:

- Windows 11
- NVIDIA Isaac Sim 5.1
- Isaac Lab `main`
- Python 3.11 with a CUDA-enabled PyTorch installation

Double-click `LaunchSimulator.exe`, or run:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
& C:\il\venv\Scripts\python.exe .\run_sim.py --max-fuel 456
```

Controls:

- `W/S`: field-forward/back
- `A/D`: field-left/right
- left/right arrows: rotate
- `I`: intake
- `N`: compact/extend
- hold `SPACE`: aim and shoot into the blue HUB
- hold `F`: ferry toward the alliance zone
- `E`: emergency stop

The GUI also exposes the mechanisms and live aim solution. Mouse clicks in the
viewport cannot select or manipulate scene components.

## Verification

```powershell
& C:\il\venv\Scripts\python.exe -m pytest -q
& C:\il\venv\Scripts\python.exe tools\validate_competition_robot.py
& C:\il\venv\Scripts\python.exe tools\validate_robot_drive_intake.py
& C:\il\venv\Scripts\python.exe tools\validate_robot_trench_mode.py
& C:\il\venv\Scripts\python.exe tools\validate_hub.py
```

Validation summaries are stored in `runs/*.json`. Generated screenshots,
logs, USD exports, and learned checkpoints are intentionally ignored.

## RL plan

The governing document is [`RL_BRAINSTORM.md`](RL_BRAINSTORM.md) (converged
decisions + implementation log); engineering details live in
[`docs/VECTORIZATION_PLAN.md`](docs/VECTORIZATION_PLAN.md). The plan, in
sequence (the original PPO/teacher-student plan in
`docs/RL_TRAINING_PLAN.md` is retained for history but superseded):

1. vectorized full-physics environment with the real controller per env
   (done: `xrc_rebuilt.rl.vec_env`);
2. DrQ-v2 pixel baseline with an asymmetric privileged critic and per-env
   n-step replay (done: `xrc_rebuilt.rl.drqv2`, `scripts/rl/train_drqv2.py`);
3. curriculum stages A-D (short acquisition episodes up to full 160 s,
   456-FUEL matches under exact rules);
4. equal-budget bake-off vs a reconstruction-free recurrent world model and
   TD-MPC2, judged on deterministic held-out evaluations
   (`scripts/rl/eval_checkpoint.py`);
5. dynamics, camera, lighting, and latency randomization hardening.

The RL stack can be checked independently of the environment:

```powershell
& C:\il\venv\Scripts\python.exe .\scripts\rl\check_stack.py
```

The initial policy contract is implemented in `xrc_rebuilt.rl`: 60 Hz physics,
10 Hz policy decisions, 1600 decisions per match, and a seven-dimensional
blue-only action space. The next implementation milestone is the vectorized
`DirectRLEnv`.

### Live training dashboard

Start the localhost-only monitor in a separate terminal:

```powershell
& C:\il\venv\Scripts\python.exe .\scripts\rl\training_dashboard.py
```

It opens `http://127.0.0.1:8765` and automatically discovers DrQ-v2 run
directories. The page distinguishes a live process from an interrupted run,
tracks transitions, updates, ETA, reward components and learning curves, and
shows CPU, RAM, RTX utilization, VRAM, temperature, and power. New training
runs also write `run_config.json` plus timestamped throughput metrics so the
dashboard can calculate progress without inferring it from file timestamps.

## Imported assets and distribution

The repository contains extracted xRC interoperability assets and supplied
robot CAD. Their copyrights and trademarks remain with their respective
owners; do not redistribute those assets separately without confirming
permission from the original owners.
