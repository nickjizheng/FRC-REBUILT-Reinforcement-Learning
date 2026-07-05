# Vectorization architecture & milestones

Last updated: 2026-07-04. Companion to `RL_BRAINSTORM.md` (Converged decisions).
This is the engineering plan for the big milestone: turning the single-env
procedural sim into a batched Isaac Lab env that clears the **≥8 policy
transitions/s** gate on the 16 GB laptop.

## The core problem

- Our sim is **procedurally built** by `SceneBuilder` (isaac_scene.py) at absolute
  `/World/...` paths: field visuals, 456 FUEL rigid bodies, the articulated robot
  (`CompetitionRobotArticulationBuilder`), and the chassis camera rig.
- Isaac Lab's `DirectRLEnv` clones an **env template** `/World/envs/env_.*` and
  spawns assets from **USD/cfg** (`ArticulationCfg`, `RigidObjectCfg`,
  `TiledCameraCfg`), then batches physics + tiled rendering natively.
- Bridging: export the procedural build to **reusable USD assets**, then drive
  cloning + batching through Isaac Lab cfgs.

## Key decisions

1. **Per-env Python controller loop, not full tensorization (first).** Amdahl:
   ~85% of step time is PhysX and the Python controller is ~2%/env, so at the
   2–4 envs that fit in 16 GB, looping `CompetitionRobotController` +
   `HubRouter` per env costs ~4–8% — acceptable. Tensorize only if the
   benchmark says the controller became the bottleneck. This avoids a risky
   multi-week rewrite of well-tested logic.
2. **USD-asset export path** (reuse `SceneBuilder`) over rewriting it as cfg.
   Export the robot articulation + field + a FUEL prototype to USD once; Isaac
   Lab references them per env.
3. **Shared, not cloned:** one `PhysicsScene` and lighting per stage. Field +
   robot + FUEL are per-env at grid offsets (`env_spacing` large enough that
   independent matches never touch — field is ~16×8 m, so spacing ≥ 20 m).
4. **Two visual profiles (RL plan §4.1).** Training clones **simplified** robot
   visuals (the 1.29 M-tri CAD must not render ×N — and the robot's own
   outward-facing cameras barely see it); FIELD/HUB/FUEL visuals are kept
   because the cameras *do* see those. Full CAD is eval/inspection only. Watch
   the train→eval visual gap; close it with domain randomization.
5. **Cameras:** reuse ChatGPT's `CAMERA_RIG` (**3-camera baseline**:
   intake+shooter+navigation, owner-validated as drivable; 640×360;
   `CAMERA_TWO_VIEW_NAMES` = the 2-cam render-cost ablation)
   via `TiledCamera` so all envs render in one pass. Extrinsics come straight
   from `CAMERA_RIG` local offsets.
6. **Replay:** hybrid D:-NVMe, episode-contiguous, image-compressed chunks;
   codec chosen by ball/tag detector recall (Converged #11).

## Milestones (each ends in a runnable check)

1. **USD asset export** — build the scene via `SceneBuilder`, export a
   cloneable template; verify it reloads. *(started)*
2. **Vectorized scene** — `DirectRLEnv._setup_scene` clones N envs (robot +
   FUEL + field), batched PhysX, step with zero actions at N=1/2/4.
3. **Cameras per env** — the 3-camera rig, rendered once per policy step
   (implemented via per-env `Camera` sensors; `TiledCamera` remains a
   later optimization).
4. **Throughput gate** — aggregate tx/s + VRAM at N=1/2/4 with a learner stub.
   **Decision:** does batching clear ≥8 tx/s? If not: cut render cost
   (resolution/visuals) or env count before tensorizing.
5. **Controller/router per-env loop** — drive, intake, mechanisms, hub scoring
   under match rules, inside the batched step.
6. **RL interface** — batched obs (tiled pixels + proprioception, Turn-4
   schema), 7-D action decode (`rl.spec`), reward = raw legal score under full
   match rules, episodic reset, curriculum hooks. Register
   `XRC-CompetitionRobot-Direct-v0`.
7. **DrQ-v2 baseline** — off-policy + compressed NVMe replay + asymmetric
   critic; first Stage-A run. Then the equal-budget bake-off.

## Measured results (2026-07-05, RTX 5080 Laptop, physics-only, GridCloner)

| N envs | FUEL/env | aggregate env-steps/s | policy tx/s (repeat 6) | gate >=8 | VRAM |
|---|---|---|---|---|---|
| 1 | 456 | 27.3 | 4.6 | no | 4.9 GB |
| 2 | 456 | 23.6 | 3.9 | no | - |
| 4 | 456 | 23.7 | 4.0 | no | - |
| 1 | 32 | 243.1 | 40.5 | yes | 2.4 GB |
| 4 | 32 | 295.4 | 49.2 | yes | 2.5 GB |
| 8 | 32 | 281.5 | 46.9 | yes | 2.7 GB |

**Conclusion: throughput is FUEL-count-bound, not env-count-bound.** The GPU
saturates on ball contacts; cloning neither helps nor hurts aggregate much.
Curriculum stages A-C (8-96 FUEL) run 40-50 policy-tx/s aggregate - the gate
is cleared where the bulk of learning happens. Full-456 (~4 tx/s) is reserved
for Stage-D fine-tuning and evaluation, as the curriculum already assumed.
Cameras render once per policy step (10 Hz), not per physics step.

## Status (2026-07-05)

- Milestones 1-3 DONE: template export (+strip singletons), GridCloner scene,
  per-env adapters (`rl/vec_env.py`: `EnvArticulationAdapter`, `EnvFuelAdapter`
  with env-origin shifting), controller+router per env, 3-camera rig per env
  rendered once per policy step.
- Controller gained `usd_root_path` so mechanism visuals / net-collision swap
  rebase onto each clone (`competition_robot.py`).
- **Smoke (N=2, 32-FUEL template, 3 cameras): 9.31 policy-tx/s aggregate —
  gate cleared with the full camera rig.** Obs: rgb (N,3,360,640,3), proprio
  (N,22), privileged (N,26). Per-env rewards independent and nonzero under a
  random policy.
- DrQ-v2 baseline implemented (`rl/drqv2.py` asymmetric critic, `rl/replay.py`
  n-step ring, `scripts/rl/train_drqv2.py`); Stage-A smoke training launched.
- **Review fixes (2026-07-05, external review):** replay now uses per-env
  rings (`PerEnvReplay`) so n-step chains never cross env streams; auto-reset
  returns the NEW episode's first (rendered) observation; the controller
  gained `reset_match_state()` clearing drive-slew memory and the shooter FSM
  between episodes; trainer logs score/collect reward components separately;
  deterministic evaluation vs random/zero baselines added
  (`scripts/rl/eval_checkpoint.py`); 5 RL unit tests added (145 total).
  **Checkpoints `runs/drqv2_stageA{,_v2}` predate these fixes and are kept as
  systems smoke-test artifacts only - not scientific baselines.**
- Known deferred items: RAM ring -> D:-NVMe chunked replay with recall-chosen
  codec; stage C/D compact-trench episodes; 2-cam vs 3-cam render ablation;
  fuel-subset randomization per reset.

## Risks

- **Render is the binding cost** (3-cam single-env measured 3.50 tx/s at 456
  FUEL; 14.0 tx/s at N=2 on the 32-FUEL template). Milestone
  4 may force simplified training visuals or ≤2–4 envs.
- **456 FUEL × N** rigid bodies stress PhysX GPU; may need `RigidObjectCollection`
  and/or fewer FUEL in early curriculum stages.
- **USD export fidelity** — colliders, materials, articulation drives must
  survive export/clone; verify against the single-env sim numbers.
