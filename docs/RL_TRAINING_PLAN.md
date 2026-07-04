# RL training plan

Last updated: 2026-07-04

Implementation status:

- RL dependencies are installed in `C:\il\venv`: Isaac Lab RL/tasks, RSL-RL
  5.0.1, and the compatible ONNX/Isaac Sim pins.
- `scripts/rl/check_stack.py` successfully boots Isaac Lab on the RTX 5080.
- `xrc_rebuilt.rl` defines and tests the first versioned action/timing contract.
- Next: implement and benchmark the full-scene `DirectRLEnv` reset/step loop.

## 1. Mission

Train a blue-alliance policy to play a complete xRC REBUILT match with the
current competition robot. The policy must:

1. leave the blue TRENCH safely from the compact starting pose;
2. find and collect dynamic FUEL without relying on teleportation;
3. retain and manage FUEL in the physical hopper and feeder;
4. choose productive scoring locations and times;
5. request shooting only when the blue HUB is legal and the shot controller
   can complete the shot;
6. use ferrying when it improves future collection/scoring throughput;
7. avoid walls, field structures, immobilization, and illegal red-HUB
   targeting;
8. score at least 200 FUEL in a full evaluation match.

The long-term goal is to replace a human driver using onboard-like perception.
The first policy will not be end-to-end pixels. It will be a privileged teacher
that establishes what good driving and match strategy look like. A camera-based
student will then learn from that teacher.

## 2. Non-negotiable invariants

- The robot alliance is hard-locked to blue in the environment and controller.
- The policy cannot directly set projectile position, speed, or score.
- Intake and shooting continue to use the current physical FUEL and mechanism
  paths.
- The policy cannot call a scripted route planner that completes the task for
  it.
- The calibrated shooter/auto-aim is a low-level robot capability, analogous to
  real robot code. RL decides when and where to use it.
- Formal evaluation begins from the accepted compact TRENCH pose with 456 total
  FUEL and the implemented match clock/rules.
- Training shortcuts must be removable curriculum settings, not hidden final
  evaluation advantages.

## 3. Why this training strategy

### 3.1 PPO first

Isaac Lab is designed to collect experience from thousands of GPU-vectorized
environments. PPO is the best first baseline here because it is stable,
well-supported by both RSL-RL and RL-Games, easy to inspect, and can exploit
that parallel simulation throughput.

The problem is primarily long-horizon task strategy over a reliable low-level
controller, not learning raw motor torques. A modest recurrent actor can be
trained faster and debugged more easily than a very large world model.

### 3.2 Asymmetric teacher

The actor receives only observations that can later be reconstructed from
cameras, odometry, AprilTags, and mechanism sensors. The critic receives
privileged simulator state: complete FUEL state, contacts, exact poses, match
state, and mechanism state. This reduces value-estimation noise without
teaching the deployed actor to depend on unavailable information.

Isaac Lab directly supports separate actor/image and privileged critic
observation groups. This architecture is also well established for
image-based robot learning.

### 3.3 Curriculum before sparse full-match learning

A successful match is a long chain:

`navigate -> acquire -> retain -> choose target -> align -> shoot -> repeat`.

Learning that chain from score-only reward would waste most samples. Training
therefore introduces one skill at a time, then removes the shaping and combines
the skills in full matches.

### 3.4 Vision second

Rendering thousands of high-detail camera views is far more expensive than
physics-only rollouts. NVIDIA's published Isaac Lab benchmark, for example,
shows 4096 state-based Cartpole environments using about 3.3 GB VRAM, while
1024 RGB-camera environments use about 16.7 GB. On this 16 GB laptop GPU, a
state teacher can use many more environments than a visual learner.

The camera student will use tiled rendering and teacher-generated targets,
then receive short RL fine-tuning. This preserves the computer-vision goal
without making perception the bottleneck before the task itself is solved.

## 4. Environment design

Create:

```text
src/xrc_rebuilt/rl/
  __init__.py
  competition_env.py
  competition_env_cfg.py
  observations.py
  actions.py
  rewards.py
  curricula.py
  terminations.py
  randomization.py
  agents/
    rsl_rl_ppo_cfg.py
scripts/rl/
  train.py
  play.py
  evaluate.py
  collect_teacher_rollouts.py
```

Use an Isaac Lab `DirectRLEnv` initially. The environment owns batched tensors
and calls the existing robot mechanism/shot logic through a vector-safe adapter.
Once behavior is stable, reusable terms may be moved into manager-based
components.

### 4.1 Two scene profiles

**Training profile**

- collision and sensor geometry only;
- simplified materials and no high-detail CAD per environment;
- batched robot/fuel state on GPU;
- reduced initial FUEL subsets during early curricula;
- no RTX rendering for the state teacher.

**Evaluation profile**

- complete accepted field and robot visuals;
- all 456 FUEL;
- exact match timing, HUB state, return routing, and lights;
- one or a small number of environments;
- optional recording and deterministic replay.

Both profiles must share dimensions, masses, collision proxies, controller
constants, rules, and reward-independent success metrics.

### 4.2 Simulation/control rates

- physics: 60 Hz;
- low-level robot controller: 60 Hz;
- RL action rate: 10 Hz (six physics steps per decision);
- episode: full 160-second match = 1600 policy decisions;
- early skill episodes: 10-40 seconds.

Ten hertz is fast enough for FRC strategic driving with the current slew-limited
swerve controller, while greatly reducing the effective planning horizon.

## 5. Action space

Use a normalized continuous `Box(-1, 1, shape=(7,))`:

| Index | Meaning | Mapping |
|---|---|---|
| 0 | field-relative forward request | normalized input to the robot drive controller |
| 1 | field-relative strafe request | normalized input to the robot drive controller |
| 2 | turn request | normalized input to the stable yaw-rate controller |
| 3 | intake command | hysteresis: on above `+0.25`, off below `-0.25` |
| 4 | storage command | compact/extend with hysteresis |
| 5 | blue-HUB shoot request | hold while above `+0.25` |
| 6 | ferry request | hold while above `+0.25` |

Shoot and ferry are mutually exclusive. If both are requested, blue-HUB score
mode wins. Action changes are rate-limited to prevent policy-induced chassis
instability and mechanism chatter.

This is a hierarchical policy: RL chooses motion and mechanism intent; the
existing controller performs swerve kinematics, automatic aim, flywheel
calibration, feeder timing, and legal blue-target enforcement.

## 6. Observation design

### 6.1 Actor/teacher observations

All quantities are robot-relative or normalized:

- chassis linear/angular velocity and acceleration;
- heading represented as sine/cosine;
- blue HUB relative bearing/range and currently active/eligible flags;
- match phase and normalized time remaining;
- storage extension, intake state, shooter state, aim error, shot readiness;
- physical magazine/feeder count estimates and recent collection/shot events;
- 24 planar obstacle range rays;
- nearest `K=24` visible FUEL relative positions/velocities with masks;
- a coarse egocentric FUEL-density grid (for example `16 x 8`);
- previous action;
- short event history or a 256-unit GRU state.

Do not provide the actor with perfect global coordinates for every FUEL. Add
range limits, occlusion/noise, and observation latency so the representation
can later be generated by vision.

### 6.2 Privileged critic observations

The critic additionally receives:

- exact robot pose and contact forces;
- complete or pooled field FUEL occupancy/velocity;
- exact intake/hopper/feeder FUEL memberships;
- exact obstacle distances and collision state;
- true randomized physics parameters;
- exact HUB queue and return state.

### 6.3 Vision student observations

Initial visual student:

- front and rear RGB or RGB-D cameras at `128 x 96`;
- AprilTag detections/poses;
- wheel odometry and IMU;
- mechanism state and match clock;
- previous action;
- recurrent state.

Use Isaac Lab tiled cameras. Domain-randomize lighting, exposure, textures,
motion blur, camera pose, tag detection noise, and 0-100 ms observation delay.

## 7. Reward specification

Rewards are logged separately; never hide them in one total.

### 7.1 Event rewards

- `+10.0` blue FUEL legally scored;
- `+1.5` new FUEL crosses the physical intake throat;
- `+0.5` FUEL reaches feeder-ready state;
- `-1.0` retained FUEL leaves the robot other than through a valid shot;
- `-2.0` shot exits when no valid blue-HUB solution exists;
- `-5.0` red-target request reaches the environment (should be impossible);
- `-2.0` hard collision above the tuned impulse threshold;
- `-10.0` immobilization/invalid physics termination.

### 7.2 Dense shaping

- potential-based progress toward a selected reachable FUEL cluster;
- after useful inventory is acquired, potential-based progress toward a valid
  scoring pose;
- alignment/readiness reward only while carrying feeder-ready FUEL;
- small reward for scoring throughput and keeping the feeder supplied;
- small action-smoothness and energy penalties;
- stagnation penalty after several seconds without collection, scoring, or
  meaningful displacement.

Potential differences rather than raw distance rewards reduce the risk of the
agent farming reward by oscillation.

### 7.3 Reward annealing

Collection/navigation shaping starts strong and decays as success improves.
Final full-match selection is based on actual legal score, not shaped return.

## 8. Curriculum

### Stage 0: environment correctness

Goal: a random policy can step/reset thousands of times without NaNs, exploding
objects, red-HUB targeting, or state leakage.

- 64 environments;
- 30 seconds;
- no learning;
- deterministic replay hash for fixed seeds.

Gate: 100,000 environment steps with zero invalid states.

### Stage 1: locomotion and obstacle safety

- 256-1024 environments;
- waypoint/velocity commands;
- no FUEL or only static marker FUEL;
- randomized starts around BUMPS and TRENCHES.

Gate: >99% target-zone arrival and >99% compact TRENCH passage.

### Stage 2: collection

- 512-2048 environments;
- 4-32 FUEL near the robot;
- intake and storage actions enabled;
- episodes end after target collection count or timeout.

Gate: median acquisition >1.5 FUEL/s in dense areas, <2% loss after intake.

### Stage 3: scoring from inventory

- robot begins with randomized 1-12 feeder/hopper FUEL;
- spawn position and velocity randomized across legal field locations;
- current auto-aim/calibration remains active;
- learn where to stop, coast, or shoot while moving.

Gate: >99% valid-shot hit rate and no red-HUB shots across 10,000 attempts.

### Stage 4: collect-and-score loops

- 32-128 FUEL;
- 30-80 second episodes;
- randomized clusters, return timing, and initial pose;
- gradually reduce navigation shaping.

Gate: median >60 legal scores per 80 seconds.

### Stage 5: complete match

- all 456 FUEL and exact 160-second timing;
- exact HUB activity and return routing;
- randomized xRC-valid initial FUEL micro-poses and HUB return jitter;
- 512-1024 environments if memory permits, otherwise 128-512.

Gate: median >=200, 10th percentile >=170 over 100 held-out seeds; zero
red-target incidents.

### Stage 6: vision distillation

- freeze the best teacher;
- collect diverse teacher rollouts, including recovery and deliberately
  perturbed states;
- train CNN/GRU student by action distribution and latent-feature distillation;
- aggregate student-visited states (DAgger-style) and relabel with teacher;
- fine-tune with asymmetric PPO.

Gate: vision student retains >=90% of teacher median score before domain
randomization, then >=85% under the full randomized evaluation suite.

## 9. Baseline network and PPO configuration

Start deliberately small:

**Actor**

- observation normalization;
- MLP `[256, 256, 256]`, ELU;
- optional GRU 256 after the first feed-forward baseline;
- separate Gaussian continuous-action heads;
- approximately 0.3-0.8 million parameters.

**Critic**

- privileged encoder plus actor observation encoder;
- MLP `[512, 512, 256]`, ELU;
- approximately 1-2 million parameters.

**Initial PPO values**

- optimizer: Adam;
- learning rate: `3e-4`, adaptive/KL schedule;
- discount `gamma`: `0.997`;
- GAE `lambda`: `0.95`;
- clip: `0.2`;
- value loss coefficient: `1.0`;
- entropy coefficient: `0.01`, anneal toward `0.001`;
- rollout: 32-64 decisions/environment;
- 4-5 epochs/update;
- 4-8 minibatches;
- gradient norm clip: `1.0`;
- desired KL: `0.01-0.02`.

Run at least three seeds for every configuration promoted beyond a curriculum
stage. A single lucky run is not a result.

## 10. RTX 5080 Laptop execution plan

The GPU has about 16 GB VRAM. Begin with:

- state teacher smoke test: 256 environments;
- scale sequentially to 512, 1024, 2048, then 4096 while recording simulation
  FPS, policy FPS, VRAM, reset time, and PhysX error count;
- maintain at least 1.5-2 GB VRAM headroom;
- full 456-FUEL environments: start at 128 and scale cautiously;
- visual student: start at 32, then 64/128 tiled-camera environments.

High-detail 1.29-million-triangle robot rendering must not be cloned into every
training environment. It is an evaluation/inspection asset.

Checkpoints every 25-50 updates, best checkpoint by held-out legal score, and a
rolling last-three checkpoint set. TensorBoard is local by default; no cloud
logging is required.

## 11. Evaluation protocol

Every promoted checkpoint runs:

1. 20 deterministic regression seeds;
2. 100 held-out randomized seeds;
3. five stress suites:
   - FUEL crowding and intake jams,
   - BUMP/TRENCH contacts,
   - lighting/camera variation,
   - mass/friction/motor/latency variation,
   - HUB return bursts.

Record:

- legal blue score;
- score per second and time to first score;
- FUEL collected, retained, fired, hit, and lost;
- shot hit rate by distance/speed bin;
- collision impulse and stuck time;
- compact/extend transitions and TRENCH failures;
- red-target requests (must be zero);
- policy inference time;
- episode video for the median, worst, and best seed.

The 200-score milestone is accepted only when the median of 100 held-out
full-match seeds is at least 200. The GUI showcase is useful, but it is not the
metric.

## 12. Frontier methods: where they fit

### Production baseline

- **Massively parallel PPO:** fastest route to a reliable first policy in
  Isaac Lab.
- **Asymmetric actor-critic:** privileged critic improves training while
  keeping deployable actor inputs realistic.
- **Teacher-student distillation:** converts the high-throughput state policy
  into a camera policy.
- **Curriculum and domain randomization:** solve the long-horizon chain and
  harden it.

### Experiments after the PPO teacher

- **DreamerV3:** learns a recurrent world model and trains behavior in imagined
  trajectories. It is attractive for image-based, sparse, long-horizon tasks,
  but is more complex and compute-heavy than the first PPO baseline.
- **TD-MPC2:** plans in the latent space of a learned implicit world model and
  has strong continuous-control results. It is a useful sample-efficiency
  comparison at the 30-80 second collect-and-score stage.
- **Planner-guided/distilled RL:** a privileged route/cluster planner can guide
  a partial-observation student, but it must remain a training-only teacher and
  cannot become a scripted final policy.

The project should compare methods with identical held-out seeds and real score,
not choose a frontier algorithm because its model is larger.

## 13. Primary references

- [Isaac Lab quickstart and vectorization](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html)
- [Isaac Lab RL performance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- [Isaac Lab DirectRLEnv API](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/direct_rl_env.html)
- [Isaac Lab asymmetric actor/critic observation mapping](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_rl/isaaclab_rl.html)
- [NVIDIA tiled rendering for vision RL](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/an-introduction-to-robot-learning-and-isaac-lab/03-available-robots-and-environments/03-tiled-rendering.html)
- [Isaac Lab teacher-student sim-to-real workflow](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/experimental-features/newton-physics-integration/sim-to-real.html)
- [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542)
- [DreamerV3: Mastering Diverse Domains through World Models](https://www.nature.com/articles/s41586-025-08744-2)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://www.tdmpc2.com/)
- [PriPG-RL: privileged planner-guided RL](https://arxiv.org/abs/2604.08036)

## 14. Immediate implementation checklist

1. Install the `isaaclab_rl`, `isaaclab_tasks`, and RSL-RL components from the
   existing `C:\il\IsaacLab` checkout into `C:\il\venv`.
2. Add the `xrc_rebuilt.rl` package and register `XRC-CompetitionRobot-Direct-v0`.
3. Extract vector-safe robot action/state adapters from the interactive scene.
4. Implement deterministic reset, batched observations, and zero-reward random
   stepping.
5. Add invariant tests for blue-only target lock, 456-FUEL accounting, reset
   reproducibility, action bounds, and no direct scoring mutation.
6. Benchmark 64/256/1024 lightweight environments.
7. Implement Stage 1 locomotion reward and train the first PPO smoke-test
   checkpoint.
8. Promote only after evaluation scripts reproduce its metrics from a saved
   seed list.
