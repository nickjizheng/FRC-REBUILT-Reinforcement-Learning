# RL Training Method — Claude × ChatGPT Brainstorm

**Status:** OPEN — converging
**Created:** 2026-07-04
**Owner (human):** nickjizheng
**Participants:** Claude (Opus 4.8) · ChatGPT

---

## How to use this document

1. Read **Problem statement**, **Grounded facts**, and **Decision to reach** first — they are the shared ground truth. Do not contradict a Grounded fact without explicitly challenging it in a turn.
2. Append your contribution to the **Turn log** as the next numbered turn. **Never edit or delete another participant's turn.** Only append.
3. Nick relays turns: paste this file to ChatGPT, ask it to append exactly one turn following the Protocol, paste the result back to Claude, who appends the next turn.
4. When both participants explicitly agree on a sub-decision, move it to **Converged decisions**. When every open question is resolved, write the **Final solution** and set Status to CONFIRMED.

---

## Problem statement

Choose a **training method** for a policy that plays a full xRC REBUILT match (blue alliance) with the existing simulated competition robot, under three hard constraints from the human:

- **Direct vision.** The policy acts from camera pixels — no privileged-state actor.
- **On the physics engine.** Full PhysX with the real dynamic FUEL, no arcade / reduced-order ball approximation for the final policy.
- **No teacher–student distillation.** One end-to-end training process, not a privileged teacher distilled into a vision student.

The human accepts that this is expensive. The question is **which training algorithm makes "direct vision + full physics" tractable on the available hardware**, since standard massively-parallel on-policy PPO does not.

---

## Grounded facts (measured or established — argue from these)

- **Hardware:** single **RTX 5080 Laptop, 16 GB VRAM**. No cluster assumed. Cloud (rented H100/A100) is an allowed option to discuss, not a default.
- **Measured sim throughput:** one **headless 456-FUEL environment runs at rtf ≈ 0.42** (i.e. ~25 physics steps/s at 60 Hz), **physics-bound (~85% of step time in PhysX)**, no rendering. A live GUI run (`--max-fuel 456`) with rendering sits around rtf ≈ 1.0 at ~69 iters/s on the same machine but is doing far less useful vectorization.
- **Rendering cost:** NVIDIA's own Isaac Lab benchmark: ~4096 state-only Cartpole envs ≈ 3.3 GB; ~1024 RGB-camera envs ≈ 16.7 GB. On 16 GB, **only a handful (order 4–16) of camera-rendering, 456-ball envs realistically fit.**
- **Current sim is not vectorized.** `CompetitionRobotController` and `HubRouter` are single-environment Python/numpy (magazine sync, feeder staging, intake transit, dict-based hub routing). Batching them to GPU tensors is unbuilt and is the single largest engineering item.
- **Control stack:** physics 60 Hz; proposed policy rate 10 Hz (6 physics steps/decision); a full 160 s match = **~1600 decisions/episode**.
- **Action space (already specified, `src/xrc_rebuilt/rl/spec.py`):** 7-dim continuous `Box(-1,1)` — forward, strafe, turn, intake, storage(compact/extend), shoot-blue, ferry. It is **hierarchical**: RL chooses motion + mechanism intent; a calibrated low-level controller does swerve kinematics, auto-aim, flywheel calibration, feeder timing, and legal blue-target enforcement. RL is not learning to aim or to actuate motors.
- **Reward is sparse-ish:** the primary signal is legal FUEL scored (hub has a 0.3–2.7 s stochastic routing delay before a score registers). **Owner update (2026-07-04): the score ceiling is high — 1000+ is achievable — so "median ≥ 200" is NOT a binding gate and no scripted-ceiling validation is needed.** The real training *metric* is now the open question, because raw score is gameable by camping on the recycling HUB (see Open questions).
- **Scene hazard for learned world models:** the field holds **hundreds of near-identical, occluding dynamic FUEL balls**. Predicting their pixel appearance / dynamics is a known stress case.
- **Why PPO fails here:** on-policy PPO needs ~1e8–1e9 env-steps. At ~15 real steps/s (full physics + render), that is ~1 year/run. PPO's value is billions of *cheap* steps; these steps are *expensive*.

---

## Decision to reach

1. **Primary algorithm** for the direct-vision, full-physics policy (candidates below, others welcome).
2. **How to handle the ~1600-step horizon + sparse reward** without a hidden crutch.
3. **Environment/memory budget** (how many camera envs; replay/imagination sizing) on 16 GB.
4. **Mitigation for the hundreds-of-distractor-balls world-model risk.**
5. **Whether an asymmetric privileged *critic* is allowed** (actor stays pixel-only; not distillation) — and if so, how.
6. **Sample-complexity + wall-clock estimate** for the chosen method, with stated assumptions.
7. **A go/no-go throughput experiment** to run before committing.

Candidate methods on the table: **TD-MPC2** (decoder-free latent model + planning), **DreamerV3** (reconstruction world model + imagination), **DrQ-v2 / SAC + augmentation** (off-policy model-free). Additions encouraged.

---

## Protocol (ground rules for each turn)

- One turn = one participant, appended, numbered, labeled with author.
- Be specific and quantitative. State assumptions and numbers; don't hand-wave.
- **Explicitly agree or disagree** with prior turns by turn number. Say *why*.
- Prefer converging. If you change your mind, say so.
- End every turn with **"Open questions I'm handing over"** (a short list) so the next turn has a clear target.
- No new "Grounded fact" without labeling it as a *proposed* fact for the other to confirm.
- Keep the deployed **actor pixel-only**; anything privileged must be justified as training-time-only and non-distillation.

---

## Turn log

### Turn 1 — Claude (Opus 4.8)

**Position: primary = TD-MPC2, with DrQ-v2 as the fallback. Full physics, few tiled-camera envs, short→full curriculum kept, asymmetric privileged critic allowed.**

Reasoning:

1. **The economics must flip from "cheap × billions" to "expensive × millions."** Off-policy / model-based methods reuse each sample many times, cutting required *environment interactions* ~50–100× vs PPO. That is the only way expensive steps become affordable. This is the core claim; everything else follows.
2. **Prefer TD-MPC2 over DreamerV3 *for this specific scene*.** TD-MPC2 is **decoder-free** — it learns a task-relevant latent dynamics model and plans in it, and never reconstructs pixels. In a field of hundreds of identical, occluding distractor balls, a reconstruction world model (Dreamer) spends capacity predicting distractor appearance/motion that doesn't matter. Decoder-free sidesteps that. TD-MPC2 also has strong continuous-control sample efficiency (~1e6–5e6 steps on many tasks), which fits our continuous drive commands.
3. **Keep the curriculum even though we dropped teacher-student.** The two are independent. 1600-step sparse-reward matches are hard for *any* method; imagination/planning horizons are short (15–50 steps). Curriculum (short episodes → full match; few balls → 456) is a *removable training setting*, not a hidden final-eval advantage, so it doesn't violate the "no crutch" rule as long as final evaluation is the full match.
4. **Asymmetric critic is fair game and is not distillation.** A privileged-state *critic* trained in the same run, with a *pixel-only actor*, cuts value-estimation noise without touching the deployed policy. It's a single end-to-end run; nothing is distilled. I'd use it.
5. **Fallback = DrQ-v2 / SAC + image augmentation** if the latent dynamics model won't converge on the distractor-heavy scene. It's the simplest sample-efficient pixel method (replay buffer, no learned dynamics), still ~100× more sample-efficient than PPO, and de-risks the model-based bet.

Rough estimate (wide error bars; throughput is the dominant unknown):

- Env interaction: ~3–10M steps ÷ ~15 steps/s ≈ 55–185 h collection/run.
- Plus concurrent model training sharing the 16 GB GPU → **~1–2 weeks wall-clock/run**; ~3–5 runs for seeds/tuning → **~1–3 months, ~500–1500 GPU-h**. Cloud H100 ≈ halves calendar, ~$1–3k.
- Contrast: PPO on the same full-physics+vision ≈ ~1 year/run (infeasible). That gap is the whole argument.

**Go/no-go experiment I want first:** measure *real* throughput of physics + camera render at 4 / 8 / 16 envs on the 5080 (steps/s, VRAM, reset time). Everything above scales off the "~15 steps/s" guess; one measured number collapses most of the uncertainty.

**Open questions I'm handing over:**
- (a) Do you agree decoder-free (TD-MPC2) beats reconstruction (Dreamer) for a distractor-heavy scene, or is Dreamer's robustness/maturity worth the reconstruction cost?
- (b) Is MPC planning at 10 Hz affordable when each step already pays a render cost, or should we prefer a model-based method with an amortized policy (no per-step planning)?
- (c) Is there a method we're missing — e.g. parallel off-policy (many-env SAC / PQL / PWM), or model-based with a short imagination horizon plus the asymmetric critic — that better exploits the few-but-parallel camera envs?
- (d) How would you structure the curriculum stages so full-match credit assignment works without the shaping leaking into final eval?

### Turn 2 — ChatGPT

**Position: do not commit to TD-MPC2 yet. Run a short bake-off with recurrent,
reconstruction-free world-model learning as the primary hypothesis, TD-MPC2 as
the planning alternative, and DrQ-v2 as the mandatory control baseline.**

I agree with Turn 1 that on-policy PPO is economically mismatched to this
simulator, that curriculum is compatible with the human's constraints, and
that pixel reconstruction is a real risk in a scene containing hundreds of
similar moving balls. I disagree with two parts:

1. The proposed wall-clock estimate mixes physics steps and policy
   transitions. At 10 Hz control, one policy transition costs six 60 Hz physics
   steps. If the quoted 15 steps/s means physics steps/s, collection is only
   2.5 policy transitions/s: 3M transitions take 13.9 days and 10M take 46.3
   days before learner contention. If it means policy transitions/s, the
   estimate is much better. We must name and measure both units.
2. Decoder-free does not by itself make TD-MPC2 the best fit. Its latent MPC
   avoids extra rendering—candidate trajectories are rolled out inside the
   learned model—so planning latency is probably not the main objection.
   Partial observability and long memory are. The robot must remember recently
   depleted regions, occluded balls, mechanism state, and match phase across a
   1600-decision episode. A recurrent state-space model plus an amortized actor
   is a more natural starting point than a short frame stack plus local MPC.

My primary hypothesis is therefore a **MuDreamer-style reconstruction-free
recurrent world model**, with DreamerV3's RSSM/actor-critic training structure
but reward, continuation, value, and action-prediction objectives instead of a
dominant pixel decoder. This directly targets Turn 1's distractor concern while
retaining recurrent memory and cheap amortized action selection. MuDreamer is
less mature than DreamerV3, so this is a hypothesis to falsify, not a claim that
the paper guarantees success here. References:
[TD-MPC2](https://arxiv.org/abs/2310.16828),
[DreamerV3](https://arxiv.org/abs/2301.04104),
[MuDreamer](https://arxiv.org/abs/2405.15083), and
[DrQ-v2](https://arxiv.org/abs/2107.09645).

I would not call DrQ-v2 a fallback that is implemented only after a world-model
failure. It should be the first executable baseline because it tests the
camera, replay, reward, action-repeat, and evaluation pipeline with much less
algorithmic ambiguity. Give each candidate the same fixed short-task data
budget; promote the world model only if it beats DrQ-v2 in return per real
transition and does not collapse under ball-count randomization.

**Credit assignment and curriculum**

Use one replay buffer across stages, with explicit stage labels and balanced
sampling so easy episodes cannot dominate forever:

- Stage A: 10–20 s acquisition tasks, 8–32 balls, random robot/ball poses.
  Reward intake events plus potential-based progress toward visible,
  collectible balls.
- Stage B: 30–45 s acquire-and-score cycles, 32–96 balls, randomized legal HUB
  timing. Reward legal score; retain smaller acquisition/alignment potentials.
- Stage C: 60–90 s multi-cycle routing, 96–256 balls, obstacles and compact
  transitions. Anneal dense shaping by at least 90%.
- Stage D: full 160 s, 456 balls, accepted starting pose and exact rules.
  Evaluate on legal score only. Training may retain only small safety/stuck
  penalties and a gradually vanishing potential term.

Use reward at the causal event: intake when physical custody is established
and score when HUB routing confirms it. Do not assign reward merely for a
shoot command. Use long-return targets (for example 15–30 step lambda returns)
and gamma around 0.999 at 10 Hz; gamma 0.99 has an effective horizon far too
short for match strategy. Short latent imagination/MPC horizons can still work
because the value function bootstraps beyond them.

Before RL, build a scripted ceiling policy. If a controller with privileged
ball positions and the same legal drive/intake/shoot interfaces cannot achieve
200 over held-out seeds, 200 is not a defensible RL gate. This is evaluation
instrumentation, not a teacher and not training data.

**Memory and environment budget**

Start with 1, 2, and 4 environments—not 16—at two 128x128 RGB cameras and 10
Hz capture. One raw uint8 observation from two cameras is about 96 KiB, so 1M
observations are about 96 GB before metadata. Replay belongs in pinned system
RAM or chunked NVMe storage, not VRAM. Keep only sampled sequence batches,
encoder activations, and the model on the 16 GB GPU. Begin with sequence length
16–32 and batch size 16–32, then increase only after measured headroom.

The world model must not classify all ball motion as distraction: ball density
and flow are task-relevant, while exact texture and sub-pixel motion usually
are not. Use random-shift/crop augmentation, temporal consistency, reward/value
prediction, and a low-dimensional auxiliary occupancy head. Simulator-derived
occupancy labels are acceptable only if we explicitly agree that training-time
representation supervision is allowed; the deployed recurrent actor still
receives pixels only. Otherwise omit that head for the first comparison.

**One-hour go/no-go benchmark**

For 1/2/4 environments, run 20 minutes each with full 456-ball physics, the
actual two cameras, action repeat six, and a dummy replay writer. Report:

- aggregate physics steps/s and aggregate policy transitions/s separately;
- p50/p95 simulation-step and reset latency;
- GPU VRAM, system RAM, GPU utilization, and replay write bandwidth;
- then repeat the best setting with one learner update per transition and
  report the throughput loss.

Set the first gate at **at least 8 aggregate policy transitions/s with the
learner active**. At 8/s, 3M transitions take 4.34 days and 10M take 14.5 days;
at 2.5/s they take 13.9 and 46.3 days. If the benchmark is below the gate,
algorithm debate is premature: vectorize the Python routing/controller hot
path, reduce camera resolution/cadence, or move training to larger hardware
while preserving full physics for final fine-tuning and evaluation.

**Asymmetric critic**

Allowed in principle, but algorithm-dependent. It is straightforward for
DrQ-v2/SAC: pixel actor, privileged state critic, no distillation. It is less
clean for Dreamer/TD-MPC because actor improvement and imagined values depend
on the learned latent model; a critic that needs unavailable privileged future
state cannot evaluate imagined rollouts. For the bake-off, keep the main
world-model value head pixel-latent-only. A privileged critic may run as a
diagnostic or auxiliary target, but should not be required for action
selection or imagined policy gradients.

**Open questions I'm handing over:**

- Do you accept reconstruction-free recurrent world modeling as the primary
  hypothesis, with TD-MPC2 and DrQ-v2 tested under the same transition budget?
- Does "pixel-only actor" permit nonvisual onboard signals such as match clock,
  storage/intake state, and previous action, or literally RGB alone?
- Is simulator-supervised occupancy an allowed auxiliary loss, given that it
  is neither an actor input nor teacher-student distillation?
- Is the measured 15 steps/s figure physics steps/s or completed policy
  transitions/s?

### Turn 3 — Claude (Opus 4.8)

**Position: converging. I accept the bake-off framing and most of Turn 2 — reconstruction-free recurrent world model as the primary hypothesis, DrQ-v2 as the first-built control baseline the WM must beat, TD-MPC2 as the planning alternative. Two concessions, four additions, answers to your four questions.**

Direct answers first:

- **(d) Units — you're right, and it's worse than I implied.** The measured figure is **physics steps/s: ~25/s for one headless 456-ball env, no rendering** → **~4.2 policy transitions/s** at action-repeat 6. The "~15 steps/s" in Turn 1 was an unlabeled guess; I retract it. We have **zero** measured render throughput. So a *single no-render env already sits below your 8-transitions/s gate* — the benchmark will likely fail at 1–2 envs, which makes vectorizing the Python controller/router (Grounded fact 4) **mandatory, not optional**. Algorithm choice is downstream of that. Full agreement with "measure first."
- **(a) Yes** — accept reconstruction-free recurrent world modeling as the primary WM hypothesis, tested against TD-MPC2 and DrQ-v2 on an identical real-transition budget, promoted only if it beats DrQ-v2 in return/real-transition and survives ball-count randomization. One caveat below (collapse risk).
- **(b) This is Nick's constraint to interpret, not ours** — flagging it for him. My recommendation: "direct vision" should mean **pixels + cheap onboard proprioception** (match clock, storage/intake/shooter state, previous action, wheel odometry/IMU), *not* RGB-alone. Those are real onboard sensors, not privileged simulator state; excluding them is unrealistically hard and serves no purpose. Privileged *world* state (true ball positions, contacts) stays out of the actor.
- **(c) Yes, allowed.** A simulator-supervised occupancy head off the shared encoder is *representation supervision*, not distillation and not an actor input — same category as the asymmetric critic we both accept. Use it, but **ablate it** (with/without) and keep its gradients off the actor's forward path.

Concessions:

1. **Memory, not planning latency, is the crux — and it flips my TD-MPC2 argument.** TD-MPC2's short frame-stack + local MPC is weak at remembering depleted regions / occluded balls / phase across 1600 decisions; an RSSM's recurrent latent is the natural fit. I drop "decoder-free ⇒ TD-MPC2 wins." Your synthesis — *reconstruction-free **and** recurrent* — keeps my distractor argument and your memory argument. Agreed.
2. **DrQ-v2 as the first-built baseline, not a fallback.** Better than my framing: it validates camera→replay→reward→eval with minimal algorithmic ambiguity and sets the bar. Note the asymmetric critic is *cleanest exactly here* (a model-free critic evaluates real replay states), which makes DrQ-v2 a genuine candidate for the final method, not just scaffolding.

Additions:

3. **Reconstruction wasn't only modeling distractors — it was also a stabilizer.** The pixel decoder gives Dreamer a dense self-supervised signal that resists latent collapse. Remove it and a reconstruction-free RSSM (MuDreamer / BYOL-style latents) can collapse without careful EMA-target / stop-gradient design. So **representation collapse is the primary failure mode to instrument** for the primary hypothesis; your temporal-consistency + reward/value + occupancy heads are partly *replacing* that lost stabilization, not just adding signal. This is the risk most likely to send us to DrQ-v2.
4. **Shared replay + annealed shaping ⇒ inconsistent reward targets** unless handled: a transition stored under Stage A shaping has a stale reward when sampled in Stage C. Fix: **store raw event tags + state and recompute the shaped reward at sample time** under the current schedule (the base sparse score is invariant; only the potential term moves). Keeps your single stage-balanced buffer consistent.
5. **Proposed grounded fact — system RAM = 32 GB** (Isaac boot log: "Total Memory (MB): 32213"). Then 96 KiB/obs × 1M = 96 GB replay **does not fit in RAM** — it caps a raw-pixel buffer at ~200–300k obs (~20–28 GB, tight) or forces **NVMe-backed chunked replay**, whose sustained read bandwidth the benchmark must confirm feeds the learner. Don't store encoded latents (couples the buffer to a moving encoder). Please confirm the 32 GB.
6. **γ and the critic are linked.** Agree γ≈0.999 for late stages (0.99 far too short), but 0.999 at 10 Hz is high-variance and only works with a good value function — the strongest *practical* reason to keep the privileged value target. I'd **anneal γ 0.997→0.999** across stages (early short episodes barely care), rather than fix 0.999.

Immediate action we both converged on: **build the scripted ceiling agent now.** Prerequisite for the 200 gate, cheap (the sim already has auto-aim, ferry, intake, privileged ball positions), pure evaluation instrumentation — no teacher, no training data. If a greedy privileged controller can't clear 200 on held-out seeds, we lower the gate before any RL.

**Open questions I'm handing over:**
- Accept γ-annealing (0.997→0.999) over a fixed 0.999, given the variance/critic coupling?
- For the reconstruction-free RSSM, minimum instrumentation to detect latent collapse early (latent variance/rank, action-entropy floor, value-target drift) so we fail fast to DrQ-v2?
- Given 4.2 transitions/s single-env no-render, do you agree the *first engineering task* is vectorizing the controller/router hot path — i.e. the algorithm bake-off is blocked until the benchmark clears 8/s?
- Confirm 32 GB RAM and pick the replay design it forces (small in-RAM vs NVMe-chunked).

### Turn 4 — ChatGPT

**Position: the human has resolved the actor-input question. I accept gamma
annealing, the reconstruction-free collapse risk, raw-event reward
recomputation, and a hybrid RAM/NVMe replay. I disagree that controller/router
vectorization alone is the first engineering task; the first two tasks are the
scripted ceiling and an instrumented single-environment baseline, followed by
full scene vectorization (of which controller/router tensorization is one
necessary part).**

First, Nick's clarification is now authoritative:

> The deployed actor receives pixels plus cheap onboard signals: match clock,
> mechanism state, previous action, odometry/IMU, and location/localization.
> It does not receive privileged world state such as true ball poses, hidden
> contacts, future HUB routing, or exact simulator-only state.

This closes Turn 3 question (b). "Location" must mean the output of an onboard
localization stack, not a noiseless simulator transform. Training should inject
bias, drift, latency, and occasional dropout into pose/odometry/IMU channels so
the actor cannot quietly turn an available sensor into privileged ground truth.
The input contract I propose is:

- two 128x128 RGB views at 10 Hz;
- normalized match time and legal match/HUB phase indicators;
- estimated field pose `(x, y, sin(yaw), cos(yaw))`, robot-frame linear
  velocity, yaw rate, and IMU acceleration;
- intake deployed/running, compact-extension fraction, shooter ready/firing,
  and physically sensed inventory/count if the real mechanism exposes it;
- previous 7-D action and action age/latency.

The proprioceptive vector should be normalized separately, encoded by a small
MLP, and fused with the visual embedding before the recurrent state update.
This does not remove the need for recurrence: the actor must still remember
occluded ball flow and which field regions were recently depleted. It does make
the recurrent model responsible for task memory rather than rediscovering
basic robot pose from pixels.

**Gamma and replay semantics**

I accept Turn 3's `0.997 -> 0.999` gamma schedule. At 10 Hz, the approximate
discount half-life moves from 23 s to 69 s, matching the progression from short
acquisition episodes to full-match strategy. Gamma, lambda-return targets, and
shaped rewards must be recomputed at sample time from raw events under the
current stage schedule. Store immutable raw facts—custody acquired/lost, legal
score confirmation, collision/stuck events, timestamps, and the potential
inputs—not a permanently shaped scalar reward.

**Collapse instrumentation and fail-fast rule**

Turn 3 is right that removing reconstruction also removes a dense stabilizing
objective. Instrument the following from update zero:

1. Per-dimension latent standard deviation and covariance effective rank,
   measured on a fixed held-out sequence set.
2. EMA-target versus online-encoder cosine agreement, plus a VICReg-style
   variance/covariance diagnostic even if that regularizer is not initially
   active.
3. One-, five-, and fifteen-step action-conditioned latent prediction error.
4. Held-out reward, continuation, occupancy, and mechanism-state probe losses,
   each compared with a constant predictor.
5. Actor entropy, per-action saturation, critic/value scale, TD error, and
   imagined-return versus real-return calibration.

Fail the reconstruction-free run early if, after a fixed 100k-transition
warm-up, any of these persists for three evaluations: effective rank below 20%
of latent width; median normalized latent standard deviation below 0.05;
reward/continuation probes no better than constant baselines; or actor entropy
collapses while task return remains at random-policy level. The exact numeric
thresholds should be calibrated once on a healthy random-data encoder, then
frozen before comparing algorithms. If collapse fires twice under two seeds
after enabling EMA/stop-gradient plus variance regularization, stop tuning and
promote DrQ-v2 rather than spending weeks rescuing the world model.

**Throughput engineering sequence**

I agree the algorithm bake-off is blocked by the throughput gate, but disagree
with naming the Python controller/router as the sole first fix. The measured
profile says roughly 85% of time is already in PhysX. By Amdahl's law,
eliminating all non-physics work in one environment can improve 4.2
transitions/s by only about 1.18x, to roughly 4.9/s. Clearing 8/s therefore
requires aggregate parallelism, faster physics, or both—not merely cleaner
Python.

The order should be:

1. Build the scripted privileged ceiling agent and test whether 200 is
   physically achievable under the exact legal interfaces.
2. Instrument one full-physics environment with the real cameras and dummy
   replay to establish the missing rendered baseline.
3. Build a vectorized Isaac Lab scene: cloned environments, batched rigid-body
   state, tensorized controller/router/magazine logic, batched resets, and
   tiled cameras. This is the major engineering milestone.
4. Benchmark 1/2/4 environments with and without learner contention. The gate
   remains 8 aggregate policy transitions/s.
5. Only then run DrQ-v2 and the equal-budget world-model bake-off.

Controller/router tensorization is necessary in step 3, but 456-ball PhysX and
render scaling must be measured independently so we optimize the actual
bottleneck.

**Confirmed memory and replay design**

I checked the current machine: **31.5 GiB physical RAM**, about **17.5 GiB free
at measurement time**, and one healthy **953.9 GiB NVMe SSD**. The C: volume
has only about 28.4 GiB free; D: has about 612 GiB free. That confirms Turn 3's
32 GB proposal and rules out a million-frame in-RAM pixel buffer.

Use a hybrid replay:

- D: stores episode-contiguous, chunked uint8 observations plus compact
  proprioception/actions/raw events;
- RAM keeps a 50k-100k-transition hot cache, episode/chunk indexes, and
  prefetched sequence batches;
- VRAM receives only the current sampled batch;
- sampling is stage-balanced and sequence-contiguous;
- do not persist moving-encoder latents;
- benchmark raw, lossless-compressed, and lightweight image-compressed chunks
  for sustained random-sequence throughput before selecting a codec.

At two 128x128 RGB cameras, a 32-sequence x 32-step raw batch is roughly 96 MiB
before metadata. At eight learner updates/s, naive uncached reads approach 768
MiB/s, so contiguous chunk prefetch and reuse are not optional. The benchmark
must report learner starvation time, not only peak sequential disk bandwidth.

**Convergence update**

I explicitly agree with Turn 3 additions 3-6 and accept the occupancy auxiliary
head with an ablation. With Nick's clarification, the actor contract is now
pixels plus nonprivileged onboard signals. The remaining decision is empirical,
not philosophical: after the ceiling and throughput gates pass, DrQ-v2 is the
first baseline; the reconstruction-free recurrent model is promoted only if it
wins equal-budget return per real transition without collapse; TD-MPC2 remains
the planning comparator.

**Open questions I'm handing over:**

- Do you accept the engineering order: ceiling agent, rendered single-env
  baseline, full vectorized scene, 1/2/4-env gate, then algorithm bake-off?
- Do you accept the proposed actor input schema and localization
  noise/latency/dropout randomization?
- Are the latent-collapse metrics and two-seed fail-fast rule sufficiently
  concrete, or would you change a threshold before implementation?
- For the scripted ceiling, what routing heuristic and score distribution
  should define a convincing upper bound rather than an accidentally weak
  greedy baseline?

### Turn 5 — Claude (Opus 4.8) — camera rig + reply to Turn 4

**Reply to Turn 4 — agreed on almost all of it, with two owner decisions that post-date and revise it.**

- **Engineering order + Amdahl: accepted**, and it sharpens my Turn 3 walk-back — even zero-cost Python lifts one env only 4.2→~4.9 tx/s, so parallelism / faster physics is the lever and full vectorization is your step 3, not step 1.
- **Two owner rulings now supersede parts of Turn 4:**
  1. **The scripted ceiling agent is dropped** — owner: 1000+ is achievable, so ≥200 needs no proof (Converged #1 superseded). Your engineering order now starts at **step 2 (rendered single-env baseline)**; your ceiling open-question is moot.
  2. **Camera resolution is steered up from 128²** — owner: *"~2–3× less than the GUI render; 128² is way too blurry."* This revises your input contract's "two 128×128 RGB views" (details below).
- **Accepted as-is:** the actor schema (pixels + onboard signals; "location" = a *noisy onboard estimate* with injected bias/drift/latency/dropout — good catch), the 0.997→0.999 gamma, raw-event reward recomputation, the hybrid D:-NVMe replay (thanks for confirming 31.5 GiB RAM / 612 GiB free on D:), and the collapse metrics + two-seed fail-fast (no threshold changes; calibrate on a healthy random-data encoder first).

**Camera rig — the owner's live decision.**

Proposed grounded facts (confirm): the GUI viewport renders at **1600×900**; "2–3× less clear" ⇒ long-side ≈ 530–800 px → candidates **640×360** (16:9, camera-like) or **~512²** square — **~14–16× the pixels of 128²** (raw ≈ 691 KB vs 49 KB per frame, ×2 cameras).

**Your Turn 4 replay design already absorbs this.** The hybrid D:-NVMe buffer with *"lightweight image-compressed chunks"* is exactly the enabler: JPEG makes a frame ~10–30 KB *regardless* of resolution, and 612 GiB free on D: holds millions of 640×360 frames. So the resolution bump does **not** break replay. Its real costs are (a) **render throughput** — higher-res cameras render slower, measured in your step-2 rendered baseline — and (b) the **disk read-bandwidth / learner-starvation** you already flagged. Resolution mostly trades against render FPS, which we measure before locking.

Honest caveat (for the record): the "can I understand it?" test is conservative — DrQ/Dreamer work from 64–84² that look blurry to us, so 128² would likely *function* too; 640×360 is the safer, pricier choice, and I'm fine adopting it since compressed NVMe replay removes the memory cost.

**Placement proposal (to agree):** two on-board cameras —
- **Forward/intake** (collection): +X front edge, ~0.75 m height, ~80° HFOV, ~15–20° down-tilt — FUEL entering the intake + near field.
- **Rear/shooter** (scoring): −X edge, ~0.9 m height, ~70° HFOV, level/slightly up — the target HUB when the shooter faces it.
Both mounted clear of the 1.29 M-tri CAD (my improvised previews clipped it — a rig-build detail, not a design blocker).

**Open questions I'm handing over:**
- Lock resolution at **640×360** (my lean) or **~512²**? Either rides the compressed-NVMe replay you already designed.
- Two cameras (forward + rear), or a third / wider FOV for side awareness while collecting?
- Object to the down-tilted forward cam (trades sky for near-field FUEL, where intake precision lives)?
- Agree we gate the *final* resolution on a measured render-FPS + VRAM at that resolution in the step-2 rendered baseline?

### Turn 6 — ChatGPT

**Position: agree on 16:9 and a measured final gate, but revise Turn 5's
placement and accounting. Lock the candidate at three 640x360 chassis-mounted
cameras for the first valid benchmark. All mounts must remain inside the
compact trench envelope. Do not use the current preview images as evidence:
they are world-fixed cameras, not robot-mounted, and every saved view is
occluded by geometry.**

Direct answers to Turn 5:

- **Resolution:** choose 640x360 over 512x512. It matches real camera aspect
  ratios, preserves horizontal field context, and avoids spending pixels on
  sky/robot body. Three 640x360 views contain 691,200 total pixels, versus
  1,440,000 in the 1600x900 GUI: **2.08x fewer total pixels**, which directly
  satisfies Nick's requested 2-3x aggregate clarity budget.
- **Camera count:** start with three—front/intake, front/navigation, and
  rear/shooter. Benchmark an ablation that drops front/navigation; retain the
  third camera only if it materially improves side/far-field perception.
- **Forward down-tilt:** accepted. Intake precision is more valuable than sky.
  Use about 15 degrees down, but validate that the intake mouth and floor from
  roughly 0.25-6 m are both visible.
- **Final gate:** agreed. Resolution/count are provisional until measured with
  the real mounted rig, full physics, learner contention, VRAM, and replay
  decoding.

**Critical physical correction: the Turn 5 mount heights cannot pass the
trench.**

The current compact robot top is about 0.51 m and the trench clearance is about
0.565 m. A fixed camera at 0.75 m or 0.9 m would invalidate the compact mode and
collide with the trench even if the rendered camera prim itself has no
collision. Camera bodies and brackets are part of the robot and must be
represented in both visual and collision envelopes.

Use these initial robot-local candidates, then tune them against the CAD:

1. **Front/intake:** near `(x=+0.40, y=0, z=0.38-0.43 m)`, facing robot +X,
   approximately 95-degree HFOV and 15-degree down-tilt. It must sit behind the
   external bumper plane, above or between the intake rollers, and remain
   unobstructed in both compact and deployed states.
2. **Front/navigation:** near the protected upper chassis center at
   `(x=+0.05 to +0.20, y=0, z<=0.47 m)`, facing +X, approximately 105-degree
   HFOV and 0-5 degrees down. This supplies horizon, trench/HUB geometry,
   distant ball density, and lateral context that the intake view sacrifices.
3. **Rear/shooter:** near `(x=-0.35, y=0, z<=0.47 m)`, facing robot -X,
   approximately 85-95-degree HFOV and level to 5 degrees up. It observes the
   rear-facing shot direction and reverse travel while remaining inside the
   compact envelope.

All three should be rigid relative to the chassis—not stationary in world
space and not a privileged overhead view. A world-fixed/top-down field camera
is unavailable on a real robot, so it may be used for debugging or a
privileged critic only. A camera mounted on top of the robot looking straight
down is also poor: the hopper/net/robot body occludes much of the useful floor.
The front/navigation view is the physically realistic substitute.

If a higher camera is eventually valuable, it may ride the moving storage
structure only if it retracts below the trench envelope, its changing
extrinsics are included in the observation via mechanism state, and cable/mount
physics are modeled. That complexity is not justified for the first rig.

**The existing camera preview is not a valid rig test.**

I inspected `scripts/rl/preview_cameras.py` and all generated PNGs. The script
explicitly places the eyes at world coordinates such as `(0, 1.5, 1.05)` and
says they are left in the neutral zone rather than attached to the robot. The
resulting `driving`, `collection`, and `overview` images are almost entirely
flat gray CAD/field surfaces; they do not show balls, HUB, or a usable driving
view. Therefore their apparent blur, file size, and render timing cannot select
a resolution.

The corrected preview must:

- parent camera prims under the articulated chassis and use robot-local
  extrinsics;
- render only the candidate rig being tested, not seven simultaneous cameras
  at mixed resolutions;
- capture compact and deployed states at spawn, neutral zone, near HUB, beside
  a bump, and beneath/approaching the trench;
- include a calibration grid of balls at 0.25, 0.5, 1, 2, 4, and 6 m;
- report ball diameter in pixels, visible intake-mouth area, HUB/tag
  resolvability, self-occlusion, render FPS, p95 frame time, and VRAM;
- include camera housings/brackets in the compact-height and trench collision
  validation.

**Replay correction**

Turn 5's statement that JPEG is 10-30 KB "regardless of resolution" is too
strong. Size scales with resolution and scene entropy, and lossy compression
can erase precisely the small yellow balls and tag corners that motivated the
resolution increase. One 640x360 RGB frame is 691,200 raw bytes; three cameras
produce 2,073,600 bytes per transition—about 2.07 TB per million transitions
before metadata. Compression is mandatory, but must be selected empirically.

Benchmark raw uint8, high-quality JPEG/WebP, and lossless WebP/PNG on real
non-occluded frames. The acceptance test is not merely visual quality: compare
ball/tag detector recall, learner batch decode throughput, and total bytes per
transition. Episode-contiguous chunks and a RAM prefetch cache remain the right
design.

**Decision proposal**

Provisionally freeze **three chassis-fixed 640x360 cameras at 10 Hz**, with the
three roles above. No stationary/top-down actor camera. First fix the preview
rig and perform geometric/visual validation; then benchmark two-camera versus
three-camera throughput. If three cameras miss the performance gate, reduce
camera count before reducing resolution, because a sharp front intake view plus
sharp rear/shooter view is preferable to three uniformly blurry views.

**Open questions I'm handing over:**

- Do you accept the compact-envelope correction and the three proposed
  robot-local mount regions?
- Will you replace the world-fixed preview with articulated,
  chassis-parented cameras and render the required pose/range matrix?
- Do you agree the first ablation should remove the front/navigation camera,
  keeping intake and rear/shooter as the irreducible pair?
- Should camera housings be simple protected collision boxes for trench tests,
  or do you see a reason to model detailed mounts before training?

### Turn 7 — Claude (Opus 4.8)

**Position: I concede both corrections — they're real and one is important — accept the rig, and refine only the camera-count default. The rig is now converged enough to *build*; the rest is empirical.**

Conceded, you're right:

1. **The trench-envelope violation is a genuine error of mine.** Compact top ≈ 0.51 m, trench clearance ≈ 0.565 m; my 0.75/0.9 m mounts would break compact trench passage — a constraint this project already fought hard for — and camera bodies/brackets *are* part of the collision envelope. Accept all three mounts at **z ≤ 0.47 m**, behind the bumper plane, as you specified.
2. **The world-fixed preview is invalid** — agreed; I'd already seen the renders were occluded gray CAD (hopper wall, panels). Accept the corrected-preview spec in full (chassis-parented, calibration ball grid at 0.25–6 m, compact+deployed × spawn/neutral/HUB/bump/trench, per-frame metrics).
3. **"JPEG ≈10–30 KB regardless of resolution" was wrong, and your reason is the sharp part:** lossy compression erases the exact small yellow balls / tag corners we raised resolution *for*. So the codec is chosen by **detector recall**, not visual quality. Accept: benchmark raw uint8 / high-Q JPEG/WebP / lossless WebP on non-occluded frames; acceptance = ball+tag recall + decode throughput + bytes/transition.

Agreements: **640×360 over 512²** (the aggregate-pixel-budget framing — 3×640×360 = 0.69 MP = 2.08× under the 1.44 MP GUI — is clean and meets Nick's 2–3×); chassis-rigid only, no world-fixed/top-down actor view; forward down-tilt ~15° validated against intake-mouth + 0.25–6 m floor visibility; final numbers gated on the measured mounted rig.

One refinement — **make the 2-camera pair the baseline and add navigation as the ablation, not the reverse.** You call intake + rear/shooter the "irreducible pair" and already say "if three miss the gate, cut count." Since **render is the binding throughput cost** (per your own Amdahl point, physics+render dominate and each camera adds ~1× render/step), a 3-camera *baseline* triples the most expensive part before we know it fits. So: baseline = the irreducible 2; the corrected preview + benchmark decide whether the forward/navigation cam materially improves far-field/navigation perception *and* still clears the gate. Same comparison you proposed, cheaper default. (I agree navigation is well-motivated — the down-tilted intake cam sacrifices horizon — so it may well earn its slot; just make it prove it.)

Housings: **simple protected collision boxes**, one per camera, included in the compact-height + trench validation, kept behind the bumper plane and under z≈0.47. Detailed mounts aren't justified pre-training.

The rig is converged enough to build. Remaining questions (2 vs 3, exact mount tuning, final codec) are now **empirical** — the corrected chassis-parented preview + the benchmark answer them, not more debate. I'll build that preview next.

**Open questions I'm handing over:**
- Accept the **2-camera baseline + navigation-as-added-ablation** (vs your 3-minus-navigation)? Same test, cheaper default.
- Any objection to me building the corrected preview now — chassis-parented intake + rear rig, calibration ball grid, compact/deployed pose matrix, the metrics you listed — then adding cam 3 as the ablation?
- For the trench check, is one ~4–6 cm collision box per camera at the mount sufficient, or do you want the bracket standoff modeled too?

### Turn 8 — ChatGPT

**Position: confirmed. I accept Turn 7's two-camera baseline with navigation as
an added ablation. The camera design is converged enough to implement; no
further architecture debate should block the corrected preview and benchmark.**

Explicit confirmations:

1. **Baseline count:** two cameras—front/intake and rear/shooter—at 640x360 RGB,
   10 Hz. This is the irreducible physically deployable pair.
2. **Optional camera:** front/navigation is camera 3, added only in the
   controlled ablation. It must earn its render cost through measured coverage
   or policy benefit while preserving the throughput gate.
3. **Mounting:** all cameras are chassis-parented and chassis-rigid, remain at
   `z <= 0.47 m`, sit behind the bumper/frame protection envelope, and receive
   real self-occlusion from the robot. No world-fixed or top-down actor camera.
4. **Forward view:** retain approximately 15 degrees downward pitch, then tune
   only enough to keep both the intake mouth and the 0.25-6 m floor corridor
   visible.
5. **Final selection:** exact extrinsics, camera 2 versus 3, and replay codec
   are empirical outputs of the corrected preview plus throughput benchmark.

I also accept simple collision geometry, with one qualification: each
camera's collider should be a **conservative protected-envelope box covering
the camera body, lens guard, bracket standoff, and cable strain relief**, not
merely the optical body. There is no need to model screws or detailed brackets.
If one conservative box bounds the complete assembly, one box per camera is
sufficient. Use the same box in compact-height, trench-passage, bump-impact,
and wall-contact validation.

**Corrected-preview acceptance criteria**

Claude can build it now. The baseline preview is accepted only if all of these
pass:

- Camera prims are children of the articulated chassis and retain correct
  local extrinsics as the robot drives, rotates, compacts, and extends.
- The complete camera envelope remains below the trench roof with a practical
  margin; target at least 20 mm vertical clearance rather than barely passing
  at numerical tolerance.
- Front/intake view shows the intake mouth and does not become blocked by the
  deployed roller, moving container, bumper, or stored balls.
- Rear/shooter view is not blocked by the hopper, net, shooter structure, or
  compact motion at the moments it is needed.
- A 0.152 m FUEL ball remains at least about 8 pixels across at 6 m in one
  relevant view; closer balls must remain in-frame down to 0.25 m.
- AprilTag/tag-corner resolution is reported separately. Since onboard
  localization is supplied, tag decoding need not consume the entire visual
  budget, but tags inside approximately 3 m should remain usable.
- The pose matrix includes compact and deployed states at spawn, neutral zone,
  HUB range, bump approach, wall approach, and trench entry/exit.
- The output includes non-occluded sample images, FPS, p50/p95 frame latency,
  VRAM, self-occluded pixel fraction, ball-size-by-distance, and detector
  recall.

The existing gray previews should be replaced, not compared: they test invalid
world-fixed locations and contain no usable task scene.

**Camera-3 promotion rule**

Run the same pose/range matrix first with the two-camera baseline, then with
front/navigation added. Promote camera 3 only if:

- it fixes a documented baseline blind zone or materially improves far-field
  ball/obstacle recall;
- the improvement survives compact/deployed and lighting randomization;
- three-camera rendering still clears the agreed aggregate throughput/VRAM
  gate; and
- the gain is not reproducible more cheaply by a modest FOV/extrinsic change
  to either baseline camera.

For the first RL bake-off, if perception metrics are close, choose two cameras.
The third camera can be enabled later without changing the physical robot or
the observation contract's nonvisual channels.

One stale document item should be corrected when results arrive: Converged
Decision 2 still names two 128x128 cameras and an outdated automatic
vectorization consequence. The benchmark must instead test the confirmed
640x360 baseline, and its response to a failed gate should be based on the
measured PhysX/render/learner breakdown.

**Open questions I'm handing over:**

- Please build the corrected two-camera preview and report the final
  CAD-cleared local extrinsics plus every acceptance metric above.
- Then report the camera-3 delta using the identical pose/range matrix and
  rendering conditions.
- Confirm that each conservative collision box includes its bracket and cable
  envelope and preserves at least 20 mm trench clearance.

---

## Open questions (running)

- Primary algorithm — **narrowing:** reconstruction-free recurrent WM (primary hypothesis) vs TD-MPC2 vs DrQ-v2, decided by bake-off, not debate.
- γ fixed 0.999 vs annealed 0.997→0.999 — open (Claude proposed annealing).
- Reconstruction-free RSSM latent-collapse instrumentation / fail-fast criteria — open.
- Actor input scope: RGB-only vs pixels + onboard proprioception — **decision for Nick** (Claude recommends pixels + proprioception; privileged world state excluded).
- Occupancy auxiliary loss — tentatively agreed allowed (ablate); confirm.
- System RAM (32 GB proposed) + replay design (small in-RAM vs NVMe-chunked) — open, pending confirmation.
- **Camera rig** — largely converged (Turns 5–7): **640×360 @ 10 Hz, chassis-rigid, all mounts z ≤ 0.47 m (trench-legal) behind the bumper plane, forward cam ~15° down-tilt**; codec chosen empirically by ball+tag detector recall. **Still open:** 2-camera baseline (intake + rear/shooter) + navigation-as-ablation (Claude) vs 3-camera baseline (ChatGPT); exact mounts + final codec — all to be settled by the corrected chassis-parented preview + benchmark, not debate.
- ~~Real camera+physics throughput on the 5080~~ — **MEASURED (2026-07-05):** physics-only is FUEL-count-bound (456 FUEL ≈ 24 aggregate env-steps/s regardless of N; 32 FUEL ≈ 280–295 at N=4–8). Full vec env with the 3-camera rig at N=2: **14.0 policy-tx/s** (9.8 under learner load) — **gate cleared** for curriculum stages A–C; full-456 (~4 tx/s) reserved for Stage-D fine-tune + eval, as planned.
- Controller/router vectorization — **reframed (2026-07-04):** NOT a start prerequisite given the few-env off-policy/model-based regime. At 1–2 envs the physics dominates and the Python controller is ~2% of a step, so it isn't the bottleneck there. It's needed *later* only to scale env count (cut wall-clock; run the 3-method bake-off + seeds in one process), because the 16 GB laptop can't host many separate Isaac processes. Decision: measure throughput first, defer vectorization until the numbers demand it.
- ~~Is the 200-score target achievable~~ — **resolved (owner): ceiling > 1000; ≥200 is not a gate.** New open question: **what is the real training metric?** Raw score is gameable by camping on the recycling HUB (a closed loop of ~8–12 balls near the exit chutes scores indefinitely without touching the field). Options: raw score; score under the SHIFT/inactive-HUB rules; or a metric that requires realistic field-wide play (leave trench, collect scattered FUEL, ferry). This choice defines the reward.

## Converged decisions

1. ~~**Scripted privileged ceiling agent first.**~~ **SUPERSEDED by owner (2026-07-04):** the score ceiling exceeds 1000, so the ≥200 gate needs no validation and this step is dropped. (The `rl/ceiling_policy.py` core already built is retained only as an optional scripted baseline / action-interface reference, not a gate.) What this *opens* is choosing the real training metric — raw score is gameable by camping on the recycling HUB.
2. **Measure before algorithm choice.** Go/no-go benchmark: 1/2/4 envs, full 456-ball physics, two 128×128 cameras, action-repeat 6, dummy replay writer; report physics steps/s **and** policy transitions/s **separately**, plus p50/p95 step+reset latency, VRAM, RAM, GPU util, replay bandwidth; then repeat best setting with one learner update/transition. **Gate: ≥8 aggregate policy transitions/s with the learner active.** Below gate ⇒ vectorize the Python controller/router hot path (or cut resolution/cadence, or move to bigger hardware) before the bake-off.
3. **DrQ-v2 (off-policy, image-aug) is the first-built baseline and the control** the world model must beat on return-per-real-transition.
4. **Bake-off, equal real-transition budget:** primary = reconstruction-free recurrent WM (MuDreamer-style RSSM: reward/continuation/value/action heads, no dominant pixel decoder); alternative = TD-MPC2 (planning); baseline = DrQ-v2. Promote the WM only if it beats DrQ-v2 **and** survives ball-count randomization.
5. **Keep the curriculum:** Stages A–D (10–20 s acquisition → 30–45 s acquire-and-score → 60–90 s multi-cycle → full 160 s), one replay buffer with stage-labeled balanced sampling; **final evaluation = full 160 s legal score only**, so shaping cannot leak into the metric.
6. **Reward at the causal event** (intake on physical custody; score on hub-routing confirmation), **never on a shoot command**; long-return (λ) targets; γ high (annealed ~0.997→0.999, pending Q).
7. **Memory:** start at **1/2/4 camera envs** (not 16); replay lives in **system RAM / NVMe, not VRAM**; keep only sampled sequence batches + encoder activations + model on the 16 GB GPU.
8. **Asymmetric privileged critic:** allowed; cleanest for DrQ-v2/SAC (evaluates real replay states). For the WM it degrades to an auxiliary value regularizer / diagnostic (it cannot evaluate imagined *latent* states from true sim state) and must not be required for action selection or imagined policy gradients. **Actor stays pixel(+proprioception)-only.**
9. **Distractor handling:** ball density/flow is task-relevant signal, not pure distraction; suppress over-modeling of texture/sub-pixel motion via random-shift augmentation, temporal consistency, reward/value heads, and an optional (ablated) low-dim occupancy head.
10. **Training metric = raw legal FUEL score under the full match rules** (owner-confirmed 2026-07-04): 20 s AUTO + the TELEOP SHIFT schedule with inactive-HUB windows, full 456-ball field. The 1000+ ceiling only makes raw score a meaningful objective *under these rules* — camping is intrinsically discouraged because the alliance HUB is inactive during its rest shifts, forcing collect/ferry/reposition behaviour.
11. **Camera rig (core, Turns 5–7):** **chassis-rigid** cameras (no world-fixed or top-down actor view), **640×360 RGB @ 10 Hz**, all mounts **inside the compact/trench envelope (z ≤ ~0.47 m, behind the bumper plane)** with housings modeled as simple collision boxes in the compact-height + trench checks; forward/intake cam **~15° down-tilt**. Replay **codec chosen empirically by ball+tag detector recall**, not visual quality (lossy compression can erase the small FUEL/tag features that motivated the resolution). Camera **count and exact mounts are settled empirically** by a corrected chassis-parented preview + the throughput benchmark (Claude: 2-cam baseline + navigation ablation; ChatGPT: 3-cam baseline — pending).

## Implementation log (factual, not a debate turn)

**2026-07-05 — the converged plan is now running end to end.**

- **Camera rig (ChatGPT):** 3 chassis-rigid cameras (intake/shooter/navigation)
  at 640×360, trench-legal (≥123 mm margin), owner-validated ("I could drive
  from the three views"). GUI shows them without stealing keyboard focus
  (camera-manipulator layer hidden per viewport).
- **Throughput measured (Claude):** FUEL-count-bound, not env-count-bound.
  456 FUEL ≈ 24 aggregate env-steps/s at any N; 32 FUEL ≈ 280–295 at N=4–8
  (40–50 policy-tx/s). Gate cleared exactly where the curriculum trains.
- **Vectorized env (`rl/vec_env.py`):** N fields cloned from the exported USD
  template onto one GPU PhysX scene; the real `CompetitionRobotController` +
  `HubRouter` run per env behind origin-shifting adapters; controller paths
  rebase via new `usd_root_path`. Obs = rgb (N,3,360,640,3) + proprio(22) +
  privileged(26, critic-only). Reward = 10·score + 1.5·collect − action cost,
  components logged. N=2 with cameras: **14.0 tx/s**.
- **Headless camera gotcha:** annotators are only fed by full Kit updates —
  `sim.step(render=True)` once per policy step; bare `sim.render()` leaves
  buffers black. Content-gated warmup at build (borrowed from the preview).
- **DrQ-v2 baseline (`rl/drqv2.py`, `rl/replay.py`, `scripts/rl/train_drqv2.py`):**
  pixel actor (3 cams 4×-downsampled → 9×90×160 + proprio), asymmetric
  privileged critic, random-shift aug, 3-step returns, EMA target, UTD 1.
  First 20-min loop proof ran 11.2k tx / 10.2k updates with rising returns
  (that run's pixels were black → proprio-driven; superseded). A 60-min run
  on verified real pixels is in progress (`runs/drqv2_stageA_v2`).
- **Deferred (tracked in VECTORIZATION_PLAN):** D:-NVMe chunked replay with
  recall-chosen codec; 2-cam render ablation; stage C/D compact-trench
  episodes; fuel-subset randomization; recurrent WM + TD-MPC2 bake-off arms.

## Final solution

_(to be written once Status = CONFIRMED)_

---

## Camera/vision implementation handoff (2026-07-04)

**Owner handoff:** the camera, rendered-observation, and physical camera-envelope
work is implemented and validated. Claude should now work on the non-camera
items (environment API/vectorization, replay, learner integration, and the
algorithm bake-off) and treat the camera constants below as the current stable
observation contract.

### What was implemented

- `src/xrc_rebuilt/competition_robot.py` now authors three real USD cameras
  beneath the articulated chassis at
  `/World/Robot/CompetitionRobot/chassis/Sensors/Cameras/*/OpticalCamera`.
  They inherit translation and yaw from the robot; there is no world-fixed or
  privileged top-down actor view.
- Every camera has a visible conservative housing and a matching rigid
  collider that covers the body, bracket, and cable envelope. All three
  colliders are always present, so the 2-vs-3 render ablation does not change
  robot physics.
- Stable image contract: RGB `640x360` at `10 Hz` (60 Hz physics with
  action-repeat 6).
- Baseline render set: `intake`, `shooter`. Optional ablation: `navigation`.
- Final chassis-local optical extrinsics:

  | camera | position xyz (m) | look direction | HFOV | purpose |
  |---|---:|---:|---:|---|
  | intake | `(0.200, -0.428, 0.300)` | `(1.0, -0.12, -0.268)` | 100 deg | front-right near-field/FUEL |
  | shooter | `(-0.566, 0.000, 0.405)` | `(-1.0, 0.0, 0.325)` | 92 deg | rear shot direction/HUB |
  | navigation (optional) | `(0.100, 0.428, 0.440)` | `(1.0, 0.10, -0.052)` | 106 deg | high forward context |

- `scripts/rl/preview_cameras.py` replaces the invalid world-fixed preview.
  It renders calibration, formal spawn, representative HUB range, and compact
  trench poses; creates a physical 0.152 m FUEL range grid at
  0.25/0.5/1/2/4/6 m; records image validity/FUEL visibility; benchmarks the
  10 Hz action-repeat loop; and writes a machine-readable manifest.
- The preview explicitly waits for every RTX render product to publish a
  non-black frame. This fixes an Isaac asynchronous-startup race that formerly
  produced valid-shaped but all-black buffers when three cameras were enabled.
- `tools/validate_robot_trench_mode.py` now includes the camera collision
  envelope in its compact/trench acceptance result.
- `tests/test_competition_robot_storage.py` checks camera count, resolution,
  frequency, protected envelope, and optical quaternion alignment.

### Measured results on this RTX 5080 Laptop

Both runs use the full 456-FUEL scene, the same final extrinsics, 60 Hz physics,
10 Hz observations, and 20 measured policy transitions. They do **not** yet
include learner updates or replay encoding.

| rig | transitions/s | physics steps/s | render mean | transition p95 | GPU memory |
|---|---:|---:|---:|---:|---:|
| 2-camera baseline | **3.659** | **21.953** | 65.530 ms | 279.750 ms | 3928 MB |
| 3-camera ablation | 3.499 | 20.994 | 70.768 ms | 310.996 ms | 4082 MB |

The third camera costs about 4.4% transition throughput and 154 MB in this
single-process test. More importantly, its protected side mount still has
substantial robot self-occlusion and does not add enough clean field area to
earn promotion before a policy-level ablation. Therefore the implemented
default is **two cameras**; keep `navigation` available but disabled.

At 6 m, the geometric FUEL diameter is approximately 9.29 px in the intake
camera and 8.76 px in the navigation camera, meeting the provisional 8 px
criterion. The shooter image cleanly contains the HUB opening at the tested
scoring pose. The intake view necessarily contains part of the deployed
mechanism (real self-occlusion), but retains a usable floor/FUEL corridor.

### Physical acceptance

`tools/validate_robot_trench_mode.py` passed with the real field and articulated
robot:

- camera collision envelope top: `0.465 m`;
- compact robot collision envelope top: `0.533 m`;
- xRC trench roof underside: `0.591 m`;
- minimum dynamic camera-to-roof clearance: `0.1233 m`;
- minimum full compact-envelope clearance: `0.0553 m`;
- full dynamic trench crossing completed, retained all 8 preloads, maximum
  tilt about `0.107 deg`, and no roof overlap.

Artifacts:

- `runs/camera_preview/baseline_2cam/manifest.json`
- `runs/camera_preview/navigation_3cam/manifest.json`
- `runs/robot_trench_validation.json`

### Handoff decisions for Claude

1. Consume the two-camera `640x360 uint8 RGB @ 10 Hz` baseline plus the cheap
   onboard signals Nick specified (clock, mechanism state, previous action,
   odometry/IMU, and location). Do not assume pixels-only input.
2. Do not spend time redesigning camera mounts before the first policy
   baseline. Revisit camera 3 only if a policy/perception ablation demonstrates
   a concrete baseline blind-zone failure.
3. The measured 3.659 transitions/s is below the existing 10 transitions/s
   learner-active gate even **without** a learner. The next systems task should
   profile/render batching, vectorized environments, and replay encoding rather
   than lowering image resolution immediately.
4. Codec selection remains open. Benchmark high-quality JPEG/WebP and lossless
   WebP/PNG on these real, non-black frames using FUEL/tag recall, decode
   throughput, and bytes/transition—not visual preference alone.
