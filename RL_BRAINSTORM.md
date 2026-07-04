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
- **Reward is sparse-ish:** the primary signal is legal FUEL scored (hub has a 0.3–2.7 s stochastic routing delay before a score registers). Target milestone: **median ≥ 200 legal scores** over 100 held-out full-match seeds. (This target is **assumed, not yet validated** by any agent.)
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

---

## Open questions (running)

- Primary algorithm: TD-MPC2 vs DreamerV3 vs off-policy model-free — unresolved.
- Per-step planning (MPC) vs amortized policy under render cost — unresolved.
- Curriculum structure that survives the horizon without crutching final eval — unresolved.
- Distractor-ball world-model mitigation — unresolved.
- Real camera+physics throughput on the 5080 — **unmeasured** (blocking experiment).
- Is the 200-score target achievable at all — **unvalidated** (needs a scripted-ceiling agent).

## Converged decisions

_(none yet)_

## Final solution

_(to be written once Status = CONFIRMED)_
