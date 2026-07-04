# Timing audit — isaac_scene.py main loop (2026-07-02)

Audit of the simulation timing semantics in `src/xrc_rebuilt/isaac_scene.py`
against the actual Isaac Sim 5.1 source in `C:\il\venv`, verified empirically
with `tools/timing_probe.py` (full probe output: `runs/timing_probe_output.txt`).

## 1. Verified `SimulationContext.step()` semantics (from source + probe)

Source files read (Isaac Sim 5.1, isaacsim.core.api):

- `C:\il\venv\Lib\site-packages\isaacsim\exts\isaacsim.core.api\isaacsim\core\api\simulation_context\simulation_context.py`
- `C:\il\venv\Lib\site-packages\isaacsim\exts\isaacsim.core.api\isaacsim\core\api\physics_context\physics_context.py`
- `C:\il\venv\Lib\site-packages\isaacsim\extscache\omni.physx-107.3.26+107.3.3.wx64.r.cp311.u353\omni\physx\bindings\_physx.pyi`

**`step(render=False)`** (simulation_context.py lines 711–713) calls
`PhysicsContext._step(current_time)` which is (physics_context.py lines 563–565):

```python
self._physx_sim_interface.simulate(self.get_physics_dt(), current_time)
self._physx_sim_interface.fetch_results()
```

`IPhysxSimulation.simulate(elapsed_time, ...)` simulates *exactly* the elapsed
time passed, "No substepping will happen" (_physx.pyi lines 1445–1465). So one
`step(render=False)` = exactly ONE 0.004 s physics step.
**Probe exp1: 250 calls → Δcurrent_time = 1.0000000, Δsteps = 250.** Confirmed.

**`step(render=True)`** (simulation_context.py line 710) calls `self._app.update()`
— a full Kit update. `set_simulation_dt` (lines 407–471) put the Kit loop runner
in manual mode with step size = `rendering_dt` = 1/60 s (lines 456–461), and
computed `substeps = max(int(rendering_dt / physics_dt), 1) = int(4.1667) = 4`
(line 437), which `set_physics_dt(0.004, 4)` turned into
`TimeStepsPerSecond = 250` and carb `persistent/simulation/minFrameRate =
int(250/4) = 62` (physics_context.py lines 281–285). omni.physx therefore clamps
each app update to at most 1/62 s of simulated time → **exactly 4 physics steps
of 0.004 s (16 ms) per `step(render=True)`; the 0.667 ms remainder of the 1/60 s
frame is discarded, never accumulated.**
**Probe exp2: 60 calls → Δcurrent_time = 0.960 s, Δsteps = 240, histogram
{4 steps: 60 of 60 calls}.** Confirmed.

**`SimulationContext.render()`** (lines 730–747) is the render-WITHOUT-physics
primitive: it wraps `app.update()` with carb `/app/player/playSimulations=False`,
so it refreshes the viewport/UI and steps zero physics.

**`SimulationContext.current_time`** exists (property, lines 266–280) and is
physics-authoritative: `_physics_timer_callback_fn` (lines 1419–1422) subscribes
to physx step events and accumulates the actual `step_size` of every physics
step, including the 4 steps inside `app.update()`. `current_time_step_index`
counts steps. Both reset to 0 on timeline STOP (lines 1424–1427).

**Warm-start offset:** `sim.reset()` (stop→play→initialize) executes 2 physics
steps before the main loop starts. Probe: after `reset()`,
`current_time = 0.008, step_index = 2`. Same after any `stop(); play()`.

## 2. Bugs

### BUG 1 — GUI-mode clock skew of exactly 1.75x (HIGH, GUI mode only)

`isaac_scene.py` line 548:

```python
sim.step(render=(not args.headless and frame % 4 == 0))
```

with every loop iteration counted as 0.004 s (`frame * 0.004`, lines 551/553/576).

- Headless: every iteration = 1 step = 0.004 s. Assumption holds
  (probe exp4: 2500 frames → drift 4.7e-7 s; only the constant +8 ms
  warm-start offset separates `frame*0.004` from `current_time`).
- GUI: every 4th iteration advances **4** physics steps (16 ms), the other three
  advance 1 step each. Per 4 iterations: actual 7 steps = 0.028 s vs assumed
  0.016 s → **skew factor 1.75** (probe exp3, exact loop pattern replicated:
  assumed 1.600 s, actual 2.800 s, 700 steps, skew 1.7500000831).

Consequences in GUI mode:
- The match clock runs 43% slow: the UI's "AUTO 20 s" lasts 35 s of physics; a
  "160 s" match is 280 s of simulated time.
- `router.step(frame * 0.004)` (line 551): HUB routing delays (triangular,
  mean 1.32 s per docs/RULES.md) are stretched 1.75x in physics time → mean
  effective delay ≈ 2.31 s, violating the calibration and the official 3 s
  processing window.
- `sim_seconds` in `runs/physics_diagnostics.json` (line 576) over-reports by
  1.75x relative to nothing — it *under*-reports actual sim time by factor 1.75.
- Robot autopilot cadence (`frame % 5`, line 535) becomes 35 ms of sim time
  instead of 20 ms — GUI and headless runs are not physically comparable.

Secondary defect of the same line: because 1/60 is not an integer multiple of
0.004, the render path silently discards 0.667 ms per rendered frame (physics
advances 16 ms while the Kit timeline advances 16.667 ms), so even timeline
time and physics time drift apart. Avoid `step(render=True)` entirely with this
dt pair (or set `rendering_dt=0.016` if it is ever reinstated).

### BUG 2 — `frame * 0.004` used instead of the physics-authoritative clock (MEDIUM)

Lines 551, 553, 576 derive time from the loop counter. `sim.current_time` is the
authoritative clock (counts every actual physics step, survives any future
change to render cadence or dt) and is off from `frame*0.004` by +8 ms
(warm-start) headless and by 1.75x in GUI. All three sites should use
`sim.current_time` (optionally minus a baseline captured after `reset()` if a
zero-based match clock is wanted; HubRouter only uses differences, so the 8 ms
constant is harmless there).

### BUG 3 — HubRouter 20 ms polling: adequate headless, degraded in GUI (LOW)

`router.step` runs every 5th iteration. Headless that is every
5 × 0.004 = 0.020 s of sim time. At the stated worst case of 4 m/s a ball moves
0.08 m between polls; the sensor volume spans 1.12 m (x) × 0.32 m (y) ×
0.38 m (z), so any trajectory with an in-volume chord ≥ 8 cm is guaranteed to
be sampled at least once. Only corner-grazing chords < 8 cm can be missed —
and xRC itself samples its trigger at Unity's FixedUpdate (default 0.02 s =
50 Hz), so the headless cadence is *exactly* equivalent to the original game's
sensor sampling. Bonus: 5 × 0.004 s = one xRC FixedUpdate, a clean 1:1.
Verdict: **not a bug headless — keep it, it is calibration-faithful.**
In GUI mode, however, 5 iterations ≈ 8.75 physics steps ≈ 35 ms of sim time
(pre-fix), coarser than xRC and jittery (the poll lands after either 1 or 2
render frames). The BUG 1 fix below restores a uniform 20 ms cadence in both
modes. Release timing granularity (pending balls released up to one poll late)
is 20 ms — negligible vs the 0–3 s routing delays.

### Non-bug (a) — TimeStepsPerSecond=250 vs physics_dt=0.004: CONSISTENT (INFO)

`build_physics_scene` (line 445) sets `TimeStepsPerSecondAttr(250)` on
`/World/PhysicsScene`. `SimulationContext(physics_dt=0.004)` →
`PhysicsContext.__init__` *reuses* the existing scene (physics_context.py lines
64–82: it traverses the stage and adopts the first PhysicsScene found, ignoring
its default `/physicsScene` path), then `set_defaults` briefly sets dt=1/60
(line 116) before line 203–204 re-applies `physics_dt=0.004` →
`TimeStepsPerSecond = int(1/0.004) = 250`. Final state (probe config):
`TimeStepsPerSecond=250, get_physics_dt=0.004, minFrameRate=62,
loop_manual_step_size=1/60`. Consistent — but note the builder's attribute is
decorative: SimulationContext's `physics_dt` is authoritative and overwrites it
(along with gravity −9.81, TGS solver, CCD on, stabilization off — all
currently matching the builder's intent). If dt is ever changed, change it in
the `SimulationContext(...)` call, not the USD attribute.

### Note (d) — 250 Hz vs xRC's Unity fixed timestep (CALIBRATION NOTE)

docs/RULES.md does not state xRC's fixed timestep, and the decompiled
Assembly-CSharp does not override `Time.fixedDeltaTime` (the value would live in
Unity project settings/globalgamemanagers, not extracted) → xRC almost certainly
runs PhysX at the Unity default **0.02 s (50 Hz)**. This sim runs 250 Hz.
Effects on ball-behavior calibration:
- Bounce (restitution 0.5) is largely dt-independent in PhysX for clean impacts,
  but contact depenetration, rolling/sliding friction transitions, and stacking
  behave "stiffer"/more accurately at 250 Hz; balls settle faster and tunnel less
  (CCD is also on). Direction of error: Isaac balls will be slightly *better*
  behaved than xRC's, not worse.
- A likelier calibration gap than the rate itself: Unity's default
  bounce-threshold velocity is 2 m/s while omni.physx's scene default is
  0.2 m/s, so slow balls that stop bouncing in xRC keep micro-bouncing in Isaac.
  Worth a one-shot drop-test comparison (drop from 1 m, compare first/second
  bounce heights vs xRC) before trusting fuel scatter for RL.
- Keep 250 Hz (integer 5:1 against xRC's 50 Hz; RL-friendly control at any
  multiple of 4 ms); calibrate materials, not the rate.

## 3. Recommended patch (for the coordinator to apply to src/xrc_rebuilt/isaac_scene.py)

Never let the render path step physics; render via `sim.render()` (steps zero
physics) and drive all clocks from `sim.current_time`. After this patch, GUI and
headless advance identically (1 step = 4 ms per iteration), the 20 ms
router/autopilot cadences hold in both modes, and the discarded-remainder issue
disappears.

```diff
--- a/src/xrc_rebuilt/isaac_scene.py
+++ b/src/xrc_rebuilt/isaac_scene.py
@@ -531,6 +531,7 @@ def main() -> None:
         frame = 0
         drive_direction = 1.0
         max_robot_height = 0.0
+        match_t0 = sim.current_time  # reset() warm-start = 2 physics steps (8 ms)
         while app.is_running() and (args.frames <= 0 or frame < args.frames):
             if not args.no_autopilot and frame % 5 == 0:
                 robot_position, _ = robot_view.get_world_poses()
@@ -545,12 +546,18 @@ def main() -> None:
                 velocity_np[:, 0] = np.clip((1.52 - robot_position_np[:, 0]) * 2.5, -0.8, 0.8)
                 velocity_np[:, 1] = 1.05 * drive_direction
                 robot_view.set_linear_velocities(velocity_np)
-            sim.step(render=(not args.headless and frame % 4 == 0))
+            # step(render=True) runs a full Kit update: with rendering_dt=1/60 and
+            # TimeStepsPerSecond=250 it advances FOUR 0.004 s physics steps (16 ms),
+            # not one -- a 1.75x clock skew in GUI mode.  Always advance physics by
+            # exactly one dt, and refresh the viewport with sim.render(), which
+            # steps zero physics (playSimulations=False around app.update()).
+            sim.step(render=False)
+            if not args.headless and frame % 4 == 0:
+                sim.render()
             frame += 1
             if frame % 5 == 0:
-                router.step(frame * 0.004)
+                router.step(sim.current_time - match_t0)
             if status_labels and frame % 15 == 0:
-                elapsed = frame * 0.004
+                elapsed = sim.current_time - match_t0
                 if elapsed < 20:
                     phase = f"AUTO {20-elapsed:05.1f} s | Both HUBS active"
                 elif elapsed < 30:
@@ -573,7 +580,7 @@ def main() -> None:
         velocity_np = as_numpy(fuel_velocity)
         diagnostics = {
             "frames": frame,
-            "sim_seconds": frame * 0.004,
+            "sim_seconds": sim.current_time - match_t0,
             "fuel_count": len(fuel_end_np),
             "fuel_moved_over_5cm": int((np.linalg.norm(fuel_end_np - fuel_start_np, axis=1) > 0.05).sum()),
```

(The `fuel_mean_displacement_m` line between the last two context lines is
unchanged; hunk shown abbreviated — apply by the `sim_seconds` line.)

Optional hardening (not required once the patch is in): assert the contract at
loop end, e.g. `assert abs((sim.current_time - match_t0) - frame * 0.004) < 1e-3`.

## 4. Probe artifacts

- `tools/timing_probe.py` — self-contained headless experiment (boots Isaac,
  rebuilds the same physics-scene config, measures all four cases).
- `runs/timing_probe_output.txt` — full JSON results. Key numbers:
  - after reset: `current_time=0.008, step_index=2`
  - config: `physics_dt=0.004, rendering_dt=0.016667, TSPS=250, minFrameRate=62`
  - 250 × `step(render=False)`: 1.000 s / 250 steps
  - 60 × `step(render=True)`: 0.960 s / 240 steps (4 steps per call, all calls)
  - 400 iterations of the exact GUI loop pattern: assumed 1.600 s,
    actual 2.800 s / 700 steps → skew 1.75
  - 2500 headless frames: `frame*0.004` vs `current_time` drift 4.7e-7 s
