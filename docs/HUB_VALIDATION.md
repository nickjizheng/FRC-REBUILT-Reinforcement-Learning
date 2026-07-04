# HUB Routing Pipeline Validation

Stress test of the FUEL sensed → holding pen → delayed 4-exit release pipeline
(`HubRouter` in `src/xrc_rebuilt/isaac_scene.py`), run headless in Isaac Sim 5.1.

- Harness: `tools/validate_hub.py` (40 s sim, physics_dt 0.004, router stepped every
  0.02 s exactly like `run_sim.py`; 64 fuel bodies; robot removed to isolate routing).
- Raw results: `runs/hub_validation.json` (per-ball episodes + diagnosis).
- Geometry probe: `tools/analyze_hub_exit_geometry.py` → `runs/hub_exit_geometry.json`.

## Stress profile

116 idle floor balls teleported into the hub sensor volumes (alternating red/blue,
58/58) between t=1 s and t=30 s: one every 0.3 s, plus a 0.1 s-interval burst during
t=5–8 s that drove the holding pen to 14 simultaneous pending balls (mean 3.6).

## Results per check

| # | Check | Result | Numbers |
|---|-------|--------|---------|
| 0 | `rules.sample_hub_routing_delay` statistics (100k samples, pure python) | PASS | mean 1.3191 s (target 1.32 ± 0.01), all samples in [0.0016, 2.983] ⊂ [0, 3], histogram peak at 0.975 s ≈ mode 0.96 s |
| 1 | Sensed → released latency vs sampled delay | PASS | 116/116 released; latency − sampled delay ∈ [+0.00001, +0.0198] s, mean +0.0100 s — exactly the [0, 0.02) router-tick quantization; measured latencies 0.24–2.84 s, mean 1.239 s |
| 1a | Sensor reliability | PASS | 116/116 injections detected on the very next router tick (inject→detect = 0.02 s every time); 0 misses |
| 2 | Post-release trajectory clears the hub | **FAIL (marginal, 97.4%)** | 113/116 reached the neutral zone (\|y\| < 2.7): median 0.96 s, p95 2.82 s, max 5.68 s after release. 3/116 stuck on hub (z > 0.3, speed < 0.05 for > 2 s), all blue side (lane 3 ×2, lane 1 ×1). 0 tunneled below floor, 0 timeouts airborne |
| 3 | Re-detection while leaving (`blocked_until_clear`) | PASS | 0 re-detections in ~5,800 router steps, even though every released ball re-enters the sensor y-band ~0.04 s after release |
| 4 | Exit-lane uniformity | PASS | counts 35 / 23 / 24 / 34, χ² = 4.21 < 7.815 (df 3, p 0.05); red 18/12/10/18, blue 17/11/14/16 |
| 5 | Holding-pen containment & release exactness | PASS | max z while pending −1.968 m (never above floor, 0 escapes; pen balls just free-fall below the field between teleports); release pose dev ≤ 1.1e-7 m, release velocity dev ≤ 1.8e-9 m/s — exactly (0, ∓1.0, −0.08) |

**Overall: the pipeline works normally at scale — 97.4% of balls physically exit and
reach the neutral zone in ~1 s — but ~2.6% wedge permanently in the exit chutes.**

## Why balls get stuck (geometry + spin diagnosis)

1. **The exit chute is a millimeter-tight slot.** Along the ballistic release path
   from (EXIT_X, ∓3.4706, 1.02) at (0, ∓1.0, −0.08) m/s, every lane passes within
   **+1.3 mm** of the chute ceiling at t = 0.04 s (y = ∓3.431, z = 1.009); the
   lane-dependent lower cone lip leaves +2.7 mm (blue lane 3, the worst) to +27 mm.
   Contact during exit is *normal*, not exceptional.
2. **Residual spin + extreme friction turns grazes into random kicks.** The release
   teleport sets only the linear velocity; angular velocity accumulated in the
   overlapping holding pen is preserved. With the FuelFoam material (static μ 10,
   dynamic μ 5), a spinning ball grazing the +1.3 mm ceiling picks up a large random
   friction impulse — measured speeds 0.02 s after release were **1.35–2.85 m/s** vs
   the commanded 1.003 m/s. Most balls still power through; occasionally one deflects
   into a chute pocket and wedges (rest points (0.461, −3.476, 1.012),
   (0.375, −3.359, 0.967), (0.013, −3.394, 0.981), against faces with normals
   (±1,0,0) and (0, ∓0.93, 0.36)).
3. **The two hubs are 180°-rotated, not mirrored.** The tight internal web face sits
   near red lane 1 / blue lane 2 (the ballistic path intersects it by 21–24 mm
   mid-flight; balls bounce off it and still clear, but the slowest clears, 5.5–5.7 s,
   were blue lanes 1–2), while blue lane 3 has the tightest lower lip (+2.7 mm) —
   matching exactly where the wedges occurred.
4. **Stuck balls are permanently lost.** Two of the three rest *inside* the blue
   sensor volume, but `blocked_until_clear` only clears at z < 0.40 or |y| < 2.70,
   which a wedged ball never satisfies — so it can never be re-detected or re-routed.

## Recommended fixes (parameters/logic for the coordinator; `isaac_scene.py` not touched)

1. **Zero the angular velocity at release** (highest leverage): in the release branch
   of `HubRouter.step`, add
   `self.view.set_angular_velocities(np.zeros((1, 3), np.float32), indices=indices)`.
   This removes the random spin-friction kick at the +1.3 mm ceiling graze.
2. **Add an unstick watchdog**: for ~4 s after release, if a ball's speed < 0.05 m/s
   while z > 0.30 and |y| > 2.70 persists for > 1 s, re-queue it (back to the holding
   pen with a fresh delay and a different exit lane). Makes the pipeline lossless.
3. **Keep exit z = 1.02 and velocity (0, ∓1.0, −0.08)**: a grid search over
   z₀ ∈ [0.98, 1.02], v_y ∈ [1.0, 1.5], v_z ∈ [−0.08, −0.30] found *no* contact-free
   corridor (best worst-case in-chute clearance −13.6 mm at z₀ = 1.01, v_y = 1.2);
   the current values are near-optimal for the ceiling (+21.9 mm at spawn). Do not
   lower z below 1.00 — that increases lower-cone contact.
4. Optional: bind a lower-friction physics material to the hub chute region if any
   wedging persists after (1)–(2); effective fuel-vs-chute μ ≈ 5+ currently amplifies
   every graze.

## Repro

```bash
cd C:/Users/nickj/Desktop/xrc-rebuilt-robot-rl
OMNI_KIT_ACCEPT_EULA=YES /c/il/venv/Scripts/python.exe tools/validate_hub.py            # full 40 s run
/c/il/venv/Scripts/python.exe tools/validate_hub.py --delay-only                        # pure-python delay stats
/c/il/venv/Scripts/python.exe tools/analyze_hub_exit_geometry.py                        # chute clearance maps
```
