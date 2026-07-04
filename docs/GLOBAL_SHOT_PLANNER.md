# Global shot planner (Phase D)

How legacy turns "score from anywhere" into a concrete, physically honest plan.
"From anywhere" means an **end-to-end scoring command from any reachable legal
pose** — not that every pose has a direct ballistic shot. Poses without a
robust direct shot drive to one.

Modules: `src/xrc_rebuilt/shot_planner.py` (solver + planner + calibration),
`src/xrc_rebuilt/field_map.py` (occupancy grid + A*). Ballistics come from the
single source of truth `robot_model.solve_shot_geometry`, the same function the
live auto-aim uses, so GUI/planner/RL never diverge (boundary #8).

## APIs (Phase D1)

```python
solve_direct_shot(robot_state, hub, field_state) -> ShotPlan | BlockedReason
plan_global_score(robot_state, hub, field_state) -> DriveAndShootPlan
```

`ShotPlan`: hub, aim, speed, dynamic pitch, muzzle pose, exit direction,
desired chassis yaw, flight time, range, vertical error, clearance margin,
calibrated uncertainty, `calibrated` flag, validity + reason.

`DriveAndShootPlan`: chosen firing pose, collision-free path, arrival
orientation, braking segment index, the final `ShotPlan`, and a replan policy.

## Direct-shot equations (exact xRC)

The shooter has **one** control parameter `aim ∈ [0,1]` (xRC `DisAim/100`):

- exit speed `v(aim) = 6 + 5·aim` m/s
- BS Outtake (3) pitch `= 70.355276 − 20·aim` deg
- the muzzle and push axis rotate about the extracted shooter pivot; world exit
  velocity adds chassis velocity (moving shots are a separately tested mode).

`solve_shot_geometry` sweeps `aim`, and for each computes the desired chassis
yaw that puts the HUB MarkBall opening on the muzzle bearing (two fixed-point
iterations absorb the muzzle's small lateral offset), then the projectile
vertical error at the opening under gravity `g = 9.81`:

```
t_flight   = ‖target_xy − muzzle_xy‖ / (v·‖dir_xy‖)
z_predicted = muzzle_z + v·dir_z·t_flight − ½·g·t_flight²
error       = z_predicted − target_z
```

The aim with least |error| is chosen. `solve_direct_shot` then gates it:

| reason | condition |
| --- | --- |
| `unsafe_side` | not on the open scoring side (blue from −y, red from +y, 0.45 m margin) |
| `under_range` / `over_range` | horizontal range outside the measured envelope 2.3–4.0 m |

The 2.3–4.0 m envelope is **measured**, not assumed: `tools/sweep_envelope.py`
fires through the full HUB geometry across range × approach angle for both HUBs
(`runs/shot_envelope.json`). Result: ~100% scoring at 2.25–2.75 m and on-axis
out to ~4 m; beyond ~4 m only favourable approach angles score (the HUB opening
favours a head-on approach), so the gate stops at 4.0 m rather than the ~5 m a
few on-axis shots reached. The shooter thus fires wherever a shot reliably
scores, not within an arbitrary narrow band.
| `no_trajectory` | best \|vertical error\| > 0.025 m within the 6–11 m/s / hood limits |

Analytic ballistics are a **seed**. The `calibrated` flag is `False` until a
matching PhysX calibration (below) is loaded; `clearance_margin_m` is then the
analytic range-envelope margin and `uncertainty_m` a conservative 0.05 m. The
true swept-sphere clearance vs HUB/net/tower and the measured dispersion come
from the PhysX pass.

## Feasibility map (Phase D3)

`feasibility_map` classifies every free floor cell for red and blue by
best-heading direct shot (heading is solved per cell, so no yaw sweep). Labels:
`direct`, `unsafe_side`, `under_range`, `over_range`, `no_trajectory`, and
occupied cells are omitted. The GUI overlay (Phase G) renders `direct` green,
drive-to-shoot amber, illegal/blocked red.

This analytic map is the seed. The **robust** evaluation required by D3 —
randomising ball friction/restitution/mass, solver timing, and launch speed and
keeping only cells whose worst case still scores — runs in PhysX and is
deferred to a GPU window. Measured rates will be reported honestly, never
labelled 100% unless they are.

## Path planner (Phase D4)

`OccupancyGrid` rasterises the **exact** extracted field colliders into a 2-D
grid (default 0.08 m), inflated by the robot footprint radius (0.42 m):

- includes static structural colliders (walls, HUB bodies, towers, sources,
  nets) whose world AABB intersects the chassis height band 0.12–0.60 m;
- **excludes** the floor and the traversable BUMPS/TRENCHES/ramps (the robot
  must cross those) and the dynamic FUEL spheres.

`plan_path` is 8-connected A* over the grid; it never crosses an occupied cell,
so it never tunnels through a structure (a straight line is never used).
`plan_global_score` fires directly if possible, else screens open-side ring
firing poses, and plans a collision-free path to the nearest reachable one,
returning a braking segment and replan policy. If nothing is reachable it
returns `unreachable` — never a blind shot.

Hybrid-A* over SE(2) is an acceptable refinement for tighter heading-aware
paths; the current grid A* is the collision-safe baseline.

## Drivetrain audit (Phase D4 — establish drive type before adding motion)

Established from the serialized `drive_params` in `robot_spec.json`, not
assumed:

- `is_FRC = 1`, `use_new_algorithm = 1`
- `SixWheelMotorScaler = 1.25`, `TankMotorScaler = 1.25`, `turn_priority = 1.0`
- **no mecanum/omni scaler is present**; six driven wheel links.

Conclusion: legacy is an **FRC six-wheel / tank differential (skid-steer)** drive.
It is **not holonomic** — there is no lateral strafe. The live model's
differential arcade drive (`RobotController.drive`, six velocity-driven
revolute wheels) matches this, and the baseline PhysX measurements (top speed
3.277 m/s, yaw 95.5 deg/s) validate it. The planner therefore produces
position paths that a differential controller follows with heading control;
**no strafe command is added.**

## Calibration versioning (Phase D2)

`CalibrationKey` = `{field_hash, robot_hash, ball_mass, ball_restitution,
ball_radius, physics_dt, isaac_version}`. `ShotCalibration.matches` refuses to
apply a LUT whose key differs from the current scene, so a calibration from a
different field/robot/physics build is never used silently. Artifacts live in
`runs/calibration/` with the key embedded. Until a PhysX pass populates the aim
offsets and per-hub uncertainty, the solver runs analytic-seed only.

## Failure modes & honesty

- `BlockedReason` gives a specific cause; the planner returns `unreachable`
  rather than firing a known miss.
- `calibrated=False` means the shot is an analytic seed, not PhysX-verified.
- Analytic `clearance_margin_m` is a range-envelope proxy, not swept-sphere.
- The occupancy grid is conservative (footprint-inflated AABBs), so it may
  reject a few truly-drivable poses; it will not accept an undrivable one.

## Deferred to a coordinated GPU window

D2 PhysX aim calibration (swept-sphere clearance, aim LUT, measured
uncertainty, red/blue independent + mirror check); D3 robust randomized
feasibility; D4 closed-loop drive-follow of planned paths; D5 acceptance matrix
(25 direct positions/HUB × 20 shots, 100 random legal starts/alliance, obstacle
adjacencies, 1/4/8-ball magazines, nominal + randomized physics).
