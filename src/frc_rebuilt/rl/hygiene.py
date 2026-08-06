"""Simulation hygiene: pure-numpy guards against PhysX blow-ups.

No Isaac imports — every function here is laptop-testable.  Production
evidence (2026-07-06, 4x RTX 4090): shared-scene PhysX either SIGSEGVs
(libomni.physx.plugin vector realloc) or goes non-finite and poisons every
subsequent observation.  The guards here are the prevention/detection half of
the containment design; ``VecCompetitionEnv`` owns the Isaac-side application.

Clamp bounds are derived from the robot/shooter specs (all cited values
verified in-code):

- ``MAX_WHEEL_SPEED_MPS = 4.59`` (swerve.py) -> robot linear 8.0 (1.7x).
- aim omega caps at 3-4 rad/s (competition_robot.py) -> robot angular 12.0.
- wheel joint at 4.59 / 0.051 m radius = 90 rad/s -> joint 150.0.
- max legit ball speed: ferry exit 11.0 m/s + chassis 4.59 -> fuel linear 16.0
  (score-mode calibration tops out at 8.37 m/s exit).
- rolling without slip at 16 m/s / 0.076 m radius = 210 rad/s -> fuel angular 250.

Poison thresholds mean "the clamp regime has already failed / the solver
injected energy": anything non-finite, positions far outside the field box, or
speeds at 2x their clamp bound (legit gameplay peaks at ~0.8x).
"""
from __future__ import annotations

import numpy as np

# -- clamp bounds (legitimate-gameplay ceilings with headroom) ----------------
ROBOT_LIN_CLAMP = 8.0     # m/s
ROBOT_ANG_CLAMP = 12.0    # rad/s
JOINT_VEL_CLAMP = 150.0   # rad/s
FUEL_LIN_CLAMP = 16.0     # m/s
FUEL_ANG_CLAMP = 250.0    # rad/s

# -- poison box (field-frame, per env; field is +-8 m) ------------------------
POS_XY_MAX = 14.0
POS_Z_MIN = -3.0
POS_Z_MAX = 6.0           # ferry apex ~4.7 m — thin but real margin
POISON_SPEED_FACTOR = 2.0

# -- hub-router holding pen ---------------------------------------------------
# Deterministic per-ball parking slots for FUEL captured by the hub, spaced so
# pen balls never touch (ball diameter 0.152 m) and NEVER leave the poison box:
# x in [9.0, 13.75] (< POS_XY_MAX), y in [0, 0.35*(F//20)] and z = -2.0
# (> POS_Z_MIN).  y stays < 2.7 for any template up to 160 balls, preserving
# the router's "on floor outside hub" funnel-entry discard for pen balls.
_PEN_COLS = 20


def pen_slot(index: int) -> np.ndarray:
    """Field-frame parking position for captured ball ``index``."""
    return np.asarray(
        [
            9.0 + (index % _PEN_COLS) * 0.25,
            0.35 * (index // _PEN_COLS),
            -2.0,
        ],
        dtype=np.float32,
    )


def pen_slots(indices: np.ndarray) -> np.ndarray:
    """(K,) template indices -> (K, 3) field-frame pen positions."""
    idx = np.asarray(indices, dtype=np.int64)
    return np.stack(
        [
            9.0 + (idx % _PEN_COLS) * 0.25,
            0.35 * (idx // _PEN_COLS),
            np.full(idx.shape, -2.0),
        ],
        axis=1,
    ).astype(np.float32)


def clamp_rows(vectors: np.ndarray, bound: float) -> tuple[np.ndarray, np.ndarray]:
    """Rescale rows of (R, 3) whose norm exceeds ``bound``.

    Non-finite rows are left untouched (they are detection's job, not the
    clamp's).  Returns ``(clamped, changed_mask)``.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    over = np.isfinite(norms) & (norms > bound)
    out = vectors.copy()
    if over.any():
        out[over] *= (bound / norms[over])[:, None]
    return out, over


def hygiene_verdict(
    robot_pos: np.ndarray,     # (N, 3) field-frame
    robot_vel6: np.ndarray,    # (N, 6) lin+ang, PRE-clamp values
    joint_vel: np.ndarray,     # (N, D) PRE-clamp values
    fuel_pos: np.ndarray,      # (N, F, 3) field-frame
    fuel_vel: np.ndarray,      # (N, F, 3) PRE-clamp values
) -> np.ndarray:
    """Per-env poison verdict: (N,) bool.

    Uses PRE-clamp speeds vs POISON_SPEED_FACTOR x the clamp bound: legit
    gameplay peaks around 0.8x each bound, so 2x means the solver injected
    energy this step regardless of what the clamp subsequently wrote.
    """
    robot_pos = np.asarray(robot_pos, np.float32)
    robot_vel6 = np.asarray(robot_vel6, np.float32)
    joint_vel = np.asarray(joint_vel, np.float32)
    fuel_pos = np.asarray(fuel_pos, np.float32)
    fuel_vel = np.asarray(fuel_vel, np.float32)

    poison = ~np.isfinite(robot_pos).all(axis=1)
    poison |= ~np.isfinite(robot_vel6).all(axis=1)
    poison |= ~np.isfinite(joint_vel).all(axis=1)
    poison |= ~np.isfinite(fuel_pos).all(axis=(1, 2))
    poison |= ~np.isfinite(fuel_vel).all(axis=(1, 2))

    # position boxes (NaN compares False; the finiteness checks above own NaN)
    with np.errstate(invalid="ignore"):
        poison |= np.abs(robot_pos[:, 0]) > POS_XY_MAX
        poison |= np.abs(robot_pos[:, 1]) > POS_XY_MAX
        poison |= robot_pos[:, 2] < POS_Z_MIN
        poison |= robot_pos[:, 2] > POS_Z_MAX
        fuel_out = (
            (np.abs(fuel_pos[..., 0]) > POS_XY_MAX)
            | (np.abs(fuel_pos[..., 1]) > POS_XY_MAX)
            | (fuel_pos[..., 2] < POS_Z_MIN)
            | (fuel_pos[..., 2] > POS_Z_MAX)
        )
        poison |= fuel_out.any(axis=1)

        lin_speed = np.linalg.norm(robot_vel6[:, :3], axis=1)
        ang_speed = np.linalg.norm(robot_vel6[:, 3:], axis=1)
        poison |= lin_speed > POISON_SPEED_FACTOR * ROBOT_LIN_CLAMP
        poison |= ang_speed > POISON_SPEED_FACTOR * ROBOT_ANG_CLAMP
        poison |= (np.abs(joint_vel) > POISON_SPEED_FACTOR * JOINT_VEL_CLAMP).any(axis=1)
        fuel_speed = np.linalg.norm(fuel_vel, axis=2)
        poison |= (fuel_speed > POISON_SPEED_FACTOR * FUEL_LIN_CLAMP).any(axis=1)
    return poison


class TripTracker:
    """Escalation bookkeeping for hygiene resets.

    Abort (-> the env raises ``SimPoisonedError``) when per-slot resets are
    demonstrably not curing the scene:

    - the same slot trips ``k`` times within ``window`` policy steps (a cured
      slot should stay healthy; fast recurrence = the shared scene is sick);
    - every slot trips on the same step (scene-wide event, per-slot resets
      provably insufficient — one shared PhysX scene);
    - a slot is STILL poisoned immediately after its reset + settle step;
    - trips occur on >= ``rate_limit`` of the steps in the window (rotating
      slots evading the per-slot k rule).
    """

    def __init__(self, num_envs: int, k: int = 3, window: int = 600, rate_limit: float = 0.25):
        self.num_envs = int(num_envs)
        self.k = int(k)
        self.window = int(window)
        self.rate_limit = float(rate_limit)
        self._trips: list[tuple[int, int]] = []       # (step, env)
        self._trip_steps: list[int] = []              # steps with >=1 trip
        self.total_trips = 0
        self.abort_reason: str | None = None

    def _prune(self, now: int) -> None:
        floor = now - self.window
        self._trips = [(s, e) for s, e in self._trips if s > floor]
        self._trip_steps = [s for s in self._trip_steps if s > floor]

    def record(self, step: int, env_indices, still_poisoned: bool = False) -> None:
        envs = [int(e) for e in env_indices]
        if not envs:
            return
        self.total_trips += len(envs)
        for e in envs:
            self._trips.append((int(step), e))
        self._trip_steps.append(int(step))
        self._prune(int(step))

        if still_poisoned:
            self.abort_reason = f"env(s) {envs} still poisoned after reset+settle"
            return
        if len(set(envs)) >= self.num_envs:
            self.abort_reason = f"all {self.num_envs} envs poisoned on step {step}"
            return
        for e in set(envs):
            hits = sum(1 for _, ee in self._trips if ee == e)
            if hits >= self.k:
                self.abort_reason = (
                    f"env {e} tripped {hits}x within {self.window} steps"
                )
                return
        if len(self._trip_steps) >= max(2, int(self.window * self.rate_limit)):
            self.abort_reason = (
                f"{len(self._trip_steps)} steps with trips in a {self.window}-step window"
            )

    def should_abort(self) -> bool:
        return self.abort_reason is not None

    def summary(self) -> str:
        return (
            f"SimPoisoned: {self.abort_reason or 'no abort condition'} "
            f"(total_trips={self.total_trips})"
        )
