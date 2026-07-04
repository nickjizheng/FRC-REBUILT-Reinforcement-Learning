"""Stable policy contract shared by the future Isaac Lab environment.

This module deliberately has no Isaac Sim dependency, so action semantics and
timing invariants can be tested before a heavyweight simulation process starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_NAMES = (
    "forward",
    "strafe",
    "turn",
    "intake",
    "storage",
    "shoot_blue",
    "ferry",
)


@dataclass(frozen=True)
class CompetitionRLSpec:
    """Versioned contract for the first privileged-state PPO teacher."""

    version: str = "0.1"
    physics_hz: int = 60
    controller_hz: int = 60
    policy_hz: int = 10
    match_duration_s: int = 160
    action_dim: int = len(ACTION_NAMES)
    action_threshold: float = 0.25
    obstacle_ray_count: int = 24
    nearest_fuel_count: int = 24
    alliance: str = "blue"

    @property
    def physics_steps_per_action(self) -> int:
        return self.physics_hz // self.policy_hz

    @property
    def episode_policy_steps(self) -> int:
        return self.match_duration_s * self.policy_hz

    def validate(self) -> None:
        if self.physics_hz % self.policy_hz:
            raise ValueError("policy_hz must divide physics_hz exactly")
        if self.controller_hz != self.physics_hz:
            raise ValueError("the accepted robot controller currently runs every physics step")
        if self.action_dim != len(ACTION_NAMES):
            raise ValueError("action_dim does not match the stable action contract")
        if self.alliance != "blue":
            raise ValueError("the training environment is hard-locked to blue")


@dataclass(frozen=True)
class PolicyActionBatch:
    """Decoded batched intent consumed by the existing robot controller."""

    driver: np.ndarray
    intake_on: np.ndarray
    storage_extended: np.ndarray
    shoot_blue: np.ndarray
    ferry: np.ndarray


def decode_policy_actions(
    actions: np.ndarray,
    spec: CompetitionRLSpec | None = None,
) -> PolicyActionBatch:
    """Clip and decode a batch of seven normalized policy actions.

    The first three values are passed to ``CompetitionRobotController.drive`` as
    normalized forward/strafe/turn requests.  The existing controller retains
    the competition robot speed, acceleration, angular-rate, and wheel desaturation
    limits.  Shoot always means the blue HUB; there is no target-alliance action.
    """

    cfg = spec or CompetitionRLSpec()
    cfg.validate()
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != cfg.action_dim:
        raise ValueError(
            f"expected actions shaped (N, {cfg.action_dim}), got {values.shape}"
        )
    values = np.clip(values, -1.0, 1.0)
    threshold = cfg.action_threshold
    shoot = values[:, 5] > threshold
    # A scoring request wins if a noisy policy activates both heads.
    ferry = (values[:, 6] > threshold) & ~shoot
    return PolicyActionBatch(
        driver=values[:, :3].copy(),
        intake_on=values[:, 3] > threshold,
        storage_extended=values[:, 4] > threshold,
        shoot_blue=shoot,
        ferry=ferry,
    )
