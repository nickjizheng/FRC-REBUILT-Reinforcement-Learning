"""Uint8 ring replay for DrQ-v2 with n-step returns.

Stage-A scale: policy-resolution frames (C,H,W uint8) + proprio + privileged
vectors in a RAM ring (~130 KB/transition at 9x90x160).  The converged plan's
D:-NVMe episode-chunk store with an empirically chosen codec replaces the RAM
ring when capacity demands it (RL_BRAINSTORM.md Converged #7/#11) - the
sampling interface here is already chunk-friendly (contiguous, index-based).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray          # (B, C, H, W) uint8
    proprio: np.ndarray      # (B, P) float32
    privileged: np.ndarray   # (B, V) float32
    action: np.ndarray       # (B, A) float32
    reward: np.ndarray       # (B,) float32  (n-step discounted sum)
    discount: np.ndarray     # (B,) float32  (gamma^n, 0 where done inside n)
    next_obs: np.ndarray     # (B, C, H, W) uint8
    next_proprio: np.ndarray
    next_privileged: np.ndarray


class ReplayRing:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        proprio_dim: int,
        privileged_dim: int,
        action_dim: int,
        n_step: int = 3,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.proprio = np.zeros((capacity, proprio_dim), np.float32)
        self.privileged = np.zeros((capacity, privileged_dim), np.float32)
        self.action = np.zeros((capacity, action_dim), np.float32)
        self.reward = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, bool)
        self.size = 0
        self.cursor = 0

    def add(self, obs, proprio, privileged, action, reward, done) -> None:
        i = self.cursor
        self.obs[i] = obs
        self.proprio[i] = proprio
        self.privileged[i] = privileged
        self.action[i] = action
        self.reward[i] = float(reward)
        self.done[i] = bool(done)
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample_indices(self, batch_size: int) -> np.ndarray:
        # valid anchors: n_step successors must exist and not wrap the cursor
        high = self.size - self.n_step - 1
        idx = self.rng.integers(0, high, size=batch_size)
        if self.size == self.capacity:
            # avoid sampling across the write head
            idx = (self.cursor + 1 + idx) % self.capacity
        return idx

    def sample(self, batch_size: int) -> Batch:
        idx = self.sample_indices(batch_size)
        reward = np.zeros(batch_size, np.float32)
        discount = np.ones(batch_size, np.float32)
        next_idx = idx.copy()
        alive = np.ones(batch_size, bool)
        for k in range(self.n_step):
            step_idx = (idx + k) % self.capacity
            reward += alive * discount * self.reward[step_idx]
            discount *= np.where(alive, self.gamma, 1.0)
            terminated = self.done[step_idx] & alive
            next_idx = np.where(alive, (idx + k + 1) % self.capacity, next_idx)
            alive &= ~self.done[step_idx]
        bootstrap = np.where(alive, discount, 0.0).astype(np.float32)
        return Batch(
            obs=self.obs[idx],
            proprio=self.proprio[idx],
            privileged=self.privileged[idx],
            action=self.action[idx],
            reward=reward,
            discount=bootstrap,
            next_obs=self.obs[next_idx],
            next_proprio=self.proprio[next_idx],
            next_privileged=self.privileged[next_idx],
        )


class PerEnvReplay:
    """One ring per environment so n-step chains NEVER cross env streams.

    Transitions from different parallel envs are separate trajectories; a
    shared ring interleaves them and corrupts every multi-step target.  Each
    env writes to its own ring (episode boundaries inside a ring are handled
    by the stored ``done`` flags), and batches are drawn proportionally to
    ring fill.
    """

    def __init__(self, num_envs: int, capacity_per_env: int, seed: int = 0, **ring_kwargs):
        self.rings = [
            ReplayRing(capacity=capacity_per_env, seed=seed + 31 * i, **ring_kwargs)
            for i in range(num_envs)
        ]
        self.rng = np.random.default_rng(seed)

    def add(self, env_index: int, *args, **kwargs) -> None:
        self.rings[env_index].add(*args, **kwargs)

    def __len__(self) -> int:
        return sum(len(ring) for ring in self.rings)

    def ready(self, min_per_env: int) -> bool:
        need = min_per_env
        return all(len(ring) > ring.n_step + 1 for ring in self.rings) and (
            len(self) >= need
        )

    def sample(self, batch_size: int) -> Batch:
        sizes = np.asarray([max(0, len(r) - r.n_step - 1) for r in self.rings], float)
        probability = sizes / sizes.sum()
        counts = self.rng.multinomial(batch_size, probability)
        parts = [
            ring.sample(int(count))
            for ring, count in zip(self.rings, counts)
            if count > 0
        ]
        return Batch(
            **{
                name: np.concatenate([getattr(part, name) for part in parts])
                for name in Batch.__dataclass_fields__
            }
        )
