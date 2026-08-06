"""Prefix-takeover (Stage-E) production wiring.

Pure-Python / numpy building blocks for the candidate-suffix learner, kept free of
Isaac and (mostly) of torch so they are unit-testable on any machine:

* ``SuffixEmitter``   -- collector-side handoff tracking + SUFFIX-ONLY chunk emission
                         (prefix transitions are never written; natural handoff is
                         non-terminal; episode-end / forced-reset closes the stream).
* ``AnchorSampler``   -- champion anchor minibatches that EXCLUDE the frozen holdout
                         episodes and phase-balance the training anchors.
* ``PBRSShaper``      -- potential-based shaping F=γΦ(s')−Φ(s), Φ=0 at a true terminal,
                         phase-aware, telescoping (no persistent penalty / repeat bonus).
* ``AlphaSchedule``   -- TD3+BC α starts at 1.0, rises toward 2.5 only after two safe
                         held-out windows; ``check_alpha`` clamps/validates [1.0, 2.5].
* ``DriftGate``       -- runs the locked frozen-holdout drift check on a candidate before
                         publication; champion identity passes, regressions hard-stop.
* ``immutable_champion_ok`` -- SHA-256 guard so the champion is never overwritten.

The frozen champion is immutable and plays the prefix deterministically; only the
candidate suffix is trained and only suffix transitions ever reach replay.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# this module is src/frc_rebuilt/rl/prefix_takeover.py -> repo root is parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Immutable Stage-C champion (plays the prefix; never trained, never overwritten).
CHAMPION_PATH = str(PROJECT_ROOT / "runs/stageC_champion_998753.pt")
# Frozen holdout episode indices from anchor_holdout_frozen.json.
FROZEN_HOLDOUT_EPISODES = (8, 10, 15, 19, 22, 23, 25)


# --------------------------------------------------------------------------- #
# SuffixEmitter -- collector-side suffix-only chunk emission
# --------------------------------------------------------------------------- #
class SuffixEmitter:
    """Buffer and emit ONLY suffix transitions, per env, in temporal order.

    The caller feeds one env-step at a time with ``unloaded`` (True once this episode
    reached first-unload: score>0 ∧ magazine empty) and ``done`` / ``forced_reset``.
    An env enters suffix mode on the step AFTER ``unloaded`` first turns True -- i.e.
    the handoff tick H (which SETS unloaded) is the last prefix step and is NOT
    buffered, and H+1 (the candidate's first action) is the first buffered transition.
    A buffered transition is marked terminal iff the episode ends or is force-reset,
    so an n-step return can neither reach back into prefix (never stored) nor bridge
    into the next episode's suffix through the unwritten prefix gap.
    """

    def __init__(self, collector_envs: int, chunk_steps: int = 12):
        self.collector_envs = int(collector_envs)
        self.chunk_steps = int(chunk_steps)
        self._buf: dict[int, list[tuple]] = {e: [] for e in range(self.collector_envs)}
        self._in_suffix = [False] * self.collector_envs
        self._n = 0

    def in_suffix(self, e: int) -> bool:
        return self._in_suffix[e]

    def observe(self, e, obs, proprio, privileged, action, reward,
                unloaded: bool, done: bool, forced_reset: bool = False) -> None:
        """Feed one env-step for local env ``e``."""
        if self._in_suffix[e]:
            terminal = bool(done or forced_reset)
            self._buf[e].append(
                (np.asarray(obs), np.asarray(proprio), np.asarray(privileged),
                 np.asarray(action), float(reward), terminal))
            self._n += 1
            if terminal:                       # episode/stream ends -> exclude next prefix
                self._in_suffix[e] = False
        else:
            # enter suffix mode on the tick AFTER first unload (H -> arm, H+1 -> buffer);
            # never arm on a step that is itself terminal (degenerate unload-at-end).
            if unloaded and not (done or forced_reset):
                self._in_suffix[e] = True

    def ready(self) -> bool:
        return self._n >= self.chunk_steps

    def pending(self) -> int:
        return self._n

    def flush(self) -> dict[str, np.ndarray] | None:
        """Return one flat suffix chunk (grouped by stream, temporal order) or None."""
        if self._n == 0:
            return None
        obs, pro, priv, act, rew, done, stream = [], [], [], [], [], [], []
        for e in range(self.collector_envs):
            for (o, p, v, a, r, d) in self._buf[e]:
                obs.append(o); pro.append(p); priv.append(v); act.append(a)
                rew.append(r); done.append(d); stream.append(e)
            self._buf[e].clear()
        self._n = 0
        return {
            "obs": np.stack(obs).astype(np.uint8),
            "proprio": np.stack(pro).astype(np.float32),
            "privileged": np.stack(priv).astype(np.float32),
            "action": np.stack(act).astype(np.float32),
            "reward": np.asarray(rew, np.float32),
            "done": np.asarray(done, dtype=bool),
            "stream": np.asarray(stream, np.int32),
        }


# --------------------------------------------------------------------------- #
# AnchorSampler -- holdout-excluding, phase-balanced champion anchors
# --------------------------------------------------------------------------- #
def _ep_index(path: Path) -> int:
    return int(re.search(r"ep(\d+)", path.name).group(1))


class AnchorSampler:
    """Phase-balanced champion-anchor minibatches drawn from the anchor dump, with the
    frozen holdout episodes EXCLUDED (they are reserved for the drift kill-switch and
    must never be trained on -- otherwise the drift gate leaks into the objective)."""

    def __init__(self, anchor_dir, holdout_episodes=FROZEN_HOLDOUT_EPISODES, seed: int = 0):
        self.anchor_dir = Path(anchor_dir)
        self.holdout = set(int(i) for i in holdout_episodes)
        self.rng = np.random.default_rng(seed)
        frames, proprio, actions, phases, episodes = [], [], [], [], []
        self.train_episodes: list[int] = []
        for p in sorted(self.anchor_dir.glob("anchor_*.npz"), key=_ep_index):
            idx = _ep_index(p)
            if idx in self.holdout:
                continue
            self.train_episodes.append(idx)
            d = np.load(p, allow_pickle=True)
            n = d["frames"].shape[0]
            frames.append(d["frames"]); proprio.append(d["proprio"])
            actions.append(d["mean_action"]); phases.append(d["phase"].astype(str))
            episodes.append(np.full(n, idx, np.int32))
        if not frames:
            raise SystemExit(f"AnchorSampler: no non-holdout anchors in {anchor_dir}")
        self.frames = np.concatenate(frames)
        self.proprio = np.concatenate(proprio)
        self.actions = np.concatenate(actions)
        self.phase = np.concatenate(phases)
        self.episode = np.concatenate(episodes)
        # per-phase index pools for balanced sampling
        self._by_phase: dict[str, np.ndarray] = {
            ph: np.flatnonzero(self.phase == ph) for ph in np.unique(self.phase)
        }

    @property
    def phases(self) -> list[str]:
        return sorted(self._by_phase)

    def excludes_holdout(self) -> bool:
        return not (set(self.episode.tolist()) & self.holdout)

    def sample(self, batch_size: int):
        """Phase-BALANCED sample: split the batch as evenly as possible across phases,
        then draw within each phase. Returns (frames, proprio, mean_action)."""
        phs = self.phases
        base, extra = divmod(batch_size, len(phs))
        idxs = []
        for j, ph in enumerate(phs):
            k = base + (1 if j < extra else 0)
            if k:
                idxs.append(self.rng.choice(self._by_phase[ph], size=k, replace=True))
        idx = np.concatenate(idxs)
        self.rng.shuffle(idx)
        self._last_idx = idx        # exposed for balance introspection / tests
        return self.frames[idx], self.proprio[idx], self.actions[idx]


# --------------------------------------------------------------------------- #
# PBRS -- potential-based shaping (telescoping, phase-aware, Φ=0 at terminal)
# --------------------------------------------------------------------------- #
STAGE_C_GAMMA = 0.999   # scripts/rl/run_distributed.sh --gamma 0.999 (NOT the 0.997 learner default)


def assert_pbrs_gamma(pbrs_gamma: float, learner_gamma: float) -> None:
    """PBRS shaping is only policy-invariant when it uses the SAME γ as the learner's
    n-step targets. Silently using a different γ (e.g. the old 0.997 default vs Stage-C's
    0.999) breaks the telescoping invariance. Fail loudly."""
    if abs(float(pbrs_gamma) - float(learner_gamma)) > 1e-12:
        raise ValueError(f"PBRS gamma {pbrs_gamma} != learner gamma {learner_gamma}")


class PBRSShaper:
    """F = γ·Φ(s') − Φ(s), with Φ(terminal)=0 forced.

    Φ is supplied by the caller (a bounded legal-gateway-graph potential; the
    curriculum phase is part of its state).  Because F is a potential difference the
    shaped return telescopes to −Φ(s_0) at a true terminal, so shaping is
    policy-invariant: it adds NO persistent empty-home penalty and NO repeatable
    crossing bonus (a there-and-back loop nets to ~0).

    ``gamma`` is REQUIRED (no default): it must equal the learner's γ, so pass the
    composed-config value and check it with ``assert_pbrs_gamma`` (design note gamma footgun).
    """

    def __init__(self, gamma: float):
        self.gamma = float(gamma)

    def shaped(self, phi_s: float, phi_next: float, next_is_terminal: bool) -> float:
        phi_next_eff = 0.0 if next_is_terminal else float(phi_next)
        return self.gamma * phi_next_eff - float(phi_s)

    def trajectory_bonus(self, phis: list[float], terminal: bool) -> float:
        """Discounted Σ_t γ^t F_t over a trajectory whose potentials are ``phis``
        (Φ(s_0..s_T)); if ``terminal`` the last state's Φ is treated as 0.
        Telescopes to (γ^T·Φ_T^eff − Φ_0)."""
        total, disc = 0.0, 1.0
        for t in range(len(phis) - 1):
            term = terminal and (t + 1 == len(phis) - 1)
            total += disc * self.shaped(phis[t], phis[t + 1], term)
            disc *= self.gamma
        return total


# --------------------------------------------------------------------------- #
# Leave-only PBRS potential Φ_leave (exact contract)
# --------------------------------------------------------------------------- #
def _path_len(path) -> float:
    import math
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


@dataclass(frozen=True)
class LeaveGeom:
    """Geometry for the blue leave-only gateway set (metres, Isaac world XY)."""
    board_y: float = 2.775            # |y| scoring/neutral line (competition_robot.NEUTRAL_ZONE_HALF_Y_M)
    gateway_margin_m: float = 0.375   # gateway sits this far INTO neutral past the board (y=-2.4 for blue,
                                      # clear of the hub-ring cell inflation that occupies y~-2.575 lanes)
    # Physical lane centres shared with eval_phase_timing.py: two ramps and two
    # trenches.  The hub centre around x=0 is deliberately excluded.
    lane_x: tuple = (-3.3582, -1.55, 1.55, 3.3582)
    ramp_abs_x_max_m: float = 2.50  # ramps are the two inner legal lanes
    trench_route_penalty_m: float = 1.25
    # The penalty is used only when selecting the fixed PBRS target. A trench
    # remains a legal fallback, but must be materially shorter to beat a ramp.
    norm_distance_m: float = 6.0      # D: ONE fixed normalization (path length clips here)
    snap_radius_m: float = 0.6        # snap an inflated-but-drivable start cell to nearest free


class LeavePotential:
    """Φ_leave(s) = -clip(d_graph(s, g*)/D, 0, 1),  Φ=0 at the leave terminal.

    ``d_graph`` is collision-aware ``field_map.plan_path``
    length; ``g*`` is the nearest LEGAL outbound gateway, selected ONCE at handoff via
    ``select_gateway`` and held FIXED for the whole leave segment (a per-tick gateway
    would inject target-switching noise).  The potential is NEGATIVE distance, so moving
    toward g* raises Φ toward 0 (a positive distance would reward moving away).  No legal
    path to the fixed g* -> failure (``None``, flagged), never NaN/inf.  For v1 the
    gateway set is fixed (no randomization).  D is a single documented constant.
    """

    def __init__(self, alliance: str = "blue", geom: LeaveGeom = LeaveGeom(), grid=None):
        from frc_rebuilt.field_map import OccupancyGrid
        self.alliance = alliance
        self.geom = geom
        self.grid = grid if grid is not None else OccupancyGrid()
        sign = -1.0 if alliance == "blue" else 1.0        # blue own court is y<0
        gy = sign * (geom.board_y - geom.gateway_margin_m)  # blue: y ≈ -2.575 (just into neutral)
        # keep only gateways whose cell is free (an occupied gateway is not a legal target)
        self._gateways = [(float(x), float(gy)) for x in geom.lane_x
                          if self.grid.is_free(float(x), float(gy))]
        self._gstar: tuple[float, float] | None = None

    @property
    def gstar(self):
        return self._gstar

    def _snap_free(self, pos_xy):
        """A live chassis position often maps to an INFLATED-but-drivable cell (close to a
        wall); snap it to the nearest free grid cell within ``snap_radius_m`` so plan_path
        has a valid start. Deeper than that inside an obstacle -> genuine failure (None)."""
        x, y = float(pos_xy[0]), float(pos_xy[1])
        if self.grid.is_free(x, y):
            return (x, y)
        res = self.grid.resolution
        rings = int(self.geom.snap_radius_m / res)
        best, best_d = None, float("inf")
        for dr in range(1, rings + 1):
            for dx in range(-dr, dr + 1):
                for dy in range(-dr, dr + 1):
                    if max(abs(dx), abs(dy)) != dr:
                        continue
                    cx, cy = x + dx * res, y + dy * res
                    if self.grid.is_free(cx, cy):
                        d = (dx * dx + dy * dy) ** 0.5
                        if d < best_d:
                            best, best_d = (cx, cy), d
            if best is not None:
                return best
        return None

    def select_gateway(self, pos_xy) -> tuple[float, float] | None:
        """Pick g* = nearest gateway with a legal plan_path from ``pos_xy``; hold it fixed.
        Returns None (a leave failure) if NO gateway is reachable."""
        from frc_rebuilt.field_map import plan_path
        start = self._snap_free(pos_xy)
        if start is None:
            self._gstar = None
            return None
        best, best_cost = None, float("inf")
        for g in self._gateways:
            p = plan_path(self.grid, start, g)
            if p is None:
                continue
            d = _path_len(p)
            route_cost = d
            if abs(float(g[0])) > self.geom.ramp_abs_x_max_m:
                route_cost += self.geom.trench_route_penalty_m
            if route_cost < best_cost:
                best, best_cost = g, route_cost
        self._gstar = best
        return best

    def potential(self, pos_xy):
        """Return Φ_leave(s) ∈ [-1, 0], or None on a no-path failure. ``select_gateway``
        must have been called (at handoff) first."""
        from frc_rebuilt.field_map import plan_path
        if self._gstar is None:
            return None
        start = self._snap_free(pos_xy)
        if start is None:
            return None
        p = plan_path(self.grid, start, self._gstar)
        if p is None:
            return None
        d = _path_len(p)
        phi = -min(max(d / self.geom.norm_distance_m, 0.0), 1.0)
        if not np.isfinite(phi):        # never emit NaN/inf shaping
            return None
        return float(phi)

    def crossing_kind(
        self,
        pos_xy,
        *,
        clear_y_m: float = 2.50,
        lane_tolerance_m: float = 0.90,
    ) -> str | None:
        """Return ``"ramp"``/``"trench"`` for a legal clear crossing, else None.

        ``g*`` remains the fixed nearest lane used for PBRS shaping, but it must
        not invalidate a different physically legal ramp/trench route selected
        by the policy.  This separation is important for the compact robot,
        which may prefer the +X trench even when the occupancy graph says the
        +X ramp has a shorter centre-line path.
        """
        x, y = float(pos_xy[0]), float(pos_xy[1])
        crossed = y >= -clear_y_m if self.alliance == "blue" else y <= clear_y_m
        if not crossed:
            return None
        legal = [g for g in self._gateways if abs(x - g[0]) <= lane_tolerance_m]
        if not legal:
            return None
        gx, _ = min(legal, key=lambda g: abs(x - g[0]))
        return "ramp" if abs(gx) <= self.geom.ramp_abs_x_max_m else "trench"

    def is_legal_crossing(
        self,
        pos_xy,
        *,
        clear_y_m: float = 2.50,
        lane_tolerance_m: float = 0.90,
    ) -> bool:
        """Return whether ``pos_xy`` cleared the board through any legal lane."""
        return self.crossing_kind(
            pos_xy,
            clear_y_m=clear_y_m,
            lane_tolerance_m=lane_tolerance_m,
        ) is not None

    def reset_segment(self) -> None:
        self._gstar = None


# --------------------------------------------------------------------------- #
# CyclePotential — second-cycle phase-PBRS
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CycleGeom:
    """Geometry for the blue second-cycle phase potential (metres, Isaac world XY)."""
    board_y: float = 2.775
    band_in_y: float = -3.05          # hysteresis: "home" once y < this (blue own court)
    band_out_y: float = -2.50         # hysteresis: "away" once y > this (crossed toward neutral)
    gateway_margin_m: float = 0.375   # outbound gateway y = -2.4 (clear of hub-ring inflation)
    lane_x: tuple = (-2.55, -2.2, -1.55, -0.9, 0.9, 1.55, 2.2, 2.55)
    collect_xy: tuple = (0.0, 1.5)    # deep-neutral collect target ("go out to the field")
    inbound_xy: tuple = (1.5, -3.05)  # own court just past the board (return); x=0 is inside the hub
    hub_xy: tuple = (1.5, -3.6)        # hub scoring approach in own court (then raw score)
    leg_norm_m: float = 6.0           # per-leg graph-distance normalization (documented constant)
    snap_radius_m: float = 0.6


# each leg occupies a contiguous 0.25-wide band so Phi is CONTINUOUS + MONOTONE across leg
# transitions (design note): leave [-1.00,-0.75] -> collect [-0.75,-0.50] -> return
# [-0.50,-0.25] -> unload [-0.25, 0.00]. Reaching a leg's subgoal puts Phi at the next
# leg's base, so no spurious reward jump at the transition.
_CYCLE_BASE = {"leave": -1.00, "collect": -0.75, "return": -0.50, "unload": -0.25}
_CYCLE_SPAN = 0.25


class CyclePotential:
    """Per-env second-cycle phase-PBRS potential. Generalizes
    LeavePotential to the whole cycle AFTER the first unload: leave -> collect -> return ->
    unload. Phi in [-1, 0], a single monotone function of (leg, graph-distance-to-leg-subgoal)
    so F = gamma*Phi(s') - Phi(s) telescopes (policy-invariant). Phi = 0 BEFORE the first
    unload (the first cycle is already good under raw score — no shaping) and at cycle
    completion. Court-side uses hysteresis so the leg never flickers at the board; the
    outbound gateway is fixed once per leave-leg (design note). collision-aware plan_path;
    no-path -> None (flagged), never NaN. One plan_path per potential() call. Blue alliance.
    """

    def __init__(self, alliance: str = "blue", geom: CycleGeom = CycleGeom(), grid=None):
        from frc_rebuilt.field_map import OccupancyGrid
        assert alliance == "blue", "CyclePotential v1 is blue-specific"
        self.geom = geom
        self.grid = grid if grid is not None else OccupancyGrid()
        gy = -(geom.board_y - geom.gateway_margin_m)          # blue outbound gateway y = -2.4
        self._gateways = [(float(x), float(gy)) for x in geom.lane_x
                          if self.grid.is_free(float(x), gy)]
        self.reset()

    def reset(self) -> None:
        """New episode: clear the first-unload flag, leg state, and fixed outbound gateway."""
        self._unloaded_once = False
        self._leg: str | None = None
        self._side = "home"
        self._gstar_out: tuple[float, float] | None = None

    def note_unload(self) -> None:
        """Caller flags the FIRST unload (score>0 and magazine emptied) — shaping starts now."""
        self._unloaded_once = True

    @property
    def leg(self):
        return self._leg

    @property
    def gstar_out(self):
        return self._gstar_out

    def _snap_free(self, pos_xy):
        x, y = float(pos_xy[0]), float(pos_xy[1])
        if self.grid.is_free(x, y):
            return (x, y)
        res = self.grid.resolution
        for dr in range(1, int(self.geom.snap_radius_m / res) + 1):
            best, best_d = None, float("inf")
            for dx in range(-dr, dr + 1):
                for dy in range(-dr, dr + 1):
                    if max(abs(dx), abs(dy)) != dr:
                        continue
                    cx, cy = x + dx * res, y + dy * res
                    if self.grid.is_free(cx, cy):
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best, best_d = (cx, cy), d
            if best is not None:
                return best
        return None

    def _select_gateway(self, pos_xy):
        from frc_rebuilt.field_map import plan_path
        start = self._snap_free(pos_xy)
        if start is None:
            return None
        best, best_d = None, float("inf")
        for g in self._gateways:
            p = plan_path(self.grid, start, g)
            if p is not None and _path_len(p) < best_d:
                best, best_d = g, _path_len(p)
        return best

    def update_leg(self, pos_xy, mag_len: int):
        """Infer the current cycle leg from state (with board hysteresis). Returns the leg
        name, or None before the first unload (no shaping). Fixes the outbound gateway on
        entry to the leave leg."""
        if not self._unloaded_once:
            self._leg = None
            return None
        y = float(pos_xy[1])
        if y < self.geom.band_in_y:
            self._side = "home"
        elif y > self.geom.band_out_y:
            self._side = "away"
        # else: keep previous side (hysteresis band)
        loaded = mag_len > 0
        if not loaded and self._side == "home":
            if self._leg != "leave":
                self._gstar_out = self._select_gateway(pos_xy)   # fix g* once per leave-leg
            self._leg = "leave"
        elif not loaded and self._side == "away":
            self._leg = "collect"
        elif loaded and self._side == "away":
            self._leg = "return"
        else:                                                    # loaded and home
            self._leg = "unload"
        return self._leg

    def _subgoal(self):
        if self._leg == "leave":
            return self._gstar_out
        if self._leg == "collect":
            return self.geom.collect_xy
        if self._leg == "return":
            return self.geom.inbound_xy
        if self._leg == "unload":
            return self.geom.hub_xy
        return None

    def potential(self, pos_xy):
        """Phi in [-1, 0] for the CURRENT leg (call update_leg first), 0 before the first
        unload / at terminal, or None on a no-path failure (never NaN)."""
        from frc_rebuilt.field_map import plan_path
        if self._leg is None:
            return 0.0
        subgoal = self._subgoal()
        if subgoal is None:
            return None
        start = self._snap_free(pos_xy)
        if start is None:
            return None
        p = plan_path(self.grid, start, (float(subgoal[0]), float(subgoal[1])))
        if p is None:
            return None
        d = _path_len(p)
        phi = _CYCLE_BASE[self._leg] + _CYCLE_SPAN * (1.0 - min(max(d / self.geom.leg_norm_m, 0.0), 1.0))
        return float(phi) if np.isfinite(phi) else None


# --------------------------------------------------------------------------- #
# Aggressive multi-cycle reward (direct, NON-invariant) — user directive 2026-07-13
# --------------------------------------------------------------------------- #
# Reward emptying the chamber, penalize lingering home-side after an unload, and pay
# an escalating bonus per ADDITIONAL ordered cycle. Unlike the policy-invariant
# CyclePotential above, these terms deliberately BIAS the policy (invariance dropped).
# Pure Python (no Isaac); one instance per env, update() once per step in the collector
# with post-env.step state. Mechanic (1) is "bonus + gated penalty" (user's choice):
# a full-clear bonus (Term A1) PLUS a mild penalty for carrying a big undumped load
# back out to neutral (Term A2). Terms A2/B/C are gated on a productive-unload latch so
# the strong FIRST cycle is shaped only by the positive full-clear bonus — no penalty
# can touch it. Magnitudes grounded in the champion's real eval (streams ~46 balls over
# ~29 s; a literal dribble penalty would erode that, hence bonus + gated). Blue alliance.
@dataclass
class AggressiveCycleCfg:
    score_floor: float = 25.0          # arm A2/B/C once cycle 1 is banked (score>=, mag 0)
    # Term A1 — whole-chamber unload bonus (fresh balls, on a genuine full clear)
    atonce_weight: float = 1.0
    atonce_min_load: int = 4
    atonce_cap: float = 50.0
    # Term A2 — gated abandon penalty: cross OUT to neutral holding a big undumped load
    abandon_load: int = 8
    abandon_weight: float = 0.3
    # Term B — post-unload cross-over linger penalty. AGGRESSIVE (2026-07-14 user directive):
    # after the 1st chamber-empty (~19 s) the rest of the episode is wasted, and cycle-1 score
    # is already banked, so punish sitting home-side hard to force the bot back onto the ramp.
    linger_penalty: float = 0.35
    linger_grace_steps: int = 5
    # NEW per-leg progress bonuses (once per episode): reward the JOURNEY out->collect->return,
    # not just completion, so the policy discovers the 2nd cycle. Once-only => no oscillation farm.
    leave_bonus: float = 15.0          # first crossing OUT after the 1st unload (go on the ramp)
    collect_bonus: float = 15.0        # first deep-neutral collect after leaving
    return_bonus: float = 15.0         # first crossing back IN loaded
    # Term C — escalating multi-cycle completion bonus (ordered, stricter than the eval)
    mc_per_ball: float = 2.0
    mc_cap: float = 40.0
    mc_escalation: float = 1.2
    mc_episode_cap: float = 150.0
    mc_min_cycle_score: int = 2
    neutral_deep_y: float = -1.5
    # --- Term D: bounded time-decay ramps (2026-07-14 user directive). ALL default to
    #     no-op (slopes/pref 0, deadline disabled) so existing tests are untouched; the
    #     live values are supplied by run_aggressive.sh / collector flags. ---
    # D-A "idle-since-empty" ramp: pressure from confirmed cycle-1-dump-end until first
    #     deep-neutral collect, position-agnostic (pushes both leave-fast and collect-fast).
    core_slope: float = 0.0            # per-step growth (enable ~0.005); 0 => Ramp A OFF
    core_step_cap: float = 0.50        # per-step penalty ceiling (~5% of a 10-pt ball)
    core_grace_steps: int = 8          # no charge for the first N steps after the clock starts
    core_freeze_confirm: int = 15      # consecutive non-score-rise steps that "confirm" dump end
    core_freeze_cap: int = 30          # hard cap: start the clock this many steps post-latch regardless
    arm_deadline_steps: int = 10**9    # deadline-arm the latch if score>=floor but mag never 0 (OFF by default)
    # D-B/D-C later-leg dawdle ramps (lighter than D-A); charge only excess dwell per leg.
    rampB_slope: float = 0.0           # stage 2 collect->return (enable ~0.0125)
    rampB_step_cap: float = 0.25
    rampB_grace: int = 30
    rampB_budget: float = 10.0
    rampC_slope: float = 0.0           # stage 3 return->score (enable ~0.010)
    rampC_step_cap: float = 0.20
    rampC_grace: int = 40
    rampC_budget: float = 8.0
    ramp_episode_cap: float = 60.0     # shared per-episode backstop across all three ramps
    # D-pref ramp-lane preference (prefer-don't-force): scaled bonus on the PRESERVED base
    #     leave/return bonus, keyed off crossing-x near a ramp lane vs a trench lane.
    ramp_pref: float = 0.0             # extra bonus at a ramp lane (enable ~4.0); 0 => pref OFF
    ramp_center: float = 1.55          # |x| of the ramp lanes
    ramp_tol: float = 0.9              # lane half-width (1.0 at center -> 0 at the ramp/trench split)


class AggressiveCycleShaper:
    """Direct (non-invariant) aggressive multi-cycle reward. One per env; call update()
    once per step AFTER env.step with the post-transition state; returns the total shaping
    reward for that step. Bands match CycleGeom (blue): board at y=-2.775, "home" once
    y<=-3.05, "away" once y>=-2.50 (hysteresis). All state is per-episode; reset() on done.
    """

    BOARD_Y = -2.775
    BAND_IN = -3.05
    BAND_OUT = -2.50

    def __init__(self, cfg: "AggressiveCycleCfg | None" = None):
        self.cfg = cfg or AggressiveCycleCfg()
        self.reset()

    def reset(self) -> None:
        self._latched = False
        self._prev_mag = 0
        self._prev_score = 0
        self._fresh_at_last_empty = 0
        self._band = "start"
        # Term B home-dwell ratchet
        self._in_dwell = False
        self._dwell_yhigh = -1e9
        self._home_linger_steps = 0
        # Term C ordered ratchet
        self._stage = 0
        self._cycle_k = 1
        self._mc_paid_ep = 0.0
        self._fresh_at_out: int | None = None
        self._collected_at_out: int | None = None
        # --- second-cycle phase telemetry (steps; x0.1 = seconds) for the live monitor ---
        self._t = 0
        self._t_unload: int | None = None      # first productive unload (funnel entry)
        self._t_leave: int | None = None        # first crossing OUT after unload
        self._t_collect: int | None = None      # first deep-neutral collect after leaving
        self._t_return: int | None = None        # first crossing back IN loaded
        self._t_score2: int | None = None         # first 2nd-cycle score (cycle complete)
        self._max_stage = 0    # 0 none | 1 left | 2 collected | 3 returned | 4 scored-again
        self._n_leaves = 0
        # --- Term D: time-decay ramp state (all per-episode) ---
        self._core_off = False                 # set True at first deep collect -> Ramp A off for the ep
        self._core_t0: int | None = None       # step the idle clock started (post dump-end confirm)
        self._core_quiet = 0                   # consecutive non-score-rise steps since latch
        self._core_since_latch = 0             # steps since latch (freeze hard-cap)
        self._t_reach_floor: int | None = None # first step score>=floor (deadline-arm)
        self._leg_entered_t = 0                # step the current later-leg (stage 2/3) began
        self._rampB_paid = 0.0                 # per-leg B budget spent
        self._rampC_paid = 0.0                 # per-leg C budget spent
        self._ramp_paid_ep = 0.0               # shared episode ramp total (backstop)

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def cycles_completed(self) -> int:
        return self._cycle_k - 1     # 0 until the first ADDITIONAL (2nd) cycle completes

    def phase_report(self) -> dict:
        """Second-cycle phase timeline for the episode SO FAR (call before reset on done).
        Times are STEP indices (x0.1 = seconds). max_stage: how far the best 2nd-cycle got —
        0 none | 1 left the field | 2 collected in neutral | 3 returned loaded | 4 scored again."""
        return {
            "t_unload": self._t_unload, "t_leave": self._t_leave,
            "t_collect": self._t_collect, "t_return": self._t_return,
            "t_score2": self._t_score2, "max_stage": int(self._max_stage),
            "n_leaves": int(self._n_leaves), "cycles_completed": int(self.cycles_completed),
            "steps": int(self._t),
        }

    def _ramp_score(self, x: float) -> float:
        """Lane-preference weight in [0,1]: 1.0 crossing at a ramp lane (|x|=ramp_center),
        linearly decaying to 0 at the ramp/trench split, 0 at a trench lane. x=0 (default
        when the collector doesn't pass a position) => 0, so the base bonus is unchanged."""
        cfg = self.cfg
        d = abs(abs(float(x)) - cfg.ramp_center)
        return max(0.0, min(1.0, (cfg.ramp_tol - d) / cfg.ramp_tol))

    def update(self, mag, score, collected, fresh_score, y, done, x=0.0) -> float:
        cfg = self.cfg
        mag = int(mag); score = int(score); collected = int(collected)
        fresh_score = int(fresh_score); y = float(y); x = float(x)
        r = 0.0
        self._t += 1
        score_rose = score > self._prev_score
        cleared = self._prev_mag > 0 and mag == 0      # chamber went loaded -> empty this step

        # (1) productive-unload latch (arms A2/B/C/D; monotone within an episode). The
        #     deadline-arm safe harbor (mitigation #4) also latches a policy that reaches
        #     the score floor but refuses to ever empty, after arm_deadline_steps.
        if self._t_reach_floor is None and score >= cfg.score_floor:
            self._t_reach_floor = self._t
        if not self._latched and score >= cfg.score_floor and (
            mag == 0
            or (self._t_reach_floor is not None
                and self._t - self._t_reach_floor >= cfg.arm_deadline_steps)):
            self._latched = True
            self._t_unload = self._t                   # telemetry: funnel entry

        # (2) Term A1 — whole-chamber unload bonus (fresh balls; NOT latch-gated so the
        #     strong first dump earns it too). Fresh-only, so it never rewards recycling.
        if cleared:
            fresh_delta = fresh_score - self._fresh_at_last_empty
            if fresh_delta >= cfg.atonce_min_load:
                r += min(cfg.atonce_cap, cfg.atonce_weight * float(fresh_delta))
            self._fresh_at_last_empty = fresh_score

        # (3) hysteretic board band + outbound/inbound edges
        prev_band = self._band
        if y <= self.BAND_IN:
            self._band = "below"
        elif y >= self.BAND_OUT:
            self._band = "above"
        outbound = (prev_band == "below" and self._band == "above")
        inbound = (prev_band == "above" and self._band == "below")

        if self._latched:
            # (4) Term A2 — gated abandon penalty: carry a big load OUT to neutral (leaving
            #     a dump undumped). One-time per crossing; never fires in a healthy cycle
            #     (outbound legs are empty, returns are inbound).
            if outbound and mag >= cfg.abandon_load:
                r -= cfg.abandon_weight * float(mag)

            # (5) Term B — cross-over linger penalty (per home-dwell high-water ratchet).
            #     Flat/Markov; grace + freeze-while-scoring protect trailing landings;
            #     the ratchet kills the board-line oscillation dodge.
            if mag == 0 and y < self.BOARD_Y:
                if not self._in_dwell:
                    self._in_dwell = True
                    self._dwell_yhigh = y
                    self._home_linger_steps = 0
                progressing = y > self._dwell_yhigh
                if y > self._dwell_yhigh:
                    self._dwell_yhigh = y
                if (not progressing) and (not score_rose):
                    self._home_linger_steps += 1
                else:
                    self._home_linger_steps = 0
                if self._home_linger_steps > cfg.linger_grace_steps:
                    r -= cfg.linger_penalty
            else:
                self._in_dwell = False
                self._home_linger_steps = 0

            # (5b) Term D-A — "idle-since-empty" ramp (user's core directive). A bounded,
            #      position-agnostic per-step penalty that grows with idle time from the
            #      CONFIRMED end of the cycle-1 dump until the first deep-neutral collect
            #      (stage 1->2 sets _core_off). Crossing OUT (leave) does NOT stop it, so it
            #      pressures BOTH "leave fast" and "collect fast". Capped per-step and by the
            #      shared per-episode backstop; re-arms each completed cycle for cycle 3+.
            self._core_since_latch += 1
            if score_rose:
                self._core_quiet = 0               # a trailing in-flight landing resets the confirm
            else:
                self._core_quiet += 1
            if self._core_t0 is None and (
                self._core_quiet >= cfg.core_freeze_confirm
                or self._core_since_latch >= cfg.core_freeze_cap):
                self._core_t0 = self._t            # clock starts once the dump is confirmed finished
            if (not self._core_off) and self._core_t0 is not None:
                elapsed = self._t - self._core_t0
                pen = min(cfg.core_step_cap,
                          cfg.core_slope * float(max(0, elapsed - cfg.core_grace_steps)))
                pen = min(pen, cfg.ramp_episode_cap - self._ramp_paid_ep)
                if pen > 0.0:
                    r -= pen
                    self._ramp_paid_ep += pen

            # (5c) Term D-B/D-C — later-leg dawdle ramps ("and so on"), lighter than D-A.
            #      Read self._stage as-of TOP-of-step (Term C below may advance it this same
            #      step). grace ~= a clean leg's duration, so a promptly-run leg pays 0; only
            #      excess dwell is charged, up to a per-leg budget and the shared episode cap.
            if self._stage == 2:
                elapsed = self._t - self._leg_entered_t
                rate = min(cfg.rampB_step_cap,
                           cfg.rampB_slope * float(max(0, elapsed - cfg.rampB_grace)))
                pen = min(rate, cfg.rampB_budget - self._rampB_paid,
                          cfg.ramp_episode_cap - self._ramp_paid_ep)
                if pen > 0.0:
                    r -= pen; self._rampB_paid += pen; self._ramp_paid_ep += pen
            elif self._stage == 3:
                elapsed = self._t - self._leg_entered_t
                rate = min(cfg.rampC_step_cap,
                           cfg.rampC_slope * float(max(0, elapsed - cfg.rampC_grace)))
                pen = min(rate, cfg.rampC_budget - self._rampC_paid,
                          cfg.ramp_episode_cap - self._ramp_paid_ep)
                if pen > 0.0:
                    r -= pen; self._rampC_paid += pen; self._ramp_paid_ep += pen

            # (6) Term C — escalating multi-cycle ordered bonus. Ratchet mirrors the eval's
            #     completed_cycle_2 but STRICTER: deep-neutral collect (< the eval's board
            #     line) + fresh-ball payout + completion on a full clear, so any earned
            #     bonus necessarily raises the eval metric (reward subset of eval).
            # a 2nd-cycle LEAVE is crossing out NOT carrying a big load (>=abandon_load is the
            # A2 abandon case above, penalized, not rewarded). Empty/light = going to collect.
            if self._stage == 0 and outbound and mag < cfg.abandon_load:
                self._stage = 1
                self._fresh_at_out = fresh_score
                self._collected_at_out = collected
                self._n_leaves += 1
                self._max_stage = max(self._max_stage, 1)
                if self._t_leave is None:
                    self._t_leave = self._t            # telemetry: left the field
                    # progress reward (once) + D-pref: prefer crossing at a ramp lane. Base
                    # is PRESERVED so a trench crossing stays net-positive (never strands).
                    r += cfg.leave_bonus + cfg.ramp_pref * self._ramp_score(x)
            elif (self._stage == 1 and y > cfg.neutral_deep_y
                  and self._collected_at_out is not None and collected > self._collected_at_out):
                self._stage = 2
                self._core_off = True                  # D-A off-switch: first deep collect ends Ramp A
                self._leg_entered_t = self._t          # D-B clock starts (collect->return leg)
                self._rampB_paid = 0.0
                self._max_stage = max(self._max_stage, 2)
                if self._t_collect is None:
                    self._t_collect = self._t          # telemetry: collected in neutral
                    r += cfg.collect_bonus             # progress reward: collected out there (once)
            elif self._stage == 2 and inbound and mag > 0:
                self._stage = 3
                self._leg_entered_t = self._t          # D-C clock starts (return->score leg)
                self._rampC_paid = 0.0
                self._max_stage = max(self._max_stage, 3)
                if self._t_return is None:
                    self._t_return = self._t           # telemetry: returned loaded
                    r += cfg.return_bonus + cfg.ramp_pref * self._ramp_score(x)
            if self._stage == 3 and cleared:
                base = self._fresh_at_out if self._fresh_at_out is not None else fresh_score
                fresh_this = fresh_score - base
                if fresh_this >= cfg.mc_min_cycle_score:
                    self._cycle_k += 1
                    self._max_stage = 4
                    if self._t_score2 is None:
                        self._t_score2 = self._t       # telemetry: 2nd-cycle score (complete)
                    bonus = (min(cfg.mc_cap, cfg.mc_per_ball * float(fresh_this))
                             * (cfg.mc_escalation ** (self._cycle_k - 2)))
                    pay = min(bonus, cfg.mc_episode_cap - self._mc_paid_ep)
                    if pay > 0.0:
                        r += pay
                        self._mc_paid_ep += pay
                self._stage = 0                      # re-arm for the next cycle (pay-once)
                self._fresh_at_out = None
                self._collected_at_out = None
                # D re-arm: cycle 3+ gets identical Ramp-A pressure from stage 0 (still
                # under the shared per-episode cap, which is NOT reset).
                self._core_off = False
                self._core_t0 = None
                self._core_quiet = 0
                self._core_since_latch = 0
                self._leg_entered_t = self._t
                self._rampB_paid = 0.0
                self._rampC_paid = 0.0
        else:
            self._in_dwell = False
            self._home_linger_steps = 0

        self._prev_mag = mag
        self._prev_score = score
        out = float(r)
        if done:
            self.reset()
        return out


# --------------------------------------------------------------------------- #
# Alpha schedule + validation (TD3+BC anchor strength)
# --------------------------------------------------------------------------- #
ALPHA_MIN, ALPHA_MAX = 1.0, 2.5


def finetune_beta(update_idx: int, critic_only_updates: int,
                  beta_start: float = 0.3, beta_end_update: int = 23000) -> float:
    """Reward-first champion BC-anchor coefficient at a given 0-based update index
    The schedule is 0 during the critic-only phase (no actor/anchor);
    ``beta_start`` at actor unlock (== ``critic_only_updates``); linear → 0 at
    ``beta_end_update``; 0 after. The encoder is frozen while this returns 0 for the
    critic-only phase, so the anchor arrives exactly when the encoder starts moving."""
    if update_idx < critic_only_updates or update_idx >= beta_end_update:
        return 0.0
    span = beta_end_update - critic_only_updates
    if span <= 0:
        return 0.0
    return float(beta_start) * (1.0 - (update_idx - critic_only_updates) / span)


def check_alpha(alpha: float) -> float:
    """Return ``alpha`` iff finite and within [1.0, 2.5]; else raise ValueError."""
    a = float(alpha)
    if not np.isfinite(a) or a < ALPHA_MIN or a > ALPHA_MAX:
        raise ValueError(f"alpha {alpha!r} outside [{ALPHA_MIN}, {ALPHA_MAX}] or non-finite")
    return a


@dataclass
class AlphaSchedule:
    """α starts at 1.0 (anchor dominant) and rises toward 2.5 only after two
    consecutive held-out suffix-success windows with SAFE drift."""
    start: float = ALPHA_MIN
    end: float = ALPHA_MAX
    increment: float = 0.25
    warmup_windows: int = 2
    alpha: float = field(init=False)
    _safe_windows: int = field(init=False, default=0)

    def __post_init__(self):
        self.alpha = check_alpha(self.start)

    def on_window(self, success: bool, safe_drift: bool) -> float:
        """Record one held-out evaluation window; return the (possibly raised) alpha."""
        if success and safe_drift:
            self._safe_windows += 1
            if self._safe_windows > self.warmup_windows:
                self.alpha = check_alpha(min(self.end, self.alpha + self.increment))
        else:
            self._safe_windows = 0          # any bad window resets the warmup
        return self.alpha


# --------------------------------------------------------------------------- #
# Immutable-champion guard
# --------------------------------------------------------------------------- #
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def immutable_champion_ok(path: str = CHAMPION_PATH, expected_sha: str | None = None) -> str:
    """Return the champion SHA-256; raise if it is missing or (when ``expected_sha`` is
    given) has changed. Call at process start and before any publish so the immutable
    champion can never be silently overwritten."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"immutable champion missing: {path}")
    sha = sha256_file(p)
    if expected_sha is not None and sha != expected_sha:
        raise RuntimeError(f"immutable champion CHANGED: {sha} != expected {expected_sha}")
    return sha


# --------------------------------------------------------------------------- #
# DriftGate -- run the locked frozen-holdout drift check before publication
# --------------------------------------------------------------------------- #
def _load_drift_bounds() -> dict:
    """Import BOUNDS from the locked eval_anchor_drift.py so there is one source."""
    spec = importlib.util.spec_from_file_location(
        "eval_anchor_drift", PROJECT_ROOT / "scripts/rl/eval_anchor_drift.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BOUNDS


def drift_decision(champ_act: np.ndarray, cand_act: np.ndarray, spec, bounds: dict) -> dict:
    """Decode both action sets and apply the OR-ed hard-stop / warning bands.

    Triggers are the three PRIMARY heads only (drive-L2 p50, shoot, storage); intake
    and ferry are reported, not triggered by the four-head contract.
    """
    from frc_rebuilt.rl.spec import decode_policy_actions
    cd, dd = decode_policy_actions(champ_act, spec), decode_policy_actions(cand_act, spec)
    dl2 = np.linalg.norm(cand_act[:, :3] - champ_act[:, :3], axis=1)
    r = {
        "drive_l2_p50": float(np.percentile(dl2, 50)),
        "shoot_disagree": float((dd.shoot_blue != cd.shoot_blue).mean()),
        "storage_disagree": float((dd.storage_extended != cd.storage_extended).mean()),
        "intake_disagree": float((dd.intake_on != cd.intake_on).mean()),
        "ferry_disagree": float((dd.ferry != cd.ferry).mean()),
    }
    hs, wn = bounds["hard_stop"], bounds["warning"]
    r["hard_stop"] = bool(r["drive_l2_p50"] > hs["drive_l2_p50"]
                          or r["shoot_disagree"] > hs["shoot_disagree"]
                          or r["storage_disagree"] > hs["storage_disagree"])
    r["warning"] = bool(r["drive_l2_p50"] > wn["drive_l2_p50"]
                        or r["shoot_disagree"] > wn["shoot_disagree"]
                        or r["storage_disagree"] > wn["storage_disagree"])
    return r


class DriftGate:
    """Precompute champion actions on the frozen holdout once, then gate each candidate
    checkpoint before it is published to collectors."""

    def __init__(self, champion_agent, anchor_dir, holdout_episodes=FROZEN_HOLDOUT_EPISODES,
                 bounds: dict | None = None):
        from frc_rebuilt.rl.spec import CompetitionRLSpec
        self.spec = CompetitionRLSpec()
        self.bounds = bounds if bounds is not None else _load_drift_bounds()
        hold = set(int(i) for i in holdout_episodes)
        frames, proprio = [], []
        for p in sorted(Path(anchor_dir).glob("anchor_*.npz"), key=_ep_index):
            if _ep_index(p) not in hold:
                continue
            d = np.load(p, allow_pickle=True)
            frames.append(d["frames"]); proprio.append(d["proprio"])
        if not frames:
            raise SystemExit(f"DriftGate: no holdout anchors {holdout_episodes} in {anchor_dir}")
        self.frames = np.concatenate(frames)
        self.proprio = np.concatenate(proprio)
        self.champ_act = champion_agent.act(self.frames, self.proprio, explore=False)

    def check(self, candidate_agent) -> dict:
        cand_act = candidate_agent.act(self.frames, self.proprio, explore=False)
        return drift_decision(self.champ_act, cand_act, self.spec, self.bounds)
