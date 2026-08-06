"""Custody-weighted reward — pure Python, with no Isaac or torch dependency.

Anti-recycling signal: a FUEL ball's FIRST legal score earns full weight; every re-score
of the SAME ball earns ``rho_score`` × weight. Likewise a ball collected after it has
already been scored (the score → chute → re-collect loop) earns ``rho_collect`` × weight,
while a genuinely fresh (never-scored) collect earns full weight. Both are keyed on a
per-slot ``ever_scored`` set, so the discount targets exactly the camping/recycling
exploit that made raw score gameable, and never touches a first productive field cycle.

The env keeps one ``CustodyState`` per slot, feeds it the blue score events emitted by
``HubRouter`` (per-ball index) and the current magazine index set, and adds the returned
rewards. Kept import-free of Isaac so the logic is unit-testable anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CustodyState:
    ever_scored: set = field(default_factory=set)   # blue ball indices scored at least once
    prev_magazine: frozenset = frozenset()          # magazine index set at the previous step
    score_events_seen: int = 0                       # cursor into router.score_events
    # cumulative per-episode diagnostics (custody-bite telemetry, design note 10k gate)
    fresh_score: int = 0
    recycled_score: int = 0
    fresh_collect: int = 0
    recycled_collect: int = 0

    def reset(self, initial_magazine) -> None:
        """Reset for a new episode. ``initial_magazine`` (e.g. a preload) is seeded into
        ``prev_magazine`` so preloaded balls are NOT credited as fresh collections."""
        self.ever_scored = set()
        self.prev_magazine = frozenset(int(i) for i in initial_magazine)
        self.score_events_seen = 0
        self.fresh_score = self.recycled_score = 0
        self.fresh_collect = self.recycled_collect = 0


def score_custody(new_blue_indices, state: CustodyState, w_score: float, rho_score: float) -> float:
    """Credit newly-scored blue ball indices (in order) and return the total score reward.
    First score of an index → ``w_score``; a repeat → ``w_score * rho_score``."""
    r = 0.0
    for idx in new_blue_indices:
        idx = int(idx)
        if idx in state.ever_scored:
            r += w_score * rho_score
            state.recycled_score += 1
        else:
            r += w_score
            state.ever_scored.add(idx)
            state.fresh_score += 1
    return r


def collect_custody(magazine_now, state: CustodyState, w_collect: float, rho_collect: float) -> float:
    """Credit balls newly appearing in the magazine (set-diff vs the previous step) and
    return the total collect reward. A ball already in ``ever_scored`` (recycled from the
    chute) → ``w_collect * rho_collect``; a fresh field ball → ``w_collect``. Firing removes
    indices from the magazine, so it never appears in the diff."""
    mag = frozenset(int(i) for i in magazine_now)
    newly = mag - state.prev_magazine
    state.prev_magazine = mag
    r = 0.0
    for idx in newly:
        if idx in state.ever_scored:
            r += w_collect * rho_collect
            state.recycled_collect += 1
        else:
            r += w_collect
            state.fresh_collect += 1
    return r
