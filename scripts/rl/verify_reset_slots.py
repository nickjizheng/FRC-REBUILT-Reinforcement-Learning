"""Standalone integration check for VecCompetitionEnv.reset_slots.

Proves the reset->fresh-observation contract only: env0 gets clock=0, prev-action=0, empty
magazine/score/router, cleared per-episode fire/dump telemetry, and a non-black camera; the
returned obs is the batch the caller must adopt; out-of-range indices FAIL FAST; valid
duplicates dedupe; an untouched env is not reset; and its shared-scene settle time is
tracked. NOTE: the terminal replay-stream boundary is NOT proven here -- that is the
collector's job (mark_last_terminal on the final suffix transition) and is proven in the
candidate=champion continuity integration test, not by reset_slots. Needs Isaac + a GPU.
Run: C:\\il\\venv\\Scripts\\python.exe scripts/rl/verify_reset_slots.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        env = VecCompetitionEnv(VecEnvCfg(
            num_envs=2, template_usd=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"),
            cameras=True, episode_len_s=90.0, preload_prob=0.0, seed=70001,
            spawn_under_trench=True, lock_storage_extended=False,
        ))
        env.reset_all()
        obs, *_ = env.step(np.zeros((2, 7), np.float32))
        drive = np.zeros((2, 7), np.float32); drive[:, 0] = 0.8
        for _ in range(20):                       # advance BOTH envs; robots move, clocks tick
            obs, *_ = env.step(drive)
        clk1_before = float(env.slots[1].clock_s)
        # inject per-episode telemetry on env0 to prove reset_slots clears it
        env.slots[0].ferry_fires = 5; env.slots[0].dumping = True; env.slots[0].dump_ticks = 9
        env.slots[0].prev_action[:] = 0.5
        print(f"before reset: env0 clock={env.slots[0].clock_s:.2f} env1 clock={clk1_before:.2f}", flush=True)

        ok = True
        def chk(name, cond):
            nonlocal ok; ok = ok and bool(cond)
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)

        # fail-fast: out-of-range indices must raise, not silently drop (design note)
        for bad in ([env.cfg.num_envs], [-1]):
            try:
                env.reset_slots(bad); raised = False
            except IndexError:
                raised = True
            chk(f"reset_slots({bad}) fails fast (IndexError)", raised)

        fresh = env.reset_slots([0, 0])            # valid duplicates dedupe -> only env 0
        s0 = env.slots[0]

        chk("returns fresh obs dict (rgb+proprio)", isinstance(fresh, dict) and "rgb" in fresh and "proprio" in fresh)
        chk("env0 clock reset to 0", abs(float(s0.clock_s)) < 1e-6)
        chk("env0 prev_action zero", float(np.abs(s0.prev_action).max()) == 0.0)
        chk("env0 magazine empty", len(s0.controller.magazine) == 0)
        chk("env0 score/router cleared", int(s0.router.scored["blue"]) == 0 and int(s0.score_seen) == 0)
        chk("env0 per-ep telemetry cleared", s0.ferry_fires == 0 and s0.dumping is False
            and s0.dump_ticks == 0 and s0.dump_mode is None)
        cam0 = float(np.asarray(fresh["rgb"][0]).std())
        chk(f"env0 camera fresh (std={cam0:.1f} > 1, not black)", cam0 > 1.0)
        chk("env1 UNTOUCHED (not reset to 0)", abs(float(env.slots[1].clock_s) - clk1_before) < 1e-6 and clk1_before > 0)
        chk("env1 forced_reset_settle_s tracked (~1/60 s)",
            abs(getattr(env.slots[1], "forced_reset_settle_s", 0.0) - 1.0 / 60.0) < 1e-3)

        print(f"\nRESET_SLOTS_VERIFY {'PASS' if ok else 'FAIL'}", flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
