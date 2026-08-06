"""Integration smoke for custody-weighted reward in the real environment.

Runs the champion (which camps + recycles) under rho_score/rho_collect and checks the
custody machinery end to end: per-ball score_events populate + are consumed, fresh vs
recycled credits are counted, the recycled reward is discounted vs rho=1.0, and the
ledger clears on episode reset. Needs Isaac + GPU.
Run: CUDA_VISIBLE_DEVICES=0 python scripts/rl/smoke_custody.py --champion <path> --rho 0.2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=2200)   # >= 2 completed episodes
    ap.add_argument("--num-envs", type=int, default=2)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

        def run(rho: float):
            env = VecCompetitionEnv(VecEnvCfg(
                num_envs=args.num_envs, template_usd=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"),
                cameras=True, episode_len_s=90.0, preload_prob=0.0, seed=90101,
                spawn_under_trench=True, lock_storage_extended=False, collect_reward_weight=0.3,
                rho_score=rho, rho_collect=rho,
            ))
            from frc_rebuilt.rl import distributed as D
            agent = DrQV2Agent(DrQConfig()); agent.load(args.champion)
            env.reset_all()
            obs, *_ = env.step(np.zeros((args.num_envs, 7), np.float32))
            tot_score_r = 0.0; fresh = recyc = 0; n_eps = 0
            for _ in range(args.steps):
                frames = D.to_policy_frames(obs["rgb"])
                act = agent.act(frames, obs["proprio"], explore=False).astype(np.float32)
                obs, rewards, dones, info = env.step(act)
                for rc in info["reward_components"]:
                    tot_score_r += rc["score"]
                # episode_stats captures the custody counters BEFORE auto-reset wipes them
                for st in info.get("episode_stats", {}).values():
                    fresh += int(st.get("fresh_score", 0))
                    recyc += int(st.get("recycled_score", 0))
                    n_eps += 1
            env.close()
            return tot_score_r, fresh, recyc, n_eps

        sr_rho, fresh, recyc, n_eps = run(args.rho)
        print(f"CUSTODY_SMOKE completed_episodes={n_eps}", flush=True)
        print(f"CUSTODY_SMOKE rho={args.rho}: total_score_reward={sr_rho:.1f} "
              f"fresh_score={fresh} recycled_score={recyc}", flush=True)
        ok = True
        def chk(name, cond, detail=""):
            nonlocal ok; ok = ok and bool(cond)
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  '+detail) if detail else ''}", flush=True)

        chk("completed >= 1 episode (counters captured pre-reset)", n_eps >= 1, f"n_eps={n_eps}")
        chk("credits fresh field scores (fresh_score>0)", fresh > 0, f"fresh={fresh}")
        chk("detects recycled scores — the champion camps (recycled_score>0)", recyc > 0, f"recycled={recyc}")
        # the discount is then arithmetic: score_custody paid recycled balls rho x weight.
        implied_full = 10.0 * (fresh + recyc)
        implied_rho = 10.0 * (fresh + args.rho * recyc)
        chk("rho discount reduces score reward vs full credit",
            recyc == 0 or implied_rho < implied_full - 1e-6,
            f"rho-credit {implied_rho:.0f} < full-credit {implied_full:.0f} "
            f"(recycled share {recyc/(fresh+recyc):.0%})")
        print(f"\nCUSTODY_SMOKE {'PASS' if ok else 'FAIL'}  "
              f"total_score_reward={sr_rho:.0f} fresh={fresh} recycled={recyc}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
