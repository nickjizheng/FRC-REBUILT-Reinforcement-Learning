"""Smoke-test the vectorized competition env: shapes, rewards, resets, tx/s.

Runs N envs with random 7-D actions for a fixed number of policy steps,
verifies observation/reward/done plumbing, saves sample camera frames, and
reports aggregate policy-transitions/s INCLUDING camera rendering - the number
that decides the DrQ-v2 collection budget (docs/VECTORIZATION_PLAN.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument(
        "--template",
        default=str(PROJECT_ROOT / "assets" / "rl" / "env_template_32.usd"),
    )
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "vec_env_smoke")
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    report: dict[str, object] = {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "template": args.template,
        "cameras": not args.no_cameras,
    }
    try:
        from xrc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=not args.no_cameras,
                episode_len_s=12.0,
            )
        )
        print("SMOKE_ENV_READY", flush=True)
        rng = np.random.default_rng(7)

        # drive-forward bias so the robot actually moves and intakes
        def random_actions() -> np.ndarray:
            a = rng.uniform(-1.0, 1.0, (args.num_envs, 7)).astype(np.float32)
            a[:, 3] = 1.0  # intake on
            return a

        obs, rewards, dones, info = env.step(random_actions())
        shapes = {k: list(v.shape) for k, v in obs.items()}
        report["obs_shapes"] = shapes
        print(f"SMOKE_SHAPES {json.dumps(shapes)}", flush=True)

        if "rgb" in obs:
            args.out.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            for cam_index, name in enumerate(env.camera_names):
                frame = obs["rgb"][0, cam_index]
                Image.fromarray(frame).save(args.out / f"smoke_env0_{name}.png")
                print(
                    f"SMOKE_FRAME {name} std={float(frame.std()):.1f}",
                    flush=True,
                )

        total_reward = np.zeros(args.num_envs, np.float32)
        reset_count = 0
        t0 = time.perf_counter()
        for step in range(args.steps):
            obs, rewards, dones, info = env.step(random_actions())
            total_reward += rewards
            reset_count += int(dones.sum())
        dt = time.perf_counter() - t0
        tx = args.steps * args.num_envs / dt
        report.update(
            {
                "policy_tx_per_s": round(tx, 2),
                "wall_s": round(dt, 1),
                "episode_resets": reset_count,
                "sum_reward_per_env": [round(float(r), 2) for r in total_reward],
                "gate_8_tx_per_s_cleared": bool(tx >= 8.0),
            }
        )
        print("SMOKE_RESULT " + json.dumps(report), flush=True)
        (PROJECT_ROOT / "runs" / "vec_env_smoke.json").write_text(
            json.dumps(report, indent=2)
        )
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
