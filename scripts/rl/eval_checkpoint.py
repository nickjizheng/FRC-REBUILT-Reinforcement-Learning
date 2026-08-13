"""Deterministic checkpoint evaluation vs random/zero baselines.

Runs E fixed-seed Stage-A episodes per policy with exploration OFF and reports
raw FUEL collected + legally scored (not just shaped return), so "did it learn
to score or only to intake" is answered directly (reviewer requirement).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from frc_rebuilt.rl.policy_v2 import (
    ACTION_POLICY,
    LEGACY_PROPRIO_DIM,
    SCHEMA_VERSION,
    apply_executed_action_policy,
    compose_phase_actions,
    validate_composite_metadata,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "models/selected/stageA_clean_latest_94627ce.pt"),
    )
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--episode-len-s", type=float, default=20.0)
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_32.usd"))
    ap.add_argument("--preload-prob", type=float, default=0.0,
                    help="fraction of episodes preloaded at a shooting pose; set 0.5 to match Stage-B training conditions")
    ap.add_argument("--spawn-under-trench", action="store_true",
                    help="Stage C: spawn compact under the blue trench (unlocks storage) -- "
                         "REQUIRED to eval a Stage C policy under its real training start; "
                         "without it the policy is evaluated from the wrong pose.")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--dump-on-press", action="store_true")
    ap.add_argument("--max-dump-ticks", type=int, default=None)
    ap.add_argument("--stagec-v2", action="store_true")
    ap.add_argument(
        "--stagec-v2-prefix-checkpoint",
        default=None,
        help="required with --stagec-v2: frozen legacy 22-proprio first-cycle policy",
    )
    ap.add_argument(
        "--stagec-v2-reset-mode",
        choices=("full", "postdump", "collect", "return"),
        default="full",
    )
    ap.add_argument("--stagec-v2-target-load", type=int, default=15)
    ap.add_argument("--stagec-v2-reserve-count", type=int, default=18)
    ap.add_argument("--stagec-v2-reserve-batches", type=int, default=3)
    ap.add_argument("--policies", default="checkpoint,random,zero")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "eval_stageA_clean.json")
    args = ap.parse_args()
    if args.stagec_v2 and not args.dump_on_press:
        ap.error("--stagec-v2 requires --dump-on-press")
    if args.stagec_v2 and not args.stagec_v2_prefix_checkpoint:
        ap.error("--stagec-v2 requires --stagec-v2-prefix-checkpoint")
    prefix_path = (
        Path(args.stagec_v2_prefix_checkpoint).resolve()
        if args.stagec_v2_prefix_checkpoint
        else None
    )
    if prefix_path is not None and not prefix_path.is_file():
        ap.error(f"Stage C v2 prefix checkpoint does not exist: {prefix_path}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        results: dict[str, dict] = {}
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                preload_prob=args.preload_prob,
                seed=args.seed,
                spawn_under_trench=bool(args.spawn_under_trench),
                lock_storage_extended=not bool(args.spawn_under_trench),
                dump_on_press=bool(args.dump_on_press),
                max_dump_ticks=(
                    int(args.max_dump_ticks)
                    if args.max_dump_ticks is not None
                    else (180 if args.stagec_v2 else 90)
                ),
                collect_reward_weight=0.3 if args.stagec_v2 else 1.5,
                empty_own_court_penalty=0.0 if args.stagec_v2 else 0.02,
                stagec_v2=bool(args.stagec_v2),
                cycle_v2_reset_modes=(args.stagec_v2_reset_mode,),
                cycle_v2_target_load=int(args.stagec_v2_target_load),
                cycle_v2_reserve_count=int(args.stagec_v2_reserve_count),
                cycle_v2_reserve_batches=int(args.stagec_v2_reserve_batches),
            )
        )
        steps_per_episode = int(args.episode_len_s * 10)

        agent = None
        prefix_agent = None
        to_frames = None
        if "checkpoint" in args.policies:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "train_drqv2", PROJECT_ROOT / "scripts" / "rl" / "train_drqv2.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            to_frames = module.to_policy_frames
            from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
            import torch

            env.reset_all()
            probe_obs, *_ = env.step(np.zeros((args.num_envs, 7), np.float32))
            probe_frames = to_frames(probe_obs["rgb"])
            agent = DrQV2Agent(
                DrQConfig(
                    frame_channels=probe_frames.shape[1],
                    frame_h=probe_frames.shape[2],
                    frame_w=probe_frames.shape[3],
                    proprio_dim=probe_obs["proprio"].shape[1],
                    privileged_dim=probe_obs["privileged"].shape[1],
                )
            )
            prefix_sha256 = None
            if args.stagec_v2:
                prefix_sha256 = _sha256_file(prefix_path)
                try:
                    candidate_payload = torch.load(
                        args.checkpoint, map_location="cpu", weights_only=True
                    )
                except TypeError:
                    candidate_payload = torch.load(args.checkpoint, map_location="cpu")
                stagec_metadata = validate_composite_metadata(
                    candidate_payload.get("stagec_v2"), prefix_sha256
                )
                if stagec_metadata.get("reward_revision") is not None:
                    raise ValueError(
                        "eval_checkpoint.py does not reproduce route-efficiency "
                        "mechanics; use eval_stagec_seedmine.py for Stage C "
                        "revision checkpoints"
                    )
                prefix_agent = DrQV2Agent(
                    DrQConfig(
                        frame_channels=probe_frames.shape[1],
                        frame_h=probe_frames.shape[2],
                        frame_w=probe_frames.shape[3],
                        proprio_dim=LEGACY_PROPRIO_DIM,
                        privileged_dim=probe_obs["privileged"].shape[1],
                    )
                )
                prefix_agent.load(str(prefix_path))
            agent.load(args.checkpoint)
            print(
                f"EVAL_LOADED {args.checkpoint} steps={agent.train_steps}"
                + (
                    f" schema={SCHEMA_VERSION} action_policy={ACTION_POLICY} "
                    f"prefix_sha256={prefix_sha256}"
                    if args.stagec_v2
                    else ""
                ),
                flush=True,
            )

        for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
            rng = np.random.default_rng(args.seed)
            env.rng = np.random.default_rng(args.seed)  # identical resets per policy
            env.reset_all()
            obs, *_ = env.step(np.zeros((args.num_envs, 7), np.float32))
            episode_stats: list[dict] = []
            returns = np.zeros(args.num_envs, np.float32)
            needed = args.episodes
            while len(episode_stats) < needed:
                if policy == "checkpoint":
                    frames = to_frames(obs["rgb"])
                    candidate_actions = agent.act(
                        frames, obs["proprio"], explore=False
                    )
                    if args.stagec_v2:
                        prefix_actions = prefix_agent.act(
                            frames,
                            obs["proprio"][:, :LEGACY_PROPRIO_DIM],
                            explore=False,
                        )
                        actions = compose_phase_actions(
                            prefix_actions, candidate_actions, obs["proprio"]
                        )
                        actions = apply_executed_action_policy(
                            actions, obs["proprio"]
                        )
                    else:
                        actions = candidate_actions
                elif policy == "random":
                    actions = rng.uniform(-1, 1, (args.num_envs, 7)).astype(np.float32)
                    actions[:, 3] = 1.0
                else:  # zero: sit still, intake off
                    actions = np.zeros((args.num_envs, 7), np.float32)
                obs, rewards, dones, info = env.step(actions.astype(np.float32))
                returns += rewards
                for i in np.flatnonzero(dones):
                    terminal = info["episode_stats"][int(i)]
                    episode_stats.append(
                        {
                            "return": round(float(returns[i]), 2),
                            "scored": terminal["scored"],
                            "collected": terminal["collected"],
                            "shots_fired": terminal["shots_fired"],
                            "cycles_completed": terminal.get("cycles_completed", 0),
                            "reset_mode": terminal.get("reset_mode", "legacy"),
                            "terminal_reason": terminal.get("terminal_reason", "done"),
                            **(
                                {
                                    key: terminal[key]
                                    for key in (
                                        "latched",
                                        "terminal_phase",
                                        "dump_attempts",
                                        "dump_empty_completions",
                                        "partial_dumps",
                                        "return_skill_preload",
                                    )
                                    if key in terminal
                                }
                                if terminal.get("stagec_v2")
                                else {}
                            ),
                        }
                    )
                    returns[i] = 0.0
            scored = [e["scored"] for e in episode_stats[:needed]]
            collected = [e["collected"] for e in episode_stats[:needed]]
            rets = [e["return"] for e in episode_stats[:needed]]
            cycles = [e["cycles_completed"] for e in episode_stats[:needed]]
            results[policy] = {
                "episodes": needed,
                "mean_return": round(float(np.mean(rets)), 2),
                "mean_scored": round(float(np.mean(scored)), 2),
                "max_scored": int(np.max(scored)),
                "pct_episodes_scored": round(
                    100.0 * float(np.mean([s >= 1 for s in scored])), 1
                ),
                "mean_collected": round(float(np.mean(collected)), 2),
                "max_collected": int(np.max(collected)),
                "cycle2_rate": round(float(np.mean([value >= 1 for value in cycles])), 3),
                "mean_cycles": round(float(np.mean(cycles)), 3),
                "pct_100_plus": round(100.0 * float(np.mean([s >= 100 for s in scored])), 1),
                "per_episode": episode_stats[:needed],
            }
            print(f"EVAL {policy} " + json.dumps(results[policy]), flush=True)

        args.out.write_text(json.dumps(results, indent=2))
        print("EVAL_DONE " + str(args.out), flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
