"""Stage C v2 full-physics collector.

This is deliberately separate from ``collector.py``: old Stage C rewards,
observations, checkpoints, and launchers keep their exact contract.  V2 owns
the phase-conditioned 30-value proprio vector, event-triggered reserve batches,
one-press dumping, and fixed per-stream curricula inside ``VecCompetitionEnv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from frc_rebuilt.rl.cycle_v2 import ROUTE_EFFICIENCY_REVISION
from frc_rebuilt.rl.policy_v2 import (
    ACTION_POLICY,
    FIELD_STRATEGY,
    LEGACY_PROPRIO_DIM,
    RETURN_SKILL_PRELOAD,
    SCHEMA_VERSION,
    apply_executed_action_policy,
    compose_phase_actions,
    validate_composite_metadata,
)

PROPRIO_FEATURE_NAMES = (
    "phase_first_cycle",
    "phase_leave",
    "phase_collect",
    "phase_return",
    "phase_score",
    "qualified_load_over_60",
    "target_load_over_60",
    "time_remaining",
)


def _true_black_camera_mask(rgb: np.ndarray) -> np.ndarray:
    """Return an (env, camera) mask for genuinely blank rendered frames."""

    rgb = np.asarray(rgb)
    if rgb.ndim != 5 or rgb.shape[-1] < 3:
        raise ValueError(
            "camera batch must have shape (env, camera, height, width, channels)"
        )
    camera_mean = rgb[..., :3].mean(axis=(2, 3, 4))
    camera_max = rgb[..., :3].max(axis=(2, 3, 4))
    return (camera_mean <= 1.0) & (camera_max <= 2)


def _parse_modes(text: str, num_envs: int) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(modes) == 1:
        modes = modes * int(num_envs)
    if len(modes) != int(num_envs):
        raise ValueError("--reset-modes must contain one mode or one per env")
    return modes


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_weight_metadata(
    blob: dict,
    args,
    proprio_dim: int,
    prefix_sha256: str,
) -> dict:
    meta = blob.get("stagec_v2")
    validate_composite_metadata(meta, prefix_sha256)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "proprio_dim": int(proprio_dim),
        "legacy_proprio_dim": LEGACY_PROPRIO_DIM,
        "dump_on_press": True,
        "target_load": int(args.target_load),
        "reserve_count": int(args.reserve_count),
        "reserve_batches": int(args.reserve_batches),
        "max_dump_ticks": int(args.max_dump_ticks),
        "cycle_score_fraction": float(args.cycle_score_fraction),
        "cycle_score_floor": int(args.cycle_score_floor),
        "collect_weight": float(args.collect_weight),
        "progress_per_m": float(args.progress_per_m),
        "progress_step_cap": float(args.progress_step_cap),
        "ramp_bonus": float(args.ramp_bonus),
        "leave_grace_steps": int(args.leave_grace_steps),
        "leave_penalty_per_step": float(args.leave_penalty_per_step),
        "leave_penalty_cap": float(args.leave_penalty_cap),
        "return_grace_steps": int(args.return_grace_steps),
        "return_penalty_per_step": float(args.return_penalty_per_step),
        "return_penalty_cap": float(args.return_penalty_cap),
        "shoot_grace_s": float(args.shoot_grace_s),
        "shoot_penalty_per_step": float(args.shoot_penalty_per_step),
        "shoot_penalty_cap": float(args.shoot_penalty_cap),
        "dump_lost_aim_grace_ticks": int(args.dump_lost_aim_grace_ticks),
        "partial_dump_penalty_per_ball": float(args.partial_dump_penalty_per_ball),
        "partial_dump_penalty_cap": float(args.partial_dump_penalty_cap),
        "prefix_sha256": str(prefix_sha256),
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "return_skill_preload": RETURN_SKILL_PRELOAD,
        "encoder_frozen": True,
    }
    if args.route_efficiency_revision:
        expected.update(
            {
                "reward_revision": ROUTE_EFFICIENCY_REVISION,
                "refresh_ramp_side_on_dump": bool(
                    args.refresh_ramp_side_on_dump
                ),
                "ramp_side_deadband_x": float(args.ramp_side_deadband_x),
                "require_ramp_out": bool(args.require_ramp_out),
                "ramp_out_half_width": float(args.ramp_out_half_width),
                "ramp_out_bonus": float(args.ramp_out_bonus),
                "off_ramp_exit_penalty": float(args.off_ramp_exit_penalty),
                "postdump_require_target_load": bool(
                    args.postdump_require_target_load
                ),
                "postdump_complete_cycle": bool(args.postdump_complete_cycle),
                "postdump_depleted_count": int(args.postdump_depleted_count),
                "postdump_depleted_prob": float(args.postdump_depleted_prob),
                "preferred_repeat_load": int(args.preferred_repeat_load),
                "collect_stall_steps": int(args.collect_stall_steps),
                "return_time_guard": float(args.return_time_guard),
                "intake_during_return": bool(args.intake_during_return),
                "repeat_load_return_bonus": float(
                    args.repeat_load_return_bonus
                ),
                "repeat_load_score_bonus": float(
                    args.repeat_load_score_bonus
                ),
                "outer_rail_enter_x": float(args.outer_rail_enter_x),
                "outer_rail_exit_x": float(args.outer_rail_exit_x),
                "outer_rail_max_x": float(args.outer_rail_max_x),
                "outer_rail_grace_steps": int(args.outer_rail_grace_steps),
                "outer_rail_penalty_per_step": float(
                    args.outer_rail_penalty_per_step
                ),
                "outer_rail_penalty_cap": float(args.outer_rail_penalty_cap),
                "outer_rail_min_scale": float(args.outer_rail_min_scale),
                "outer_rail_escalation_steps": int(
                    args.outer_rail_escalation_steps
                ),
                "outer_rail_max_multiplier": float(
                    args.outer_rail_max_multiplier
                ),
                "intake_substeps": int(args.intake_substeps),
            }
        )
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"weight metadata mismatch for {key}: {meta.get(key)!r} != {value!r}"
            )
    for key in ("stddev_start", "stddev_end", "stddev_steps"):
        actual = float(meta.get(key, float("nan")))
        wanted = float(getattr(args, key))
        if not np.isfinite(actual) or abs(actual - wanted) > 1e-9:
            raise ValueError(f"weight metadata mismatch for {key}: {actual} != {wanted}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collector-id", type=int, required=True)
    ap.add_argument("--root", default="/dev/shm/frc_stagec_v2")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--reset-modes", required=True)
    ap.add_argument(
        "--stagec-v2-prefix-checkpoint",
        required=True,
        help="frozen legacy 22-proprio checkpoint used for the protected first cycle",
    )
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"))
    ap.add_argument("--episode-len-s", type=float, default=120.0)
    ap.add_argument("--chunk-steps", type=int, default=12)
    ap.add_argument("--weight-reload-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=600.0)
    ap.add_argument("--target-load", type=int, default=15)
    ap.add_argument("--reserve-count", type=int, default=18)
    ap.add_argument("--reserve-batches", type=int, default=3)
    ap.add_argument("--collect-weight", type=float, default=0.3)
    ap.add_argument("--dump-on-press", action="store_true")
    ap.add_argument("--max-dump-ticks", type=int, default=180)
    ap.add_argument("--cycle-score-fraction", type=float, default=0.75)
    ap.add_argument("--cycle-score-floor", type=int, default=6)
    ap.add_argument("--progress-per-m", type=float, default=5.0)
    ap.add_argument("--progress-step-cap", type=float, default=0.75)
    ap.add_argument("--ramp-bonus", type=float, default=6.0)
    ap.add_argument("--route-efficiency-revision", action="store_true")
    ap.add_argument("--refresh-ramp-side-on-dump", action="store_true")
    ap.add_argument("--ramp-side-deadband-x", type=float, default=0.25)
    ap.add_argument("--require-ramp-out", action="store_true")
    ap.add_argument("--ramp-out-half-width", type=float, default=0.90)
    ap.add_argument("--ramp-out-bonus", type=float, default=0.0)
    ap.add_argument("--off-ramp-exit-penalty", type=float, default=0.0)
    ap.add_argument("--postdump-require-target-load", action="store_true")
    ap.add_argument("--postdump-complete-cycle", action="store_true")
    ap.add_argument("--postdump-depleted-count", type=int, default=0)
    ap.add_argument("--postdump-depleted-prob", type=float, default=0.0)
    ap.add_argument("--preferred-repeat-load", type=int, default=0)
    ap.add_argument("--collect-stall-steps", type=int, default=0)
    ap.add_argument("--return-time-guard", type=float, default=0.0)
    ap.add_argument("--intake-during-return", action="store_true")
    ap.add_argument("--repeat-load-return-bonus", type=float, default=0.0)
    ap.add_argument("--repeat-load-score-bonus", type=float, default=0.0)
    ap.add_argument("--outer-rail-enter-x", type=float, default=2.85)
    ap.add_argument("--outer-rail-exit-x", type=float, default=2.55)
    ap.add_argument("--outer-rail-max-x", type=float, default=3.60)
    ap.add_argument("--outer-rail-grace-steps", type=int, default=20)
    ap.add_argument("--outer-rail-penalty-per-step", type=float, default=0.0)
    ap.add_argument("--outer-rail-penalty-cap", type=float, default=8.0)
    ap.add_argument("--outer-rail-min-scale", type=float, default=0.0)
    ap.add_argument("--outer-rail-escalation-steps", type=int, default=0)
    ap.add_argument("--outer-rail-max-multiplier", type=float, default=1.0)
    ap.add_argument("--intake-substeps", type=int, default=1)
    ap.add_argument("--leave-grace-steps", type=int, default=5)
    ap.add_argument("--leave-penalty-per-step", type=float, default=0.03)
    ap.add_argument("--leave-penalty-cap", type=float, default=5.0)
    ap.add_argument("--return-grace-steps", type=int, default=10)
    ap.add_argument("--return-penalty-per-step", type=float, default=0.02)
    ap.add_argument("--return-penalty-cap", type=float, default=5.0)
    ap.add_argument("--shoot-grace-s", type=float, default=2.0)
    ap.add_argument("--shoot-penalty-per-step", type=float, default=0.05)
    ap.add_argument("--shoot-penalty-cap", type=float, default=5.0)
    ap.add_argument("--dump-lost-aim-grace-ticks", type=int, default=15)
    ap.add_argument("--partial-dump-penalty-per-ball", type=float, default=0.5)
    ap.add_argument("--partial-dump-penalty-cap", type=float, default=15.0)
    ap.add_argument("--stddev-start", type=float, default=1.0)
    ap.add_argument("--stddev-end", type=float, default=0.30)
    ap.add_argument("--stddev-steps", type=int, default=150_000)
    ap.add_argument(
        "--deterministic-suffix",
        action="store_true",
        help="execute the suffix actor mean without exploration noise; the "
        "frozen first-cycle prefix is deterministic in either mode",
    )
    ap.add_argument("--telemetry", type=Path, default=None)
    args = ap.parse_args()
    if not args.dump_on_press:
        ap.error("Stage C v2 requires --dump-on-press for train/eval parity")
    if args.route_efficiency_revision:
        if not args.refresh_ramp_side_on_dump:
            ap.error(
                "--route-efficiency-revision requires "
                "--refresh-ramp-side-on-dump"
            )
        if float(args.outer_rail_penalty_per_step) <= 0.0:
            ap.error(
                "--route-efficiency-revision requires a positive "
                "--outer-rail-penalty-per-step"
            )
        if args.require_ramp_out and float(args.ramp_out_bonus) <= 0.0:
            ap.error("--require-ramp-out requires a positive --ramp-out-bonus")
        if args.postdump_complete_cycle and (
            not args.postdump_require_target_load or not args.require_ramp_out
        ):
            ap.error(
                "--postdump-complete-cycle requires target-load and ramp-out gates"
            )
        if int(args.postdump_depleted_count) < 0:
            ap.error("--postdump-depleted-count cannot be negative")
        if not (0.0 <= float(args.postdump_depleted_prob) <= 1.0):
            ap.error("--postdump-depleted-prob must be in [0, 1]")
        if int(args.preferred_repeat_load) and not (
            int(args.target_load) < int(args.preferred_repeat_load) <= 60
        ):
            ap.error(
                "--preferred-repeat-load must exceed --target-load and be <= 60"
            )
        if (
            ROUTE_EFFICIENCY_REVISION == "score_efficiency_v9"
            and int(args.preferred_repeat_load)
            and (
                int(args.collect_stall_steps) <= 0
                and float(args.return_time_guard) <= 0.0
            )
        ):
            ap.error(
                "--preferred-repeat-load requires --collect-stall-steps or "
                "--return-time-guard"
            )
        if (
            ROUTE_EFFICIENCY_REVISION == "score_efficiency_v10_return_intake"
            and not bool(args.intake_during_return)
        ):
            ap.error(
                "score_efficiency_v10_return_intake requires "
                "--intake-during-return"
            )
        if (
            float(args.repeat_load_return_bonus) > 0.0
            or float(args.repeat_load_score_bonus) > 0.0
        ) and not int(args.preferred_repeat_load):
            ap.error("repeat-load bonuses require --preferred-repeat-load")
    elif (
        args.refresh_ramp_side_on_dump
        or args.require_ramp_out
        or float(args.ramp_out_bonus) != 0.0
        or float(args.off_ramp_exit_penalty) != 0.0
        or args.postdump_require_target_load
        or args.postdump_complete_cycle
        or int(args.postdump_depleted_count) != 0
        or float(args.postdump_depleted_prob) != 0.0
        or int(args.preferred_repeat_load) != 0
        or int(args.collect_stall_steps) != 0
        or float(args.return_time_guard) != 0.0
        or args.intake_during_return
        or float(args.repeat_load_return_bonus) != 0.0
        or float(args.repeat_load_score_bonus) != 0.0
        or float(args.outer_rail_penalty_per_step) != 0.0
        or int(args.intake_substeps) != 1
    ):
        ap.error(
            "route-efficiency settings require --route-efficiency-revision"
        )
    if not (
        0.0
        <= float(args.outer_rail_exit_x)
        < float(args.outer_rail_enter_x)
        < float(args.outer_rail_max_x)
    ):
        ap.error("outer-rail geometry must satisfy 0 <= exit < enter < max")
    if int(args.outer_rail_grace_steps) < 0:
        ap.error("--outer-rail-grace-steps must be non-negative")
    if int(args.outer_rail_escalation_steps) < 0:
        ap.error("--outer-rail-escalation-steps must be non-negative")
    if (
        not np.isfinite(args.outer_rail_penalty_per_step)
        or float(args.outer_rail_penalty_per_step) < 0.0
        or not np.isfinite(args.outer_rail_penalty_cap)
        or float(args.outer_rail_penalty_cap) < 0.0
    ):
        ap.error("outer-rail penalty and cap must be finite and non-negative")
    if (
        not np.isfinite(args.outer_rail_min_scale)
        or not 0.0 <= float(args.outer_rail_min_scale) <= 1.0
    ):
        ap.error("--outer-rail-min-scale must be in [0, 1]")
    if (
        not np.isfinite(args.outer_rail_max_multiplier)
        or float(args.outer_rail_max_multiplier) < 1.0
    ):
        ap.error("--outer-rail-max-multiplier must be at least 1")
    if not 1 <= int(args.intake_substeps) <= 3:
        ap.error("--intake-substeps must be in [1, 3]")
    if int(args.collect_stall_steps) < 0:
        ap.error("--collect-stall-steps cannot be negative")
    if not 0.0 <= float(args.return_time_guard) <= 1.0:
        ap.error("--return-time-guard must be in [0, 1]")
    if (
        not np.isfinite(args.ramp_out_half_width)
        or float(args.ramp_out_half_width) <= 0.0
    ):
        ap.error("--ramp-out-half-width must be finite and positive")
    for name in (
        "ramp_out_bonus",
        "off_ramp_exit_penalty",
        "repeat_load_return_bonus",
        "repeat_load_score_bonus",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    modes = _parse_modes(args.reset_modes, args.num_envs)
    prefix_path = Path(args.stagec_v2_prefix_checkpoint).resolve()
    if not prefix_path.is_file():
        ap.error(f"Stage C v2 prefix checkpoint does not exist: {prefix_path}")
    prefix_sha256 = _sha256_file(prefix_path)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import torch

        from frc_rebuilt.rl import distributed as D
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cdir = D.collector_dir(args.root, args.collector_id)
        wdir = D.weights_dir(args.root)
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                preload_prob=0.0,
                spawn_under_trench=True,
                lock_storage_extended=False,
                mask_illegal_fire=True,
                collect_reward_weight=float(args.collect_weight),
                rho_score=1.0,
                rho_collect=1.0,
                empty_own_court_penalty=0.0,
                dump_on_press=True,
                max_dump_ticks=int(args.max_dump_ticks),
                stagec_v2=True,
                cycle_v2_reset_modes=modes,
                cycle_v2_target_load=int(args.target_load),
                cycle_v2_reserve_count=int(args.reserve_count),
                cycle_v2_reserve_batches=int(args.reserve_batches),
                cycle_v2_score_fraction=float(args.cycle_score_fraction),
                cycle_v2_score_floor=int(args.cycle_score_floor),
                cycle_v2_progress_per_m=float(args.progress_per_m),
                cycle_v2_progress_step_cap=float(args.progress_step_cap),
                cycle_v2_ramp_bonus=float(args.ramp_bonus),
                cycle_v2_refresh_ramp_side_on_dump=bool(
                    args.refresh_ramp_side_on_dump
                ),
                cycle_v2_ramp_side_deadband_x=float(args.ramp_side_deadband_x),
                cycle_v2_require_ramp_out=bool(args.require_ramp_out),
                cycle_v2_ramp_out_half_width=float(args.ramp_out_half_width),
                cycle_v2_ramp_out_bonus=float(args.ramp_out_bonus),
                cycle_v2_off_ramp_exit_penalty=float(
                    args.off_ramp_exit_penalty
                ),
                cycle_v2_postdump_require_target_load=bool(
                    args.postdump_require_target_load
                ),
                cycle_v2_postdump_complete_cycle=bool(
                    args.postdump_complete_cycle
                ),
                cycle_v2_postdump_depleted_count=int(
                    args.postdump_depleted_count
                ),
                cycle_v2_postdump_depleted_prob=float(
                    args.postdump_depleted_prob
                ),
                cycle_v2_preferred_repeat_load=int(
                    args.preferred_repeat_load
                ),
                cycle_v2_collect_until_preferred=bool(
                    ROUTE_EFFICIENCY_REVISION == "score_efficiency_v9"
                    and int(args.preferred_repeat_load)
                    and (
                        int(args.collect_stall_steps) > 0
                        or float(args.return_time_guard) > 0.0
                    )
                ),
                cycle_v2_collect_stall_steps=int(args.collect_stall_steps),
                cycle_v2_return_time_guard=float(args.return_time_guard),
                cycle_v2_intake_during_return=bool(
                    args.intake_during_return
                ),
                cycle_v2_repeat_load_return_bonus=float(
                    args.repeat_load_return_bonus
                ),
                cycle_v2_repeat_load_score_bonus=float(
                    args.repeat_load_score_bonus
                ),
                cycle_v2_outer_rail_enter_x=float(args.outer_rail_enter_x),
                cycle_v2_outer_rail_exit_x=float(args.outer_rail_exit_x),
                cycle_v2_outer_rail_max_x=float(args.outer_rail_max_x),
                cycle_v2_outer_rail_grace_steps=int(args.outer_rail_grace_steps),
                cycle_v2_outer_rail_penalty_per_step=float(
                    args.outer_rail_penalty_per_step
                ),
                cycle_v2_outer_rail_penalty_cap=float(
                    args.outer_rail_penalty_cap
                ),
                cycle_v2_outer_rail_min_scale=float(
                    args.outer_rail_min_scale
                ),
                cycle_v2_outer_rail_escalation_steps=int(
                    args.outer_rail_escalation_steps
                ),
                cycle_v2_outer_rail_max_multiplier=float(
                    args.outer_rail_max_multiplier
                ),
                cycle_v2_intake_substeps=int(args.intake_substeps),
                cycle_v2_leave_grace_steps=int(args.leave_grace_steps),
                cycle_v2_leave_penalty_per_step=float(args.leave_penalty_per_step),
                cycle_v2_leave_penalty_cap=float(args.leave_penalty_cap),
                cycle_v2_return_grace_steps=int(args.return_grace_steps),
                cycle_v2_return_penalty_per_step=float(args.return_penalty_per_step),
                cycle_v2_return_penalty_cap=float(args.return_penalty_cap),
                cycle_v2_shoot_grace_steps=int(round(float(args.shoot_grace_s) * 10.0)),
                cycle_v2_shoot_penalty_per_step=float(args.shoot_penalty_per_step),
                cycle_v2_shoot_penalty_cap=float(args.shoot_penalty_cap),
                cycle_v2_dump_lost_aim_grace_ticks=int(args.dump_lost_aim_grace_ticks),
                cycle_v2_partial_dump_penalty_per_ball=float(
                    args.partial_dump_penalty_per_ball
                ),
                cycle_v2_partial_dump_penalty_cap=float(args.partial_dump_penalty_cap),
                seed=args.seed,
            )
        )
        if not bool(getattr(env, "_camera_ready", False)):
            raise RuntimeError("collector camera initialization failed")

        n = args.num_envs
        obs, _, _, _ = env.step(np.zeros((n, 7), np.float32))
        frames = D.to_policy_frames(obs["rgb"])
        startup_mean = obs["rgb"].mean(axis=(2, 3, 4))
        startup_max = obs["rgb"].max(axis=(2, 3, 4))
        startup_black = _true_black_camera_mask(obs["rgb"])
        if bool(startup_black.any()):
            bad = np.argwhere(startup_black).tolist()
            raise RuntimeError(
                "collector camera produced truly black startup frames: "
                f"views={bad} means={startup_mean.round(2).tolist()} "
                f"max={startup_max.tolist()}"
            )
        cfg = DrQConfig(
            frame_channels=frames.shape[1],
            frame_h=frames.shape[2],
            frame_w=frames.shape[3],
            proprio_dim=obs["proprio"].shape[1],
            privileged_dim=obs["privileged"].shape[1],
            stddev_start=args.stddev_start,
            stddev_end=args.stddev_end,
            stddev_steps=args.stddev_steps,
        )
        if cfg.proprio_dim != 30:
            raise RuntimeError(f"Stage C v2 expected 30 proprio values, got {cfg.proprio_dim}")
        agent = DrQV2Agent(cfg)
        prefix_agent = DrQV2Agent(
            DrQConfig(
                frame_channels=frames.shape[1],
                frame_h=frames.shape[2],
                frame_w=frames.shape[3],
                proprio_dim=LEGACY_PROPRIO_DIM,
                privileged_dim=obs["privileged"].shape[1],
            )
        )
        prefix_agent.load(str(prefix_path))

        for _ in range(600):
            if D.latest_weights(wdir):
                break
            time.sleep(1.0)
        loaded_step = -1

        def maybe_reload() -> None:
            nonlocal loaded_step
            got = D.latest_weights(wdir)
            if not got or got[1] == loaded_step:
                return
            path, step = got
            blob = torch.load(path, map_location=agent.device)
            _validate_weight_metadata(blob, args, cfg.proprio_dim, prefix_sha256)
            agent.encoder.load_state_dict(blob["encoder"], strict=True)
            agent.actor.load_state_dict(blob["actor"], strict=True)
            agent.train_steps = int(blob["train_steps"])
            agent.explore_offset = int(blob["explore_offset"])
            loaded_step = int(step)

        maybe_reload()
        if loaded_step < 0:
            raise RuntimeError("learner did not publish initial Stage C v2 weights")
        print(
            f"COLLECTOR_V2_READY id={args.collector_id} device={agent.device} "
            f"modes={','.join(modes)} proprio={cfg.proprio_dim} stddev={agent.stddev():.3f} "
            f"schema={SCHEMA_VERSION} action_policy={ACTION_POLICY} "
            f"prefix_sha256={prefix_sha256} intake_substeps={args.intake_substeps} "
            f"intake_during_return={bool(args.intake_during_return)}",
            f" suffix_action={'mean' if args.deterministic_suffix else 'explore'}",
            flush=True,
        )

        deadline = time.time() + args.minutes * 60.0
        seq = step_count = 0
        buf: dict[str, list] = {key: [] for key in D.FIELD_KEYS}
        pending_eps: list[dict] = []
        ep_return = np.zeros(n, np.float32)
        ep_score = np.zeros(n, np.float32)
        ep_collect = np.zeros(n, np.float32)
        episode_seq = np.zeros(n, np.int64)
        # Track each (environment, camera) independently.  The previous single
        # global streak killed a whole collector whenever any legitimate
        # low-variance wall/floor view persisted for ten steps.
        black_streak = np.zeros(
            (n, int(obs["rgb"].shape[1])),
            dtype=np.int32,
        )

        while time.time() < deadline:
            candidate_actions = agent.act(
                frames,
                obs["proprio"],
                explore=not bool(args.deterministic_suffix),
            ).astype(np.float32)
            prefix_actions = prefix_agent.act(
                frames,
                obs["proprio"][:, :LEGACY_PROPRIO_DIM],
                explore=False,
            ).astype(np.float32)
            actions = compose_phase_actions(
                prefix_actions, candidate_actions, obs["proprio"]
            )
            actions = apply_executed_action_policy(
                actions,
                obs["proprio"],
                intake_during_return=bool(args.intake_during_return),
            )
            next_obs, rewards, dones, info = env.step(actions)
            next_frames = D.to_policy_frames(next_obs["rgb"])
            camera_mean = next_obs["rgb"].mean(axis=(2, 3, 4))
            camera_max = next_obs["rgb"].max(axis=(2, 3, 4))
            truly_black = _true_black_camera_mask(next_obs["rgb"])
            black_streak = np.where(
                truly_black, black_streak + 1, 0
            ).astype(np.int32)
            if bool((black_streak >= 10).any()):
                bad = np.argwhere(black_streak >= 10).tolist()
                raise RuntimeError(
                    "collector cameras truly black for 10 consecutive steps: "
                    f"views={bad} streaks={black_streak.tolist()} "
                    f"means={camera_mean.round(2).tolist()} "
                    f"max={camera_max.tolist()}"
                )

            buf["obs"].append(frames)
            buf["proprio"].append(obs["proprio"].copy())
            buf["privileged"].append(obs["privileged"].copy())
            buf["action"].append(actions)
            buf["reward"].append(rewards.astype(np.float32))
            buf["done"].append(dones.copy())
            for i in range(n):
                parts = info["reward_components"][i]
                ep_return[i] += float(rewards[i])
                ep_score[i] += float(parts["score"])
                ep_collect[i] += float(parts["collect"])
            for i in np.flatnonzero(dones):
                terminal = dict(info.get("episode_stats", {}).get(int(i), {}))
                terminal.update(
                    {
                        "return": round(float(ep_return[i]), 3),
                        "score_reward": round(float(ep_score[i]), 3),
                        "collect_reward": round(float(ep_collect[i]), 3),
                        "collector": int(args.collector_id),
                        "env_index": int(i),
                        "episode_seq": int(episode_seq[i]),
                        "stream_mode": modes[int(i)],
                        "suffix_action_mode": (
                            "mean" if args.deterministic_suffix else "explore"
                        ),
                        "policy_train_steps": int(agent.train_steps),
                    }
                )
                pending_eps.append(terminal)
                if args.telemetry is not None:
                    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
                    with args.telemetry.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(terminal, sort_keys=True) + "\n")
                ep_return[i] = ep_score[i] = ep_collect[i] = 0.0
                episode_seq[i] += 1

            obs, frames = next_obs, next_frames
            step_count += 1
            if step_count % args.weight_reload_steps == 0:
                maybe_reload()
            if len(buf["reward"]) >= args.chunk_steps:
                arrays = {key: np.stack(buf[key], axis=1) for key in D.FIELD_KEYS}
                D.write_chunk(cdir, seq, arrays, pending_eps)
                seq += 1
                for values in buf.values():
                    values.clear()
                pending_eps = []

        env.close()
    except Exception as exc:
        print(
            f"COLLECTOR_V2_FATAL id={args.collector_id} "
            f"type={type(exc).__name__} error={exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
