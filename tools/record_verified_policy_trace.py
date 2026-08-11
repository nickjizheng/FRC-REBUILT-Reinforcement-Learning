"""Pass 1: record a complete visual-state trace of a verified 200+ policy run.

The evaluator is reconstructed from the custody-verified archive with its
original policy cameras and no additional render product.  By default both
environment slots run the exact frozen composite policy closed-loop, including
Stage D's pinned legacy-prefix observation.  Archived actions remain available
as an explicitly selected diagnostic, never as an implicit fallback.  Nothing
is published unless the fresh live target run reaches the 160 s horizon with
score >= 200.  Optional run-seed overrides provide a fail-closed closed-loop
seed search while keeping the archived score-202 bundles as immutable source
custody; that mode is explicitly classified as a seed search, not a replay.
An optional dual-environment race captures complete visual state for both
simulator slots during that same rollout, then atomically publishes only the
highest-scoring healthy 200+ horizon (and never splices data between slots).
For archived evaluator rows from the next reset generation, the opt-in
``--advance-full-horizons 1`` mode executes one complete healthy synchronized
horizon and reaches the next generation only through the environment's normal
terminal auto-reset.  It is deliberately restricted to the dual closed-loop
race path so a global evaluator row ordinal cannot be mistaken for a per-slot
episode count.
When evaluator slots reset asynchronously, the mutually exclusive
``--advance-selected-env-horizons 1`` mode instead follows only ``--env-index``
through its healthy first horizon and normal auto-reset while continuing to
drive the other slot closed-loop without making its health a selection gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from render_verified_topdown_replay import (  # noqa: E402
    _metadata_number,
    build_exact_cfg,
    load_replay_bundle,
    require_file_sha,
    sha256_file,
    stage_d_prefix_view,
    verify_code_snapshot,
)
from verified_topdown_trace import (  # noqa: E402
    FUEL_COUNT,
    MIN_LIVE_SCORE,
    STEPS,
    TRACE_PROVENANCE_SCHEMA,
    TRACE_SCHEMA,
    atomic_json,
    atomic_save_trace,
    field_declarations,
    normalized_quaternion_error,
)


def _phase_name(slot: Any) -> str:
    phase = getattr(getattr(slot, "cycle_v2", None), "phase", None)
    value = getattr(phase, "value", None) or getattr(phase, "name", None) or str(phase)
    return str(value).upper()


def _seed_mode(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve source/run seeds and reject ambiguous search requests."""

    run_env_seed = getattr(args, "run_env_seed", None)
    run_action_seed = getattr(args, "run_action_seed", None)
    has_env = run_env_seed is not None
    has_action = run_action_seed is not None
    if has_env != has_action:
        raise ValueError("--run-env-seed and --run-action-seed must be supplied together")
    search = bool(has_env)
    if search and args.action_source != "closed-loop":
        raise ValueError("seed search requires --action-source closed-loop")
    if getattr(args, "race_both_envs", False) and args.action_source != "closed-loop":
        raise ValueError("dual-environment race requires --action-source closed-loop")
    advance_full_horizons = int(getattr(args, "advance_full_horizons", 0))
    advance_selected_env_horizons = int(
        getattr(args, "advance_selected_env_horizons", 0)
    )
    if advance_full_horizons and advance_selected_env_horizons:
        raise ValueError(
            "full-horizon and selected-env advancement modes are mutually exclusive"
        )
    if advance_full_horizons and not getattr(args, "race_both_envs", False):
        raise ValueError("full-horizon advancement requires --race-both-envs")
    if advance_full_horizons and args.action_source != "closed-loop":
        raise ValueError("full-horizon advancement requires --action-source closed-loop")
    if advance_selected_env_horizons and getattr(args, "race_both_envs", False):
        raise ValueError("selected-env advancement is mutually exclusive with --race-both-envs")
    if advance_selected_env_horizons and args.action_source != "closed-loop":
        raise ValueError("selected-env advancement requires --action-source closed-loop")
    source = (int(args.expected_env_seed), int(args.expected_action_seed))
    run = (
        (int(run_env_seed), int(run_action_seed))
        if search
        else source
    )
    if search and any(value < 0 or value > 0xFFFFFFFF for value in run):
        raise ValueError("run seeds must be within the uint32 range [0, 4294967295]")
    if search and run == source:
        raise ValueError("seed search must use a fresh env/action seed pair")
    return {
        "search": search,
        "source_env_seed": source[0],
        "source_action_seed": source[1],
        "run_env_seed": run[0],
        "run_action_seed": run[1],
    }


def _slot_telemetry(slot: Any) -> tuple[float, int, int, int, str, bool, int]:
    return (
        float(slot.clock_s),
        int(slot.router.scored["blue"]),
        int(slot.controller.balls_collected),
        int(len(slot.controller.magazine)),
        _phase_name(slot),
        bool(slot.router._score_eligible("blue", float(slot.clock_s))),
        int(getattr(getattr(slot, "cycle_v2", None), "cycles_completed", 0)),
    )


def _black_policy_camera_envs(obs: dict[str, np.ndarray]) -> list[int]:
    """Return slots with at least one black policy-camera product.

    This is telemetry, not a reset-time polling primitive.  The exact evaluator
    consumes the observation returned by terminal ``env.step`` immediately;
    extra rendering or physics steps here would change the historical episode.
    """

    rgb = np.asarray(obs.get("rgb"))
    if rgb.ndim != 5 or rgb.shape[0] != 2:
        raise RuntimeError(f"policy RGB batch shape changed: {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise RuntimeError("policy RGB batch contains non-finite values")
    black_products = rgb.std(axis=(2, 3, 4)) <= 1.0
    return [int(index) for index in np.flatnonzero(black_products.any(axis=1))]


def _preallocate(joint_count: int, *, steps: int = STEPS) -> dict[str, np.ndarray]:
    return {
        "robot_position": np.empty((steps, 3), np.float32),
        "robot_orientation_wxyz": np.empty((steps, 4), np.float32),
        "robot_joint_position": np.empty((steps, joint_count), np.float32),
        "robot_joint_velocity": np.empty((steps, joint_count), np.float32),
        "fuel_position": np.empty((steps, FUEL_COUNT, 3), np.float32),
        "fuel_orientation_wxyz": np.empty((steps, FUEL_COUNT, 4), np.float32),
        "mechanism": np.empty((steps, 3), np.float32),
        "clock_s": np.empty(steps, np.float32),
        "score": np.empty(steps, np.int32),
        "collected": np.empty(steps, np.int32),
        "magazine": np.empty(steps, np.int16),
        "phase": np.empty(steps, dtype="<U24"),
        "hub_active": np.empty(steps, bool),
        "cycles": np.empty(steps, np.int16),
        "action": np.empty((steps, 2, 7), np.float32),
        "proprio": np.empty((steps, 30), np.float32),
        "privileged": np.empty((steps, 26), np.float32),
        "reward": np.empty(steps, np.float32),
        "done": np.empty(steps, bool),
    }


def _warmup_horizon_summary(
    dones: np.ndarray,
    episode_stats: dict[int, dict[str, Any]],
    *,
    transition: int,
) -> dict[str, Any]:
    """Validate the one allowed generation-advance horizon.

    Historical ``episode_index`` is a global evaluator output-row ordinal.  A
    generation advance is therefore valid only when both slots complete the
    same healthy 1600-transition horizon.  The runtime calls this immediately
    after ``env.step`` and relies on that call's normal auto-reset; no reset API
    is invoked here or by the caller.
    """

    done_mask = np.asarray(dones, dtype=bool)
    if done_mask.shape != (2,):
        raise RuntimeError(f"warmup done mask shape changed: {done_mask.shape}")
    if int(transition) != STEPS:
        raise RuntimeError(
            f"warmup horizon ended at transition {transition}, expected {STEPS}"
        )
    if not bool(done_mask.all()):
        raise RuntimeError(
            "warmup horizon was not synchronized across env0/env1: "
            f"dones={done_mask.astype(int).tolist()}"
        )
    outcomes: dict[str, dict[str, Any]] = {}
    for env_index in (0, 1):
        terminal = dict(episode_stats.get(env_index, {}))
        reason = terminal.get("terminal_reason")
        if reason != "horizon":
            raise RuntimeError(
                f"warmup env{env_index} was not a healthy horizon: {terminal!r}"
            )
        outcomes[str(env_index)] = {
            "env_index": env_index,
            "terminal_reason": str(reason),
            "scored": int(terminal.get("scored", -1)),
            "collected": int(terminal.get("collected", -1)),
            "cycles_completed": int(terminal.get("cycles_completed", 0)),
        }
    return {
        "policy_transitions": int(transition),
        "synchronized": True,
        "healthy": True,
        "normal_terminal_auto_reset": True,
        "outcomes": outcomes,
    }


def _selected_env_warmup_summary(
    dones: np.ndarray,
    episode_stats: dict[int, dict[str, Any]],
    *,
    transition: int,
    selected_env_index: int,
) -> dict[str, Any]:
    """Validate one selected slot's exact first-generation horizon.

    The unselected slot is deliberately not a health gate: the exact evaluator
    keeps driving it and may auto-reset it independently.  Only the selected
    slot must finish a healthy 1600-transition horizon before its normal reset
    observation becomes the generation-1 capture start.
    """

    done_mask = np.asarray(dones, dtype=bool)
    selected_env_index = int(selected_env_index)
    if done_mask.shape != (2,):
        raise RuntimeError(f"warmup done mask shape changed: {done_mask.shape}")
    if selected_env_index not in (0, 1):
        raise RuntimeError(f"selected env index changed: {selected_env_index}")
    if int(transition) != STEPS:
        raise RuntimeError(
            f"selected env warmup ended at transition {transition}, expected {STEPS}"
        )
    if not bool(done_mask[selected_env_index]):
        raise RuntimeError(
            f"selected env{selected_env_index} did not terminate at its warmup boundary"
        )
    selected_terminal = dict(episode_stats.get(selected_env_index, {}))
    if selected_terminal.get("terminal_reason") != "horizon":
        raise RuntimeError(
            f"selected warmup env{selected_env_index} was not a healthy horizon: "
            f"{selected_terminal!r}"
        )
    other_index = 1 - selected_env_index
    other_terminal_raw = (
        dict(episode_stats.get(other_index, {})) if bool(done_mask[other_index]) else None
    )
    other_terminal = (
        {
            "terminal_reason": str(other_terminal_raw.get("terminal_reason", "")),
            "scored": int(other_terminal_raw.get("scored", -1)),
            "collected": int(other_terminal_raw.get("collected", -1)),
            "cycles_completed": int(other_terminal_raw.get("cycles_completed", 0)),
        }
        if other_terminal_raw is not None
        else None
    )
    return {
        "policy_transitions": int(transition),
        "selected_env_index": selected_env_index,
        "selected_terminal": {
            "terminal_reason": "horizon",
            "scored": int(selected_terminal.get("scored", -1)),
            "collected": int(selected_terminal.get("collected", -1)),
            "cycles_completed": int(selected_terminal.get("cycles_completed", 0)),
        },
        "other_env_index": other_index,
        "other_env_done_on_selected_boundary": bool(done_mask[other_index]),
        "other_env_terminal_on_selected_boundary": other_terminal,
        "normal_terminal_auto_reset": True,
    }


def _trim_advanced_capture(
    arrays_by_env: dict[int, dict[str, np.ndarray]],
    *,
    transitions: int,
) -> tuple[dict[int, dict[str, np.ndarray]], int]:
    """Normalize a second-generation horizon to the fixed 1600-frame schema.

    A freshly auto-reset slot can require 1601 policy transitions because the
    160 s clock is accumulated in 1/30 s control ticks.  The exact evaluator
    records this edge as a 1601-transition episode.  For the fixed 10 fps,
    1600-frame video trace we omit only its leading t=0 pre-action sample and
    retain the final 1600 action/state rows, including the real terminal row.
    Nothing is interpolated, copied between slots, or relabelled.
    """

    transitions = int(transitions)
    if transitions not in (STEPS, STEPS + 1):
        raise RuntimeError(
            "advanced capture must terminate in 1600 or 1601 transitions, got "
            f"{transitions}"
        )
    omitted = transitions - STEPS
    normalized: dict[int, dict[str, np.ndarray]] = {}
    for env_index, arrays in arrays_by_env.items():
        trimmed: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            array = np.asarray(value)
            if array.shape[0] != STEPS + 1:
                raise RuntimeError(
                    f"advanced env{env_index} field {key} allocation changed: "
                    f"{array.shape[0]} != {STEPS + 1}"
                )
            trimmed[key] = array[omitted:transitions]
        if int(trimmed["score"][0]) != 0:
            raise RuntimeError(
                f"advanced env{env_index} retained trace did not start at score zero"
            )
        if bool(trimmed["done"][:-1].any()) or not bool(trimmed["done"][-1]):
            raise RuntimeError(
                f"advanced env{env_index} retained trace has an invalid done boundary"
            )
        normalized[env_index] = trimmed
    return normalized, omitted


def _capture_visual_state(
    arrays: dict[str, np.ndarray], step: int, slot: Any, obs: dict[str, np.ndarray]
) -> None:
    position, orientation = slot.articulation.get_world_pose()
    joints = slot.articulation.get_joint_positions()
    joint_velocity = slot.articulation.get_joint_velocities()
    fuel_position, fuel_orientation = slot.fuel.get_world_poses()
    if fuel_position.shape != (FUEL_COUNT, 3) or fuel_orientation.shape != (FUEL_COUNT, 4):
        raise RuntimeError(
            "exact target field no longer exposes all 456 FUEL poses: "
            f"{fuel_position.shape}/{fuel_orientation.shape}"
        )
    arrays["robot_position"][step] = position
    arrays["robot_orientation_wxyz"][step] = orientation
    arrays["robot_joint_position"][step] = joints
    arrays["robot_joint_velocity"][step] = joint_velocity
    arrays["fuel_position"][step] = fuel_position
    arrays["fuel_orientation_wxyz"][step] = fuel_orientation
    controller = slot.controller
    arrays["mechanism"][step] = (
        float(controller.storage_position),
        float(controller.container_extension),
        float(controller.intake_extension),
    )
    (
        arrays["clock_s"][step],
        arrays["score"][step],
        arrays["collected"][step],
        arrays["magazine"][step],
        arrays["phase"][step],
        arrays["hub_active"][step],
        arrays["cycles"][step],
    ) = _slot_telemetry(slot)
    arrays["proprio"][step] = np.asarray(obs["proprio"][slot.index], np.float32)
    arrays["privileged"][step] = np.asarray(obs["privileged"][slot.index], np.float32)


def _select_race_winner(
    arrays_by_env: dict[int, dict[str, np.ndarray]],
    terminals: dict[int, dict[str, Any] | None],
    early_termination_steps: dict[int, int | None],
    *,
    preferred_env_index: int,
) -> tuple[int, dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, Any]]]:
    """Select one complete healthy 200+ trace without mixing simulator slots.

    The returned arrays object is the exact object owned by the selected slot.
    This helper is intentionally Isaac-free so the fail-closed race gate can be
    covered by focused CPU tests.
    """

    outcomes: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[int, int]] = []
    for env_index in sorted(arrays_by_env):
        arrays = arrays_by_env[env_index]
        terminal = terminals.get(env_index)
        early_step = early_termination_steps.get(env_index)
        start_score = int(np.asarray(arrays["score"])[0])
        array_score = int(np.asarray(arrays["score"])[-1])
        terminal_score = int(terminal.get("scored", -1)) if terminal else -1
        terminal_reason = terminal.get("terminal_reason") if terminal else None
        final_done = bool(np.asarray(arrays["done"])[-1])
        rejection_reasons: list[str] = []
        if start_score != 0:
            rejection_reasons.append("nonzero_start_score")
        if early_step is not None:
            rejection_reasons.append("early_termination")
        if not final_done:
            rejection_reasons.append("missing_final_done")
        if terminal_reason != "horizon":
            rejection_reasons.append("unhealthy_terminal")
        if terminal_score != array_score:
            rejection_reasons.append("terminal_score_mismatch")
        if terminal_score < MIN_LIVE_SCORE:
            rejection_reasons.append("below_publication_gate")
        is_eligible = not rejection_reasons
        outcomes[str(env_index)] = {
            "env_index": env_index,
            "start_score": start_score,
            "array_terminal_score": array_score,
            "live_terminal_score": terminal_score,
            "terminal_reason": terminal_reason,
            "early_termination_step": early_step,
            "final_done": final_done,
            "eligible": is_eligible,
            "rejection_reasons": rejection_reasons,
        }
        if is_eligible:
            eligible.append((terminal_score, env_index))

    if not eligible:
        raise RuntimeError(
            "dual-environment race produced no healthy 200+ horizon: "
            + json.dumps(outcomes, sort_keys=True)
        )
    # Score is the primary key.  A tie respects --env-index, then uses the
    # lower slot index for deterministic selection.
    _, winner = max(
        eligible,
        key=lambda item: (
            item[0],
            int(item[1] == int(preferred_env_index)),
            -item[1],
        ),
    )
    selected_terminal = terminals[winner]
    if selected_terminal is None:  # guarded above; keeps the type invariant explicit
        raise RuntimeError("selected race winner is missing terminal metadata")
    return winner, arrays_by_env[winner], selected_terminal, outcomes


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _seed_mode(args)
    if os.environ.get("FRC_POLICY_SPEED_SCALE", "1.0") != "1.0":
        raise RuntimeError("FRC_POLICY_SPEED_SCALE must be exactly 1.0")
    template_sha = require_file_sha(args.template, args.expected_template_sha256, "template")
    checkpoint_sha = require_file_sha(
        args.checkpoint, args.expected_checkpoint_sha256, "policy checkpoint"
    )
    prefix_sha = require_file_sha(
        args.prefix_checkpoint,
        args.expected_prefix_checkpoint_sha256,
        "prefix checkpoint",
    )
    code = verify_code_snapshot(
        args.code_root, args.code_archive, args.expected_code_archive_sha256
    )
    source_checkpoint_sha = (
        args.expected_source_checkpoint_sha256
        or args.expected_checkpoint_sha256
    )
    compatible_checkpoint = source_checkpoint_sha.lower() != checkpoint_sha.lower()
    if compatible_checkpoint and not args.allow_compatible_checkpoint:
        raise ValueError(
            "source bundle and policy checkpoint differ; "
            "pass --allow-compatible-checkpoint for an explicit closed-loop search"
        )
    if args.allow_compatible_checkpoint and not seeds["search"]:
        raise ValueError("compatible-checkpoint mode requires fresh run seeds")
    if args.allow_compatible_checkpoint and args.action_source != "closed-loop":
        raise ValueError("compatible-checkpoint mode requires closed-loop actions")

    target = load_replay_bundle(
        args.bundle,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_checkpoint_sha256=source_checkpoint_sha,
        expected_env_seed=args.expected_env_seed,
        expected_action_seed=args.expected_action_seed,
        # The immutable score-202 source bundle is always env1.  A seed-search
        # or dual-environment race may select either live simulator slot
        # without relabelling that immutable source.
        expected_env_index=1,
    )
    companion = load_replay_bundle(
        args.companion_bundle,
        expected_bundle_sha256=args.expected_companion_bundle_sha256,
        expected_checkpoint_sha256=source_checkpoint_sha,
        expected_env_seed=args.expected_env_seed,
        expected_action_seed=args.expected_action_seed,
        expected_env_index=0,
        expected_steps=1601,
    )
    if not seeds["search"] and not args.race_both_envs and args.env_index != 1:
        raise ValueError("the custody-verified score-202 target must use env index 1")
    if str(target.episode.get("prefix_sha256", "")).lower() != prefix_sha:
        raise ValueError("archived target prefix SHA256 differs from supplied checkpoint")
    if args.trace_out.suffix.lower() != ".npz":
        raise ValueError("--trace-out must end in .npz")
    for path in (args.trace_out, args.provenance_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")

    sys.path.insert(0, str((args.code_root / "src").resolve()))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    env = None
    try:
        import torch
        from frc_rebuilt.rl import distributed as D
        from frc_rebuilt.rl import stage_d as _stage_d
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.policy_v2 import (
            LEGACY_PROPRIO_DIM,
            apply_executed_action_policy,
            compose_phase_actions,
            validate_composite_metadata,
        )
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        # No extra camera, annotator, or Replicator call is made in pass one.
        np.random.seed(seeds["run_action_seed"])
        torch.manual_seed(seeds["run_action_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds["run_action_seed"])
        try:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(args.checkpoint, map_location="cpu")
        policy_metadata = validate_composite_metadata(payload.get("stagec_v2"), prefix_sha)
        if not compatible_checkpoint and policy_metadata != target.episode.get("stagec_v2_metadata"):
            raise ValueError("checkpoint Stage-C metadata differs from source contract")
        run_episode = dict(target.episode)
        run_episode["env_seed"] = seeds["run_env_seed"]
        run_episode["action_seed"] = seeds["run_action_seed"]
        if compatible_checkpoint:
            # The compact FSG8 bundle supplies the immutable simulator contract,
            # while the actual checkpoint supplies its own embedded Stage-C
            # policy metadata.  This is a new closed-loop evaluation, never a
            # replay or relabelled historical episode.
            run_episode["checkpoint_sha256"] = checkpoint_sha
            run_episode["stagec_v2_metadata"] = policy_metadata
        env = VecCompetitionEnv(
            build_exact_cfg(VecEnvCfg, template=args.template, episode=run_episode)
        )
        if not bool(getattr(env, "_camera_ready", False)):
            raise RuntimeError("original policy-camera initialization failed")
        expected_camera_products = int(target.episode["num_envs"]) * len(env.camera_names)
        if len(env.cameras) != expected_camera_products:
            raise RuntimeError(
                f"policy-camera contract changed: {len(env.cameras)} != {expected_camera_products}"
            )

        # Match the exact evaluator's constructor reset and zero-action startup step.
        obs, _, _, _ = env.step(np.zeros((2, 7), np.float32))
        if bool((obs["rgb"].std(axis=(2, 3, 4)) <= 1.0).any()):
            raise RuntimeError("exact evaluator produced a black startup policy frame")
        policy_frames = D.to_policy_frames(obs["rgb"])
        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=policy_frames.shape[1],
                frame_h=policy_frames.shape[2],
                frame_w=policy_frames.shape[3],
                proprio_dim=obs["proprio"].shape[1],
                privileged_dim=obs["privileged"].shape[1],
                stddev_start=_metadata_number(policy_metadata, "stddev_start", float),
                stddev_end=_metadata_number(policy_metadata, "stddev_end", float),
                stddev_steps=_metadata_number(policy_metadata, "stddev_steps", int),
            )
        )
        prefix_agent = DrQV2Agent(
            DrQConfig(
                frame_channels=policy_frames.shape[1],
                frame_h=policy_frames.shape[2],
                frame_w=policy_frames.shape[3],
                proprio_dim=LEGACY_PROPRIO_DIM,
                privileged_dim=obs["privileged"].shape[1],
            )
        )
        agent.load(str(args.checkpoint))
        prefix_agent.load(str(args.prefix_checkpoint))
        np.random.seed(seeds["run_action_seed"])
        torch.manual_seed(seeds["run_action_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds["run_action_seed"])

        def _closed_loop_actions(current_obs: dict[str, np.ndarray]) -> np.ndarray:
            current_frames = D.to_policy_frames(current_obs["rgb"])
            candidate = agent.act(
                current_frames,
                current_obs["proprio"],
                explore=args.policy_mode == "explore",
            ).astype(np.float32)
            prefix_view = stage_d_prefix_view(
                _stage_d,
                current_obs["proprio"],
                episode_len_s=float(target.episode["episode_len_s"]),
                legacy_dim=LEGACY_PROPRIO_DIM,
            )
            prefix = prefix_agent.act(
                current_frames, prefix_view, explore=False
            ).astype(np.float32)
            composed = compose_phase_actions(
                prefix, candidate, current_obs["proprio"]
            )
            return apply_executed_action_policy(
                composed,
                current_obs["proprio"],
                intake_during_return=bool(
                    policy_metadata.get("intake_during_return", False)
                ),
            ).astype(np.float32)

        advance_full_horizons = int(args.advance_full_horizons)
        advance_selected_env_horizons = int(args.advance_selected_env_horizons)
        advance_generation = advance_full_horizons or advance_selected_env_horizons
        warmup_summary: dict[str, Any] | None = None
        post_reset_camera_status: dict[str, Any] | None = None
        evaluator_rows_seen = 0
        if advance_full_horizons:
            # Execute the complete preceding evaluator generation.  The final
            # env.step performs the exact environment's normal terminal
            # auto-reset and returns the first observation of generation 1.
            # Calling reset/reset_slots here would consume a different RNG
            # sequence and is intentionally forbidden.
            for warmup_step in range(STEPS):
                warmup_actions = _closed_loop_actions(obs)
                obs, _, warmup_dones, warmup_info = env.step(warmup_actions)
                if bool(np.asarray(warmup_dones, dtype=bool).any()):
                    warmup_summary = _warmup_horizon_summary(
                        warmup_dones,
                        warmup_info.get("episode_stats", {}),
                        transition=warmup_step + 1,
                    )
                    evaluator_rows_seen += int(
                        np.asarray(warmup_dones, dtype=bool).sum()
                    )
                    break
                if warmup_step % 100 == 0:
                    print(
                        "VERIFIED_TRACE_WARMUP_PROGRESS "
                        f"step={warmup_step + 1}/{STEPS}",
                        flush=True,
                    )
            if warmup_summary is None:
                raise RuntimeError(
                    f"warmup horizon did not terminate after exactly {STEPS} transitions"
                )
            post_reset_black_envs = _black_policy_camera_envs(obs)
            post_reset_camera_status = {
                "black_env_indices_on_auto_reset_observation": post_reset_black_envs,
                "extra_poll_or_render_steps": 0,
                "exact_auto_reset_observation_retained": True,
                "recovery_policy_transition": {
                    str(env_index): (0 if env_index not in post_reset_black_envs else None)
                    for env_index in (0, 1)
                },
            }
            for warmup_env_index, warmup_slot in enumerate(env.slots):
                clock_s, score, *_ = _slot_telemetry(warmup_slot)
                if abs(float(clock_s)) > 1e-6 or int(score) != 0:
                    raise RuntimeError(
                        "normal warmup auto-reset did not return a fresh generation: "
                        f"env{warmup_env_index} clock={clock_s} score={score}"
                    )
            print(
                "VERIFIED_TRACE_WARMUP_DONE "
                + json.dumps(warmup_summary, sort_keys=True),
                flush=True,
            )
        elif advance_selected_env_horizons:
            selected_warmup_env = int(args.env_index)
            for warmup_step in range(STEPS):
                warmup_actions = _closed_loop_actions(obs)
                obs, _, warmup_dones, warmup_info = env.step(warmup_actions)
                done_indices = [
                    int(index)
                    for index in np.flatnonzero(
                        np.asarray(warmup_dones, dtype=bool)
                    )
                ]
                if selected_warmup_env in done_indices:
                    warmup_summary = _selected_env_warmup_summary(
                        warmup_dones,
                        warmup_info.get("episode_stats", {}),
                        transition=warmup_step + 1,
                        selected_env_index=selected_warmup_env,
                    )
                    evaluator_rows_seen += len(done_indices)
                    break
                # Preserve the evaluator's global output-row ordering while
                # allowing the unselected slot to reset asynchronously.
                evaluator_rows_seen += len(done_indices)
                if warmup_step % 100 == 0:
                    print(
                        "VERIFIED_TRACE_SELECTED_WARMUP_PROGRESS "
                        f"env={selected_warmup_env} step={warmup_step + 1}/{STEPS} "
                        f"other_rows={evaluator_rows_seen}",
                        flush=True,
                    )
            if warmup_summary is None:
                raise RuntimeError(
                    f"selected env{selected_warmup_env} warmup did not terminate "
                    f"after exactly {STEPS} transitions"
                )
            post_reset_black_envs = _black_policy_camera_envs(obs)
            post_reset_camera_status = {
                "black_env_indices_on_auto_reset_observation": post_reset_black_envs,
                "extra_poll_or_render_steps": 0,
                "exact_auto_reset_observation_retained": True,
                "recovery_policy_transition": {
                    str(env_index): (0 if env_index not in post_reset_black_envs else None)
                    for env_index in (0, 1)
                },
            }
            selected_slot = env.slots[selected_warmup_env]
            clock_s, score, *_ = _slot_telemetry(selected_slot)
            if abs(float(clock_s)) > 1e-6 or int(score) != 0:
                raise RuntimeError(
                    "selected env normal auto-reset did not return a fresh generation: "
                    f"env{selected_warmup_env} clock={clock_s} score={score}"
                )
            warmup_summary["evaluator_rows_completed_before_capture"] = (
                evaluator_rows_seen
            )
            print(
                "VERIFIED_TRACE_SELECTED_WARMUP_DONE "
                + json.dumps(warmup_summary, sort_keys=True),
                flush=True,
            )

        record_indices = (0, 1) if args.race_both_envs else (int(args.env_index),)
        slots = {env_index: env.slots[env_index] for env_index in record_indices}
        joint_names = [
            str(value) for value in slots[record_indices[0]].articulation.dof_names
        ]
        joint_count = len(joint_names)
        if joint_count <= 0:
            raise RuntimeError("target robot articulation contains no joints")
        for env_index, slot in slots.items():
            slot_joint_names = [str(value) for value in slot.articulation.dof_names]
            if slot_joint_names != joint_names:
                raise RuntimeError(
                    f"env{env_index} articulation joint order differs from race peer"
                )
            if int(slot.fuel.count) != FUEL_COUNT:
                raise RuntimeError(
                    f"env{env_index} field FUEL count changed: {slot.fuel.count}"
                )
        capture_capacity = STEPS + int(bool(advance_generation))
        arrays_by_env = {
            env_index: _preallocate(joint_count, steps=capture_capacity)
            for env_index in record_indices
        }
        references = {0: companion, 1: target}
        compare_archived = not seeds["search"] and not advance_generation
        divergence = {
            env_index: {
                "max_proprio_error": 0.0,
                "max_privileged_error": 0.0,
                "max_reward_error": 0.0,
                "score_timeline_mismatches": 0,
                "max_score_timeline_error": 0,
            }
            for env_index in record_indices
        }
        max_target_action_error = 0.0
        max_companion_action_error = 0.0
        terminals: dict[int, dict[str, Any] | None] = {
            env_index: None for env_index in record_indices
        }
        early_termination_steps: dict[int, int | None] = {
            env_index: None for env_index in record_indices
        }
        capture_transitions: int | None = None
        capture_global_episode_indices: dict[int, int] = {}
        for step in range(capture_capacity):
            if post_reset_camera_status is not None:
                black_now = set(_black_policy_camera_envs(obs))
                recovery = post_reset_camera_status["recovery_policy_transition"]
                for env_index in (0, 1):
                    key = str(env_index)
                    if recovery[key] is None and env_index not in black_now:
                        recovery[key] = step
            expected_scores: dict[int, int | None] = {
                env_index: None for env_index in record_indices
            }
            for env_index in record_indices:
                arrays_for_env = arrays_by_env[env_index]
                _capture_visual_state(
                    arrays_for_env, step, slots[env_index], obs
                )
                if step == 0 and int(arrays_for_env["score"][step]) != 0:
                    if args.race_both_envs:
                        raise RuntimeError(
                            f"fresh policy run env{env_index} did not start at score zero"
                        )
                    raise RuntimeError("fresh policy run did not start at score zero")
            if compare_archived:
                for env_index in record_indices:
                    arrays_for_env = arrays_by_env[env_index]
                    reference = references[env_index]
                    metrics = divergence[env_index]
                    metrics["max_proprio_error"] = max(
                        metrics["max_proprio_error"],
                        float(
                            np.max(
                                np.abs(
                                    arrays_for_env["proprio"][step]
                                    - reference.proprio[step]
                                )
                            )
                        ),
                    )
                    metrics["max_privileged_error"] = max(
                        metrics["max_privileged_error"],
                        float(
                            np.max(
                                np.abs(
                                    arrays_for_env["privileged"][step]
                                    - reference.privileged[step]
                                )
                            )
                        ),
                    )
                    expected_score = int(
                        round(float(reference.proprio[step, 14]) * 20.0)
                    )
                    expected_scores[env_index] = expected_score
                    score_error = abs(
                        int(arrays_for_env["score"][step]) - expected_score
                    )
                    metrics["score_timeline_mismatches"] += int(score_error != 0)
                    metrics["max_score_timeline_error"] = max(
                        metrics["max_score_timeline_error"], score_error
                    )

            if args.action_source == "closed-loop":
                actions = _closed_loop_actions(obs)
            else:
                actions = np.stack(
                    (companion.action[step], target.action[step]), axis=0
                ).astype(np.float32, copy=False)
            if compare_archived:
                max_target_action_error = max(
                    max_target_action_error,
                    float(np.max(np.abs(actions[1] - target.action[step]))),
                )
                max_companion_action_error = max(
                    max_companion_action_error,
                    float(np.max(np.abs(actions[0] - companion.action[step]))),
                )
            for arrays_for_env in arrays_by_env.values():
                arrays_for_env["action"][step] = actions
            obs, rewards, dones, info = env.step(actions)
            episode_stats = info.get("episode_stats", {})
            for done_index in np.flatnonzero(np.asarray(dones, dtype=bool)):
                done_index = int(done_index)
                capture_global_episode_indices.setdefault(
                    done_index, evaluator_rows_seen
                )
                evaluator_rows_seen += 1
            for env_index in record_indices:
                arrays_for_env = arrays_by_env[env_index]
                arrays_for_env["reward"][step] = float(rewards[env_index])
                arrays_for_env["done"][step] = bool(dones[env_index])
                if compare_archived:
                    metrics = divergence[env_index]
                    metrics["max_reward_error"] = max(
                        metrics["max_reward_error"],
                        abs(
                            float(rewards[env_index])
                            - float(references[env_index].reward[step])
                        ),
                    )
                if bool(dones[env_index]):
                    if terminals[env_index] is None:
                        terminals[env_index] = dict(episode_stats.get(env_index, {}))
                    if step + 1 < STEPS:
                        if not args.race_both_envs:
                            raise RuntimeError(
                                f"target terminated early at transition {step + 1}"
                            )
                        if early_termination_steps[env_index] is None:
                            early_termination_steps[env_index] = step + 1
            if advance_full_horizons and bool(np.asarray(dones, dtype=bool).any()):
                if not bool(np.asarray(dones, dtype=bool).all()):
                    raise RuntimeError(
                        "advanced target horizon was not synchronized across env0/env1: "
                        f"transition={step + 1} "
                        f"dones={np.asarray(dones, dtype=int).tolist()}"
                    )
                capture_transitions = step + 1
                break
            if advance_selected_env_horizons and bool(dones[int(args.env_index)]):
                capture_transitions = step + 1
                break
            if step % 100 == 0:
                progress_scores = " ".join(
                    f"score{env_index}="
                    f"{int(arrays_by_env[env_index]['score'][step])}"
                    for env_index in record_indices
                )
                selected_metrics = divergence[int(args.env_index)]
                print(
                    f"VERIFIED_TRACE_PROGRESS step={step + 1}/{capture_capacity} "
                    + (
                        f"{progress_scores} "
                        if args.race_both_envs
                        else f"score={int(arrays_by_env[int(args.env_index)]['score'][step])} "
                    )
                    + (
                        "archived_comparison=skipped_advanced_generation"
                        if advance_generation
                        else "archived_comparison=skipped_seed_search"
                        if seeds["search"]
                        else (
                            f"expected_score={expected_scores[int(args.env_index)]} "
                            f"max_proprio_abs={selected_metrics['max_proprio_error']:.6g} "
                            f"max_privileged_abs={selected_metrics['max_privileged_error']:.6g} "
                            f"max_target_action_abs={max_target_action_error:.6g} "
                            f"max_companion_action_abs={max_companion_action_error:.6g}"
                        )
                    ),
                    flush=True,
                )

        leading_frames_omitted = 0
        if advance_generation:
            if capture_transitions is None:
                raise RuntimeError(
                    "advanced target horizon did not terminate within "
                    f"{capture_capacity} transitions"
                )
            arrays_by_env, leading_frames_omitted = _trim_advanced_capture(
                arrays_by_env,
                transitions=capture_transitions,
            )
        else:
            capture_transitions = STEPS

        race_outcomes = None
        if args.race_both_envs:
            selected_env_index, arrays, terminal, race_outcomes = _select_race_winner(
                arrays_by_env,
                terminals,
                early_termination_steps,
                preferred_env_index=int(args.env_index),
            )
            print(
                "VERIFIED_TRACE_RACE_WINNER "
                + json.dumps(
                    {
                        "selected_env_index": selected_env_index,
                        "live_terminal_score": int(terminal["scored"]),
                        "outcomes": race_outcomes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            selected_env_index = int(args.env_index)
            arrays = arrays_by_env[selected_env_index]
            terminal = terminals[selected_env_index]
            if not bool(arrays["done"][-1]):
                raise RuntimeError("target did not terminate at the 1600-transition horizon")
            if not terminal or terminal.get("terminal_reason") != "horizon":
                raise RuntimeError(f"target did not finish a healthy horizon: {terminal!r}")
            live_score = int(terminal.get("scored", -1))
            if live_score < MIN_LIVE_SCORE:
                run_kind = (
                    "closed-loop seed search"
                    if seeds["search"]
                    else f"{args.action_source} re-simulation"
                )
                raise RuntimeError(
                    f"fresh {run_kind} scored {live_score}; refusing "
                    f"to publish below {MIN_LIVE_SCORE}"
                )
        live_score = int(terminal.get("scored", -1))
        selected_global_episode_index = capture_global_episode_indices.get(
            selected_env_index
        )
        if selected_global_episode_index is None:
            raise RuntimeError(
                f"selected env{selected_env_index} is missing its evaluator row ordinal"
            )
        robot_quat_error = normalized_quaternion_error(
            arrays["robot_orientation_wxyz"]
        )
        fuel_quat_error = normalized_quaternion_error(
            arrays["fuel_orientation_wxyz"]
        )
        if robot_quat_error > 1e-3 or fuel_quat_error > 1e-3:
            raise RuntimeError(
                "captured orientation convention/normalization changed: "
                f"robot={robot_quat_error} fuel={fuel_quat_error}"
            )

        selected_reference = references[selected_env_index]
        # Seed searches retain the immutable env1 target bundle as source
        # custody.  An exact-source dual race that selects env0 instead points
        # render custody at the env0 companion bundle explicitly.
        publication_bundle = target if seeds["search"] else selected_reference
        source_contract = dict(target.episode["stage_d_contract"])
        selected_metrics = divergence[selected_env_index]
        metadata: dict[str, Any] = {
            "schema": TRACE_SCHEMA,
            "steps": STEPS,
            "fps": 10.0,
            "episode_len_s": 160.0,
            "fuel_count": FUEL_COUNT,
            "joint_count": joint_count,
            "joint_names": joint_names,
            "orientation_convention": "WXYZ",
            "mechanism_columns": [
                "storage_position",
                "container_extension",
                "intake_extension",
            ],
            "checkpoint_sha256": checkpoint_sha,
            "prefix_checkpoint_sha256": prefix_sha,
            "bundle_sha256": publication_bundle.sha256,
            "companion_bundle_sha256": companion.sha256,
            "template_sha256": template_sha,
            "code_archive_sha256": code["archive_sha256"],
            "code_verified_files": int(code["verified_files"]),
            "env_seed": seeds["run_env_seed"],
            "action_seed": seeds["run_action_seed"],
            "env_index": selected_env_index,
            "episode_generation": int(bool(advance_generation)),
            "evaluator_global_episode_index": selected_global_episode_index,
            "contract": source_contract,
            "source_capture": {
                "target_env_seed": int(target.episode["env_seed"]),
                "target_action_seed": int(target.episode["action_seed"]),
                "target_env_index": int(target.episode["env_index"]),
                "target_episode_index": int(target.episode.get("episode_index", -1)),
                "target_bundle_sha256": target.sha256,
                "companion_env_index": int(companion.episode["env_index"]),
                "companion_episode_index": int(
                    companion.episode.get("episode_index", -1)
                ),
                "companion_bundle_sha256": companion.sha256,
            },
            "classification": {
                "type": (
                    (
                        "compatible_checkpoint_closed_loop_seed_search"
                        if compatible_checkpoint
                        else "deterministic_closed_loop_seed_search"
                    )
                    if seeds["search"]
                    else (
                        "deterministic_closed_loop_policy_resimulation"
                        if args.action_source == "closed-loop"
                        else "deterministic_archived_action_resimulation_diagnostic"
                    )
                ),
                "bit_exact_replay": False,
                "replay": False,
                "seed_search": seeds["search"],
                "policy_cameras_retained": len(env.cameras),
                "added_render_products": 0,
                "extra_physics_steps_for_capture": (
                    int(bool(advance_generation))
                    * STEPS
                    * int(env.spec.physics_steps_per_action)
                ),
                "pre_capture_policy_transitions": (
                    int(bool(advance_generation)) * STEPS
                ),
                "full_horizon_advance": advance_full_horizons,
                "selected_env_horizon_advance": advance_selected_env_horizons,
                "action_source": args.action_source,
                "policy_mode": args.policy_mode,
                "compatible_checkpoint": compatible_checkpoint,
                "prefix_observation": "stage_d.pin_prefix_view",
            },
            "policy_contract": {
                "checkpoint_stagec_v2_metadata": policy_metadata,
                "source_bundle_checkpoint_sha256": source_checkpoint_sha.lower(),
                "source_bundle_stagec_v2_metadata": target.episode.get(
                    "stagec_v2_metadata"
                ),
            },
            "publication_min_score": MIN_LIVE_SCORE,
            "live_terminal_score": live_score,
            "live_terminal": terminal,
            "archived_source_terminal": {
                key: (
                    target.episode.get(key)
                    if seeds["search"]
                    else selected_reference.episode.get(key)
                )
                for key in ("scored", "collected", "cycles_completed", "terminal_reason")
            },
            "measured_divergence": (
                {
                    "performed": False,
                    "reason": "advanced_generation_has_no_archived_per_step_reference",
                }
                if advance_generation
                else {
                    "performed": False,
                    "reason": "fresh_seed_search_has_no_archived_per_step_reference",
                }
                if seeds["search"]
                else {
                    "performed": True,
                    "selected_env_index": selected_env_index,
                    "max_proprio_abs": selected_metrics["max_proprio_error"],
                    "max_privileged_abs": selected_metrics["max_privileged_error"],
                    "max_reward_abs": selected_metrics["max_reward_error"],
                    "max_target_action_abs": max_target_action_error,
                    "max_companion_action_abs": max_companion_action_error,
                    "score_timeline_mismatch_frames": selected_metrics[
                        "score_timeline_mismatches"
                    ],
                    "max_score_timeline_abs": selected_metrics[
                        "max_score_timeline_error"
                    ],
                }
            ),
            "quaternion_max_norm_error": {
                "robot": robot_quat_error,
                "fuel": fuel_quat_error,
            },
            "executed_action_sha256": hashlib.sha256(
                np.ascontiguousarray(arrays["action"]).tobytes()
            ).hexdigest(),
            "recorder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        }
        if advance_full_horizons:
            metadata["full_horizon_advancement"] = {
                "requested_full_horizons": advance_full_horizons,
                "completed_full_horizons": advance_full_horizons,
                "reset_method": "normal_terminal_auto_reset",
                "manual_reset_calls": 0,
                "warmup": warmup_summary,
                "post_reset_policy_cameras": post_reset_camera_status,
                "captured_generation": advance_full_horizons,
                "captured_policy_transitions": capture_transitions,
                "trace_frames": STEPS,
                "leading_pre_action_frames_omitted": leading_frames_omitted,
                "frame_omission_reason": (
                    "none"
                    if leading_frames_omitted == 0
                    else "1601_transition_float_clock_edge_fixed_1600_frame_schema"
                ),
            }
        elif advance_selected_env_horizons:
            metadata["selected_env_horizon_advancement"] = {
                "requested_selected_env_horizons": advance_selected_env_horizons,
                "completed_selected_env_horizons": advance_selected_env_horizons,
                "selected_env_index": int(args.env_index),
                "other_env_health_gate": False,
                "other_env_driven_closed_loop": True,
                "other_env_async_auto_resets_allowed": True,
                "reset_method": "normal_terminal_auto_reset",
                "manual_reset_calls": 0,
                "warmup": warmup_summary,
                "post_reset_policy_cameras": post_reset_camera_status,
                "captured_generation": 1,
                "captured_policy_transitions": capture_transitions,
                "trace_frames": STEPS,
                "leading_pre_action_frames_omitted": leading_frames_omitted,
                "frame_omission_reason": (
                    "none"
                    if leading_frames_omitted == 0
                    else "1601_transition_float_clock_edge_fixed_1600_frame_schema"
                ),
                "evaluator_global_episode_index": selected_global_episode_index,
            }
        if args.race_both_envs:
            metadata["source_capture"].update(
                {
                    "selected_live_env_index": selected_env_index,
                    "publication_bundle_role": (
                        "target_env1"
                        if publication_bundle is target
                        else "companion_env0"
                    ),
                }
            )
            metadata["classification"]["dual_target_race"] = True
            metadata["dual_target_race"] = {
                "enabled": True,
                "recorded_env_indices": list(record_indices),
                "preferred_env_index_on_tie": int(args.env_index),
                "selected_env_index": selected_env_index,
                "selection_rule": "highest_score_among_healthy_200plus_horizons",
                "outcomes": race_outcomes,
            }
        # atomic_save_trace adds declarations before validation/publication.
        trace_sha = atomic_save_trace(
            args.trace_out, metadata, arrays, overwrite=args.overwrite
        )
        metadata["fields"] = field_declarations(arrays)
        provenance = {
            "schema": TRACE_PROVENANCE_SCHEMA,
            "trace": {
                "path": str(args.trace_out.resolve()),
                "sha256": trace_sha,
                "bytes": args.trace_out.stat().st_size,
            },
            "inputs": {
                "target_bundle": {"path": str(target.path), "sha256": target.sha256},
                "companion_bundle": {
                    "path": str(companion.path),
                    "sha256": companion.sha256,
                },
                "checkpoint": {
                    "path": str(args.checkpoint.resolve()),
                    "sha256": checkpoint_sha,
                },
                "prefix_checkpoint": {
                    "path": str(args.prefix_checkpoint.resolve()),
                    "sha256": prefix_sha,
                },
                "template": {"path": str(args.template.resolve()), "sha256": template_sha},
                "exact_code": {
                    "root": str(args.code_root.resolve()),
                    "archive": str(args.code_archive.resolve()),
                    **code,
                },
            },
            "trace_metadata": metadata,
        }
        if args.race_both_envs:
            provenance["inputs"]["publication_bundle"] = {
                "path": str(publication_bundle.path),
                "sha256": publication_bundle.sha256,
                "env_index": int(publication_bundle.episode["env_index"]),
            }
        atomic_json(args.provenance_out, provenance)
        print("VERIFIED_TRACE_DONE " + json.dumps(provenance, sort_keys=True), flush=True)
        return provenance
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--companion-bundle", type=Path, required=True)
    parser.add_argument("--expected-companion-bundle-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--expected-source-checkpoint-sha256",
        default=None,
        help=(
            "checkpoint SHA declared by the immutable source bundles when the "
            "evaluated checkpoint differs"
        ),
    )
    parser.add_argument(
        "--allow-compatible-checkpoint",
        action="store_true",
        help=(
            "run a different exact checkpoint against the source Stage-D "
            "simulator contract; requires fresh seeds and closed-loop actions"
        ),
    )
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-prefix-checkpoint-sha256", required=True)
    parser.add_argument("--expected-env-seed", type=int, required=True)
    parser.add_argument("--expected-action-seed", type=int, required=True)
    parser.add_argument(
        "--run-env-seed",
        type=int,
        default=None,
        help="fresh environment seed for closed-loop seed search; requires --run-action-seed",
    )
    parser.add_argument(
        "--run-action-seed",
        type=int,
        default=None,
        help="fresh action/RNG seed for closed-loop seed search; requires --run-env-seed",
    )
    parser.add_argument("--env-index", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--race-both-envs",
        action="store_true",
        help=(
            "capture env0 and env1 in one closed-loop rollout and atomically "
            "publish the higher-scoring healthy 200+ horizon; --env-index is "
            "used only as the deterministic tie preference"
        ),
    )
    parser.add_argument(
        "--advance-full-horizons",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "execute exactly one healthy synchronized 1600-transition horizon "
            "through normal auto-reset before recording generation 1; requires "
            "--race-both-envs and closed-loop actions"
        ),
    )
    parser.add_argument(
        "--advance-selected-env-horizons",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "execute exactly one healthy 1600-transition horizon for --env-index "
            "through its normal auto-reset, then record only that slot's next "
            "generation while the other slot runs/resets normally; mutually "
            "exclusive with --race-both-envs"
        ),
    )
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--expected-code-archive-sha256", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--trace-out", type=Path, required=True)
    parser.add_argument(
        "--action-source",
        choices=("closed-loop", "archived"),
        default="closed-loop",
        help="closed-loop is the publication path; archived is diagnostic only",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("mean", "explore"),
        default="mean",
        help="candidate action mode for the fresh closed-loop evaluation",
    )
    parser.add_argument("--provenance-out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    try:
        _seed_mode(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.provenance_out is None:
        args.provenance_out = args.trace_out.with_suffix(".provenance.json")
    outputs = {args.trace_out.resolve(), args.provenance_out.resolve()}
    protected = {
        args.bundle.resolve(),
        args.companion_bundle.resolve(),
        args.checkpoint.resolve(),
        args.prefix_checkpoint.resolve(),
        args.code_archive.resolve(),
        args.template.resolve(),
    }
    if len(outputs) != 2 or outputs & protected:
        raise SystemExit("trace/provenance outputs must be distinct from every input")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
