r"""Reproducible, read-only Stage C v2.3 checkpoint evaluation.

This entrypoint never opens replay transport and never publishes weights.  It
loads one fixed candidate checkpoint plus its immutable first-cycle prefix,
runs at most two full-physics environments, and writes one self-contained JSON
record per completed episode.

Examples (PowerShell, from the repository root)::

    $env:OMNI_KIT_ACCEPT_EULA = "YES"
    C:\il\venv\Scripts\python.exe scripts/rl/eval_stagec_seedmine.py `
      --checkpoint runs/stagec_v23_cycle1_v2_000075000.pt `
      --prefix-checkpoint runs/aggressive_near_high75_ft_000165000.pt `
      --template assets/rl/env_template_200.usd `
      --mode full --episodes 16 --num-envs 2 `
      --env-seed 750000 --action-seed 750000 `
      --action-mode policy-noise `
      --out runs/seedmine/v2_75k_full_750000.jsonl `
      --capture-dir runs/seedmine/training_episodes `
      --trajectory-out runs/seedmine/v2_75k_full_750000_success.jsonl

``policy-noise`` uses the checkpoint's saved exploration schedule and a fixed
Torch seed.  ``fixed-gaussian`` uses a separate fixed NumPy generator and an
explicit ``--noise-std``.  ``deterministic`` uses the actor mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


SCHEMA = "stagec_seed_eval_v1"
CAPTURE_SCHEMA = "stagec_training_episode_v1"
ACTION_MODES = (
    "deterministic",
    "policy-noise",
    "fixed-gaussian",
    "smooth-drive",
)
RESET_MODES = ("full", "return")
TRAINING_FIELD_KEYS = ("obs", "proprio", "privileged", "action", "reward", "done")


class SmoothDriveNoise:
    """Per-environment AR(1) exploration for suffix navigation only."""

    def __init__(
        self,
        num_envs: int,
        std: float,
        cap: float,
        correlation: float,
        phase_indices: set[int],
        rng: np.random.Generator,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("smooth-drive num_envs must be positive")
        if not np.isfinite(std) or float(std) <= 0.0:
            raise ValueError("smooth-drive std must be finite and positive")
        if not np.isfinite(cap) or float(cap) <= 0.0:
            raise ValueError("smooth-drive cap must be finite and positive")
        if float(std) > float(cap):
            raise ValueError("smooth-drive std cannot exceed its cap")
        if not np.isfinite(correlation) or not 0.0 <= float(correlation) < 1.0:
            raise ValueError("smooth-drive correlation must be in [0, 1)")
        if not phase_indices or not set(phase_indices).issubset({1, 2, 3}):
            raise ValueError("smooth-drive phases must be LEAVE/COLLECT/RETURN")
        self.std = float(std)
        self.cap = float(cap)
        self.correlation = float(correlation)
        self.phase_indices = frozenset(int(value) for value in phase_indices)
        self.rng = rng
        self.state = np.zeros((int(num_envs), 3), dtype=np.float32)
        self.previous_phase = np.full(int(num_envs), -1, dtype=np.int8)

    def sample(self, proprio: np.ndarray) -> np.ndarray:
        values = np.asarray(proprio, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != self.state.shape[0]:
            raise ValueError("smooth-drive proprio batch does not match noise state")
        if values.shape[1] < 27:
            raise ValueError("smooth-drive requires Stage C v2 phase proprio")
        innovation_scale = self.std * math.sqrt(
            max(0.0, 1.0 - self.correlation * self.correlation)
        )
        innovation = self.rng.normal(
            0.0, innovation_scale, size=self.state.shape
        ).astype(np.float32)
        phases = np.argmax(values[:, 22:27], axis=1)
        phase_changed = phases != self.previous_phase
        self.state[phase_changed] = 0.0
        self.state = np.clip(
            self.correlation * self.state + innovation,
            -self.cap,
            self.cap,
        ).astype(np.float32)
        selected = np.isin(phases, tuple(self.phase_indices))
        self.state[~selected] = 0.0
        self.previous_phase = phases.astype(np.int8, copy=True)
        return np.where(selected[:, None], self.state, 0.0).astype(
            np.float32, copy=False
        )

    def reset(self, indices: np.ndarray | list[int]) -> None:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(selected):
            self.state[selected] = 0.0
            self.previous_phase[selected] = -1


def _parse_noise_phases(text: str) -> tuple[str, ...]:
    phases = tuple(part.strip().lower() for part in str(text).split(",") if part.strip())
    unknown = sorted(set(phases) - {"leave", "collect", "return"})
    if not phases or unknown:
        raise ValueError(
            "smooth-drive phases must be a non-empty comma-separated subset of "
            "leave,collect,return"
        )
    return tuple(dict.fromkeys(phases))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy values into strict JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def episode_succeeded(stats: dict[str, Any]) -> bool:
    """A v2 success is an ordered, qualified second-cycle score."""

    return bool(
        int(stats.get("cycles_completed", 0) or 0) >= 1
        or int((stats.get("milestones") or {}).get("cycle_scored", 0) or 0) >= 1
    )


def capture_tier(
    stats: dict[str, Any], *, include_returned_home: bool = False
) -> str | None:
    """Select an episode for standalone training capture.

    A qualified cycle is always the top tier.  ``returned_home`` is optional
    because it is useful as a lower-tier return example but is not proof that
    the robot completed the second dump.
    """

    if episode_succeeded(stats):
        return "cycle"
    returned_home = int(
        (stats.get("milestones") or {}).get("returned_home", 0) or 0
    )
    if include_returned_home and returned_home > 0:
        return "returned_home"
    return None


def new_capture_buffer() -> dict[str, list[Any]]:
    return {key: [] for key in TRAINING_FIELD_KEYS}


def append_capture_transition(
    buffer: dict[str, list[Any]],
    *,
    obs: np.ndarray,
    proprio: np.ndarray,
    privileged: np.ndarray,
    action: np.ndarray,
    reward: float,
    done: bool,
) -> None:
    """Append one ReplayRing-compatible transition to an episode buffer."""

    if tuple(buffer) != TRAINING_FIELD_KEYS:
        raise ValueError("capture buffer fields do not match ReplayRing schema")
    buffer["obs"].append(np.asarray(obs, dtype=np.uint8).copy())
    buffer["proprio"].append(np.asarray(proprio, dtype=np.float32).copy())
    buffer["privileged"].append(np.asarray(privileged, dtype=np.float32).copy())
    buffer["action"].append(np.asarray(action, dtype=np.float32).copy())
    buffer["reward"].append(np.float32(reward))
    buffer["done"].append(bool(done))


def stack_capture_buffer(buffer: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    """Pack one completed episode into the exact fields consumed by ReplayRing."""

    if tuple(buffer) != TRAINING_FIELD_KEYS:
        raise ValueError("capture buffer fields do not match ReplayRing schema")
    lengths = {key: len(buffer[key]) for key in TRAINING_FIELD_KEYS}
    if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
        raise ValueError(f"capture fields have inconsistent or empty lengths: {lengths}")
    arrays = {
        "obs": np.stack(buffer["obs"]).astype(np.uint8, copy=False),
        "proprio": np.stack(buffer["proprio"]).astype(np.float32, copy=False),
        "privileged": np.stack(buffer["privileged"]).astype(np.float32, copy=False),
        "action": np.stack(buffer["action"]).astype(np.float32, copy=False),
        "reward": np.asarray(buffer["reward"], dtype=np.float32),
        "done": np.asarray(buffer["done"], dtype=bool),
    }
    if arrays["obs"].ndim != 4:
        raise ValueError(f"capture obs must have shape (T,C,H,W), got {arrays['obs'].shape}")
    if arrays["proprio"].ndim != 2 or arrays["privileged"].ndim != 2:
        raise ValueError("capture proprio and privileged fields must be rank two")
    if arrays["action"].ndim != 2 or arrays["action"].shape[1] != 7:
        raise ValueError(f"capture action must have shape (T,7), got {arrays['action'].shape}")
    if not bool(arrays["done"][-1]) or bool(arrays["done"][:-1].any()):
        raise ValueError("capture must contain exactly one terminal flag on its final row")
    for key in ("proprio", "privileged", "action", "reward"):
        if not bool(np.isfinite(arrays[key]).all()):
            raise ValueError(f"capture field {key!r} contains non-finite values")
    return arrays


def capture_basename(record: dict[str, Any], tier: str) -> str:
    if tier not in ("cycle", "returned_home"):
        raise ValueError(f"unknown capture tier: {tier!r}")
    checkpoint_tag = str(record.get("checkpoint_sha256", "unknown"))[:12]
    return (
        f"episode_{int(record.get('episode_index', 0)):06d}_{tier}"
        f"_score{int(record.get('scored', 0)):03d}"
        f"_env{int(record.get('env_index', 0))}"
        f"_es{int(record.get('env_seed', 0))}"
        f"_as{int(record.get('action_seed', 0))}_{checkpoint_tag}.npz"
    )


def build_capture_metadata(
    record: dict[str, Any], arrays: dict[str, np.ndarray], tier: str
) -> dict[str, Any]:
    return {
        "schema": CAPTURE_SCHEMA,
        "capture_tier": str(tier),
        "field_keys": list(TRAINING_FIELD_KEYS),
        "length": int(arrays["reward"].shape[0]),
        "fields": {
            key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
            for key in TRAINING_FIELD_KEYS
        },
        "episode": _jsonable(record),
    }


def atomic_save_capture(
    capture_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    basename: str,
) -> Path:
    """Compress and atomically publish one standalone episode archive."""

    if tuple(arrays) != TRAINING_FIELD_KEYS:
        raise ValueError("capture arrays do not match ReplayRing field order")
    capture_dir.mkdir(parents=True, exist_ok=True)
    proposed = capture_dir / basename
    final = proposed
    collision = 0
    while final.exists():
        collision += 1
        final = proposed.with_name(f"{proposed.stem}_{collision:03d}{proposed.suffix}")
    tmp = capture_dir / f".{final.name}.{os.getpid()}.tmp"
    payload = {key: arrays[key] for key in TRAINING_FIELD_KEYS}
    payload["metadata"] = np.frombuffer(
        json.dumps(_jsonable(metadata), sort_keys=True, allow_nan=False).encode("utf-8"),
        dtype=np.uint8,
    )
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, final)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return final


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episodes = list(records)
    scored = [int(item.get("scored", 0) or 0) for item in episodes]
    collected = [int(item.get("collected", 0) or 0) for item in episodes]
    successes = sum(bool(item.get("success", episode_succeeded(item))) for item in episodes)

    def milestone_count(name: str) -> int:
        return sum(
            int((item.get("milestones") or {}).get(name, 0) or 0) > 0
            for item in episodes
        )

    count = len(episodes)
    return {
        "schema": SCHEMA,
        "episodes": count,
        "successes": int(successes),
        "success_rate": round(float(successes) / count, 6) if count else 0.0,
        "mean_scored": round(float(np.mean(scored)), 3) if scored else 0.0,
        "max_scored": max(scored, default=0),
        "mean_collected": round(float(np.mean(collected)), 3) if collected else 0.0,
        "max_collected": max(collected, default=0),
        "cycles_completed": sum(int(item.get("cycles_completed", 0) or 0) for item in episodes),
        "dump_attempts": sum(int(item.get("dump_attempts", 0) or 0) for item in episodes),
        "clean_dumps": sum(
            int(item.get("dump_empty_completions", 0) or 0) for item in episodes
        ),
        "partial_dumps": sum(int(item.get("partial_dumps", 0) or 0) for item in episodes),
        "captured_cycle_episodes": sum(
            item.get("capture_tier") == "cycle" for item in episodes
        ),
        "captured_returned_home_episodes": sum(
            item.get("capture_tier") == "returned_home" for item in episodes
        ),
        "milestone_episodes": {
            name: milestone_count(name)
            for name in (
                "latched",
                "left_home",
                "target_load",
                "returned_home",
                "cycle_dumped",
                "cycle_scored",
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prefix-checkpoint",
        "--stagec-v2-prefix-checkpoint",
        dest="prefix_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "assets" / "rl" / "env_template_200.usd",
    )
    parser.add_argument("--mode", choices=RESET_MODES, default="full")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--episode-len-s",
        type=float,
        default=None,
        help="defaults to 120 seconds for full and 75 seconds for return",
    )
    parser.add_argument("--env-seed", "--seed", dest="env_seed", type=int, default=7000)
    parser.add_argument(
        "--action-seed",
        type=int,
        default=None,
        help="Torch/NumPy action seed; defaults to --env-seed",
    )
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="deterministic")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="required with fixed-gaussian or smooth-drive",
    )
    parser.add_argument(
        "--noise-correlation",
        type=float,
        default=None,
        help="AR(1) correlation for smooth-drive (default: 0.95)",
    )
    parser.add_argument(
        "--noise-cap",
        type=float,
        default=None,
        help="hard L-infinity cap for smooth-drive (default: 0.05)",
    )
    parser.add_argument(
        "--noise-phases",
        type=str,
        default=None,
        help="comma-separated smooth-drive phases (default: leave)",
    )
    parser.add_argument("--out", type=Path, required=True, help="per-episode JSONL")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="summary JSON; defaults beside --out with .summary.json suffix",
    )
    parser.add_argument(
        "--trajectory-out",
        type=Path,
        default=None,
        help="optional JSONL containing compact trajectories for successful episodes only",
    )
    parser.add_argument(
        "--trajectory-all-episodes",
        action="store_true",
        help=(
            "write failed as well as successful episodes to --trajectory-out; "
            "use this for mechanics-gate diagnosis"
        ),
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help=(
            "optional standalone .npz directory; complete transition episodes are "
            "saved atomically only for qualified second cycles"
        ),
    )
    parser.add_argument(
        "--capture-returned-home",
        action="store_true",
        help="also capture returned_home episodes as a clearly labelled lower tier",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace evaluation output files if they already exist",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not 1 <= args.num_envs <= 2:
        parser.error("--num-envs must be 1 or 2; Stage C camera evaluation is capped at 2")
    if args.episode_len_s is not None and args.episode_len_s <= 0:
        parser.error("--episode-len-s must be positive")
    if args.action_seed is None:
        args.action_seed = int(args.env_seed)
    if args.action_mode in ("fixed-gaussian", "smooth-drive"):
        if args.noise_std is None or args.noise_std <= 0:
            parser.error(f"{args.action_mode} requires a positive --noise-std")
    elif args.noise_std is not None:
        parser.error(
            "--noise-std is only valid with --action-mode fixed-gaussian or smooth-drive"
        )
    if args.action_mode == "smooth-drive":
        if args.noise_correlation is None:
            args.noise_correlation = 0.95
        if args.noise_cap is None:
            args.noise_cap = 0.05
        if args.noise_phases is None:
            args.noise_phases = ("leave",)
        else:
            try:
                args.noise_phases = _parse_noise_phases(args.noise_phases)
            except ValueError as exc:
                parser.error(str(exc))
        if not 0.0 <= float(args.noise_correlation) < 1.0:
            parser.error("--noise-correlation must be in [0, 1)")
        if not np.isfinite(args.noise_cap) or float(args.noise_cap) <= 0.0:
            parser.error("--noise-cap must be finite and positive")
        if float(args.noise_std) > float(args.noise_cap):
            parser.error("--noise-std cannot exceed --noise-cap")
    elif args.noise_correlation is not None:
        parser.error("--noise-correlation is only valid with --action-mode smooth-drive")
    elif args.noise_cap is not None or args.noise_phases is not None:
        parser.error(
            "--noise-cap and --noise-phases are only valid with --action-mode smooth-drive"
        )
    if args.capture_returned_home and args.capture_dir is None:
        parser.error("--capture-returned-home requires --capture-dir")
    if args.trajectory_all_episodes and args.trajectory_out is None:
        parser.error("--trajectory-all-episodes requires --trajectory-out")
    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(".summary.json")
    outputs = [args.out, args.summary_out]
    if args.trajectory_out is not None:
        outputs.append(args.trajectory_out)
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("--out, --summary-out, and --trajectory-out must be distinct")
    protected = {
        args.checkpoint.resolve(),
        args.prefix_checkpoint.resolve(),
        args.template.resolve(),
    }
    if protected.intersection(resolved_outputs):
        parser.error("evaluation outputs must not overwrite a checkpoint or template")
    if args.capture_dir is not None and args.capture_dir.resolve() in protected:
        parser.error("--capture-dir must not alias a checkpoint or template")
    return args


def resolved_episode_len_s(args: argparse.Namespace) -> float:
    if args.episode_len_s is not None:
        return float(args.episode_len_s)
    return 160.0 if args.mode == "full" else 75.0


def _prepare_output(path: Path | None, overwrite: bool) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _open_jsonl(path: Path, overwrite: bool):
    _prepare_output(path, overwrite)
    return path.open("w" if overwrite else "x", encoding="utf-8", buffering=1)


def _metadata_number(metadata: dict[str, Any], key: str, cast):
    if key not in metadata:
        raise ValueError(f"checkpoint Stage C metadata is missing {key!r}")
    return cast(metadata[key])


def _trajectory_point(slot: Any, action: np.ndarray, policy_step: int) -> dict[str, Any]:
    position, quat = slot.articulation.get_world_pose()
    w, x, y, z = (float(value) for value in quat)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    phase = getattr(getattr(slot, "cycle_v2", None), "phase", None)
    return {
        "step": int(policy_step),
        "x": round(float(position[0]), 4),
        "y": round(float(position[1]), 4),
        "yaw": round(float(yaw), 4),
        "magazine": int(len(slot.controller.magazine)),
        "scored": int(slot.router.scored["blue"]),
        "collected": int(slot.controller.balls_collected),
        "phase": getattr(phase, "value", str(phase) if phase is not None else None),
        "action": np.asarray(action, dtype=np.float32).round(4).tolist(),
    }


def _candidate_actions(
    agent: Any,
    frames: np.ndarray,
    proprio: np.ndarray,
    *,
    action_mode: str,
    noise_std: float | None,
    action_rng: np.random.Generator,
    smooth_noise: SmoothDriveNoise | None = None,
) -> np.ndarray:
    if action_mode == "policy-noise":
        return agent.act(frames, proprio, explore=True).astype(np.float32)
    actions = agent.act(frames, proprio, explore=False).astype(np.float32)
    if action_mode == "fixed-gaussian":
        actions = np.clip(
            actions
            + action_rng.normal(0.0, float(noise_std), size=actions.shape),
            -1.0,
            1.0,
        ).astype(np.float32)
    elif action_mode == "smooth-drive":
        if smooth_noise is None:
            raise ValueError("smooth-drive requires a SmoothDriveNoise state")
        actions = actions.copy()
        actions[:, :3] += smooth_noise.sample(proprio)
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
    return actions


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Boot Isaac and evaluate one immutable checkpoint."""

    for label, path in (
        ("checkpoint", args.checkpoint),
        ("prefix checkpoint", args.prefix_checkpoint),
        ("template", args.template),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    _prepare_output(args.out, args.overwrite)
    _prepare_output(args.summary_out, args.overwrite)
    _prepare_output(args.trajectory_out, args.overwrite)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    env = None
    try:
        import torch

        from frc_rebuilt.rl import distributed as D
        from frc_rebuilt.rl.cycle_v2 import (
            COLLECT_UNTIL_PREFERRED_REVISIONS,
            POSTDUMP_COMPLETE_CYCLE_REVISIONS,
            POSTDUMP_TARGET_REVISIONS,
            RAMP_OUT_REVISIONS,
            RETURN_INTAKE_REVISIONS,
            SCORE_EFFICIENCY_REVISIONS,
            SUPPORTED_ROUTE_EFFICIENCY_REVISIONS,
        )
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.policy_v2 import (
            LEGACY_PROPRIO_DIM,
            apply_executed_action_policy,
            compose_phase_actions,
            validate_composite_metadata,
        )
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        if tuple(D.FIELD_KEYS) != TRAINING_FIELD_KEYS:
            raise RuntimeError(
                f"capture schema drift: {tuple(D.FIELD_KEYS)!r} != {TRAINING_FIELD_KEYS!r}"
            )

        # Environment randomness and action randomness are deliberately separate.
        np.random.seed(int(args.action_seed))
        torch.manual_seed(int(args.action_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.action_seed))

        try:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(args.checkpoint, map_location="cpu")
        prefix_sha256 = _sha256_file(args.prefix_checkpoint)
        metadata = validate_composite_metadata(payload.get("stagec_v2"), prefix_sha256)
        checkpoint_sha256 = _sha256_file(args.checkpoint)
        template_sha256 = _sha256_file(args.template)
        reward_revision = metadata.get("reward_revision")
        if reward_revision not in (None, *SUPPORTED_ROUTE_EFFICIENCY_REVISIONS):
            raise ValueError(
                f"unsupported Stage C reward revision: {reward_revision!r}"
            )
        route_efficiency = reward_revision in SUPPORTED_ROUTE_EFFICIENCY_REVISIONS
        route_efficiency_v2 = reward_revision in (
            "outer_rail_v2",
            "outer_rail_v3",
            *RAMP_OUT_REVISIONS,
        )
        ramp_out_revision = reward_revision in RAMP_OUT_REVISIONS
        return_intake_revision = reward_revision in RETURN_INTAKE_REVISIONS
        return_intake_enabled = (
            bool(metadata["intake_during_return"])
            if return_intake_revision
            else False
        )
        if return_intake_revision and not return_intake_enabled:
            raise ValueError(
                f"{reward_revision} requires intake_during_return=true"
            )

        cfg = VecEnvCfg(
            num_envs=int(args.num_envs),
            template_usd=str(args.template),
            cameras=True,
            episode_len_s=resolved_episode_len_s(args),
            preload_prob=0.0,
            spawn_under_trench=True,
            lock_storage_extended=False,
            mask_illegal_fire=True,
            collect_reward_weight=_metadata_number(metadata, "collect_weight", float),
            rho_score=1.0,
            rho_collect=1.0,
            empty_own_court_penalty=0.0,
            dump_on_press=True,
            max_dump_ticks=_metadata_number(metadata, "max_dump_ticks", int),
            stagec_v2=True,
            cycle_v2_reset_modes=(str(args.mode),),
            cycle_v2_target_load=_metadata_number(metadata, "target_load", int),
            cycle_v2_reserve_count=_metadata_number(metadata, "reserve_count", int),
            cycle_v2_reserve_batches=_metadata_number(metadata, "reserve_batches", int),
            cycle_v2_score_fraction=_metadata_number(
                metadata, "cycle_score_fraction", float
            ),
            cycle_v2_score_floor=_metadata_number(metadata, "cycle_score_floor", int),
            cycle_v2_progress_per_m=_metadata_number(metadata, "progress_per_m", float),
            cycle_v2_progress_step_cap=_metadata_number(
                metadata, "progress_step_cap", float
            ),
            cycle_v2_ramp_bonus=_metadata_number(metadata, "ramp_bonus", float),
            cycle_v2_refresh_ramp_side_on_dump=(
                bool(metadata["refresh_ramp_side_on_dump"])
                if route_efficiency
                else False
            ),
            cycle_v2_ramp_side_deadband_x=(
                _metadata_number(metadata, "ramp_side_deadband_x", float)
                if route_efficiency
                else 0.25
            ),
            cycle_v2_require_ramp_out=(
                bool(metadata["require_ramp_out"])
                if ramp_out_revision
                else False
            ),
            cycle_v2_ramp_out_half_width=(
                _metadata_number(metadata, "ramp_out_half_width", float)
                if ramp_out_revision
                else 0.90
            ),
            cycle_v2_ramp_out_bonus=(
                _metadata_number(metadata, "ramp_out_bonus", float)
                if ramp_out_revision
                else 0.0
            ),
            cycle_v2_off_ramp_exit_penalty=(
                _metadata_number(metadata, "off_ramp_exit_penalty", float)
                if ramp_out_revision
                else 0.0
            ),
            cycle_v2_postdump_require_target_load=(
                bool(metadata["postdump_require_target_load"])
                if reward_revision in POSTDUMP_TARGET_REVISIONS
                else False
            ),
            cycle_v2_postdump_complete_cycle=(
                bool(metadata["postdump_complete_cycle"])
                if reward_revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS
                else False
            ),
            cycle_v2_postdump_depleted_count=(
                int(metadata["postdump_depleted_count"])
                if reward_revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS
                else 0
            ),
            cycle_v2_postdump_depleted_prob=(
                float(metadata["postdump_depleted_prob"])
                if reward_revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS
                else 0.0
            ),
            cycle_v2_preferred_repeat_load=(
                int(metadata["preferred_repeat_load"])
                if reward_revision in SCORE_EFFICIENCY_REVISIONS
                else 0
            ),
            cycle_v2_collect_until_preferred=(
                reward_revision in COLLECT_UNTIL_PREFERRED_REVISIONS
            ),
            cycle_v2_collect_stall_steps=(
                int(metadata["collect_stall_steps"])
                if reward_revision in COLLECT_UNTIL_PREFERRED_REVISIONS
                else 0
            ),
            cycle_v2_return_time_guard=(
                float(metadata["return_time_guard"])
                if reward_revision in COLLECT_UNTIL_PREFERRED_REVISIONS
                else 0.0
            ),
            cycle_v2_intake_during_return=(
                return_intake_enabled
            ),
            cycle_v2_repeat_load_return_bonus=(
                float(metadata["repeat_load_return_bonus"])
                if reward_revision in SCORE_EFFICIENCY_REVISIONS
                else 0.0
            ),
            cycle_v2_repeat_load_score_bonus=(
                float(metadata["repeat_load_score_bonus"])
                if reward_revision in SCORE_EFFICIENCY_REVISIONS
                else 0.0
            ),
            cycle_v2_outer_rail_enter_x=(
                _metadata_number(metadata, "outer_rail_enter_x", float)
                if route_efficiency
                else 2.85
            ),
            cycle_v2_outer_rail_exit_x=(
                _metadata_number(metadata, "outer_rail_exit_x", float)
                if route_efficiency
                else 2.55
            ),
            cycle_v2_outer_rail_max_x=(
                _metadata_number(metadata, "outer_rail_max_x", float)
                if route_efficiency
                else 3.60
            ),
            cycle_v2_outer_rail_grace_steps=(
                _metadata_number(metadata, "outer_rail_grace_steps", int)
                if route_efficiency
                else 20
            ),
            cycle_v2_outer_rail_penalty_per_step=(
                _metadata_number(
                    metadata, "outer_rail_penalty_per_step", float
                )
                if route_efficiency
                else 0.0
            ),
            cycle_v2_outer_rail_penalty_cap=(
                _metadata_number(metadata, "outer_rail_penalty_cap", float)
                if route_efficiency
                else 8.0
            ),
            cycle_v2_outer_rail_min_scale=(
                _metadata_number(metadata, "outer_rail_min_scale", float)
                if route_efficiency_v2
                else 0.0
            ),
            cycle_v2_outer_rail_escalation_steps=(
                _metadata_number(
                    metadata, "outer_rail_escalation_steps", int
                )
                if route_efficiency_v2
                else 0
            ),
            cycle_v2_outer_rail_max_multiplier=(
                _metadata_number(
                    metadata, "outer_rail_max_multiplier", float
                )
                if route_efficiency_v2
                else 1.0
            ),
            cycle_v2_intake_substeps=(
                _metadata_number(metadata, "intake_substeps", int)
                if route_efficiency_v2
                else 1
            ),
            cycle_v2_leave_grace_steps=_metadata_number(
                metadata, "leave_grace_steps", int
            ),
            cycle_v2_leave_penalty_per_step=_metadata_number(
                metadata, "leave_penalty_per_step", float
            ),
            cycle_v2_leave_penalty_cap=_metadata_number(
                metadata, "leave_penalty_cap", float
            ),
            cycle_v2_return_grace_steps=_metadata_number(
                metadata, "return_grace_steps", int
            ),
            cycle_v2_return_penalty_per_step=_metadata_number(
                metadata, "return_penalty_per_step", float
            ),
            cycle_v2_return_penalty_cap=_metadata_number(
                metadata, "return_penalty_cap", float
            ),
            cycle_v2_shoot_grace_steps=int(
                round(_metadata_number(metadata, "shoot_grace_s", float) * 10.0)
            ),
            cycle_v2_shoot_penalty_per_step=_metadata_number(
                metadata, "shoot_penalty_per_step", float
            ),
            cycle_v2_shoot_penalty_cap=_metadata_number(
                metadata, "shoot_penalty_cap", float
            ),
            cycle_v2_dump_lost_aim_grace_ticks=_metadata_number(
                metadata, "dump_lost_aim_grace_ticks", int
            ),
            cycle_v2_partial_dump_penalty_per_ball=_metadata_number(
                metadata, "partial_dump_penalty_per_ball", float
            ),
            cycle_v2_partial_dump_penalty_cap=_metadata_number(
                metadata, "partial_dump_penalty_cap", float
            ),
            seed=int(args.env_seed),
        )
        env = VecCompetitionEnv(cfg)
        if not bool(getattr(env, "_camera_ready", False)):
            raise RuntimeError("Stage C evaluator camera initialization failed")

        # Match collector startup: the constructor resets; take one zero-action step.
        obs, _, _, _ = env.step(np.zeros((args.num_envs, 7), np.float32))
        if bool((obs["rgb"].std(axis=(2, 3, 4)) <= 1.0).any()):
            raise RuntimeError("Stage C evaluator produced black startup frames")

        frames = D.to_policy_frames(obs["rgb"])
        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=frames.shape[1],
                frame_h=frames.shape[2],
                frame_w=frames.shape[3],
                proprio_dim=obs["proprio"].shape[1],
                privileged_dim=obs["privileged"].shape[1],
                stddev_start=_metadata_number(metadata, "stddev_start", float),
                stddev_end=_metadata_number(metadata, "stddev_end", float),
                stddev_steps=_metadata_number(metadata, "stddev_steps", int),
            )
        )
        prefix_agent = DrQV2Agent(
            DrQConfig(
                frame_channels=frames.shape[1],
                frame_h=frames.shape[2],
                frame_w=frames.shape[3],
                proprio_dim=LEGACY_PROPRIO_DIM,
                privileged_dim=obs["privileged"].shape[1],
            )
        )
        agent.load(str(args.checkpoint))
        prefix_agent.load(str(args.prefix_checkpoint))

        # Agent construction consumes Torch randomness.  Reset immediately before
        # rollout so policy-noise is a stable function of --action-seed.
        np.random.seed(int(args.action_seed))
        torch.manual_seed(int(args.action_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.action_seed))
        action_rng = np.random.default_rng(int(args.action_seed))
        smooth_noise = (
            SmoothDriveNoise(
                int(args.num_envs),
                float(args.noise_std),
                float(args.noise_cap),
                float(args.noise_correlation),
                {
                    {"leave": 1, "collect": 2, "return": 3}[name]
                    for name in args.noise_phases
                },
                action_rng,
            )
            if args.action_mode == "smooth-drive"
            else None
        )

        common = {
            "schema": SCHEMA,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_train_steps": int(agent.train_steps),
            "checkpoint_v2_updates": int(payload.get("v2_updates", 0)),
            "stagec_v2_metadata": _jsonable(metadata),
            "prefix_checkpoint": str(args.prefix_checkpoint.resolve()),
            "prefix_sha256": prefix_sha256,
            "template": str(args.template.resolve()),
            "template_sha256": template_sha256,
            "mode": str(args.mode),
            "episode_len_s": resolved_episode_len_s(args),
            "env_seed": int(args.env_seed),
            "action_seed": int(args.action_seed),
            "action_mode": str(args.action_mode),
            "noise_std": (
                float(args.noise_std)
                if args.action_mode in ("fixed-gaussian", "smooth-drive")
                else (float(agent.stddev()) if args.action_mode == "policy-noise" else 0.0)
            ),
            "noise_correlation": (
                float(args.noise_correlation)
                if args.action_mode == "smooth-drive"
                else None
            ),
            "noise_cap": (
                float(args.noise_cap) if args.action_mode == "smooth-drive" else None
            ),
            "noise_phases": (
                list(args.noise_phases) if args.action_mode == "smooth-drive" else []
            ),
            "num_envs": int(args.num_envs),
            "capture_dir": (
                str(args.capture_dir.resolve()) if args.capture_dir is not None else None
            ),
            "capture_returned_home": bool(args.capture_returned_home),
        }
        print("SEED_EVAL_READY " + json.dumps(common, sort_keys=True), flush=True)

        episode_returns = np.zeros(args.num_envs, dtype=np.float64)
        episode_steps = np.zeros(args.num_envs, dtype=np.int64)
        trajectories: list[list[dict[str, Any]]] = [
            [] for _ in range(args.num_envs)
        ]
        capture_buffers = [new_capture_buffer() for _ in range(args.num_envs)]
        cycle_success_steps: list[list[int]] = [
            [] for _ in range(args.num_envs)
        ]
        cycles_completed_seen = np.zeros(args.num_envs, dtype=np.int64)
        episode_sequences = np.zeros(args.num_envs, dtype=np.int64)
        max_capture_steps = int(math.ceil(resolved_episode_len_s(args) * 10.0)) + 2
        records: list[dict[str, Any]] = []

        with _open_jsonl(args.out, args.overwrite) as episode_handle:
            trajectory_handle = (
                _open_jsonl(args.trajectory_out, args.overwrite)
                if args.trajectory_out is not None
                else None
            )
            try:
                while len(records) < args.episodes:
                    frames = D.to_policy_frames(obs["rgb"])
                    candidate = _candidate_actions(
                        agent,
                        frames,
                        obs["proprio"],
                        action_mode=args.action_mode,
                        noise_std=args.noise_std,
                        action_rng=action_rng,
                        smooth_noise=smooth_noise,
                    )
                    prefix = prefix_agent.act(
                        frames,
                        obs["proprio"][:, :LEGACY_PROPRIO_DIM],
                        explore=False,
                    ).astype(np.float32)
                    actions = compose_phase_actions(prefix, candidate, obs["proprio"])
                    actions = apply_executed_action_policy(
                        actions,
                        obs["proprio"],
                        intake_during_return=return_intake_enabled,
                    )

                    if trajectory_handle is not None:
                        for index, slot in enumerate(env.slots):
                            trajectories[index].append(
                                _trajectory_point(slot, actions[index], int(episode_steps[index]))
                            )

                    capture_proprio = (
                        obs["proprio"].copy() if args.capture_dir is not None else None
                    )
                    capture_privileged = (
                        obs["privileged"].copy() if args.capture_dir is not None else None
                    )
                    obs, rewards, dones, info = env.step(actions.astype(np.float32))
                    if args.capture_dir is not None:
                        for index in range(args.num_envs):
                            append_capture_transition(
                                capture_buffers[index],
                                obs=frames[index],
                                proprio=capture_proprio[index],
                                privileged=capture_privileged[index],
                                action=actions[index],
                                reward=float(rewards[index]),
                                done=bool(dones[index]),
                            )
                            if len(capture_buffers[index]["reward"]) > max_capture_steps:
                                raise RuntimeError(
                                    f"capture env {index} exceeded bounded episode buffer "
                                    f"({max_capture_steps} transitions)"
                                )
                    for index, slot in enumerate(env.slots):
                        terminal_stats = info.get("episode_stats", {}).get(
                            index, {}
                        )
                        completed = int(
                            terminal_stats.get(
                                "cycles_completed",
                                slot.cycle_v2.cycles_completed,
                            )
                        )
                        if completed > int(cycles_completed_seen[index]):
                            cycle_success_steps[index].extend(
                                [int(episode_steps[index])]
                                * (completed - int(cycles_completed_seen[index]))
                            )
                            cycles_completed_seen[index] = completed
                    episode_returns += rewards
                    episode_steps += 1
                    for index in np.flatnonzero(dones):
                        index = int(index)
                        terminal = dict(info.get("episode_stats", {}).get(index, {}))
                        if len(records) < args.episodes:
                            record = {
                                **common,
                                "episode_index": len(records),
                                "env_index": index,
                                "env_episode_sequence": int(episode_sequences[index]),
                                "episode_steps": int(episode_steps[index]),
                                "cycle_success_steps": list(
                                    cycle_success_steps[index]
                                ),
                                "return": round(float(episode_returns[index]), 6),
                                **_jsonable(terminal),
                            }
                            record["success"] = episode_succeeded(record)
                            selected_tier = capture_tier(
                                record,
                                include_returned_home=bool(args.capture_returned_home),
                            )
                            tier = selected_tier if args.capture_dir is not None else None
                            record["capture_tier"] = tier
                            record["capture_path"] = None
                            if args.capture_dir is not None and tier is not None:
                                arrays = stack_capture_buffer(capture_buffers[index])
                                metadata = build_capture_metadata(record, arrays, tier)
                                capture_path = atomic_save_capture(
                                    args.capture_dir,
                                    arrays,
                                    metadata,
                                    capture_basename(record, tier),
                                )
                                record["capture_path"] = str(capture_path.resolve())
                            episode_handle.write(
                                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                            )
                            records.append(record)
                            print(
                                "SEED_EVAL_EPISODE "
                                f"{len(records)}/{args.episodes} env={index} "
                                f"score={record.get('scored', 0)} "
                                f"collect={record.get('collected', 0)} "
                                f"shots={record.get('shots_fired', 0)} "
                                f"cycles={record.get('cycles_completed', 0)} "
                                f"success={int(record['success'])}",
                                flush=True,
                            )
                            if trajectory_handle is not None and (
                                record["success"] or args.trajectory_all_episodes
                            ):
                                trajectory_handle.write(
                                    json.dumps(
                                        {
                                            **common,
                                            "episode": record,
                                            "trajectory": trajectories[index],
                                        },
                                        sort_keys=True,
                                        allow_nan=False,
                                    )
                                    + "\n"
                                )
                        episode_returns[index] = 0.0
                        episode_steps[index] = 0
                        trajectories[index] = []
                        capture_buffers[index] = new_capture_buffer()
                        cycle_success_steps[index] = []
                        cycles_completed_seen[index] = 0
                        episode_sequences[index] += 1
                    if smooth_noise is not None:
                        smooth_noise.reset(np.flatnonzero(dones))
            finally:
                if trajectory_handle is not None:
                    trajectory_handle.close()

        summary = {**common, **summarize(records)}
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("SEED_EVAL_DONE " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if env is not None:
            env.close()
        app.close()


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
