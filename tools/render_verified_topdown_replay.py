"""Render a 1920x1080 GUI-view deterministic Stage-D re-simulation.

The archived score-202 capture pins the exact two-slot environment, seeds,
checkpoint, prefix checkpoint, code, and template.  The renderer can either run
the frozen policy closed-loop or execute the custody-verified archived actions
for both simulator slots while one native 1920x1080 oblique simulator frame is
recorded per 0.1 s policy step.  Both modes are deterministic re-simulations,
not claims that pixels or PhysX floats are bit-identical across GPU/driver
stacks.  A final artifact is published only if the live run starts at score zero
and itself finishes with score 200+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np


CAPTURE_SCHEMA = "stagec_training_episode_v1"
PROVENANCE_SCHEMA = "stage_d_native_hd_deterministic_resimulation_v2"
STEPS = 1600
FPS = 10.0
CAMERA_SIZE = (1920, 1080)
SIDEBAR_WIDTH = 0
CODEC = "MJPG"
# These are reporting thresholds, not publication gates.  The added native-HD
# render graph can alter PhysX floating-point settling across GPU/driver stacks.
STATE_ATOL = 3e-3
REWARD_ATOL = 1e-4
MIN_LIVE_SCORE = 200
STRICT_CONTRACT = {
    "stage_d": True,
    "first_inactive": "blue",
    "ferry": False,
    "return_when_live": False,
    "owncourt_loop": False,
}
GUI_CAMERA_STATE_SCHEMA = "frc-rebuilt-gui-camera-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be one complete lowercase/uppercase SHA256")
    return normalized


def require_file_sha(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(Path(path))
    expected = _expected_sha(expected, f"expected {label} SHA256")
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def verify_code_snapshot(code_root: Path, archive: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify every regular archive member against the supplied unpacked root."""

    archive_sha256 = require_file_sha(archive, expected_sha256, "exact code archive")
    root = Path(code_root).resolve()
    if not (root / "src" / "frc_rebuilt" / "rl" / "vec_env.py").is_file():
        raise FileNotFoundError(f"code root is not an exact runtime snapshot: {root}")
    checked = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            while parts and parts[0] in (".", "exact"):
                parts = parts[1:]
            if not parts:
                raise ValueError(f"unsafe empty code-archive member: {member.name!r}")
            destination = (root / Path(*parts)).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe code-archive member: {member.name!r}") from exc
            extracted = bundle.extractfile(member)
            if extracted is None or not destination.is_file():
                raise ValueError(f"code root is missing archive member {member.name!r}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != sha256_file(destination):
                raise ValueError(f"code-root content differs for {member.name!r}")
            checked += 1
    if checked == 0:
        raise ValueError("exact code archive contains no regular files")
    return {"archive_sha256": archive_sha256, "verified_files": checked}


def _exact_float(value: Any, expected: float, label: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {value!r} != {expected!r}")


@dataclass(frozen=True)
class ReplayBundle:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    episode: dict[str, Any]
    action: np.ndarray
    proprio: np.ndarray
    privileged: np.ndarray
    reward: np.ndarray
    done: np.ndarray


def archived_action_batch(
    target: ReplayBundle,
    companion: ReplayBundle,
    step: int,
) -> np.ndarray:
    """Return the exact two-slot archived action batch in live env order."""

    if not 0 <= int(step) < STEPS:
        raise IndexError(f"archived action step out of range: {step}")
    if target.action.shape[0] < STEPS or companion.action.shape[0] < STEPS:
        raise ValueError("both archived action traces must cover the full 1600-step horizon")
    actions = np.stack((companion.action[step], target.action[step]), axis=0)
    if actions.shape != (2, 7):
        raise ValueError(f"archived two-slot action shape mismatch: {actions.shape}")
    return np.ascontiguousarray(actions, dtype=np.float32)


def load_replay_bundle(
    path: Path,
    *,
    expected_bundle_sha256: str,
    expected_checkpoint_sha256: str,
    expected_env_seed: int,
    expected_action_seed: int,
    expected_env_index: int,
    expected_steps: int = STEPS,
) -> ReplayBundle:
    bundle_sha256 = require_file_sha(path, expected_bundle_sha256, "replay bundle")
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {"action", "proprio", "privileged", "reward", "done", "metadata"}
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"compact replay keys mismatch: {sorted(archive.files)} != {sorted(expected_keys)}"
            )
        metadata = json.loads(
            np.asarray(archive["metadata"], dtype=np.uint8).tobytes().decode("utf-8")
        )
        arrays = {key: np.asarray(archive[key]).copy() for key in expected_keys - {"metadata"}}
    if not isinstance(metadata, dict) or metadata.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("unexpected replay metadata schema")
    if int(metadata.get("length", -1)) != int(expected_steps):
        raise ValueError(f"replay metadata must declare exactly {expected_steps} transitions")
    episode = metadata.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("replay metadata has no episode object")
    expected_checkpoint_sha256 = _expected_sha(
        expected_checkpoint_sha256, "expected checkpoint SHA256"
    )
    exact_values = {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "env_seed": int(expected_env_seed),
        "action_seed": int(expected_action_seed),
        "env_index": int(expected_env_index),
        "episode_steps": int(expected_steps),
        "num_envs": 2,
        "mode": "full",
        "reset_mode": "full",
        "action_mode": "deterministic",
        "terminal_reason": "horizon",
    }
    for key, expected in exact_values.items():
        actual = episode.get(key)
        if key.endswith("sha256"):
            actual = str(actual).lower()
        if actual != expected:
            raise ValueError(f"episode {key} mismatch: {actual!r} != {expected!r}")
    _exact_float(episode.get("episode_len_s"), 160.0, "episode_len_s")
    contract = episode.get("stage_d_contract")
    if not isinstance(contract, dict):
        raise ValueError("episode is missing stage_d_contract")
    for key, expected in STRICT_CONTRACT.items():
        actual = contract.get(key)
        if type(expected) is bool:
            valid = type(actual) is bool and actual is expected
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(f"stage_d_contract.{key} mismatch: {actual!r} != {expected!r}")
    _exact_float(contract.get("policy_speed_scale"), 1.0, "policy_speed_scale")
    _exact_float(contract.get("prefix_rescue_s"), 35.0, "prefix_rescue_s")
    expected_shapes = {
        "action": (expected_steps, 7),
        "proprio": (expected_steps, 30),
        "privileged": (expected_steps, 26),
        "reward": (expected_steps,),
        "done": (expected_steps,),
    }
    expected_dtypes = {
        "action": np.dtype("float32"),
        "proprio": np.dtype("float32"),
        "privileged": np.dtype("float32"),
        "reward": np.dtype("float32"),
        "done": np.dtype("bool"),
    }
    for key in expected_shapes:
        if arrays[key].shape != expected_shapes[key] or arrays[key].dtype != expected_dtypes[key]:
            raise ValueError(
                f"replay {key} contract mismatch: {arrays[key].shape}/{arrays[key].dtype}"
            )
        declared = (metadata.get("fields") or {}).get(key)
        if not isinstance(declared, dict):
            raise ValueError(f"metadata lacks field declaration for {key}")
        if declared.get("shape") != list(expected_shapes[key]) or declared.get("dtype") != str(expected_dtypes[key]):
            raise ValueError(f"metadata declaration mismatch for {key}")
    for key in ("action", "proprio", "privileged", "reward"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"replay {key} contains non-finite values")
    if np.any(np.abs(arrays["action"]) > 1.000001):
        raise ValueError("replay actions exceed [-1,1]")
    if arrays["done"][:-1].any() or not bool(arrays["done"][-1]):
        raise ValueError("replay must terminate only at action 1600")
    return ReplayBundle(
        path=Path(path).resolve(),
        sha256=bundle_sha256,
        metadata=metadata,
        episode=episode,
        **arrays,
    )


def _metadata_number(metadata: dict[str, Any], key: str, cast):
    if key not in metadata:
        raise ValueError(f"stagec_v2_metadata is missing {key!r}")
    return cast(metadata[key])


def build_exact_cfg(VecEnvCfg, *, template: Path, episode: dict[str, Any]):
    """Reconstruct the archived evaluator's full no-auxiliary Stage-D config."""

    from frc_rebuilt.rl.cycle_v2 import (
        COLLECT_UNTIL_PREFERRED_REVISIONS,
        POSTDUMP_COMPLETE_CYCLE_REVISIONS,
        POSTDUMP_TARGET_REVISIONS,
        RAMP_OUT_REVISIONS,
        RETURN_INTAKE_REVISIONS,
        SCORE_EFFICIENCY_REVISIONS,
        SUPPORTED_ROUTE_EFFICIENCY_REVISIONS,
    )

    metadata = episode.get("stagec_v2_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("episode lacks stagec_v2_metadata")
    revision = metadata.get("reward_revision")
    if revision not in SUPPORTED_ROUTE_EFFICIENCY_REVISIONS:
        raise ValueError(f"unsupported archived reward revision: {revision!r}")
    route_v2 = revision in ("outer_rail_v2", "outer_rail_v3", *RAMP_OUT_REVISIONS)
    ramp = revision in RAMP_OUT_REVISIONS
    return_intake = revision in RETURN_INTAKE_REVISIONS
    return VecEnvCfg(
        num_envs=2,
        template_usd=str(Path(template).resolve()),
        # The archived evaluator built its six policy cameras before reset.
        # Preserve that initialization path because it affects Isaac's startup
        # settling; the added overhead render product is read separately.
        cameras=True,
        episode_len_s=160.0,
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
        cycle_v2_reset_modes=("full",),
        cycle_v2_target_load=_metadata_number(metadata, "target_load", int),
        cycle_v2_reserve_count=_metadata_number(metadata, "reserve_count", int),
        cycle_v2_reserve_batches=_metadata_number(metadata, "reserve_batches", int),
        cycle_v2_score_fraction=_metadata_number(metadata, "cycle_score_fraction", float),
        cycle_v2_score_floor=_metadata_number(metadata, "cycle_score_floor", int),
        cycle_v2_progress_per_m=_metadata_number(metadata, "progress_per_m", float),
        cycle_v2_progress_step_cap=_metadata_number(metadata, "progress_step_cap", float),
        cycle_v2_ramp_bonus=_metadata_number(metadata, "ramp_bonus", float),
        cycle_v2_refresh_ramp_side_on_dump=bool(metadata["refresh_ramp_side_on_dump"]),
        cycle_v2_ramp_side_deadband_x=_metadata_number(metadata, "ramp_side_deadband_x", float),
        cycle_v2_require_ramp_out=bool(metadata["require_ramp_out"]) if ramp else False,
        cycle_v2_ramp_out_half_width=_metadata_number(metadata, "ramp_out_half_width", float) if ramp else 0.9,
        cycle_v2_ramp_out_bonus=_metadata_number(metadata, "ramp_out_bonus", float) if ramp else 0.0,
        cycle_v2_off_ramp_exit_penalty=_metadata_number(metadata, "off_ramp_exit_penalty", float) if ramp else 0.0,
        cycle_v2_postdump_require_target_load=bool(metadata["postdump_require_target_load"]) if revision in POSTDUMP_TARGET_REVISIONS else False,
        cycle_v2_postdump_complete_cycle=bool(metadata["postdump_complete_cycle"]) if revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS else False,
        cycle_v2_postdump_depleted_count=int(metadata["postdump_depleted_count"]) if revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS else 0,
        cycle_v2_postdump_depleted_prob=float(metadata["postdump_depleted_prob"]) if revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS else 0.0,
        cycle_v2_preferred_repeat_load=int(metadata["preferred_repeat_load"]) if revision in SCORE_EFFICIENCY_REVISIONS else 0,
        cycle_v2_collect_until_preferred=revision in COLLECT_UNTIL_PREFERRED_REVISIONS,
        cycle_v2_collect_stall_steps=int(metadata["collect_stall_steps"]) if revision in COLLECT_UNTIL_PREFERRED_REVISIONS else 0,
        cycle_v2_return_time_guard=float(metadata["return_time_guard"]) if revision in COLLECT_UNTIL_PREFERRED_REVISIONS else 0.0,
        cycle_v2_intake_during_return=bool(metadata["intake_during_return"]) if return_intake else False,
        cycle_v2_repeat_load_return_bonus=float(metadata["repeat_load_return_bonus"]) if revision in SCORE_EFFICIENCY_REVISIONS else 0.0,
        cycle_v2_repeat_load_score_bonus=float(metadata["repeat_load_score_bonus"]) if revision in SCORE_EFFICIENCY_REVISIONS else 0.0,
        cycle_v2_outer_rail_enter_x=_metadata_number(metadata, "outer_rail_enter_x", float),
        cycle_v2_outer_rail_exit_x=_metadata_number(metadata, "outer_rail_exit_x", float),
        cycle_v2_outer_rail_max_x=_metadata_number(metadata, "outer_rail_max_x", float),
        cycle_v2_outer_rail_grace_steps=_metadata_number(metadata, "outer_rail_grace_steps", int),
        cycle_v2_outer_rail_penalty_per_step=_metadata_number(metadata, "outer_rail_penalty_per_step", float),
        cycle_v2_outer_rail_penalty_cap=_metadata_number(metadata, "outer_rail_penalty_cap", float),
        cycle_v2_outer_rail_min_scale=_metadata_number(metadata, "outer_rail_min_scale", float) if route_v2 else 0.0,
        cycle_v2_outer_rail_escalation_steps=_metadata_number(metadata, "outer_rail_escalation_steps", int) if route_v2 else 0,
        cycle_v2_outer_rail_max_multiplier=_metadata_number(metadata, "outer_rail_max_multiplier", float) if route_v2 else 1.0,
        cycle_v2_intake_substeps=_metadata_number(metadata, "intake_substeps", int) if route_v2 else 1,
        cycle_v2_leave_grace_steps=_metadata_number(metadata, "leave_grace_steps", int),
        cycle_v2_leave_penalty_per_step=_metadata_number(metadata, "leave_penalty_per_step", float),
        cycle_v2_leave_penalty_cap=_metadata_number(metadata, "leave_penalty_cap", float),
        cycle_v2_return_grace_steps=_metadata_number(metadata, "return_grace_steps", int),
        cycle_v2_return_penalty_per_step=_metadata_number(metadata, "return_penalty_per_step", float),
        cycle_v2_return_penalty_cap=_metadata_number(metadata, "return_penalty_cap", float),
        cycle_v2_shoot_grace_steps=int(round(_metadata_number(metadata, "shoot_grace_s", float) * 10.0)),
        cycle_v2_shoot_penalty_per_step=_metadata_number(metadata, "shoot_penalty_per_step", float),
        cycle_v2_shoot_penalty_cap=_metadata_number(metadata, "shoot_penalty_cap", float),
        cycle_v2_dump_lost_aim_grace_ticks=_metadata_number(metadata, "dump_lost_aim_grace_ticks", int),
        cycle_v2_partial_dump_penalty_per_ball=_metadata_number(metadata, "partial_dump_penalty_per_ball", float),
        cycle_v2_partial_dump_penalty_cap=_metadata_number(metadata, "partial_dump_penalty_cap", float),
        stage_d=True,
        stage_d_first_inactive="blue",
        stage_d_synthetic_red_auto=(0, 0),
        stage_d_ferry=False,
        stage_d_return_when_live=False,
        stage_d_owncourt_loop=False,
        stage_d_prefix_rescue_s=35.0,
        seed=int(episode["env_seed"]),
    )


def telemetry(slot: Any, step: int) -> dict[str, Any]:
    phase = getattr(getattr(slot, "cycle_v2", None), "phase", None)
    phase = getattr(phase, "value", None) or getattr(phase, "name", None) or str(phase)
    active = bool(slot.router._score_eligible("blue", float(slot.clock_s)))
    return {
        "step": int(step),
        "elapsed_s": round(step / FPS, 1),
        "remaining_s": round(max(0.0, 160.0 - step / FPS), 1),
        "score": int(slot.router.scored["blue"]),
        "collected": int(slot.controller.balls_collected),
        "magazine": int(len(slot.controller.magazine)),
        "phase": str(phase).upper(),
        "hub": "ACTIVE" if active else "INACTIVE",
        "cycles": int(getattr(getattr(slot, "cycle_v2", None), "cycles_completed", 0)),
    }


def stage_d_prefix_view(
    stage_d_module: Any,
    proprio: np.ndarray,
    *,
    episode_len_s: float,
    legacy_dim: int,
) -> np.ndarray:
    """Build the frozen prefix input exactly as the Stage-D evaluator does.

    The prefix checkpoint was trained with a 90-second clock and a permanently
    eligible blue hub.  Stage-D's 160-second observation therefore cannot be
    sliced directly: ``pin_prefix_view`` rescales legacy clock channel 7 and
    restores the training-time hub value in channel 12.  Keeping this adapter
    explicit prevents a renderer from silently evaluating a different policy.
    """

    view = stage_d_module.pin_prefix_view(
        proprio,
        episode_len_s=float(episode_len_s),
        legacy_dim=int(legacy_dim),
    )
    view = np.asarray(view, dtype=np.float32)
    expected = (np.asarray(proprio).shape[0], int(legacy_dim))
    if view.shape != expected or not np.isfinite(view).all():
        raise RuntimeError(
            f"Stage-D pinned prefix view contract changed: {view.shape} != {expected}"
        )
    return view


class VideoSink:
    def __init__(self, frame_source: Any, output: Path):
        import cv2

        self.cv2 = cv2
        self.frame_source = frame_source
        self.output = Path(output)
        self.temp = self.output.with_name(f".{self.output.stem}.{os.getpid()}.partial.avi")
        self.first_frame_path = self.output.with_suffix(".first-frame.png")
        self.temp.unlink(missing_ok=True)
        self.first_frame_path.unlink(missing_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.temp),
            cv2.VideoWriter_fourcc(*CODEC),
            FPS,
            (CAMERA_SIZE[0] + SIDEBAR_WIDTH, CAMERA_SIZE[1]),
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"cannot open MJPG intermediate {self.temp}")
        prop = getattr(cv2, "VIDEOWRITER_PROP_QUALITY", None)
        if prop is not None:
            self.writer.set(prop, 100)
        self.frames = 0

    @staticmethod
    def _clock(seconds: float) -> str:
        ticks = max(0, int(round(seconds * 10)))
        return f"{ticks // 600:02d}:{(ticks // 10) % 60:02d}.{ticks % 10}"

    @staticmethod
    def _panel(
        canvas: np.ndarray,
        cv2: Any,
        frame_chw: np.ndarray,
        *,
        x: int,
        title: str,
        badge: str,
    ) -> None:
        y, width, height, header = 890, 330, 174, 30
        frame = np.asarray(frame_chw)
        if frame.shape != (3, 90, 160) or frame.dtype != np.uint8:
            raise RuntimeError(f"invalid policy camera panel {frame.shape}/{frame.dtype}")
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (63, 69, 75), -1)
        cv2.rectangle(canvas, (x + 2, y + 2), (x + width - 2, y + header), (29, 33, 37), -1)
        cv2.putText(canvas, "v", (x + 13, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 211, 215), 1, cv2.LINE_AA)
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)[0]
        cv2.putText(
            canvas,
            title,
            (x + (width - title_size[0]) // 2, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (238, 242, 244),
            2,
            cv2.LINE_AA,
        )
        bgr = np.ascontiguousarray(np.transpose(frame, (1, 2, 0))[..., ::-1])
        resized = cv2.resize(bgr, (width - 4, height - header - 4), interpolation=cv2.INTER_CUBIC)
        canvas[y + header + 2 : y + height - 2, x + 2 : x + width - 2] = resized
        cv2.rectangle(canvas, (x + 10, y + height - 32), (x + 72, y + height - 7), (20, 24, 29), -1)
        cv2.rectangle(canvas, (x + 10, y + height - 32), (x + 72, y + height - 7), (114, 124, 132), 1)
        cv2.putText(canvas, badge, (x + 16, y + height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 239, 241), 1, cv2.LINE_AA)

    def write(self, data: dict[str, Any], policy_rgb: np.ndarray) -> None:
        rgba = np.asarray(self.frame_source())
        width, height = CAMERA_SIZE
        if rgba.shape not in ((height, width, 3), (height, width, 4)) or rgba.dtype != np.uint8:
            raise RuntimeError(f"invalid native camera frame {rgba.shape}/{rgba.dtype}")
        rgb = np.ascontiguousarray(rgba[..., :3])
        if float(rgb.std()) <= 1.0:
            raise RuntimeError("native top camera returned a black frame")
        canvas = np.ascontiguousarray(rgb[..., ::-1])
        cv2 = self.cv2
        cv2.rectangle(canvas, (0, 0), (width, 44), (38, 43, 48), -1)
        cv2.putText(canvas, "Isaac Sim Python 5.1.0", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 247, 248), 2, cv2.LINE_AA)
        toolbar = ("RTX - Real-Time", "Perspective", "17 - 35 mm", "Zoom  15.800", "AE", "ISO  100.0")
        tx = 200
        for label in toolbar:
            box_width = max(55, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] + 24)
            cv2.rectangle(canvas, (tx, 7), (tx + box_width, 36), (22, 26, 30), -1)
            cv2.rectangle(canvas, (tx, 7), (tx + box_width, 36), (76, 84, 91), 1)
            cv2.putText(canvas, label, (tx + 11, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (238, 241, 243), 1, cv2.LINE_AA)
            tx += box_width + 12
        cv2.rectangle(canvas, (1771, 7), (1905, 36), (22, 26, 30), -1)
        cv2.rectangle(canvas, (1771, 7), (1905, 36), (76, 84, 91), 1)
        cv2.circle(canvas, (1790, 21), 7, (74, 204, 220), -1)
        cv2.putText(canvas, "Stage Lights", (1805, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (239, 242, 218), 1, cv2.LINE_AA)

        x0, y0, sw, sh = 28, 72, 500, 620
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + sw, y0 + sh), (23, 27, 31), -1)
        cv2.addWeighted(overlay, 0.91, canvas, 0.09, 0.0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + sw, y0 + sh), (92, 98, 104), 1)
        sx = x0 + 23
        cv2.putText(canvas, "REBUILT  |  Competition Robot", (sx, y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (69, 185, 231), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{str(data['phase']).upper()}  {self._clock(data['remaining_s'])}", (sx, y0 + 105), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (244, 246, 248), 2, cv2.LINE_AA)
        cv2.putText(canvas, "closed-loop policy control enabled", (sx, y0 + 139), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (211, 216, 219), 1, cv2.LINE_AA)
        cv2.putText(canvas, "RED HUB - ACTIVE", (sx, y0 + 178), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (63, 216, 112), 2, cv2.LINE_AA)
        blue_color = (63, 216, 112) if data["hub"] == "ACTIVE" else (94, 104, 235)
        cv2.putText(canvas, f"BLUE HUB - {data['hub']}", (sx + 265, y0 + 178), cv2.FONT_HERSHEY_SIMPLEX, 0.51, blue_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"FUEL scored   RED 0   BLUE {data['score']}", (sx, y0 + 225), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (242, 244, 246), 2, cv2.LINE_AA)
        cv2.putText(canvas, "FUEL total 456  |  field 448 + robot 8", (sx, y0 + 258), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (216, 221, 224), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Collected {data['collected']}  |  hopper {data['magazine']}  |  cycles {data['cycles']}", (sx, y0 + 292), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (216, 221, 224), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LIVE POLICY", (sx, y0 + 335), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (74, 190, 230), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Drive - intake - aim - shoot controlled by policy", (sx, y0 + 369), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 229, 231), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Full 160-second match - real-time score", (sx, y0 + 402), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 229, 231), 1, cv2.LINE_AA)
        buttons = (("COMPACT / EXTEND", "INTAKE ON/OFF"), ("AUTO AIM + SHOOT", "POLICY ACTIVE"))
        by = y0 + 472
        for left, right in buttons:
            for bx, label in ((sx, left), (sx + 233, right)):
                cv2.rectangle(canvas, (bx, by), (bx + 220, by + 38), (38, 42, 46), -1)
                cv2.rectangle(canvas, (bx, by), (bx + 220, by + 38), (70, 77, 82), 1)
                size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
                cv2.putText(canvas, label, (bx + (220 - size[0]) // 2, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 238, 240), 1, cv2.LINE_AA)
            by += 48
        cv2.rectangle(canvas, (sx, y0 + 568), (x0 + sw - 23, y0 + 602), (38, 42, 46), -1)
        cv2.rectangle(canvas, (sx, y0 + 568), (x0 + sw - 23, y0 + 602), (70, 77, 82), 1)
        cv2.putText(canvas, "RESET MATCH", (x0 + 197, y0 + 591), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 238, 240), 1, cv2.LINE_AA)

        panels = np.asarray(policy_rgb)
        if panels.shape != (3, 3, 90, 160):
            raise RuntimeError(f"unexpected live policy-camera tensor {panels.shape}")
        self._panel(canvas, cv2, panels[0], x=700, title="Viewport Intake", badge="INTAKE")
        self._panel(canvas, cv2, panels[1], x=1045, title="Viewport Shooter", badge="SHOOTER")
        self._panel(canvas, cv2, panels[2], x=1390, title="Viewport Navigation", badge="NAV")
        if self.frames == 0 and not cv2.imwrite(str(self.first_frame_path), canvas):
            raise RuntimeError(f"cannot write first-frame QA image {self.first_frame_path}")
        self.writer.write(canvas)
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def abort(self) -> None:
        self.close()
        self.temp.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_gui_camera_state(path: Path) -> tuple[dict[str, Any], str]:
    """Validate the exact main-viewport pose saved by the interactive GUI."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"GUI camera state does not exist: {resolved}")
    digest = sha256_file(resolved)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"GUI camera state is invalid JSON: {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != GUI_CAMERA_STATE_SCHEMA:
        raise ValueError(f"GUI camera state schema must be {GUI_CAMERA_STATE_SCHEMA}")
    if int(payload.get("fuel_template_count", -1)) != 456:
        raise ValueError("GUI camera state must come from the exact 456-FUEL scene")
    if payload.get("projection") != "perspective":
        raise ValueError("GUI camera state must use perspective projection")

    vectors: dict[str, np.ndarray] = {}
    for key in ("eye_xyz", "target_xyz", "up_xyz"):
        value = np.asarray(payload.get(key), dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"GUI camera state {key} must contain three finite values")
        vectors[key] = value
    forward = vectors["target_xyz"] - vectors["eye_xyz"]
    if float(np.linalg.norm(forward)) <= 1e-8:
        raise ValueError("GUI camera eye and target must be distinct")
    if float(np.linalg.norm(vectors["up_xyz"])) <= 1e-8:
        raise ValueError("GUI camera up vector must be non-zero")
    if float(np.linalg.norm(np.cross(forward, vectors["up_xyz"]))) <= 1e-8:
        raise ValueError("GUI camera up vector must not be parallel to its view")

    for key in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm"):
        value = float(payload[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"GUI camera state {key} must be finite and positive")
    for key in (
        "horizontal_aperture_offset_mm",
        "vertical_aperture_offset_mm",
        "exposure",
    ):
        value = float(payload.get(key, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"GUI camera state {key} must be finite")
    clipping = np.asarray(payload.get("clipping_range"), dtype=np.float64)
    if (
        clipping.shape != (2,)
        or not np.all(np.isfinite(clipping))
        or clipping[0] <= 0.0
        or clipping[1] <= clipping[0]
    ):
        raise ValueError("GUI camera clipping_range must be finite and increasing")
    resolution = np.asarray(payload.get("viewport_resolution"), dtype=np.int64)
    if resolution.shape != (2,) or bool((resolution <= 0).any()):
        raise ValueError("GUI camera viewport_resolution must contain two positive values")
    if not math.isclose(
        float(resolution[0] / resolution[1]),
        float(CAMERA_SIZE[0] / CAMERA_SIZE[1]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("GUI camera aspect ratio differs from the HD video renderer")
    return payload, digest


def run(args: argparse.Namespace) -> dict[str, Any]:
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
    code = verify_code_snapshot(args.code_root, args.code_archive, args.expected_code_archive_sha256)
    camera_state = camera_state_sha256 = None
    if args.camera_state_json is not None:
        camera_state, camera_state_sha256 = load_gui_camera_state(args.camera_state_json)
    bundle = load_replay_bundle(
        args.bundle,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_env_seed=args.expected_env_seed,
        expected_action_seed=args.expected_action_seed,
        expected_env_index=args.env_index,
    )
    companion = load_replay_bundle(
        args.companion_bundle,
        expected_bundle_sha256=args.expected_companion_bundle_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_env_seed=args.expected_env_seed,
        expected_action_seed=args.expected_action_seed,
        expected_env_index=0,
        expected_steps=1601,
    )
    if args.env_index != 1:
        raise ValueError("the verified score-202 target must be replayed in env index 1")
    if str(bundle.episode.get("prefix_sha256", "")).lower() != prefix_sha:
        raise ValueError("archived episode prefix SHA256 does not match supplied checkpoint")
    for path in (args.output, args.provenance_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")
    if args.output.suffix.lower() != ".avi":
        raise ValueError("--output must be .avi for the high-bitrate MJPG intermediate")

    sys.path.insert(0, str((args.code_root / "src").resolve()))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    env = None
    sink = None
    render_product = None
    rgb_annotator = None
    try:
        import omni.replicator.core as rep
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

        np.random.seed(int(bundle.episode["action_seed"]))
        torch.manual_seed(int(bundle.episode["action_seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(bundle.episode["action_seed"]))
        try:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(args.checkpoint, map_location="cpu")
        metadata = validate_composite_metadata(payload.get("stagec_v2"), prefix_sha)
        if metadata != bundle.episode.get("stagec_v2_metadata"):
            raise ValueError("checkpoint Stage-C metadata differs from archived score-202 contract")
        env = VecCompetitionEnv(build_exact_cfg(VecEnvCfg, template=args.template, episode=bundle.episode))
        policy_cameras = tuple(env.cameras.values())
        if not bool(getattr(env, "_camera_ready", False)):
            raise RuntimeError("policy camera initialization failed")

        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        origin = np.asarray(env.env_origins[args.env_index], np.float32)
        camera_path = f"/World/VerifiedTopCamera_env_{args.env_index}"
        stage = omni.usd.get_context().get_stage()
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        if camera_state is None:
            eye_offset = np.asarray([7.2, -10.2, 4.6], np.float64)
            target_offset = np.asarray([0.0, -3.9, 0.65], np.float64)
            camera_up = np.asarray([0.0, 0.0, 1.0], np.float64)
            focal_length_mm = 15.8
            horizontal_aperture_mm = 20.955
            vertical_aperture_mm = 20.955 * CAMERA_SIZE[1] / CAMERA_SIZE[0]
            horizontal_aperture_offset_mm = 0.0
            vertical_aperture_offset_mm = 0.0
            camera_exposure = 0.0
            clipping_range = np.asarray([0.1, 1000.0], np.float64)
        else:
            eye_offset = np.asarray(camera_state["eye_xyz"], np.float64)
            target_offset = np.asarray(camera_state["target_xyz"], np.float64)
            camera_up = np.asarray(camera_state["up_xyz"], np.float64)
            focal_length_mm = float(camera_state["focal_length_mm"])
            horizontal_aperture_mm = float(camera_state["horizontal_aperture_mm"])
            vertical_aperture_mm = float(camera_state["vertical_aperture_mm"])
            horizontal_aperture_offset_mm = float(
                camera_state.get("horizontal_aperture_offset_mm", 0.0)
            )
            vertical_aperture_offset_mm = float(
                camera_state.get("vertical_aperture_offset_mm", 0.0)
            )
            camera_exposure = float(camera_state.get("exposure", 0.0))
            clipping_range = np.asarray(camera_state["clipping_range"], np.float64)
        camera_prim.CreateFocalLengthAttr(focal_length_mm)
        camera_prim.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        camera_prim.CreateVerticalApertureAttr(vertical_aperture_mm)
        camera_prim.CreateHorizontalApertureOffsetAttr(horizontal_aperture_offset_mm)
        camera_prim.CreateVerticalApertureOffsetAttr(vertical_aperture_offset_mm)
        camera_prim.CreateExposureAttr(camera_exposure)
        camera_prim.CreateClippingRangeAttr(
            Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1]))
        )
        view = Gf.Matrix4d()
        eye = origin.astype(np.float64) + eye_offset
        target = origin.astype(np.float64) + target_offset
        view.SetLookAt(
            Gf.Vec3d(*map(float, eye)),
            Gf.Vec3d(*map(float, target)),
            Gf.Vec3d(*map(float, camera_up)),
        )
        UsdGeom.Xformable(camera_prim).AddTransformOp().Set(view.GetInverse())
        dome = UsdLux.DomeLight.Define(stage, "/World/VideoGuiDome")
        dome.CreateIntensityAttr(850.0)
        dome.CreateColorAttr(Gf.Vec3f(0.76, 0.82, 0.95))
        key = UsdLux.DistantLight.Define(stage, "/World/VideoGuiKey")
        key.CreateIntensityAttr(2600.0)
        key.CreateAngleAttr(0.8)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(42.0, -28.0, -24.0))
        render_product = rep.create.render_product(camera_path, CAMERA_SIZE)
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        # Match the evaluator's constructor reset plus zero-action startup step,
        # then build both frozen agents from the live observation contract.
        obs, _, _, _ = env.step(np.zeros((2, 7), np.float32))
        if bool((obs["rgb"].std(axis=(2, 3, 4)) <= 1.0).any()):
            raise RuntimeError("policy evaluator produced black startup frames")
        frames = D.to_policy_frames(obs["rgb"])
        agent = prefix_agent = None
        if not args.archived_actions:
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
        np.random.seed(int(bundle.episode["action_seed"]))
        torch.manual_seed(int(bundle.episode["action_seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(bundle.episode["action_seed"]))
        render_time = float(env.sim.current_time)
        rep.orchestrator.step(delta_time=0.0, rt_subframes=8, pause_timeline=False)
        if float(env.sim.current_time) != render_time or not env.sim.is_playing():
            raise RuntimeError("renderer prime changed simulation time or paused physics")
        sink = VideoSink(rgb_annotator.get_data, args.output)
        first = last = None
        max_proprio_error = 0.0
        max_privileged_error = 0.0
        max_reward_error = 0.0
        max_companion_state_error = 0.0
        max_companion_reward_error = 0.0
        max_target_action_error = 0.0
        max_companion_action_error = 0.0
        live_action_digest = hashlib.sha256()
        score_timeline_mismatches = 0
        max_score_timeline_error = 0
        terminal_flag_mismatches = 0
        for step in range(STEPS):
            live_proprio = np.asarray(obs["proprio"][args.env_index], np.float32)
            live_privileged = np.asarray(obs["privileged"][args.env_index], np.float32)
            p_error = float(np.max(np.abs(live_proprio - bundle.proprio[step])))
            v_error = float(np.max(np.abs(live_privileged - bundle.privileged[step])))
            max_proprio_error = max(max_proprio_error, p_error)
            max_privileged_error = max(max_privileged_error, v_error)
            companion_p = float(np.max(np.abs(obs["proprio"][0] - companion.proprio[step])))
            companion_v = float(np.max(np.abs(obs["privileged"][0] - companion.privileged[step])))
            max_companion_state_error = max(max_companion_state_error, companion_p, companion_v)
            frame_data = telemetry(env.slots[args.env_index], step)
            expected_score = int(round(float(bundle.proprio[step, 14]) * 20.0))
            score_error = abs(int(frame_data["score"]) - expected_score)
            max_score_timeline_error = max(max_score_timeline_error, score_error)
            score_timeline_mismatches += int(score_error != 0)
            if step == 0 and frame_data["score"] != 0:
                raise RuntimeError(f"live re-simulation score must start at 0, got {frame_data['score']}")
            sink.write(frame_data, np.asarray(obs["rgb"][args.env_index], np.uint8))
            first = frame_data if first is None else first
            last = frame_data
            if args.archived_actions:
                actions = archived_action_batch(bundle, companion, step)
            else:
                assert agent is not None and prefix_agent is not None
                frames = D.to_policy_frames(obs["rgb"])
                candidate = agent.act(frames, obs["proprio"], explore=False).astype(np.float32)
                prefix_view = stage_d_prefix_view(
                    _stage_d,
                    obs["proprio"],
                    episode_len_s=float(bundle.episode["episode_len_s"]),
                    legacy_dim=LEGACY_PROPRIO_DIM,
                )
                prefix = prefix_agent.act(
                    frames,
                    prefix_view,
                    explore=False,
                ).astype(np.float32)
                actions = compose_phase_actions(prefix, candidate, obs["proprio"])
                actions = apply_executed_action_policy(
                    actions,
                    obs["proprio"],
                    intake_during_return=bool(metadata.get("intake_during_return", False)),
                ).astype(np.float32)
            live_action_digest.update(np.ascontiguousarray(actions).tobytes())
            max_target_action_error = max(
                max_target_action_error,
                float(np.max(np.abs(actions[args.env_index] - bundle.action[step]))),
            )
            max_companion_action_error = max(
                max_companion_action_error,
                float(np.max(np.abs(actions[0] - companion.action[step]))),
            )
            obs, rewards, dones, info = env.step(actions)
            if step + 1 < STEPS:
                render_time = float(env.sim.current_time)
                rep.orchestrator.step(delta_time=0.0, rt_subframes=1, pause_timeline=False)
                if float(env.sim.current_time) != render_time or not env.sim.is_playing():
                    raise RuntimeError(f"renderer changed simulation time at step {step}")
            reward_error = abs(float(rewards[args.env_index]) - float(bundle.reward[step]))
            max_reward_error = max(max_reward_error, reward_error)
            terminal_flag_mismatches += int(bool(dones[args.env_index]) is not bool(bundle.done[step]))
            companion_reward_error = abs(float(rewards[0]) - float(companion.reward[step]))
            max_companion_reward_error = max(max_companion_reward_error, companion_reward_error)
            if step % 100 == 0:
                print(f"POLICY_RENDER_PROGRESS step={step + 1}/{STEPS} score={frame_data['score']}", flush=True)
        if sink.frames != STEPS:
            raise RuntimeError(f"video frame count mismatch: {sink.frames}/{STEPS}")
        terminal = dict(info.get("episode_stats", {}).get(args.env_index, {}))
        live_score = int(terminal.get("scored", last["score"] if last else -1))
        if terminal.get("terminal_reason") != "horizon":
            raise RuntimeError(f"live re-simulation did not reach the horizon: {terminal!r}")
        if live_score < MIN_LIVE_SCORE:
            raise RuntimeError(
                f"live re-simulation scored {live_score}; refusing to publish below {MIN_LIVE_SCORE}"
            )
        sink.close()
        if not sink.temp.is_file() or sink.temp.stat().st_size <= 0:
            raise RuntimeError("video encoder produced an empty intermediate")
        result = {
            "schema": PROVENANCE_SCHEMA,
            "bundle": {"path": str(bundle.path), "sha256": bundle.sha256},
            "companion_bundle": {
                "path": str(companion.path),
                "sha256": companion.sha256,
                "source_steps": 1601,
                "reference_steps": 1600,
                "archived_action_sha256_first1600": hashlib.sha256(
                    companion.action[:STEPS].tobytes()
                ).hexdigest(),
            },
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_sha},
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
            "contract": bundle.episode["stage_d_contract"],
            "env_seed": int(bundle.episode["env_seed"]),
            "action_seed": int(bundle.episode["action_seed"]),
            "env_index": int(args.env_index),
            "executed_actions": {
                "shape": [STEPS, 2, 7],
                "sha256": live_action_digest.hexdigest(),
                "source": "archived_exact_episode" if args.archived_actions else "live_policy",
                "max_target_abs_from_archived_trace": max_target_action_error,
                "max_companion_abs_from_archived_trace": max_companion_action_error,
            },
            "archived_target_actions": {
                "shape": [STEPS, 7],
                "sha256": hashlib.sha256(bundle.action.tobytes()).hexdigest(),
            },
            "classification": {
                "type": (
                    "deterministic_archived_action_resimulation"
                    if args.archived_actions
                    else "deterministic_closed_loop_policy_resimulation"
                ),
                "bit_exact_replay": False,
                "publication_gate": f"live terminal score >= {MIN_LIVE_SCORE}",
                "prefix_observation": None if args.archived_actions else "stage_d.pin_prefix_view",
            },
            "archived_source_terminal": {
                key: bundle.episode.get(key)
                for key in ("scored", "collected", "cycles_completed", "terminal_reason")
            },
            "measured_divergence": {
                "state_reference_atol": STATE_ATOL,
                "reward_reference_atol": REWARD_ATOL,
                "max_proprio_abs": max_proprio_error,
                "max_privileged_abs": max_privileged_error,
                "max_reward_abs": max_reward_error,
                "max_companion_state_abs": max_companion_state_error,
                "max_companion_reward_abs": max_companion_reward_error,
                "score_timeline_mismatch_frames": score_timeline_mismatches,
                "max_score_timeline_abs": max_score_timeline_error,
                "terminal_flag_mismatches": terminal_flag_mismatches,
            },
            "video": {
                "path": str(args.output.resolve()),
                "sha256": sha256_file(sink.temp),
                "bytes": sink.temp.stat().st_size,
                "codec": CODEC,
                "high_bitrate_intermediate": True,
                "fps": FPS,
                "frames": STEPS,
                "duration_s": 160.0,
                "camera_resolution": list(CAMERA_SIZE),
                "sidebar_width": SIDEBAR_WIDTH,
                "output_resolution": [CAMERA_SIZE[0] + SIDEBAR_WIDTH, CAMERA_SIZE[1]],
                "native_simulator_pixels": True,
                "policy_observation_upscale": True,
                "policy_panel_source_resolution": [160, 90],
                "policy_panel_count": 3,
                "extra_physics_steps_for_capture": 0,
                "added_overhead_render_products": 1,
                "retained_policy_render_products": len(policy_cameras),
                "zero_delta_render_calls": STEPS,
                "camera_eye_offset": eye_offset.tolist(),
                "camera_target_offset": target_offset.tolist(),
                "camera_eye_world": eye.tolist(),
                "camera_target_world": target.tolist(),
                "camera_up": camera_up.tolist(),
                "camera_focal_length": focal_length_mm,
                "camera_horizontal_aperture": horizontal_aperture_mm,
                "camera_vertical_aperture": vertical_aperture_mm,
                "camera_horizontal_aperture_offset": horizontal_aperture_offset_mm,
                "camera_vertical_aperture_offset": vertical_aperture_offset_mm,
                "camera_exposure": camera_exposure,
                "camera_clipping_range": clipping_range.tolist(),
                "camera_state_source": (
                    {
                        "path": str(args.camera_state_json.resolve()),
                        "sha256": camera_state_sha256,
                        "schema": camera_state["schema"],
                    }
                    if camera_state is not None
                    else None
                ),
                "lighting": {
                    "dome_intensity": 850.0,
                    "dome_color": [0.76, 0.82, 0.95],
                    "distant_intensity": 2600.0,
                    "distant_angle_deg": 0.8,
                    "distant_rotation_xyz_deg": [42.0, -28.0, -24.0],
                },
            },
            "first_frame": first,
            "last_frame": last,
            "live_resimulation_terminal": terminal,
        }
        provenance_temp = args.provenance_out.with_name(
            f".{args.provenance_out.name}.{os.getpid()}.ready"
        )
        _atomic_json(provenance_temp, result)
        os.replace(sink.temp, args.output)
        os.replace(provenance_temp, args.provenance_out)
        print("CLOSED_LOOP_HD_RENDER_DONE " + json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException:
        # Isaac Sim's shutdown can terminate the process before Python gets a
        # chance to display an uncaught exception.  Emit it before closing Kit
        # so fail-closed replay/render errors remain diagnosable.
        traceback.print_exc()
        raise
    finally:
        if sink is not None:
            sink.abort()
        if rgb_annotator is not None:
            try:
                if render_product is not None:
                    rgb_annotator.detach([render_product])
            except Exception:
                pass
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
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
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-prefix-checkpoint-sha256", required=True)
    parser.add_argument("--expected-env-seed", type=int, required=True)
    parser.add_argument("--expected-action-seed", type=int, required=True)
    parser.add_argument("--env-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--expected-code-archive-sha256", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, default=None)
    parser.add_argument(
        "--camera-state-json",
        type=Path,
        default=None,
        help="exact main-viewport pose saved from the 456-FUEL GUI with P",
    )
    parser.add_argument(
        "--archived-actions",
        action="store_true",
        help="execute the custody-verified target and companion action traces",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.provenance_out is None:
        args.provenance_out = args.output.with_suffix(".provenance.json")
    outputs = {args.output.resolve(), args.provenance_out.resolve()}
    protected = {
        args.bundle.resolve(),
        args.companion_bundle.resolve(),
        args.checkpoint.resolve(),
        args.prefix_checkpoint.resolve(),
        args.code_archive.resolve(),
        args.template.resolve(),
    }
    if args.camera_state_json is not None:
        protected.add(args.camera_state_json.resolve())
    if len(outputs) != 2 or outputs & protected:
        raise SystemExit("video/provenance outputs must be distinct and cannot overwrite inputs")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
