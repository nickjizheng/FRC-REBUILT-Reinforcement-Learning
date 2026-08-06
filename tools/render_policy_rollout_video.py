#!/usr/bin/env python3
"""Render an archived three-camera policy rollout as a reviewable video.

The evaluator capture is the source of truth.  This utility does not synthesize
or interpolate observations: each stored RGB observation is shown once, in
order, alongside a provenance panel derived from the capture metadata.

Requires NumPy and OpenCV (``pip install numpy opencv-python``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PHASE_NAMES = ("LEAVE", "COLLECT", "RETURN", "SCORE")
PHASE_COLORS = {
    "OPENING": (255, 193, 7),
    "LEAVE": (255, 183, 77),
    "COLLECT": (66, 165, 245),
    "RETURN": (171, 71, 188),
    "SCORE": (76, 175, 80),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def put_text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (235, 238, 242),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def camera_panel(
    frame: np.ndarray,
    camera_index: int,
    label: str,
    width: int,
    height: int,
    resampling: str,
    sharpen_amount: float,
) -> np.ndarray:
    rgb = np.transpose(frame[camera_index * 3 : (camera_index + 1) * 3], (1, 2, 0))
    interpolation = {
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
    }[resampling]
    panel = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (width, height),
        interpolation=interpolation,
    )
    if sharpen_amount:
        # A restrained, deterministic unsharp mask applied only to the recorded
        # camera pixels. This does not synthesize or interpolate new frames.
        blurred = cv2.GaussianBlur(panel, (0, 0), 0.8)
        panel = cv2.addWeighted(
            panel,
            1.0 + sharpen_amount,
            blurred,
            -sharpen_amount,
            0,
        )
    bar_height = max(30, height // 10)
    cv2.rectangle(panel, (0, 0), (width, bar_height), (10, 14, 20), -1)
    cv2.rectangle(panel, (0, 0), (max(5, width // 100), height), (45, 156, 219), -1)
    put_text(
        panel,
        label.upper(),
        (max(18, width // 32), int(bar_height * 0.72)),
        max(0.55, width / 1_000),
        (245, 248, 250),
        2,
    )
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Exact evaluator capture (.npz)")
    parser.add_argument("output", type=Path, help="Output .mp4 or .webm")
    parser.add_argument("--poster", type=Path, help="Optional PNG poster output")
    parser.add_argument("--provenance", type=Path, help="Optional JSON sidecar output")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc, e.g. mp4v or VP90")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS; capture is 10 Hz")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera-resampling",
        choices=("cubic", "lanczos", "nearest"),
        default="cubic",
        help="Deterministic camera-pixel resize filter",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=0.0,
        help="Restrained unsharp-mask amount applied after camera resize (0 to disable)",
    )
    parser.add_argument("--poster-second", type=float, default=60.0)
    parser.add_argument("--policy-label", default="Promoted policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width % 2 or args.height % 2:
        raise ValueError("Width and height must be even for the 2x2 layout")
    if len(args.codec) != 4:
        raise ValueError("Codec must be a four-character OpenCV fourcc")
    if not 0.0 <= args.sharpen_amount <= 0.5:
        raise ValueError("Sharpen amount must be between 0.0 and 0.5")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.capture, allow_pickle=False) as archive:
        observations = archive["obs"]
        proprio = archive["proprio"]
        metadata = json.loads(bytes(archive["metadata"]).decode("utf-8"))["episode"]
        if observations.ndim != 4 or observations.shape[1] != 9:
            raise ValueError(f"Expected [frames, 9, height, width], got {observations.shape}")
        if proprio.shape[0] != observations.shape[0] or proprio.shape[1] < 27:
            raise ValueError(f"Unexpected proprio shape {proprio.shape}")

        half_width, half_height = args.width // 2, args.height // 2
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*args.codec),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open {args.output} with codec {args.codec}")

        poster_frame: np.ndarray | None = None
        for index, frame in enumerate(observations):
            canvas = np.full((args.height, args.width, 3), (14, 18, 24), np.uint8)
            canvas[:half_height, :half_width] = camera_panel(
                frame,
                0,
                "Intake camera",
                half_width,
                half_height,
                args.camera_resampling,
                args.sharpen_amount,
            )
            canvas[:half_height, half_width:] = camera_panel(
                frame,
                1,
                "Shooter camera",
                half_width,
                half_height,
                args.camera_resampling,
                args.sharpen_amount,
            )
            canvas[half_height:, :half_width] = camera_panel(
                frame,
                2,
                "Navigation camera",
                half_width,
                half_height,
                args.camera_resampling,
                args.sharpen_amount,
            )

            panel = canvas[half_height:, half_width:]
            cv2.rectangle(panel, (0, 0), (half_width - 1, half_height - 1), (20, 26, 35), -1)
            cv2.rectangle(panel, (0, 0), (max(7, half_width // 80), half_height - 1), (45, 156, 219), -1)

            elapsed = index / args.fps
            phase_values = proprio[index, 23:27]
            phase = (
                PHASE_NAMES[int(np.argmax(phase_values))]
                if float(phase_values.max()) > 0.5
                else "OPENING"
            )
            accent = PHASE_COLORS[phase]
            final_seconds = float(metadata["episode_len_s"])
            ui_scale = min(half_width / 640.0, half_height / 360.0)
            scaled_y = lambda value: int(round(value * ui_scale))
            x = max(20, int(round(30 * ui_scale)), half_width // 22)
            put_text(
                panel,
                "FIXED-CHECKPOINT ROLLOUT",
                (x, scaled_y(38)),
                0.75 * ui_scale,
                (255, 255, 255),
                2,
            )
            put_text(
                panel,
                args.policy_label,
                (x, scaled_y(72)),
                0.72 * ui_scale,
                (79, 195, 247),
                2,
            )
            put_text(
                panel,
                f"MATCH  {int(elapsed) // 60:02d}:{elapsed % 60:04.1f} / "
                f"{int(final_seconds) // 60:02d}:{final_seconds % 60:04.1f}",
                (x, scaled_y(120)),
                0.78 * ui_scale,
                (245, 248, 250),
                2,
            )
            put_text(
                panel,
                f"PHASE  {phase}",
                (x, scaled_y(158)),
                0.78 * ui_scale,
                accent,
                2,
            )

            x0, y0, x1, y1 = (
                x,
                scaled_y(178),
                half_width - x - 4,
                scaled_y(198),
            )
            cv2.rectangle(panel, (x0, y0), (x1, y1), (55, 64, 75), -1)
            progress_x = x0 + int((x1 - x0) * min(1.0, elapsed / final_seconds))
            cv2.rectangle(panel, (x0, y0), (progress_x, y1), accent, -1)
            cv2.rectangle(panel, (x0, y0), (x1, y1), (100, 112, 126), 1)

            put_text(
                panel,
                "VERIFIED FINAL OUTCOME",
                (x, scaled_y(236)),
                0.62 * ui_scale,
                (176, 190, 203),
                1,
            )
            put_text(
                panel,
                f"{metadata['scored']} scored  |  {metadata['collected']} collected",
                (x, scaled_y(270)),
                0.68 * ui_scale,
                (255, 255, 255),
                2,
            )
            put_text(
                panel,
                f"{metadata['cycles_completed']} completed cycles  |  "
                f"repeat load {metadata['repeat_scored_load_max']}",
                (x, scaled_y(302)),
                0.58 * ui_scale,
                (225, 230, 235),
                1,
            )
            put_text(
                panel,
                f"checkpoint {metadata['checkpoint_sha256'][:12]}  |  "
                f"seed {metadata['env_seed']}  |  env {metadata['env_index']}",
                (x, scaled_y(335)),
                0.43 * ui_scale,
                (156, 168, 180),
                1,
            )
            writer.write(canvas)
            if index == min(len(observations) - 1, round(args.poster_second * args.fps)):
                poster_frame = canvas.copy()
        writer.release()

    if args.poster and poster_frame is not None:
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.poster), poster_frame, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise RuntimeError(f"Could not write poster {args.poster}")

    sidecar = {
        "schema": "frc_rebuilt_public_rollout_video_v1",
        "video_file": args.output.name,
        "video_sha256": sha256(args.output),
        "video_bytes": args.output.stat().st_size,
        "duration_seconds": len(observations) / args.fps,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "codec_fourcc": args.codec,
        "camera_resampling": args.camera_resampling,
        "sharpen_amount": args.sharpen_amount,
        "audio": False,
        "source_capture": args.capture.name,
        "source_capture_sha256": sha256(args.capture),
        "source_capture_bytes": args.capture.stat().st_size,
        "source_observation_shape": list(observations.shape),
        "source_observation_dtype": str(observations.dtype),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_train_steps": metadata["checkpoint_train_steps"],
        "checkpoint_v2_updates": metadata["checkpoint_v2_updates"],
        "env_seed": metadata["env_seed"],
        "action_seed": metadata["action_seed"],
        "env_index": metadata["env_index"],
        "action_mode": metadata["action_mode"],
        "episode_len_s": metadata["episode_len_s"],
        "episode_steps": metadata["episode_steps"],
        "policy_speed_scale": metadata["stage_d_contract"]["policy_speed_scale"],
        "scored": metadata["scored"],
        "collected": metadata["collected"],
        "cycles_completed": metadata["cycles_completed"],
        "repeat_scored_load_max": metadata["repeat_scored_load_max"],
        "terminal_reason": metadata["terminal_reason"],
        "stage_d_contract": metadata["stage_d_contract"],
        "layout": (
            "three synchronized RGB actor-observation cameras plus provenance panel; "
            "no frames omitted"
        ),
        "render_note": (
            "Source camera tensors are native 160x90 RGB at 10 Hz and are "
            f"{args.camera_resampling}-upscaled for presentation; camera-only "
            f"unsharp-mask amount is {args.sharpen_amount}."
        ),
    }
    provenance_path = args.provenance or args.output.with_suffix(".provenance.json")
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
