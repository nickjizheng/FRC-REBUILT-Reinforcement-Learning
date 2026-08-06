#!/usr/bin/env python3
"""Offline BC learner for an exact-V4 Stage C phase-drive residual.

This process has no replay transport, critic, collector publication, or active
model pointer.  It consumes only paired, curated full-match captures and emits
immutable candidate snapshots for external held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frc_rebuilt.rl.drqv2 import Actor, Encoder
from frc_rebuilt.rl.phase_drive_residual import (
    FrozenBasePhaseDrivePolicy,
    PhaseDriveResidual,
    RESIDUAL_CAP,
    build_phase_drive_residual_checkpoint,
)
from frc_rebuilt.rl.policy_v2 import LEGACY_PROPRIO_DIM, PolicyPhase, V2_PROPRIO_DIM


TRAINING_SCHEMA = "stagec_phase_drive_residual_training_v1"
CAPTURE_SCHEMA = "stagec_training_episode_v1"
PHASE_INDEX = {
    "leave": int(PolicyPhase.LEAVE),
    "collect": int(PolicyPhase.COLLECT),
    "return": int(PolicyPhase.RETURN),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    decoded = json.loads(bytes(value).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("capture metadata must be a JSON object")
    return decoded


def _last_successful_row(episode: dict[str, Any], length: int) -> int:
    steps = episode.get("cycle_success_steps") or []
    if not steps:
        raise ValueError("curated capture has no verified cycle success step")
    row = int(steps[-1])
    if not 0 <= row < int(length):
        raise ValueError("cycle success step lies outside its capture")
    return row


def _selected_rows(
    proprio: np.ndarray,
    episode: dict[str, Any],
    phase: str,
) -> np.ndarray:
    values = np.asarray(proprio)
    if values.ndim != 2 or values.shape[1] != V2_PROPRIO_DIM:
        raise ValueError("capture proprio is not Stage C v2 width 30")
    phase_values = values[
        :, LEGACY_PROPRIO_DIM : LEGACY_PROPRIO_DIM + len(PolicyPhase)
    ]
    if not np.isfinite(phase_values).all():
        raise ValueError("capture phase values are non-finite")
    decoded = np.argmax(phase_values, axis=1)
    strict = np.isclose(phase_values, 0.0, atol=1e-6) | np.isclose(
        phase_values, 1.0, atol=1e-6
    )
    if not bool(strict.all()) or not bool(
        np.isclose(phase_values, 1.0, atol=1e-6).sum(axis=1).min() == 1
    ) or not bool(
        np.isclose(phase_values, 1.0, atol=1e-6).sum(axis=1).max() == 1
    ):
        raise ValueError("capture phase features are not strict one-hot")
    last = _last_successful_row(episode, len(values))
    return np.flatnonzero(
        (decoded == PHASE_INDEX[phase]) & (np.arange(len(values)) <= last)
    )


def _episode_weight(decision: dict[str, Any]) -> float:
    score_gain = max(
        0,
        int(decision.get("candidate_score", 0))
        - int(decision.get("control_score", 0)),
    )
    cycle_gain = max(
        0,
        int(decision.get("candidate_cycles", 0))
        - int(decision.get("control_cycles", 0)),
    )
    return float(min(4.0, 1.0 + 0.05 * score_gain + 0.5 * cycle_gain))


@dataclass
class EpisodeRows:
    path: Path
    obs: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    weight: float


def _load_episode(
    path: Path,
    *,
    phase: str,
    base_sha256: str,
    decision: dict[str, Any],
) -> EpisodeRows:
    with np.load(path, allow_pickle=False) as archive:
        required = {"obs", "proprio", "action", "done", "metadata"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing capture fields: {sorted(missing)}")
        metadata = _decode_metadata(archive["metadata"])
        if metadata.get("schema") != CAPTURE_SCHEMA:
            raise ValueError(f"{path} has an unsupported capture schema")
        episode = metadata.get("episode")
        if not isinstance(episode, dict):
            raise ValueError(f"{path} is missing episode metadata")
        if episode.get("checkpoint_sha256") != base_sha256:
            raise ValueError(f"{path} was not mined from the immutable base")
        if episode.get("action_mode") != "smooth-drive":
            raise ValueError(f"{path} was not generated with smooth-drive")
        if episode.get("noise_phases") != [phase]:
            raise ValueError(f"{path} was not generated for phase {phase}")
        if float(episode.get("noise_cap", 1.0)) > RESIDUAL_CAP:
            raise ValueError(f"{path} exploration exceeded the residual cap")
        obs = np.asarray(archive["obs"])
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
        action = np.asarray(archive["action"], dtype=np.float32)
        done = np.asarray(archive["done"], dtype=np.bool_)
        if not (len(obs) == len(proprio) == len(action) == len(done)):
            raise ValueError(f"{path} capture fields have inconsistent lengths")
        if not len(done) or not bool(done[-1]) or bool(done[:-1].any()):
            raise ValueError(f"{path} has an invalid terminal boundary")
        if action.ndim != 2 or action.shape[1] != 7:
            raise ValueError(f"{path} actions are not width 7")
        rows = _selected_rows(proprio, episode, phase)
        if not len(rows):
            raise ValueError(f"{path} has no selected {phase} rows before success")
        return EpisodeRows(
            path=path,
            obs=obs[rows].copy(),
            proprio=proprio[rows].copy(),
            action=action[rows].copy(),
            weight=_episode_weight(decision),
        )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _base_tensors_unchanged(
    payload: dict[str, Any], encoder: Encoder, actor: Actor
) -> bool:
    for name, module in (("encoder", encoder), ("actor", actor)):
        source = payload[name]
        current = module.state_dict()
        if source.keys() != current.keys():
            return False
        if any(not torch.equal(source[key].cpu(), current[key].cpu()) for key in source):
            return False
    return True


def _sample_batch(
    episodes: list[EpisodeRows],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode_indices = rng.integers(0, len(episodes), size=int(batch_size))
    obs, proprio, action, weights = [], [], [], []
    for episode_index in episode_indices:
        episode = episodes[int(episode_index)]
        row = int(rng.integers(0, len(episode.proprio)))
        obs.append(episode.obs[row])
        proprio.append(episode.proprio[row])
        action.append(episode.action[row])
        weights.append(episode.weight)
    return (
        np.stack(obs),
        np.stack(proprio),
        np.stack(action),
        np.asarray(weights, dtype=np.float32),
    )


def _mean_loss(
    policy: FrozenBasePhaseDrivePolicy,
    episodes: list[EpisodeRows],
    device: torch.device,
    *,
    max_rows_per_episode: int = 256,
) -> float:
    losses: list[float] = []
    policy.residual.eval()
    with torch.no_grad():
        for episode in episodes:
            count = min(len(episode.proprio), int(max_rows_per_episode))
            indices = np.linspace(0, len(episode.proprio) - 1, count, dtype=np.int64)
            obs = torch.as_tensor(episode.obs[indices], device=device)
            proprio = torch.as_tensor(episode.proprio[indices], device=device)
            target = torch.as_tensor(episode.action[indices, :3], device=device)
            predicted = policy.mean_actions(obs, proprio)[:, :3]
            losses.append(float(F.mse_loss(predicted, target).item()))
    policy.residual.train()
    return float(np.mean(losses)) if losses else float("nan")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_sha256 = _sha256(args.base_checkpoint)
    curation_path = args.curation_dir / "curation.json"
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    if curation.get("schema") != "stagec_residual_curation_v1":
        raise ValueError("curation schema mismatch")
    if curation.get("status") != "READY" or curation.get("phase") != args.phase:
        raise ValueError("curation is not READY for the selected phase")
    accepted = [item for item in curation["decisions"] if item.get("accepted")]
    if len(accepted) < int(args.min_episodes):
        raise ValueError(
            f"need at least {args.min_episodes} accepted episodes, got {len(accepted)}"
        )
    decisions_by_name = {
        Path(item["capture"]).name: item for item in accepted
    }
    capture_paths = sorted((args.curation_dir / "captures").glob("*.npz"))
    if set(path.name for path in capture_paths) != set(decisions_by_name):
        raise ValueError("curation decisions and capture directory differ")

    try:
        base_payload = torch.load(
            args.base_checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:
        base_payload = torch.load(args.base_checkpoint, map_location="cpu")
    if not isinstance(base_payload, dict):
        raise ValueError("base checkpoint is not a mapping")
    episodes = [
        _load_episode(
            path,
            phase=args.phase,
            base_sha256=base_sha256,
            decision=decisions_by_name[path.name],
        )
        for path in capture_paths
    ]
    random.Random(args.seed).shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * args.validation_fraction)))
    if validation_count >= len(episodes):
        validation_count = 1
    validation = episodes[:validation_count]
    training = episodes[validation_count:]
    if not training:
        raise ValueError("residual training split has no training episodes")

    sample = episodes[0]
    frame_channels, frame_h, frame_w = map(int, sample.obs.shape[1:])
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    encoder = Encoder(frame_channels).to(device)
    with torch.no_grad():
        probe = torch.zeros(1, frame_channels, frame_h, frame_w, device=device)
        feat_dim = int(encoder(probe).shape[1])
    actor = Actor(feat_dim, V2_PROPRIO_DIM, 7).to(device)
    encoder.load_state_dict(base_payload["encoder"], strict=True)
    actor.load_state_dict(base_payload["actor"], strict=True)
    trunk_dim = int(actor.trunk[0].out_features)
    residual = PhaseDriveResidual(trunk_dim).to(device)
    selected_head = residual.heads[args.phase]
    for name, head in residual.heads.items():
        if name != args.phase:
            for parameter in head.parameters():
                parameter.requires_grad_(False)
    policy = FrozenBasePhaseDrivePolicy(encoder, actor, residual)
    optimizer = torch.optim.AdamW(
        selected_head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    rng = np.random.default_rng(args.seed + 1)

    training_meta = {
        "schema": TRAINING_SCHEMA,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": base_sha256,
        "curation": str(curation_path.resolve()),
        "curation_sha256": _sha256(curation_path),
        "phase": args.phase,
        "cap": RESIDUAL_CAP,
        "seed": int(args.seed),
        "training_episodes": [str(item.path) for item in training],
        "validation_episodes": [str(item.path) for item in validation],
        "updates_requested": int(args.updates),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(training_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    initial_payload = build_phase_drive_residual_checkpoint(
        base_payload,
        residual,
        base_checkpoint_sha256=base_sha256,
        updates=0,
        optimizer_state=optimizer.state_dict(),
    )
    initial_payload["phase_drive_residual_training"] = dict(training_meta)
    _atomic_torch_save(initial_payload, args.out_dir / "initial_zero.pt")

    best_loss = float("inf")
    best_update = 0
    log_path = args.out_dir / "train.jsonl"
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        for update in range(1, int(args.updates) + 1):
            obs_np, proprio_np, action_np, weight_np = _sample_batch(
                training, args.batch_size, rng
            )
            obs = torch.as_tensor(obs_np, device=device)
            proprio = torch.as_tensor(proprio_np, device=device)
            target = torch.as_tensor(action_np[:, :3], device=device)
            weights = torch.as_tensor(weight_np, device=device)
            predicted = policy.mean_actions(obs, proprio)[:, :3]
            row_loss = (predicted - target).square().mean(dim=1)
            loss = (row_loss * weights).sum() / weights.sum().clamp_min(1e-6)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite residual loss at update {update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                selected_head.parameters(), float(args.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f"non-finite residual gradient at update {update}")
            optimizer.step()
            if not _base_tensors_unchanged(base_payload, encoder, actor):
                raise RuntimeError("immutable base encoder/actor changed during residual BC")

            if update == 1 or update % int(args.report_interval) == 0:
                train_loss = _mean_loss(policy, training, device)
                validation_loss = _mean_loss(policy, validation, device)
                entry = {
                    "update": update,
                    "batch_loss": float(loss.item()),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "grad_norm": float(grad_norm.item()),
                }
                log.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_update = update
                    candidate = build_phase_drive_residual_checkpoint(
                        base_payload,
                        residual,
                        base_checkpoint_sha256=base_sha256,
                        updates=update,
                        optimizer_state=optimizer.state_dict(),
                    )
                    candidate["phase_drive_residual_training"] = {
                        **training_meta,
                        "best_validation_loss": best_loss,
                        "best_update": best_update,
                    }
                    _atomic_torch_save(candidate, args.out_dir / "best_candidate.pt")
            if update % int(args.snapshot_interval) == 0:
                candidate = build_phase_drive_residual_checkpoint(
                    base_payload,
                    residual,
                    base_checkpoint_sha256=base_sha256,
                    updates=update,
                    optimizer_state=optimizer.state_dict(),
                )
                candidate["phase_drive_residual_training"] = dict(training_meta)
                _atomic_torch_save(
                    candidate, args.out_dir / f"candidate_{update:06d}.pt"
                )

    final_payload = build_phase_drive_residual_checkpoint(
        base_payload,
        residual,
        base_checkpoint_sha256=base_sha256,
        updates=int(args.updates),
        optimizer_state=optimizer.state_dict(),
    )
    final_payload["phase_drive_residual_training"] = {
        **training_meta,
        "best_validation_loss": best_loss,
        "best_update": best_update,
    }
    _atomic_torch_save(final_payload, args.out_dir / "final_candidate.pt")
    summary = {
        **training_meta,
        "status": "CANDIDATES_REQUIRE_EXTERNAL_GATE",
        "best_validation_loss": best_loss,
        "best_update": best_update,
        "base_encoder_actor_unchanged": _base_tensors_unchanged(
            base_payload, encoder, actor
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--curation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASE_INDEX), required=True)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--min-episodes", type=int, default=4)
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    for name in ("base_checkpoint",):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist")
    if not args.curation_dir.is_dir():
        parser.error("--curation-dir does not exist")
    if args.updates <= 0 or args.batch_size <= 0 or args.min_episodes < 2:
        parser.error("updates/batch-size must be positive and min-episodes at least 2")
    if args.report_interval <= 0 or args.snapshot_interval <= 0:
        parser.error("report/snapshot intervals must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    for name in ("learning_rate", "grad_clip"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    summary = run(parse_args(argv))
    print("RESIDUAL_TRAINING_DONE " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
