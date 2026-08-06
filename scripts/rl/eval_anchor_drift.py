"""Read-only anchor-drift evaluator -- the repeatable
calculation behind the prefix-takeover learner drift kill-switch.

Deterministic, UNAUGMENTED per-action drift of candidate checkpoints vs the frozen
champion on a frozen, episode-level, phase-stratified anchor-validation holdout. Writes a
machine-readable JSON artifact: checkpoint paths + SHA-256, anchor-manifest hash, held-out
episode indices, state count + phase counts, action thresholds, per-action errors,
decision disagreements, and the warning/hard-stop boundaries. CPU torch; NO Isaac.

Boundaries are failure-correlated (design note): calibrated conservatively below the
mildest MEASURED full-start regression (1019053: drive-L2 p50 0.75 / shoot 16% / storage
24%). Layer 1 (identity: champion vs champion) must be ~0 or the pipeline is faulted.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BOUNDS = {
    "warning": {"drive_l2_p50": 0.35, "shoot_disagree": 0.05, "storage_disagree": 0.08},
    "hard_stop": {"drive_l2_p50": 0.50, "shoot_disagree": 0.08, "storage_disagree": 0.12},
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ep(p: Path) -> int:
    return int(re.search(r"ep(\d+)", p.name).group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default=str(PROJECT_ROOT / "runs/stageC_champion_998753.pt"))
    ap.add_argument("--candidates",
                    default="runs/stageC_recent_latest.pt,runs/stageC_deterministic_retest_20260712_1730.pt",
                    help="comma-list of candidate .pt to compare vs champion")
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--holdout", default="",
                    help="comma-list of held-out EPISODE indices (frozen manifest); default = last 20%%")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch  # noqa: F401
    from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
    from frc_rebuilt.rl.spec import CompetitionRLSpec, decode_policy_actions

    def load(p):
        a = DrQV2Agent(DrQConfig(device="cpu"))
        a.load(str(p))
        return a

    adir = Path(args.anchor_dir)
    chunks = sorted(adir.glob("anchor_*.npz"), key=_ep)
    if not chunks:
        raise SystemExit(f"no anchor_*.npz in {adir}")
    if args.holdout.strip():
        hold_idx = sorted(int(x) for x in args.holdout.split(","))
    else:
        hold_idx = sorted(_ep(p) for p in chunks[-max(1, len(chunks) // 5):])
    hold = [p for p in chunks if _ep(p) in hold_idx]

    fr, pr, phs = [], [], []
    for p in hold:
        d = np.load(p, allow_pickle=True)
        fr.append(d["frames"]); pr.append(d["proprio"]); phs.append(d["phase"])
    fr = np.concatenate(fr); pr = np.concatenate(pr)
    phase_counts = dict(Counter(np.concatenate(phs).tolist()))

    manifest_p = adir / "manifest.json"
    manifest_hash = _sha256(manifest_p) if manifest_p.exists() else None

    champ = load(args.champion)
    champ_act = champ.act(fr, pr, explore=False)
    policy_spec = CompetitionRLSpec()
    champ_decoded = decode_policy_actions(champ_act, policy_spec)

    results = {}
    cands = {"champion_identity": args.champion}
    cands.update({Path(c).stem: c for c in args.candidates.split(",") if c.strip()})
    for name, p in cands.items():
        act = load(p).act(fr, pr, explore=False)
        decoded = decode_policy_actions(act, policy_spec)
        dl2 = np.linalg.norm(act[:, :3] - champ_act[:, :3], axis=1)
        r = {
            "checkpoint": str(p), "sha256": _sha256(p),
            "drive_l2_p50": round(float(np.percentile(dl2, 50)), 4),
            "drive_l2_p95": round(float(np.percentile(dl2, 95)), 4),
            "drive_l2_max": round(float(dl2.max()), 4),
            # Compare the REAL decoded mechanism intents.  All four discrete heads
            # use CompetitionRLSpec.action_threshold (currently 0.25), and ferry is
            # suppressed when shoot is active.  Duplicating raw thresholds here once
            # made storage use 0.0 and invalidated its calibration (design note).
            "intake_disagree": round(float((decoded.intake_on != champ_decoded.intake_on).mean()), 4),
            "storage_disagree": round(float((decoded.storage_extended != champ_decoded.storage_extended).mean()), 4),
            "shoot_disagree": round(float((decoded.shoot_blue != champ_decoded.shoot_blue).mean()), 4),
            "ferry_disagree": round(float((decoded.ferry != champ_decoded.ferry).mean()), 4),
        }
        b = BOUNDS["hard_stop"]
        r["hard_stop_triggered"] = bool(
            r["drive_l2_p50"] > b["drive_l2_p50"] or r["shoot_disagree"] > b["shoot_disagree"]
            or r["storage_disagree"] > b["storage_disagree"])
        results[name] = r

    identity_ok = results["champion_identity"]["drive_l2_max"] < 1e-4 \
        and results["champion_identity"]["intake_disagree"] == 0.0 \
        and results["champion_identity"]["shoot_disagree"] == 0.0 \
        and results["champion_identity"]["storage_disagree"] == 0.0 \
        and results["champion_identity"]["ferry_disagree"] == 0.0
    out = {
        "champion": {"checkpoint": args.champion, "sha256": _sha256(args.champion)},
        "anchor_dir": str(adir), "anchor_manifest_sha256": manifest_hash,
        "holdout_episode_indices": hold_idx, "n_holdout_episodes": len(hold),
        "n_states": int(len(fr)), "holdout_phase_counts": phase_counts,
        "action_contract": {
            "version": policy_spec.version,
            "discrete_threshold": policy_spec.action_threshold,
            "shoot_precedes_ferry": True,
        },
        "boundaries": BOUNDS,
        "identity_pipeline_ok": identity_ok,
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"ANCHOR_DRIFT_DONE {args.out}  identity_ok={identity_ok}  "
          f"holdout_eps={hold_idx}  states={len(fr)}", flush=True)
    for name, r in results.items():
        print(f"  {name:34s} driveL2 p50={r['drive_l2_p50']:.3f} p95={r['drive_l2_p95']:.2f} "
              f"intake={r['intake_disagree']*100:.1f}% stor={r['storage_disagree']*100:.1f}% "
              f"shoot={r['shoot_disagree']*100:.1f}% ferry={r['ferry_disagree']*100:.1f}% "
              f"{'HARD-STOP' if r['hard_stop_triggered'] else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
