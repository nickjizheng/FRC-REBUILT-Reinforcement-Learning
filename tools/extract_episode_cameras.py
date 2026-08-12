#!/usr/bin/env python3
"""Extract the three genuine RGB policy-camera frames from an elite episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    with np.load(args.episode, allow_pickle=False) as archive:
        obs = archive["obs"]
        if obs.ndim != 4 or obs.shape[1:] != (9, 90, 160):
            raise RuntimeError(f"unexpected observation tensor {obs.shape}")
        if not 0 <= args.step < obs.shape[0]:
            raise ValueError("step is outside the episode")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        labels = ("intake", "shooter", "navigation")
        for index, label in enumerate(labels):
            rgb = np.transpose(obs[args.step, index * 3 : (index + 1) * 3], (1, 2, 0))
            path = args.output_dir / f"camera_{label}_step{args.step:04d}.png"
            if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1])):
                raise RuntimeError(f"could not write {path}")
            print(path)


if __name__ == "__main__":
    main()
