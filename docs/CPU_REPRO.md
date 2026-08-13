# CPU-only verification

Everything on this page runs with Python 3.11 and the repository's CPU test
dependencies—no NVIDIA Isaac Sim, CUDA, or GPU. It covers (1) the CPU
regression suite and (2) a consistency check of the published per-episode
evaluation statistics.

## Why Isaac Sim is required for training and evaluation

The full physics environment (`run_sim.py` and the Stage A-D curriculum) is
built on NVIDIA Isaac Sim 5.1 / Omniverse Kit. This project's published
training and evaluation workflow:

- uses a CUDA-capable NVIDIA GPU and was developed on Windows 11 with RTX-class
  hardware;
- has not been designed or validated for CPU-only execution;
- does not run on macOS because the required simulator stack is unavailable
  there.

A policy is exercised inside the simulator, so the checkpoints listed in
models/SELECTED_MODELS.tsv cannot be rolled out on a machine without that
simulator. The CPU artifacts below verify the non-physics parts of the
project, including that the published Stage-A numbers are internally
consistent.

## 1. CPU regression suite

The same suite runs in CI (.github/workflows/cpu-tests.yml).

    python3 -m venv .venv
    .venv/bin/pip install -e ".[test]"
    .venv/bin/python -m pytest -q

The exact pass/skip count can vary when optional packages such as PyTorch or
OpenCV, or frozen local artifacts, are present. The GitHub Actions badge and
workflow log show the current Linux reference result. A skip is expected only
when its test declares the optional dependency or artifact explicitly.

## 2. Published evaluation statistics consistency check

runs/eval_stageA_clean.json publishes per-episode rows plus summary headers.
Recompute the summaries from the rows and diff them against the headers:

    .venv/bin/python scripts/rl/verify_eval_stats.py

This checks the episode count, mean return, mean/max collected, and mean/max
scored against the JSON headers. It also recomputes the sample standard
deviations (`ddof=1`) shown in the technical report. The source JSON does not
store SD headers, so those values are displayed and can be independently
compared with the report rather than reported as header checks. Stored summary
headers are rounded to two decimals. Verified values are:

| Block | Collected (mean +/- SD) | Return (mean +/- SD) | Scored (mean) | Max collected |
|---|---:|---:|---:|---:|
| checkpoint | 16.50 +/- 2.59 | 31.65 +/- 28.57 | 1.17 | 21 |
| random | 4.83 +/- 5.23 | 5.23 +/- 7.83 | 0.00 | 12 |
| zero | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 | 0 |

## Notes for contributors

- The checkpoints themselves (.pt files) are intentionally not committed;
  see models/README.md.
- Isaac Sim workflows are Windows-first; keep new tools that validate
  published artifacts runnable with plain NumPy where possible.
