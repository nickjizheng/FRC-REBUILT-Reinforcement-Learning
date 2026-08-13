# Experiment results

Snapshot: 2026-08-06.

The complete Stage A-D analysis is in the
[technical experiment report](TECHNICAL_REPORT.md). The sanitized source data
and generated figures are reproducible from:

- [`results/stage_abcd_experiment_results.json`](../results/stage_abcd_experiment_results.json)
- [`results/stage_d_checkpoint_evaluations.json`](../results/stage_d_checkpoint_evaluations.json)
- [`tools/generate_experiment_report_assets.py`](../tools/generate_experiment_report_assets.py)

The repository also includes a
[complete 160-second score-201 match video](../README.md#demo) with a
[public checksum record](media/score201-public-provenance.json).

## Evidence by curriculum stage

| Stage | Contract | Public evidence | Main result |
|---|---|---|---|
| A | 20 s, 32 FUEL | Fixed diagnostic, 6 episodes per policy | Checkpoint collected 16.50 +/- 2.59 FUEL versus 4.83 +/- 5.23 for random. Scoring was sparse: 5 of 6 checkpoint episodes scored zero. |
| B | 36 s, 96 FUEL | Six-minute resume/training smoke test | 2,450 transitions at 6.636 transitions/s. No fixed Stage-B policy evaluation exists, so no performance or promotion claim is made. |
| C | 90 s and 120 s, 200 FUEL | Several fixed deterministic diagnostics | The 90-second champion averaged 41.50 scored FUEL (n=8); the separate 120-second v2 block averaged 68.80 (n=10). These contracts and action semantics differ and are not pooled. |
| D | 160 s, 456 FUEL | Immutable same-seed paired checkpoint blocks | Two candidates increased observed paired mean by +20.41 and +9.20; candidate 3270 regressed by -11.34 and was rejected. |

Raw scores must not be compared across stages. Match length, FUEL count, reset
distribution, action semantics, mechanics, and evaluator contracts changed as
the curriculum progressed.

## Stage-D fixed-checkpoint decisions

Each row used 64 deterministic full-match evaluations per model. Pairing is
restricted to keys that reached an exact-contract healthy horizon for both
models; healthy zero scores remain included.

| Candidate vs baseline | Common healthy pairs | Candidate mean +/- SD | Baseline mean +/- SD | Observed delta | W-L-T | Decision |
|---|---:|---:|---:|---:|---:|---|
| Stage-D v8 vs source 3185 | 44 | 121.16 +/- 48.05 | 100.75 +/- 49.83 | +20.41 | 28-16-0 | Promoted |
| Stage-D v9 vs v8 | 46 | 122.78 +/- 41.35 | 113.59 +/- 49.49 | +9.20 | 26-20-0 | Promoted |
| Stage-D v10 candidate 3270 vs v9 | 41 | 117.73 +/- 35.54 | 129.07 +/- 43.38 | -11.34 | 13-28-0 | Rejected |

The descriptive 95% Student-t intervals derived from the paired aggregates all
span zero. These are fixed-block engineering promotion decisions, not claims of
statistical significance or broad generalization.

## Interpretation limits

- Evaluator episodes are not independent training seeds.
- Stage B currently has an evidence gap.
- Historical evaluator revisions were not embedded in every artifact.
- Checkpoint screening introduces selection bias.
- Evaluation health is exact-horizon completion, not task success.
- No physical-robot or sim-to-real result is claimed.

The report retains negative results, unhealthy counts, exclusions, and custody
hashes so that promising live telemetry is not mistaken for promotion evidence.
