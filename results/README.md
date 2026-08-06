# Public experiment data

This directory is the sanitized publication layer for the experiment. It does
not contain checkpoints, raw simulator captures, crash dumps, private machine
paths, or mutable training state.

## Files

- [`stage_abcd_experiment_results.json`](stage_abcd_experiment_results.json)
  contains the available Stage A-D contracts, per-episode rows, descriptive
  statistics, evidence classifications, integrity hashes, and limitations.
- [`stage_d_checkpoint_evaluations.json`](stage_d_checkpoint_evaluations.json)
  is the original immutable-checkpoint Stage-D decision snapshot.
- [`stage_c_archival_evidence.json`](stage_c_archival_evidence.json) is the
  sanitized Stage-C publication source. It preserves the evaluated rows and
  origin-artifact hashes without requiring private archive paths.

Regenerate the consolidated data and all report figures from the repository
root:

```bash
python tools/generate_experiment_report_assets.py
```

The report is [`docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md).

## Interpretation

Stage A-D raw scores are not one continuous metric. Horizons, FUEL counts,
starts, action semantics, mechanics, and evaluation contracts change across
the curriculum. Compare models only inside a clearly frozen result block.

Evidence labels have fixed meanings:

- **fixed paired:** immutable candidate and baseline on identical evaluator
  keys;
- **fixed diagnostic:** frozen policy evidence with limited sample size or
  incomplete historical provenance;
- **directional:** moving-policy telemetry used for checkpoint screening;
- **smoke:** infrastructure validation without a competence claim.
