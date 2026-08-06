#!/usr/bin/env python3
"""Build the public Stage A-D experiment dataset and SVG figures.

The generator intentionally uses only Python's standard library.  It reads the
archival result files that are already present in the repository workspace,
removes machine-specific paths, derives descriptive statistics, and writes a
single reviewable JSON snapshot plus publication-ready SVG figures.

Run from the repository root:

    python tools/generate_experiment_report_assets.py
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "stage_abcd_experiment_results.json"
FIGURE_DIR = ROOT / "docs" / "figures"


def load_json(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl(relative: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / relative).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: Sequence[float | int]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    result: dict[str, float | int] = {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }
    result["sample_stddev"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
    return result


def rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def t95(df: int) -> float:
    # Two-sided 95% Student-t critical values for the sample sizes used here.
    known = {40: 2.021075, 43: 2.016692, 45: 2.014103}
    if df in known:
        return known[df]
    return 1.96 if df >= 120 else 2.0


def paired_ci(mean: float, sample_stddev: float, n: int) -> list[float]:
    half_width = t95(n - 1) * sample_stddev / math.sqrt(n)
    return [mean - half_width, mean + half_width]


def source(relative: str, note: str | None = None) -> dict[str, str]:
    item = {"path": relative.replace("\\", "/"), "sha256": sha256(relative)}
    if note:
        item["note"] = note
    return item


def build_dataset() -> dict[str, Any]:
    a_summary = load_json("runs/drqv2_stageA_clean/summary.json")
    a_eval = load_json("runs/eval_stageA_clean.json")
    a_metrics = load_jsonl("runs/drqv2_stageA_clean/metrics.jsonl")
    b_summary = load_json("runs/drqv2_stageB_smoke/summary.json")
    b_config = load_json("runs/drqv2_stageB_smoke/run_config.json")
    c_public = load_json("results/stage_c_archival_evidence.json")
    c_legacy_source = c_public["legacy_90_second"]
    c_v2_source = c_public["v2_120_second_single_block"]
    c_progression_source = c_public["v2_120_second_matched_progression"]
    c_v4_source = c_public["v4_multiseed_90_second_bank"]
    c_gate_source = c_public["matched_two_row_gate"]
    d_public = load_json("results/stage_d_checkpoint_evaluations.json")

    c_matched_progression = []
    for item in c_progression_source["models"]:
        c_matched_progression.append(
            {
                **item,
                "scored_fuel": describe(item["scores"]),
                "collected_fuel": describe(item["collected"]),
            }
        )

    c_v4_bank_rows = c_v4_source["rows"]
    c_v4_bank_healthy = [
        row for row in c_v4_bank_rows if row["terminal_reason"] == "horizon"
    ]

    a_policies: dict[str, Any] = {}
    for policy_name in ("checkpoint", "random", "zero"):
        rows = a_eval[policy_name]["per_episode"]
        a_policies[policy_name] = {
            "episodes": rows,
            "return": describe([row["return"] for row in rows]),
            "collected_fuel": describe([row["collected"] for row in rows]),
            "scored_fuel": describe([row["scored"] for row in rows]),
            "shots_fired": describe([row["shots_fired"] for row in rows]),
        }

    b_returns = list(b_summary["first5_returns"]) + [b_summary["last5_returns"][-1]]

    c_legacy: dict[str, Any] = {}
    for record in c_legacy_source["models"]:
        label = record["model_id"].replace("-", "_")
        rows = record["episodes"]
        c_legacy[label] = {
            "checkpoint_sha256": record["checkpoint_sha256"],
            "train_steps": record["train_steps"],
            "episodes": rows,
            "scored_fuel": describe([row["scored"] for row in rows]),
            "collected_fuel": describe([row["collected"] for row in rows]),
            "shots_fired": describe([row["shots_fired"] for row in rows]),
            "aggregate_score_per_shot": record["aggregate_score_per_shot"],
        }

    c_v2_public_rows = c_v2_source["episodes"]

    d_comparisons = []
    for comparison in d_public["comparisons"]:
        copied = json.loads(json.dumps(comparison))
        paired = copied["paired_scores"]
        n = copied["samples"]["common_healthy_pairs"]
        paired["candidate_minus_baseline_95pct_t_interval"] = paired_ci(
            paired["candidate_minus_baseline_mean"],
            paired["paired_delta_sample_stddev"],
            n,
        )
        paired["interval_note"] = (
            "Descriptive two-sided Student-t interval from the published paired "
            "delta aggregate; not adjusted for checkpoint screening."
        )
        d_comparisons.append(copied)

    throughput = []
    for env_count in (1, 2, 4, 8):
        relative = f"runs/vec_throughput_n{env_count}.json"
        row = load_json(relative)
        throughput.append(
            {
                "num_envs": row["num_envs"],
                "fuel_bodies": row["fuel_count"],
                "aggregate_policy_transitions_per_second": row[
                    "aggregate_policy_tx_per_s"
                ],
                "aggregate_environment_steps_per_second": row[
                    "aggregate_env_steps_per_s"
                ],
                "peak_vram_mb": row["vram_mb_peak"],
                "throughput_gate_8_tx_per_s": row["gate_8_tx_per_s_cleared"],
                "source": source(relative),
            }
        )

    return rounded(
        {
            "schema": "frc_rebuilt_stage_abcd_experiment_results_v1",
            "snapshot_date": "2026-08-06",
            "scope": "Sanitized public evidence available for Stages A-D",
            "interpretation_rule": (
                "Raw scores must not be compared across stages because horizons, FUEL "
                "counts, reset distributions, action semantics, and match contracts differ."
            ),
            "evidence_tiers": {
                "fixed_paired": "Immutable checkpoints on identical seed keys within one contract.",
                "fixed_diagnostic": "Frozen checkpoint evaluation with limited sample size or provenance.",
                "directional": "Moving-policy training telemetry used only for screening.",
                "smoke": "Infrastructure/resume validation; not a policy-performance conclusion.",
            },
            "curriculum": [
                {
                    "stage": "A",
                    "horizon_seconds": 20,
                    "fuel_template_count": 32,
                    "objective": "Acquire FUEL from camera observations",
                    "primary_public_evidence": "fixed_diagnostic",
                },
                {
                    "stage": "B",
                    "horizon_seconds": 36,
                    "fuel_template_count": 96,
                    "objective": "Acquire and score from mixed starts",
                    "primary_public_evidence": "smoke",
                },
                {
                    "stage": "C",
                    "horizon_seconds": [90, 120],
                    "fuel_template_count": 200,
                    "objective": "Repeat collection, return, and scoring cycles",
                    "primary_public_evidence": "fixed_diagnostic",
                },
                {
                    "stage": "D",
                    "horizon_seconds": 160,
                    "fuel_template_count": 456,
                    "objective": "Exact full-match behavior",
                    "primary_public_evidence": "fixed_paired",
                },
            ],
            "stages": {
                "A": {
                    "evidence_tier": "fixed_diagnostic",
                    "contract": {
                        "horizon_seconds": 20,
                        "fuel_template_count": 32,
                        "evaluation_episodes_per_policy": 6,
                        "environment_count": 2,
                        "intended_measure": "FUEL acquisition",
                    },
                    "training_telemetry": {
                        **a_summary,
                        "series": [
                            {
                                "transitions": row["transitions"],
                                "episodes": row["episodes"],
                                "recent_return_mean": row["recent_return_mean"],
                                "recent_collect_reward": row["recent_collect_reward"],
                                "recent_score_reward": row["recent_score_reward"],
                            }
                            for row in a_metrics
                        ],
                        "evidence_tier": "directional",
                    },
                    "fixed_diagnostic": {
                        "checkpoint_sha256": (
                            "94627ce74e30f96a378bb25288ffb69bb7122b687100d49c73a5c3ffd0b83c65"
                        ),
                        "checkpoint_identity_note": (
                            "Recovered from the contemporaneous repository commit; the "
                            "evaluation JSON did not embed the checkpoint hash."
                        ),
                        "policies": a_policies,
                    },
                    "sources": [
                        source("runs/drqv2_stageA_clean/summary.json"),
                        source("runs/drqv2_stageA_clean/metrics.jsonl"),
                        source(
                            "runs/eval_stageA_clean.json",
                            "Historical file omits evaluator revision and embedded checkpoint hash.",
                        ),
                    ],
                    "caveats": [
                        "The checkpoint score mean is driven by one seven-score episode; five of six scored zero.",
                        "Acquisition, not scoring, was the intended Stage-A outcome.",
                        "This is one small evaluator-seed block, not independent training-seed replication.",
                    ],
                },
                "B": {
                    "evidence_tier": "smoke",
                    "contract": {
                        "horizon_seconds": b_config["episode_len_s"],
                        "fuel_template_count": 96,
                        "environment_count": b_config["num_envs"],
                        "preloaded_start_probability": b_config["preload_prob"],
                        "collection_weight_schedule": [
                            b_config["collect_weight_start"],
                            b_config["collect_weight_end"],
                        ],
                    },
                    "smoke_run": {
                        "elapsed_seconds": b_summary["elapsed_s"],
                        "transitions": b_summary["transitions"],
                        "updates": b_summary["updates"],
                        "episodes": b_summary["episodes"],
                        "transitions_per_second": b_summary["transitions_per_s"],
                        "episode_returns": b_returns,
                        "return": describe(b_returns),
                        "recent_score_reward": b_summary["mean_score_reward_last20"],
                        "recent_collection_reward": b_summary[
                            "mean_collect_reward_last20"
                        ],
                    },
                    "sources": [
                        source("runs/drqv2_stageB_smoke/summary.json"),
                        source(
                            "runs/drqv2_stageB_smoke/run_config.json",
                            "The public dataset copies only sanitized contract fields.",
                        ),
                    ],
                    "caveats": [
                        "No fixed Stage-B checkpoint evaluation is available.",
                        "The six-minute run validates resume/training throughput, not acquire-and-score competence.",
                        "Collection reward is shaped reward and is not a raw FUEL count.",
                    ],
                },
                "C": {
                    "evidence_tier": "fixed_diagnostic",
                    "legacy_90_second": {
                        "contract": c_legacy_source["contract"],
                        "models": c_legacy,
                        "decision": "Retain champion 998753; do not promote retest 1019053.",
                    },
                    "v2_120_second_single_block": {
                        "contract": c_v2_source["contract"],
                        "checkpoint_sha256": c_v2_source["checkpoint_sha256"],
                        "checkpoint_train_steps": c_v2_source["checkpoint_train_steps"],
                        "checkpoint_v2_updates": c_v2_source["checkpoint_v2_updates"],
                        "episodes": c_v2_public_rows,
                        "scored_fuel": describe(
                            [row["scored"] for row in c_v2_public_rows]
                        ),
                        "collected_fuel": describe(
                            [row["collected"] for row in c_v2_public_rows]
                        ),
                        "shots_fired": describe(
                            [row["shots_fired"] for row in c_v2_public_rows]
                        ),
                        "cycles_completed_total": sum(
                            row["cycles_completed"] for row in c_v2_public_rows
                        ),
                        "successes": sum(
                            1 for row in c_v2_public_rows if row["success"]
                        ),
                        "success_rate": sum(
                            1 for row in c_v2_public_rows if row["success"]
                        )
                        / len(c_v2_public_rows),
                        "row_bundle_sha256": c_v2_source["row_bundle_sha256"],
                        "summary_artifact_sha256": c_v2_source[
                            "summary_artifact_sha256"
                        ],
                    },
                    "v2_120_second_matched_progression": {
                        "contract": c_progression_source["contract"],
                        "models": c_matched_progression,
                        "custody": {
                            "archive_summary_sha256": c_progression_source[
                                "archive_summary_sha256"
                            ],
                            "checkpoint_manifest_sha256": c_progression_source[
                                "checkpoint_manifest_sha256"
                            ],
                            "prefix_checkpoint_sha256": c_progression_source[
                                "prefix_checkpoint_sha256"
                            ],
                        },
                        "caveat": c_progression_source["exclusion_note"],
                    },
                    "v4_multiseed_90_second_bank": {
                        "contract": c_v4_source["contract"],
                        "checkpoint_sha256": c_v4_source["checkpoint_sha256"],
                        "rows": c_v4_bank_rows,
                        "all_rows": {
                            "scored_fuel": describe([row["scored"] for row in c_v4_bank_rows]),
                            "collected_fuel": describe([row["collected"] for row in c_v4_bank_rows]),
                            "cycles_completed_total": sum(row["cycles_completed"] for row in c_v4_bank_rows),
                        },
                        "healthy_rows": {
                            "n": len(c_v4_bank_healthy),
                            "scored_fuel": describe([row["scored"] for row in c_v4_bank_healthy]),
                            "collected_fuel": describe([row["collected"] for row in c_v4_bank_healthy]),
                            "cycles_completed": describe([row["cycles_completed"] for row in c_v4_bank_healthy]),
                        },
                        "capture_manifest_sha256": c_v4_source[
                            "all_capture_manifest_sha256"
                        ],
                        "qualified_capture_manifest_sha256": c_v4_source[
                            "qualified_capture_manifest_sha256"
                        ],
                        "caveat": "Two of eight rows terminated unhealthy; all-row and healthy-only summaries are both retained.",
                    },
                    "matched_two_row_gate": {
                        "contract": c_gate_source["contract"],
                        "models": c_gate_source["models"],
                        "decision": c_gate_source["decision"],
                        "evidence_note": c_gate_source["evidence_note"],
                        "source_artifact_sha256": c_gate_source[
                            "source_artifact_sha256"
                        ],
                    },
                    "sources": [
                        source(
                            "results/stage_c_archival_evidence.json",
                            "Sanitized publication layer; origin artifact hashes are embedded.",
                        ),
                    ],
                    "caveats": [
                        "The 90-second and 120-second blocks use different action semantics and must not be pooled.",
                        "The repeated seed-424242 rows are evaluator episodes, not independent training seeds.",
                        "The n=2 checkpoint gate is an engineering diagnostic, not a statistical comparison.",
                    ],
                },
                "D": {
                    "evidence_tier": "fixed_paired",
                    "shared_protocol": d_public["shared_protocol"],
                    "comparisons": d_comparisons,
                    "directional_training_telemetry": d_public[
                        "directional_training_telemetry"
                    ],
                    "sources": [source("results/stage_d_checkpoint_evaluations.json")],
                    "caveats": d_public["caveats"]
                    + [
                        "All derived 95% intervals span zero and do not account for checkpoint-selection multiplicity."
                    ],
                },
            },
            "engineering": {
                "vectorized_throughput": throughput,
                "interpretation": (
                    "The two-environment row accidentally instantiated 912 FUEL bodies and is a "
                    "documented overload case, not evidence that two environments are intrinsically slower."
                ),
            },
            "report_wide_limitations": [
                "No 3-5 independent training-seed replication.",
                "No consistent random or rule-based baseline after Stage A.",
                "Historical evaluator revisions were not embedded in every artifact.",
                "Checkpoint screening introduces selection bias.",
                "Healthy-pair filtering can be missing-not-at-random; all-healthy counts are reported separately.",
                "No physical-robot or sim-to-real evaluation is claimed.",
            ],
        }
    )


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_document(title: str, description: str, width: int, height: int, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(description)}</desc>
  <style>
    :root {{ --bg:#ffffff; --fg:#172033; --muted:#5f6b7a; --grid:#d8dee8; --blue:#2563eb; --orange:#ea580c; --green:#15803d; --red:#b91c1c; --purple:#7c3aed; --soft:#f3f6fa; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f172a; --fg:#f1f5f9; --muted:#b7c0cf; --grid:#3a475c; --blue:#60a5fa; --orange:#fb923c; --green:#4ade80; --red:#f87171; --purple:#c4b5fd; --soft:#172033; }} }}
    .bg{{fill:var(--bg)}} .fg{{fill:var(--fg)}} .muted{{fill:var(--muted)}} .soft{{fill:var(--soft)}}
    .grid{{stroke:var(--grid);stroke-width:1}} .axis{{stroke:var(--fg);stroke-width:1.25}}
    .blue{{fill:var(--blue);stroke:var(--blue)}} .orange{{fill:var(--orange);stroke:var(--orange)}}
    .green{{fill:var(--green);stroke:var(--green)}} .red{{fill:var(--red);stroke:var(--red)}}
    .purple{{fill:var(--purple);stroke:var(--purple)}}
    text{{font-family:Inter,Segoe UI,Arial,sans-serif;fill:var(--fg)}}
    .title{{font-size:26px;font-weight:600}} .subtitle{{font-size:15px;fill:var(--muted)}}
    .label{{font-size:14px}} .small{{font-size:12px;fill:var(--muted)}} .value{{font-size:14px;font-weight:600}}
  </style>
  <rect class="bg" x="0" y="0" width="{width}" height="{height}"/>
  {body}
</svg>
'''


def write_svg(name: str, title: str, description: str, width: int, height: int, body: str) -> None:
    (FIGURE_DIR / name).write_text(
        svg_document(title, description, width, height, body),
        encoding="utf-8",
        newline="\n",
    )


def linear(value: float, domain: tuple[float, float], output: tuple[float, float]) -> float:
    low, high = domain
    out_low, out_high = output
    if high == low:
        return (out_low + out_high) / 2
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def chart_frame(x: float, y: float, width: float, height: float) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" class="grid"/>'


def curriculum_figure(dataset: dict[str, Any]) -> None:
    items = dataset["curriculum"]
    colors = ["blue", "orange", "green", "purple"]
    parts = [
        '<text class="title" x="48" y="48">Curriculum contracts: evidence grows while the task changes</text>',
        '<text class="subtitle" x="48" y="75">Scores are comparable within a contract, not across stages.</text>',
    ]
    for index, (item, color) in enumerate(zip(items, colors)):
        x = 48 + index * 286
        parts.append(f'<rect class="soft" x="{x}" y="112" width="246" height="190" rx="10"/>')
        parts.append(f'<circle class="{color}" cx="{x + 34}" cy="143" r="18"/>')
        parts.append(f'<text x="{x + 34}" y="149" text-anchor="middle" style="fill:var(--bg);font-size:16px;font-weight:600">{item["stage"]}</text>')
        horizon = item["horizon_seconds"]
        horizon_text = f'{horizon[0]} / {horizon[1]} s' if isinstance(horizon, list) else f'{horizon} s'
        parts.append(f'<text class="value" x="{x + 62}" y="140">{horizon_text} · {item["fuel_template_count"]} FUEL</text>')
        objective_words = item["objective"].split()
        line_a = " ".join(objective_words[:5])
        line_b = " ".join(objective_words[5:])
        parts.append(f'<text class="label" x="{x + 22}" y="190">{esc(line_a)}</text>')
        if line_b:
            parts.append(f'<text class="label" x="{x + 22}" y="212">{esc(line_b)}</text>')
        evidence_label = item["primary_public_evidence"].replace("_", " ")
        parts.append(f'<text class="small" x="{x + 22}" y="268">Evidence: {esc(evidence_label)}</text>')
        if index < 3:
            parts.append(f'<path d="M {x + 248} 207 H {x + 279}" class="axis" fill="none"/>')
            parts.append(f'<path d="M {x + 273} 201 L {x + 281} 207 L {x + 273} 213" class="axis" fill="none"/>')
    write_svg(
        "curriculum-contracts.svg",
        "Stage A-D curriculum contracts",
        "Four curriculum stages with increasing horizon and FUEL count, each with a different evidence tier.",
        1200,
        350,
        "\n".join(parts),
    )


def stage_a_learning_figure(dataset: dict[str, Any]) -> None:
    rows = dataset["stages"]["A"]["training_telemetry"]["series"]
    x0, y0, width, height = 92, 112, 1028, 340
    max_x = max(row["transitions"] for row in rows)
    all_y = [row["recent_return_mean"] for row in rows] + [
        row["recent_collect_reward"] for row in rows
    ]
    min_y, max_y = min(all_y), max(all_y)
    min_y = math.floor(min_y / 5) * 5
    max_y = math.ceil(max_y / 5) * 5
    parts = [
        '<text class="title" x="48" y="48">Stage A moving-policy training telemetry</text>',
        '<text class="subtitle" x="48" y="75">Recent means are directional; the fixed diagnostic is reported separately.</text>',
        chart_frame(x0, y0, width, height),
    ]
    for tick in range(0, 6):
        value = min_y + (max_y - min_y) * tick / 5
        y = linear(value, (min_y, max_y), (y0 + height, y0))
        parts.append(f'<path class="grid" d="M {x0} {y:.1f} H {x0 + width}"/>')
        parts.append(f'<text class="small" x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    for tick in range(0, 6):
        value = max_x * tick / 5
        x = linear(value, (0, max_x), (x0, x0 + width))
        parts.append(f'<text class="small" x="{x:.1f}" y="{y0 + height + 26}" text-anchor="middle">{value / 1000:.0f}k</text>')
    for key, css, label in (
        ("recent_return_mean", "blue", "Recent return mean"),
        ("recent_collect_reward", "orange", "Recent collection reward"),
    ):
        points = " ".join(
            f'{linear(row["transitions"], (0, max_x), (x0, x0 + width)):.1f},{linear(row[key], (min_y, max_y), (y0 + height, y0)):.1f}'
            for row in rows
        )
        parts.append(
            f'<polyline points="{points}" class="{css}" '
            'style="fill:none" stroke-width="2.5"/>'
        )
        legend_x = 760 if key == "recent_return_mean" else 940
        parts.append(f'<path d="M {legend_x} 74 h 24" class="{css}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{legend_x + 30}" y="79">{label}</text>')
    parts.append(f'<text class="label" x="{x0 + width / 2}" y="505" text-anchor="middle">Environment transitions</text>')
    parts.append(f'<text class="label" x="24" y="{y0 + height / 2}" text-anchor="middle" transform="rotate(-90 24 {y0 + height / 2})">Recent reward / return</text>')
    write_svg(
        "stage-a-learning.svg",
        "Stage A learning telemetry",
        "Line chart of recent return and collection reward across approximately 106 thousand transitions.",
        1200,
        540,
        "\n".join(parts),
    )


def strip_panel(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    groups: list[tuple[str, Sequence[float], str]],
    y_label: str,
) -> None:
    all_values = [float(value) for _, values, _ in groups for value in values]
    low = min(0.0, min(all_values))
    high = max(all_values) * 1.12 if max(all_values) else 1.0
    parts.append(f'<text class="value" x="{x}" y="{y - 22}">{esc(title)}</text>')
    parts.append(chart_frame(x, y, width, height))
    for tick in range(0, 5):
        value = low + (high - low) * tick / 4
        yy = linear(value, (low, high), (y + height, y))
        parts.append(f'<path class="grid" d="M {x} {yy:.1f} H {x + width}"/>')
        parts.append(f'<text class="small" x="{x - 10}" y="{yy + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    jitter = (-14, -8, -3, 3, 8, 14, -11, 11, 0, 6, -6, 16)
    for group_index, (label, values, css) in enumerate(groups):
        center = x + width * (group_index + 0.5) / len(groups)
        for index, value in enumerate(values):
            cx = center + jitter[index % len(jitter)]
            cy = linear(float(value), (low, high), (y + height, y))
            parts.append(f'<circle class="{css}" cx="{cx:.1f}" cy="{cy:.1f}" r="5" opacity="0.8"/>')
        mean_value = statistics.fmean(float(value) for value in values)
        mean_y = linear(mean_value, (low, high), (y + height, y))
        parts.append(f'<path d="M {center - 24:.1f} {mean_y:.1f} H {center + 24:.1f}" class="{css}" stroke-width="4"/>')
        parts.append(f'<text class="small" x="{center:.1f}" y="{y + height + 24}" text-anchor="middle">{esc(label)}</text>')
        parts.append(f'<text class="value" x="{center:.1f}" y="{mean_y - 10:.1f}" text-anchor="middle">μ {mean_value:.1f}</text>')
    parts.append(f'<text class="label" x="{x - 46}" y="{y + height / 2}" text-anchor="middle" transform="rotate(-90 {x - 46} {y + height / 2})">{esc(y_label)}</text>')


def stage_a_diagnostic_figure(dataset: dict[str, Any]) -> None:
    policies = dataset["stages"]["A"]["fixed_diagnostic"]["policies"]
    groups_collected = [
        (
            label,
            [row["collected"] for row in policies[key]["episodes"]],
            css,
        )
        for label, key, css in (
            ("Checkpoint", "checkpoint", "blue"),
            ("Random", "random", "orange"),
            ("Zero", "zero", "red"),
        )
    ]
    groups_scored = [
        (
            label,
            [row["scored"] for row in policies[key]["episodes"]],
            css,
        )
        for label, key, css in (
            ("Checkpoint", "checkpoint", "blue"),
            ("Random", "random", "orange"),
            ("Zero", "zero", "red"),
        )
    ]
    parts = [
        '<text class="title" x="48" y="48">Stage A diagnostic: acquisition improved; scoring stayed sparse</text>',
        '<text class="subtitle" x="48" y="75">Six deterministic 20-second episodes per policy. Horizontal marks show means.</text>',
    ]
    strip_panel(parts, 96, 130, 450, 300, "FUEL collected", groups_collected, "Collected FUEL")
    strip_panel(parts, 700, 130, 410, 300, "FUEL scored", groups_scored, "Scored FUEL")
    write_svg(
        "stage-a-diagnostic.svg",
        "Stage A fixed diagnostic",
        "Strip plots compare checkpoint, random and zero policies for collected and scored FUEL.",
        1200,
        520,
        "\n".join(parts),
    )


def stage_c_figure(dataset: dict[str, Any]) -> None:
    stage = dataset["stages"]["C"]
    legacy = stage["legacy_90_second"]["models"]
    v2 = stage["v2_120_second_single_block"]
    parts = [
        '<text class="title" x="48" y="48">Stage C deterministic scoring under two separate contracts</text>',
        '<text class="subtitle" x="48" y="75">Panels are deliberately separate: 90-second one-ball presses and 120-second dump-on-press cannot be pooled.</text>',
    ]
    legacy_groups = [
        (
            "Champion 998753",
            [row["scored"] for row in legacy["champion_998753"]["episodes"]],
            "green",
        ),
        (
            "Retest 1019053",
            [row["scored"] for row in legacy["retest_1019053"]["episodes"]],
            "red",
        ),
    ]
    v2_groups = [
        (
            "v2 checkpoint 465424",
            [row["scored"] for row in v2["episodes"]],
            "purple",
        )
    ]
    strip_panel(parts, 96, 130, 480, 310, "Legacy 90-second episodes", legacy_groups, "Scored FUEL")
    strip_panel(parts, 708, 130, 400, 310, "v2 120-second episodes", v2_groups, "Scored FUEL")
    write_svg(
        "stage-c-deterministic.svg",
        "Stage C deterministic evaluations",
        "Separate strip plots show legacy 90-second champion and retest scores, and 120-second v2 scores.",
        1200,
        540,
        "\n".join(parts),
    )


def stage_c_progression_figure(dataset: dict[str, Any]) -> None:
    models = dataset["stages"]["C"]["v2_120_second_matched_progression"]["models"]
    colors = ["blue", "orange", "green", "purple"]
    groups = [
        (model["model_id"], model["scores"], color)
        for model, color in zip(models, colors)
    ]
    parts = [
        '<text class="title" x="48" y="48">Stage C 120-second matched checkpoint progression</text>',
        '<text class="subtitle" x="48" y="75">Twelve identical evaluator keys per checkpoint across two seed blocks; points are episode scores.</text>',
    ]
    strip_panel(parts, 104, 130, 1000, 320, "Fixed deterministic block", groups, "Scored FUEL")
    write_svg(
        "stage-c-matched-progression.svg",
        "Stage C matched checkpoint progression",
        "Strip plot of twelve deterministic scores for four Stage C checkpoints on the same evaluator keys.",
        1200,
        550,
        "\n".join(parts),
    )


def stage_d_delta_figure(dataset: dict[str, Any]) -> None:
    comparisons = dataset["stages"]["D"]["comparisons"]
    x0, x1 = 420, 1080
    domain = (-40.0, 50.0)
    parts = [
        '<text class="title" x="48" y="48">Stage D paired checkpoint deltas</text>',
        '<text class="subtitle" x="48" y="75">Candidate minus baseline mean score; whiskers are descriptive 95% t intervals. All span zero.</text>',
    ]
    zero_x = linear(0, domain, (x0, x1))
    parts.append(f'<path d="M {zero_x:.1f} 108 V 400" class="axis"/>')
    for tick in (-40, -20, 0, 20, 40):
        x = linear(tick, domain, (x0, x1))
        parts.append(f'<path d="M {x:.1f} 400 V 408" class="axis"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="430" text-anchor="middle">{tick:+d}</text>')
    labels = ["v8 vs source", "v9 vs v8", "v10 / 3270 vs v9"]
    for index, (label, comparison) in enumerate(zip(labels, comparisons)):
        y = 150 + index * 95
        paired = comparison["paired_scores"]
        mean = paired["candidate_minus_baseline_mean"]
        low, high = paired["candidate_minus_baseline_95pct_t_interval"]
        css = "green" if comparison["decision"] == "promote" else "red"
        low_x = linear(low, domain, (x0, x1))
        high_x = linear(high, domain, (x0, x1))
        mean_x = linear(mean, domain, (x0, x1))
        parts.append(f'<text class="value" x="48" y="{y + 5}">{label}</text>')
        parts.append(f'<text class="small" x="48" y="{y + 27}">n={comparison["samples"]["common_healthy_pairs"]} · {comparison["paired_scores"]["wins"]}W/{comparison["paired_scores"]["losses"]}L · {comparison["decision"]}</text>')
        parts.append(f'<path d="M {low_x:.1f} {y} H {high_x:.1f}" class="{css}" stroke-width="3"/>')
        parts.append(f'<path d="M {low_x:.1f} {y - 8} V {y + 8} M {high_x:.1f} {y - 8} V {y + 8}" class="{css}" stroke-width="2"/>')
        parts.append(f'<circle class="{css}" cx="{mean_x:.1f}" cy="{y}" r="8"/>')
        parts.append(f'<text class="value" x="{mean_x + (12 if mean >= 0 else -12):.1f}" y="{y - 14}" text-anchor="{("start" if mean >= 0 else "end")}">{mean:+.2f}</text>')
    parts.append(f'<text class="label" x="{(x0 + x1) / 2}" y="470" text-anchor="middle">Paired score delta (candidate − baseline)</text>')
    write_svg(
        "stage-d-paired-deltas.svg",
        "Stage D paired checkpoint deltas",
        "Forest plot of three paired checkpoint mean score deltas with descriptive 95 percent t intervals.",
        1200,
        500,
        "\n".join(parts),
    )


def stage_d_health_figure(dataset: dict[str, Any]) -> None:
    comparisons = dataset["stages"]["D"]["comparisons"]
    labels = ["v8 / source", "v9 / v8", "v10-3270 / v9"]
    x0, y0, width, height = 100, 120, 1030, 310
    parts = [
        '<text class="title" x="48" y="48">Stage D evaluation health populations</text>',
        '<text class="subtitle" x="48" y="75">Healthy means exact-contract horizon completion, not task success. Every model ran 64 rows.</text>',
        chart_frame(x0, y0, width, height),
    ]
    for tick in range(0, 5):
        value = tick * 25
        y = linear(value, (0, 100), (y0 + height, y0))
        parts.append(f'<path class="grid" d="M {x0} {y:.1f} H {x0 + width}"/>')
        parts.append(f'<text class="small" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{value}%</text>')
    group_width = width / 3
    for index, (label, comparison) in enumerate(zip(labels, comparisons)):
        center = x0 + group_width * (index + 0.5)
        candidate_rate = comparison["samples"]["candidate_healthy"] / 64 * 100
        baseline_rate = comparison["samples"]["baseline_healthy"] / 64 * 100
        for offset, rate, css, model_label in (
            (-45, candidate_rate, "blue", "Candidate"),
            (45, baseline_rate, "orange", "Baseline"),
        ):
            bar_x = center + offset - 30
            bar_y = linear(rate, (0, 100), (y0 + height, y0))
            parts.append(f'<rect class="{css}" x="{bar_x:.1f}" y="{bar_y:.1f}" width="60" height="{y0 + height - bar_y:.1f}" opacity="0.82"/>')
            parts.append(f'<text class="value" x="{center + offset:.1f}" y="{bar_y - 8:.1f}" text-anchor="middle">{rate:.1f}%</text>')
        parts.append(f'<text class="small" x="{center:.1f}" y="{y0 + height + 28}" text-anchor="middle">{label}</text>')
    parts.append('<rect class="blue" x="792" y="62" width="14" height="14"/><text class="small" x="814" y="74">Candidate</text>')
    parts.append('<rect class="orange" x="908" y="62" width="14" height="14"/><text class="small" x="930" y="74">Baseline</text>')
    write_svg(
        "stage-d-health.svg",
        "Stage D evaluation health rates",
        "Grouped bars compare candidate and baseline exact-contract horizon completion rates in three comparisons.",
        1200,
        500,
        "\n".join(parts),
    )


def throughput_figure(dataset: dict[str, Any]) -> None:
    rows = dataset["engineering"]["vectorized_throughput"]
    x0, y0, width, height = 100, 120, 1030, 310
    parts = [
        '<text class="title" x="48" y="48">Vectorized simulator throughput diagnostics</text>',
        '<text class="subtitle" x="48" y="75">The two-env run is an overload case with 912 FUEL bodies; the other rows use 32 FUEL per environment.</text>',
        chart_frame(x0, y0, width, height),
    ]
    for tick in range(0, 6):
        value = tick * 10
        y = linear(value, (0, 50), (y0 + height, y0))
        parts.append(f'<path class="grid" d="M {x0} {y:.1f} H {x0 + width}"/>')
        parts.append(f'<text class="small" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{value}</text>')
    for row in rows:
        x = linear(row["num_envs"], (0.5, 8.5), (x0, x0 + width))
        value = row["aggregate_policy_transitions_per_second"]
        y = linear(value, (0, 50), (y0 + height, y0))
        css = "red" if row["fuel_bodies"] > row["num_envs"] * 32 else "blue"
        parts.append(f'<circle class="{css}" cx="{x:.1f}" cy="{y:.1f}" r="9"/>')
        parts.append(f'<text class="value" x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle">{value:.2f}</text>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{y0 + height + 26}" text-anchor="middle">{row["num_envs"]} env</text>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{y0 + height + 44}" text-anchor="middle">{row["fuel_bodies"]} FUEL</text>')
    parts.append(f'<text class="label" x="{x0 + width / 2}" y="500" text-anchor="middle">Vectorized environment count</text>')
    parts.append(f'<text class="label" x="28" y="{y0 + height / 2}" text-anchor="middle" transform="rotate(-90 28 {y0 + height / 2})">Policy transitions / second</text>')
    write_svg(
        "vectorized-throughput.svg",
        "Vectorized simulator throughput",
        "Scatter plot of aggregate policy transitions per second by environment count, with FUEL body count annotations.",
        1200,
        540,
        "\n".join(parts),
    )


def build_figures(dataset: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    curriculum_figure(dataset)
    stage_a_learning_figure(dataset)
    stage_a_diagnostic_figure(dataset)
    stage_c_figure(dataset)
    stage_c_progression_figure(dataset)
    stage_d_delta_figure(dataset)
    stage_d_health_figure(dataset)
    throughput_figure(dataset)


def main() -> None:
    dataset = build_dataset()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    build_figures(dataset)
    print(f"wrote {RESULT_PATH.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.svg")):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
