#!/usr/bin/env python
"""Headless stress validation of the HUB FUEL routing pipeline.

Boots Isaac Sim headless, builds the REBUILT scene with a reduced fuel count,
and continuously injects idle floor balls into BOTH hub sensor volumes
(alternating red/blue) while measuring, per routed ball:

  1. sensed -> released latency vs the sampled triangular delay
  2. post-release trajectory: does it physically clear the hub and reach the
     neutral zone (|y| < 2.7) or at least the floor (z < 0.15) within 6 s
  3. re-detection of the same ball while it is still leaving
     (blocked_until_clear failures)
  4. exit-lane distribution uniformity (chi-square vs uniform over 4 exits)
  5. holding-pen behavior: escapes above the floor (z > 0) while pending, and
     exactness of the release teleport pose/velocity (0, +/-1, -0.08)

Also validates rules.sample_hub_routing_delay statistically (pure python).

Usage (bash):
  cd <project> && OMNI_KIT_ACCEPT_EULA=YES /c/il/venv/Scripts/python.exe \
      tools/validate_hub.py [--duration 40] [--max-fuel 64] [--out runs/hub_validation.json]

  --delay-only runs the pure-python distribution check without booting Isaac.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROUTER_PERIOD_S = 0.02  # HubRouter.step cadence in run_sim.py (every 5 frames @ 250 Hz)
INJECT_Z = 0.95         # inside the sensor z-band [0.78, 1.16]
FLOOR_Z = 0.15          # "reached the floor" threshold (ball rest center = 0.076)
NEUTRAL_ABS_Y = 2.70    # "reached the neutral zone" threshold
STUCK_Z = 0.30
STUCK_SPEED = 0.05
STUCK_HOLD_S = 2.0
TRACK_WINDOW_S = 6.0


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def validate_delay_distribution(n: int = 100_000, seed: int = 20260702) -> dict:
    """Pure-python statistical check of rules.sample_hub_routing_delay."""
    from xrc_rebuilt import rules

    rng = random.Random(seed)
    samples = [rules.sample_hub_routing_delay(rng) for _ in range(n)]
    mean = sum(samples) / n
    lo, hi = min(samples), max(samples)
    stdev = math.sqrt(sum((s - mean) ** 2 for s in samples) / (n - 1))
    bins = 60
    width = 3.0 / bins
    hist = [0] * bins
    for s in samples:
        hist[min(bins - 1, int(s / width))] += 1
    peak_bin = max(range(bins), key=hist.__getitem__)
    peak_center = (peak_bin + 0.5) * width
    checks = {
        "mean_within_0.01_of_1.32": abs(mean - rules.HUB_ROUTING_DELAY_MEAN_S) <= 0.01,
        "all_samples_in_0_to_3": (lo >= 0.0) and (hi <= 3.0),
        "hist_peak_near_mode_0.96": abs(peak_center - rules.HUB_ROUTING_DELAY_MODE_S) <= 0.10,
    }
    return {
        "n": n,
        "mean_s": mean,
        "stdev_s": stdev,
        "min_s": lo,
        "max_s": hi,
        "expected_mean_s": rules.HUB_ROUTING_DELAY_MEAN_S,
        "expected_mode_s": rules.HUB_ROUTING_DELAY_MODE_S,
        "hist_bin_width_s": width,
        "hist_peak_center_s": peak_center,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_sim_validation(args, delay_stats: dict) -> dict:
    sys.argv = [sys.argv[0]]  # keep Kit from parsing our CLI flags
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    print("HUBVAL_APP_READY", flush=True)
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim

        from xrc_rebuilt.isaac_scene import HubRouter, SceneBuilder

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        builder = SceneBuilder(stage, max_fuel=args.max_fuel)
        scene_stats = builder.build()
        # Keep the routing measurement free of robot contacts.
        stage.RemovePrim("/World/Robot")
        print("HUBVAL_SCENE", json.dumps(scene_stats), flush=True)

        fuel_view = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        sim = SimulationContext(physics_dt=0.004, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel_view.initialize()
        count = len(builder.ball_prims)
        router = HubRouter(fuel_view, count)

        def as_np(v):
            return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        dt = 0.004
        total_frames = int(round(args.duration / dt))
        inj_rng = random.Random(20260704)

        episodes: list[dict] = []
        episode_by_index: dict[int, dict] = {}
        inject_wait: dict[int, dict] = {}
        injections: list[dict] = []
        sensor_misses: list[dict] = []
        skipped_injections = 0
        redetect_count = 0
        pen_max_z = -1e9
        pen_escape_details: list[dict] = []
        pending_hist: list[int] = []
        release_pos_dev_max = 0.0
        release_vel_dev_max = 0.0
        next_inject_t = args.inject_start
        inject_n = 0

        positions_np = as_np(fuel_view.get_world_poses()[0]).copy()
        velocities_np = as_np(fuel_view.get_linear_velocities()).copy()

        for frame in range(1, total_frames + 1):
            elapsed = frame * dt
            sim.step(render=False)
            if frame % 5:
                continue

            pending_before = dict(router.pending)
            router.step(elapsed)
            pending_after = dict(router.pending)
            positions_np = as_np(fuel_view.get_world_poses()[0])
            velocities_np = as_np(fuel_view.get_linear_velocities())

            # --- new detections --------------------------------------------
            for idx, item in pending_after.items():
                if idx in pending_before:
                    continue
                release_t, alliance, exit_i, delay = item
                inj = inject_wait.pop(idx, None)
                old = episode_by_index.get(idx)
                if old is not None and old["outcome"] is None and old["release_s"] is not None:
                    # ball got re-captured by the sensor while still leaving
                    old["outcome"] = "re-detected_while_leaving"
                    old["outcome_s"] = elapsed
                    redetect_count += 1
                detect_s = float(release_t - delay)  # == elapsed at this step
                ep = {
                    "index": int(idx),
                    "alliance": alliance,
                    "exit": int(exit_i),
                    "sampled_delay_s": float(delay),
                    "detect_s": detect_s,
                    "injected": inj is not None,
                    "inject_to_detect_s": (detect_s - inj["t"]) if inj else None,
                    "release_s": None,
                    "measured_latency_s": None,
                    "latency_minus_delay_s": None,
                    "release_pos_dev_m": None,
                    "release_vel_dev_mps": None,
                    "outcome": None,
                    "outcome_s": None,
                    "clear_s": None,
                    "first_floor_s": None,
                    "min_abs_y": None,
                    "min_z": None,
                    "max_z": None,
                    "pen_escaped": False,
                    "stuck_pos": None,
                    "path": [],
                    "_slow_since": None,
                    "_last_path_s": -1e9,
                }
                episode_by_index[idx] = ep
                episodes.append(ep)

            # --- releases ---------------------------------------------------
            for idx in pending_before:
                if idx in pending_after:
                    continue
                ep = episode_by_index.get(idx)
                if ep is None:
                    continue
                ep["release_s"] = elapsed
                lat = elapsed - ep["detect_s"]
                ep["measured_latency_s"] = lat
                ep["latency_minus_delay_s"] = lat - ep["sampled_delay_s"]
                hub = HubRouter.HUBS[ep["alliance"]]
                exp_pos = np.array([HubRouter.EXIT_X[ep["exit"]], hub["exit_y"], 1.02], dtype=np.float64)
                exp_vel = np.array([0.0, hub["direction_y"] * 1.0, -0.08], dtype=np.float64)
                pos_dev = float(np.abs(positions_np[idx].astype(np.float64) - exp_pos).max())
                vel_dev = float(np.abs(velocities_np[idx].astype(np.float64) - exp_vel).max())
                ep["release_pos_dev_m"] = pos_dev
                ep["release_vel_dev_mps"] = vel_dev
                release_pos_dev_max = max(release_pos_dev_max, pos_dev)
                release_vel_dev_max = max(release_vel_dev_max, vel_dev)

            # --- holding-pen monitoring ------------------------------------
            pending_hist.append(len(pending_after))
            for idx in pending_after:
                z = float(positions_np[idx, 2])
                pen_max_z = max(pen_max_z, z)
                ep = episode_by_index.get(idx)
                if z > 0.0 and ep is not None and not ep["pen_escaped"]:
                    ep["pen_escaped"] = True
                    pen_escape_details.append(
                        {"index": int(idx), "t": elapsed, "pos": [round(float(c), 3) for c in positions_np[idx]]}
                    )

            # --- post-release trajectory tracking --------------------------
            for idx, ep in list(episode_by_index.items()):
                if ep["outcome"] is not None or ep["release_s"] is None:
                    continue
                t_since = elapsed - ep["release_s"]
                if t_since <= 0.0:
                    continue
                p = positions_np[idx]
                v = velocities_np[idx]
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                speed = float(np.linalg.norm(v))
                ep["min_abs_y"] = abs(y) if ep["min_abs_y"] is None else min(ep["min_abs_y"], abs(y))
                ep["min_z"] = z if ep["min_z"] is None else min(ep["min_z"], z)
                ep["max_z"] = z if ep["max_z"] is None else max(ep["max_z"], z)
                if elapsed - ep["_last_path_s"] >= 0.1 and len(ep["path"]) < 70:
                    ep["path"].append([round(t_since, 3), round(x, 3), round(y, 3), round(z, 3), round(speed, 3)])
                    ep["_last_path_s"] = elapsed
                if ep["first_floor_s"] is None and z < FLOOR_Z:
                    ep["first_floor_s"] = t_since
                if z < -0.05:
                    ep["outcome"] = "tunneled_below_floor"
                    ep["outcome_s"] = elapsed
                    continue
                if abs(y) < NEUTRAL_ABS_Y:
                    ep["outcome"] = "reached_neutral_zone"
                    ep["outcome_s"] = elapsed
                    ep["clear_s"] = t_since
                    continue
                if z > STUCK_Z and abs(y) > NEUTRAL_ABS_Y and speed < STUCK_SPEED:
                    if ep["_slow_since"] is None:
                        ep["_slow_since"] = elapsed
                    elif elapsed - ep["_slow_since"] > STUCK_HOLD_S:
                        ep["outcome"] = "stuck_on_hub"
                        ep["outcome_s"] = elapsed
                        ep["stuck_pos"] = [round(x, 3), round(y, 3), round(z, 3)]
                        continue
                else:
                    ep["_slow_since"] = None
                if t_since >= TRACK_WINDOW_S:
                    ep["outcome"] = (
                        "reached_floor_not_neutral_in_6s" if ep["first_floor_s"] is not None
                        else "did_not_clear_in_6s"
                    )
                    ep["outcome_s"] = elapsed

            # --- sensor-miss sweep ------------------------------------------
            for idx in list(inject_wait):
                if elapsed - inject_wait[idx]["t"] > 0.5:
                    rec = inject_wait.pop(idx)
                    sensor_misses.append(
                        {
                            "index": int(idx),
                            "t_injected": rec["t"],
                            "alliance": rec["alliance"],
                            "pos_now": [round(float(c), 3) for c in positions_np[idx]],
                        }
                    )

            # --- injection ---------------------------------------------------
            if elapsed >= next_inject_t and elapsed < args.inject_stop:
                busy = set(router.pending) | set(router.blocked_until_clear) | set(inject_wait)
                active = {i for i, e in episode_by_index.items() if e["outcome"] is None}
                candidates = [
                    i for i in range(count)
                    if i not in busy and i not in active
                    and float(positions_np[i, 2]) < FLOOR_Z
                    and abs(float(positions_np[i, 1])) < 2.9
                    and float(np.linalg.norm(velocities_np[i])) < 0.8
                ]
                if candidates:
                    idx = inj_rng.choice(candidates)
                    alliance = "red" if inject_n % 2 == 0 else "blue"
                    hub = HubRouter.HUBS[alliance]
                    pos = np.asarray(
                        [[inj_rng.uniform(-0.30, 0.30), hub["sensor_y"], INJECT_Z]], dtype=np.float32
                    )
                    indices = np.asarray([idx], dtype=np.int32)
                    fuel_view.set_world_poses(positions=pos, indices=indices)
                    fuel_view.set_linear_velocities(np.zeros((1, 3), dtype=np.float32), indices=indices)
                    inject_wait[idx] = {"t": elapsed, "alliance": alliance}
                    injections.append(
                        {"index": int(idx), "t": round(elapsed, 3), "alliance": alliance, "x": float(pos[0, 0])}
                    )
                    inject_n += 1
                else:
                    skipped_injections += 1
                in_burst = args.burst_start <= elapsed < args.burst_stop
                next_inject_t += args.burst_interval if in_burst else args.inject_interval

            if frame % 2500 == 0:
                print(
                    f"HUBVAL_PROGRESS t={elapsed:.1f}s injected={inject_n} "
                    f"detected={router.detected} released={router.released} pending={len(router.pending)}",
                    flush=True,
                )

        # --- finalize open episodes ------------------------------------------
        for ep in episodes:
            if ep["outcome"] is None:
                if ep["release_s"] is None:
                    ep["outcome"] = "still_pending_at_sim_end"
                elif ep["first_floor_s"] is not None:
                    ep["outcome"] = "reached_floor_tracking_cut_by_sim_end"
                else:
                    ep["outcome"] = "tracking_cut_by_sim_end"

        sim.stop()

        # --- summarize ---------------------------------------------------------
        released_eps = [e for e in episodes if e["release_s"] is not None]
        lat_deltas = [e["latency_minus_delay_s"] for e in released_eps]
        latencies = [e["measured_latency_s"] for e in released_eps]
        outcome_counts = Counter(e["outcome"] for e in episodes)
        success_outcomes = {
            "reached_neutral_zone",
            "reached_floor_not_neutral_in_6s",
            "reached_floor_tracking_cut_by_sim_end",
        }
        undecided = {"tracking_cut_by_sim_end", "still_pending_at_sim_end"}
        decided = [e for e in episodes if e["outcome"] not in undecided and e["release_s"] is not None]
        successes = [e for e in decided if e["outcome"] in success_outcomes]
        clear_times = [e["clear_s"] for e in episodes if e["clear_s"] is not None]
        exit_counts = Counter(e["exit"] for e in released_eps)
        n_rel = len(released_eps)
        chi2 = (
            sum((exit_counts.get(k, 0) - n_rel / 4.0) ** 2 / (n_rel / 4.0) for k in range(4))
            if n_rel else None
        )
        per_alliance_exit = {
            a: dict(Counter(e["exit"] for e in released_eps if e["alliance"] == a))
            for a in ("red", "blue")
        }
        pen_escapes = len(pen_escape_details)

        checks = {
            "at_least_50_injections": inject_n >= 50,
            "no_sensor_misses": len(sensor_misses) == 0,
            # Since the per-chute serialization fix, a ball whose chute is
            # still clearing its previous FUEL is deliberately HELD past its
            # sampled routing delay (EXIT_INTERVAL 0.22-0.42 s per hold), so
            # latency = delay + step quantization + bounded chute holds.
            # Never early; burst-window queues make ~1 s of excess legitimate.
            "latency_never_early_and_chute_holds_bounded": bool(lat_deltas)
            and all(-1e-6 <= d <= 4.0 for d in lat_deltas)
            and float(np.mean(lat_deltas)) <= 0.25,
            "all_decided_balls_reach_neutral_or_floor": bool(decided) and len(successes) == len(decided),
            "no_stuck_on_hub": outcome_counts.get("stuck_on_hub", 0) == 0,
            "no_tunneling_below_floor": outcome_counts.get("tunneled_below_floor", 0) == 0,
            "no_redetection_while_leaving": redetect_count == 0,
            "exit_distribution_uniform_chi2_lt_7.815": chi2 is not None and chi2 < 7.815,
            "pen_no_escape_above_floor": pen_escapes == 0,
            "release_position_jitter_bounded": release_pos_dev_max <= 0.020,
            "release_velocity_jitter_bounded": release_vel_dev_max <= 0.170,
        }

        # strip internals; keep paths only for problematic outcomes
        for e in episodes:
            for k in [k for k in e if k.startswith("_")]:
                del e[k]
            if e["outcome"] == "reached_neutral_zone":
                e.pop("path", None)

        result = {
            "config": {
                "duration_s": args.duration,
                "max_fuel": args.max_fuel,
                "physics_dt": dt,
                "router_period_s": ROUTER_PERIOD_S,
                "inject_start_s": args.inject_start,
                "inject_stop_s": args.inject_stop,
                "inject_interval_s": args.inject_interval,
                "burst_window_s": [args.burst_start, args.burst_stop],
                "burst_interval_s": args.burst_interval,
                "robot_removed": True,
            },
            "scene_stats": scene_stats,
            "counts": {
                "fuel_bodies": count,
                "injections": inject_n,
                "skipped_injections": skipped_injections,
                "sensor_misses": len(sensor_misses),
                "detections_total": router.detected,
                "released_total": router.released,
                "pending_at_sim_end": len(router.pending),
                "redetected_while_leaving": redetect_count,
                "episodes": len(episodes),
                "decided_episodes": len(decided),
                "successful_exits": len(successes),
            },
            "latency": {
                "n": len(latencies),
                "measured_mean_s": float(np.mean(latencies)) if latencies else None,
                "measured_min_s": float(np.min(latencies)) if latencies else None,
                "measured_max_s": float(np.max(latencies)) if latencies else None,
                "delta_vs_sampled_min_s": float(np.min(lat_deltas)) if lat_deltas else None,
                "delta_vs_sampled_max_s": float(np.max(lat_deltas)) if lat_deltas else None,
                "delta_vs_sampled_mean_s": float(np.mean(lat_deltas)) if lat_deltas else None,
            },
            "outcomes": dict(outcome_counts),
            "clear_time_to_neutral_s": {
                "n": len(clear_times),
                "mean": float(np.mean(clear_times)) if clear_times else None,
                "p50": float(np.percentile(clear_times, 50)) if clear_times else None,
                "p95": float(np.percentile(clear_times, 95)) if clear_times else None,
                "max": float(np.max(clear_times)) if clear_times else None,
            },
            "exit_distribution": {
                "counts_by_exit": {str(k): exit_counts.get(k, 0) for k in range(4)},
                "per_alliance": per_alliance_exit,
                "chi2_df3": chi2,
                "chi2_critical_p05": 7.815,
            },
            "holding_pen": {
                "max_z_while_pending_m": pen_max_z if pen_max_z > -1e8 else None,
                "escapes_above_floor": pen_escapes,
                "escape_details": pen_escape_details[:20],
                "max_simultaneous_pending": max(pending_hist) if pending_hist else 0,
                "mean_pending": float(np.mean(pending_hist)) if pending_hist else 0.0,
            },
            "release_teleport": {
                "max_pos_dev_m": release_pos_dev_max,
                "max_vel_dev_mps": release_vel_dev_max,
                "expected_velocity": "bounded jitter around (0, -/+1.0, -0.08)",
            },
            "sensor_miss_details": sensor_misses,
            "router_stats": router.stats(),
            "checks": checks,
            "passed": all(checks.values()),
            "episodes": episodes,
        }
        summary = {k: v for k, v in result.items() if k not in ("episodes", "scene_stats")}
        print("HUBVAL_RESULT", json.dumps(sanitize(summary)), flush=True)
        # Write inside the try block: SimulationApp.close() (fastShutdown) can
        # terminate the process before anything after it runs.
        payload = {
            "delay_distribution": delay_stats,
            "sim": result,
            "passed": bool(delay_stats["passed"] and result["passed"]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
        print("HUBVAL_WROTE", str(args.out), flush=True)
        return result
    except BaseException as error:
        import traceback
        print("HUBVAL_ERROR", repr(error), flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the HUB FUEL routing pipeline at scale")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--max-fuel", type=int, default=64)
    parser.add_argument("--inject-start", type=float, default=1.0)
    parser.add_argument("--inject-stop", type=float, default=30.0)
    parser.add_argument("--inject-interval", type=float, default=0.3)
    parser.add_argument("--burst-start", type=float, default=5.0)
    parser.add_argument("--burst-stop", type=float, default=8.0)
    parser.add_argument("--burst-interval", type=float, default=0.1)
    parser.add_argument("--delay-only", action="store_true", help="pure-python delay check only")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "hub_validation.json")
    args = parser.parse_args()

    delay_stats = validate_delay_distribution()
    print("HUBVAL_DELAY", json.dumps(sanitize(delay_stats)), flush=True)
    if args.delay_only:
        return

    run_sim_validation(args, delay_stats)


if __name__ == "__main__":
    main()
