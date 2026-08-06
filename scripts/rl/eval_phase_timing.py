"""Read-only Stage C phase-timing evaluator + BC anchor dumper.

Under the converged evaluation contract, the immutable champion plays under-trench
Stage C episodes with one-click/one-shot fire and UNCHANGED reward/action. Per episode
it logs a full phase timeline (hysteretic board crossings with lane classification,
collect-8/16/32/48/60 timestamps, first/last shot, chamber-empty-after-score, second
outbound/inbound/score), HONEST motion telemetry (policy_idle / commanded_no_motion /
firing_stationary intervals -- never "collision"), max simultaneous magazine, time
remaining at each milestone, and additional_score_needed = 100 - score_at_first_unload
with the required scoring rate for the remaining time.

It ALSO dumps the BC anchor set: at every PRE-handoff step (before the first
chamber-empty-after-score) the exact checkpoint input (9x90x160 uint8 frame + 22-D
proprio) + the champion deterministic mean action + t + seed + phase label, in lossless
episode-contiguous npz chunks with a manifest + per-chunk sha256.

NO training-code change. Champion frozen. Read-only. Seeds are split so anchor/dev seeds
are never reused for promotion (design note): anchor_dev=70001, suffix_eval=70002,
full_eval=70003 (disjoint base seeds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np

BOARD_Y = -2.775          # blue scoring-legality line (signed; alliance sign matters)
BAND_IN = -3.05           # hysteresis: "below" region (cleared the structure inward)
BAND_OUT = -2.50          # hysteresis: "above" region (cleared the structure outward)
DT = 0.1                  # 10 Hz policy step
EP_LEN_S = 90.0
MIN_RUN = 5               # >=5 consecutive ticks (0.5 s) to log a motion interval
V_STILL = 0.15            # m/s
CMD_LOW = 0.40            # ||action[0:2]|| below this = little translation commanded
SHOOT_THRESH = 0.25       # action[5] > this = shoot requested
COLLECT_THRESHOLDS = [8, 16, 32, 48, 60]
# lane gateway centers (design note): ramps x=+-1.55, trenches x=+-3.3582
LANES = [("left_ramp", -1.55), ("right_ramp", 1.55), ("left_trench", -3.3582), ("right_trench", 3.3582)]
LANE_TOL = 0.9
# widely-separated bases (design note): adjacent bases overlap via HubRouter(seed+i)
# and the shared per-reset RNG, leaking anchor/suffix/full sets into each other.
SEED_SETS = {"anchor_dev": 70001, "suffix_eval": 170001, "full_eval": 270001}


def lane_of(x: float) -> str:
    best, bd = "unknown", LANE_TOL
    for name, cx in LANES:
        d = abs(x - cx)
        if d < bd:
            best, bd = name, d
    return best


def phase_label(n_outbound: int, y: float, mag: int, shooting: bool) -> str:
    if n_outbound == 0:
        return "trench_exit"
    if shooting:
        return "shooting"
    if y > BOARD_Y:
        return "neutral_collect"
    if mag > 0:
        return "aiming"
    return "inbound_return"


class EpisodeTracker:
    def __init__(self, seed_set: str, index: int):
        self.seed_set, self.index = seed_set, index
        self.t = 0
        self.crossings: list[dict] = []
        self.collect_t = {k: None for k in COLLECT_THRESHOLDS}
        self.first_shot_t = self.last_shot_t = None
        self.chamber_empty_t = None
        self.score_at_unload = None
        self.collected_at_unload = None
        self.first_load_settled = False
        self.unload_step = None
        self.second_outbound_t = self.second_inbound_t = self.second_score_t = None
        # ORDERED cycle-2 gate (design note): a real 2nd cycle must go out -> collect in
        # neutral -> return with inventory -> score. Recycling/home-side score that
        # skips the field trip is counted separately, NOT as a completed cycle.
        self.completed_cycle_2_t = None
        self.recycling_post_unload_score = 0
        self._c2_stage = 0            # 0 none | 1 out | 2 neutral-collected | 3 returned-w/inv
        self._c2_collected_at_out = None
        self.max_magazine = 0
        self.n_outbound = 0
        self._band = "start"
        self._prev_score = self._prev_collected = self._prev_shots = 0
        self._prev_cext = None   # unknown until first tick (avoids spurious storage_moving)
        self._run = {k: None for k in ("policy_idle", "commanded_no_motion", "firing_stationary")}
        self.intervals = {k: [] for k in self._run}

    def step(self, x, y, vxy, act_trans, shoot_req, mag, score, collected, shots, cext) -> str:
        t_s = self.t * DT
        score_event = score > self._prev_score
        collect_event = collected > self._prev_collected
        shot_event = shots > self._prev_shots
        storage_moving = (self._prev_cext is not None) and abs(cext - self._prev_cext) > 0.005
        self.max_magazine = max(self.max_magazine, mag)
        # hysteretic crossings (blue signed board)
        if y <= BAND_IN and self._band != "below":
            if self._band == "above":
                self.crossings.append({"t_s": round(t_s, 2), "dir": "inbound", "lane": lane_of(x),
                                       "x": round(x, 2), "y": round(y, 2)})
                if self.chamber_empty_t is not None and self.second_inbound_t is None:
                    self.second_inbound_t = round(t_s, 2)
                if self._c2_stage == 2 and mag > 0:          # returned WITH inventory
                    self._c2_stage = 3
            self._band = "below"
        elif y >= BAND_OUT and self._band != "above":
            if self._band == "below":
                self.crossings.append({"t_s": round(t_s, 2), "dir": "outbound", "lane": lane_of(x),
                                       "x": round(x, 2), "y": round(y, 2)})
                self.n_outbound += 1
                if self.chamber_empty_t is not None and self.second_outbound_t is None:
                    self.second_outbound_t = round(t_s, 2)
                if self.chamber_empty_t is not None and self._c2_stage == 0:  # 2nd trip out
                    self._c2_stage = 1
                    self._c2_collected_at_out = collected
            self._band = "above"
        for k in COLLECT_THRESHOLDS:
            if self.collect_t[k] is None and collected >= k:
                self.collect_t[k] = round(t_s, 2)
        if shot_event:
            if self.first_shot_t is None:
                self.first_shot_t = round(t_s, 2)
            self.last_shot_t = round(t_s, 2)
        # chamber-empty-after-score = champion has LAUNCHED its whole first load. The
        # magazine empties at launch but score credits ~0.3-0.5 s later as balls land,
        # so fix the handoff TIME at mag==0 yet keep accumulating score_at_first_unload
        # until a genuine 2nd collection begins (balls_collected rises past the unload
        # count). Trailing first-load landings are NOT a second cycle.
        if self.chamber_empty_t is None and score > 0 and mag == 0:
            self.chamber_empty_t = round(t_s, 2)
            self.unload_step = self.t
            self.collected_at_unload = collected
            self.score_at_unload = int(score)
        elif self.chamber_empty_t is not None:
            if not self.first_load_settled:
                if collected > self.collected_at_unload:
                    self.first_load_settled = True       # a real 2nd collection began
                elif score_event:
                    self.score_at_unload = int(score)    # trailing first-load ball landed
            elif score_event and self.second_score_t is None:
                self.second_score_t = round(t_s, 2)      # genuine 2nd-load score (loose)
        # ORDERED completed-cycle-2 gate + recycling split (design note)
        if self.chamber_empty_t is not None:
            if (self._c2_stage == 1 and y > BOARD_Y and self._c2_collected_at_out is not None
                    and collected > self._c2_collected_at_out):
                self._c2_stage = 2                        # collected in neutral after 2nd outbound
            if self.first_load_settled and score_event and self._c2_stage < 3:
                # scored after a real 2nd collection but WITHOUT the ordered field
                # trip -> home-side/recycling, not a completed cycle (trailing first-
                # load balls are excluded: they update score_at_unload above).
                self.recycling_post_unload_score += int(score) - int(self._prev_score)
            elif self._c2_stage == 3 and score_event and self.completed_cycle_2_t is None:
                self.completed_cycle_2_t = round(t_s, 2)  # first genuine 2nd-CYCLE score
        # honest motion telemetry
        still = vxy < V_STILL
        firing = bool(shoot_req or shot_event)
        low_cmd = act_trans < CMD_LOW
        pose = [round(x, 2), round(y, 2)]
        ph = phase_label(self.n_outbound, y, mag, shot_event)
        self._tel("firing_stationary", still and firing, t_s, pose, mag, score, act_trans, ph)
        self._tel("policy_idle", still and low_cmd and not firing and not score_event and not collect_event,
                  t_s, pose, mag, score, act_trans, ph)
        self._tel("commanded_no_motion", still and (not low_cmd) and not firing and not storage_moving,
                  t_s, pose, mag, score, act_trans, ph)
        self._prev_score, self._prev_collected, self._prev_shots = score, collected, shots
        self._prev_cext = cext
        self.t += 1
        return ph

    def _tel(self, key, active, t_s, pose, mag, score, act_trans, ph):
        run = self._run[key]
        if active:
            if run is None:
                self._run[key] = {"start_t": round(t_s, 2), "_n": 1, "pose": pose,
                                  "magazine": int(mag), "score": int(score),
                                  "action_trans_mean": act_trans, "phase": ph}
            else:
                run["_n"] += 1
                run["action_trans_mean"] += act_trans
        elif run is not None:
            self._close(key, run, t_s)
            self._run[key] = None

    def _close(self, key, run, t_s):
        if run["_n"] >= MIN_RUN:
            run["end_t"] = round(t_s, 2)
            run["duration_s"] = round(run["_n"] * DT, 2)
            run["action_trans_mean"] = round(run["action_trans_mean"] / run["_n"], 3)
            del run["_n"]
            self.intervals[key].append(run)

    def finalize(self, score, collected, shots):
        for key, run in self._run.items():
            if run is not None:
                self._close(key, run, self.t * DT)
        self.final = {"score": int(score), "collected": int(collected), "shots_fired": int(shots)}

    def record(self):
        tr = round(EP_LEN_S - self.chamber_empty_t, 2) if self.chamber_empty_t is not None else None
        addl = (100 - self.score_at_unload) if self.score_at_unload is not None else None
        req = round(addl / tr, 3) if (addl is not None and tr and tr > 0) else None
        idle_s = round(sum(iv["duration_s"] for iv in self.intervals["policy_idle"]), 1)
        cmd_s = round(sum(iv["duration_s"] for iv in self.intervals["commanded_no_motion"]), 1)
        _trm = lambda ts: round(EP_LEN_S - ts, 2) if ts is not None else None
        milestone_tr = {
            **{f"collect_{k}": _trm(self.collect_t[k]) for k in COLLECT_THRESHOLDS},
            "first_shot": _trm(self.first_shot_t), "last_shot": _trm(self.last_shot_t),
            "chamber_empty": _trm(self.chamber_empty_t),
            "second_outbound": _trm(self.second_outbound_t), "second_score": _trm(self.second_score_t),
        }
        return {
            "seed_set": self.seed_set, "seed_base": SEED_SETS[self.seed_set], "index": self.index,
            **self.final, "unloaded": self.chamber_empty_t is not None,
            "max_simultaneous_magazine": int(self.max_magazine),
            "crossings": self.crossings, "collect_timestamps": self.collect_t,
            "first_shot_t": self.first_shot_t, "last_shot_t": self.last_shot_t,
            "chamber_empty_after_score_t": self.chamber_empty_t,
            "score_at_first_unload": self.score_at_unload,
            "time_remaining_at_first_unload_s": tr,
            "additional_score_needed": addl, "required_score_rate_per_s": req,
            "second_outbound_t": self.second_outbound_t, "second_inbound_t": self.second_inbound_t,
            "second_score_t": self.second_score_t,
            "completed_cycle_2_t": self.completed_cycle_2_t,
            "recycling_post_unload_score": int(self.recycling_post_unload_score),
            "total_idle_s": idle_s, "total_commanded_no_motion_s": cmd_s,
            "milestone_time_remaining": milestone_tr,
            "intervals": self.intervals,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed-set", default="anchor_dev", choices=list(SEED_SETS))
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dump-anchors", action="store_true",
                    help="write pre-handoff (frame,proprio,mean-action) anchor chunks")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        import importlib.util
        _s = importlib.util.spec_from_file_location("train_drqv2", PROJECT_ROOT / "scripts" / "rl" / "train_drqv2.py")
        _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
        to_frames = _m.to_policy_frames
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

        base = SEED_SETS[args.seed_set]
        env = VecCompetitionEnv(VecEnvCfg(
            num_envs=args.num_envs, template_usd=args.template, cameras=True,
            episode_len_s=EP_LEN_S, preload_prob=0.0, seed=base,
            spawn_under_trench=True, lock_storage_extended=False,
        ))
        agent = DrQV2Agent(DrQConfig()); agent.load(args.checkpoint)
        print(f"PHASE_EVAL_LOADED {args.checkpoint} steps={agent.train_steps} "
              f"seed_set={args.seed_set} base={base} envs={args.num_envs}", flush=True)

        anchor_dir = args.out.parent / (args.out.stem + "_anchors")
        if args.dump_anchors:
            anchor_dir.mkdir(parents=True, exist_ok=True)
        n = args.num_envs
        env.reset_all()
        obs, *_ = env.step(np.zeros((n, 7), np.float32))
        trackers = [EpisodeTracker(args.seed_set, i) for i in range(n)]
        buf = [[] for _ in range(n)]      # per-env pre-handoff (frame, proprio, mean_action, t, phase)
        completed: list[dict] = []
        manifest: list[dict] = []
        next_index = n
        while len(completed) < args.episodes:
            frames = to_frames(obs["rgb"])
            actions = agent.act(frames, obs["proprio"], explore=False).astype(np.float32)
            for i in range(n):
                tr = trackers[i]
                slot = env.slots[i]; c = slot.controller
                pos, _ = c.chassis_pose(); lin, _ = c.chassis_velocity()
                x, y = float(pos[0]), float(pos[1])
                vxy = float(np.hypot(float(lin[0]), float(lin[1])))
                mag = len(c.magazine); score = int(slot.router.scored["blue"])
                collected = int(c.balls_collected); shots = int(c.shots_fired)
                cext = float(c.container_extension)
                act_trans = float(np.hypot(actions[i, 0], actions[i, 1]))
                shoot_req = float(actions[i, 5]) > SHOOT_THRESH
                pre = None
                if args.dump_anchors and tr.chamber_empty_t is None:
                    pre = (frames[i].copy(), obs["proprio"][i].copy(),
                           actions[i].copy(), round(tr.t * DT, 2))
                ph = tr.step(x, y, vxy, act_trans, shoot_req, mag, score, collected, shots, cext)
                # append strictly PRE-handoff, with the phase the tracker assigned (so
                # 'shooting' is captured); the tick that sets chamber_empty is excluded
                if pre is not None and tr.chamber_empty_t is None:
                    buf[i].append(pre + (ph,))
            obs, rewards, dones, info = env.step(actions)
            for i in np.flatnonzero(dones):
                i = int(i)
                if len(completed) >= args.episodes:
                    break
                st = info["episode_stats"][i]
                trackers[i].finalize(st["scored"], st["collected"], st.get("shots_fired", 0))
                rec = trackers[i].record()
                if args.dump_anchors and buf[i]:
                    fr = np.stack([b[0] for b in buf[i]]).astype(np.uint8)
                    pr = np.stack([b[1] for b in buf[i]]).astype(np.float32)
                    ac = np.stack([b[2] for b in buf[i]]).astype(np.float32)
                    ts = np.asarray([b[3] for b in buf[i]], np.float32)
                    phs = np.asarray([b[4] for b in buf[i]])
                    cpath = anchor_dir / f"anchor_{args.seed_set}_ep{trackers[i].index:03d}.npz"
                    np.savez_compressed(cpath, frames=fr, proprio=pr, mean_action=ac, t_s=ts, phase=phs,
                                        seed_base=base, index=trackers[i].index)
                    counts = {p: int((phs == p).sum()) for p in np.unique(phs).tolist()}
                    manifest.append({"file": cpath.name, "episode_index": trackers[i].index,
                                     "n_anchors": int(fr.shape[0]), "sha256": _sha256(cpath),
                                     "phase_counts": counts, "unloaded": rec["unloaded"]})
                completed.append(rec)
                print(f"PHASE_EP {len(completed)}/{args.episodes} idx={trackers[i].index} "
                      f"score={rec['score']} unload_t={rec['chamber_empty_after_score_t']} "
                      f"score@unload={rec['score_at_first_unload']} 2nd_score={rec['second_score_t']} "
                      f"anchors={buf[i].__len__() if args.dump_anchors else 0}", flush=True)
                trackers[i] = EpisodeTracker(args.seed_set, next_index); next_index += 1
                buf[i] = []

        eps = completed[:args.episodes]
        scored = np.asarray([e["score"] for e in eps], float)
        unloaded = [e for e in eps if e["unloaded"]]
        unload_tr = np.asarray([e["time_remaining_at_first_unload_s"] for e in unloaded], float)
        req_rate = np.asarray([e["required_score_rate_per_s"] for e in unloaded
                               if e["required_score_rate_per_s"] is not None], float)
        # throughput reference rates: shooting-phase (raw firing capability) and
        # full-first-cycle (sustained, incl. collect+travel+trench-escape). The 2nd
        # cycle skips trench escape, so the full-first-cycle rate under-states it.
        shoot_rate, full_rate = [], []
        for e in unloaded:
            ce, fs, sc = e["chamber_empty_after_score_t"], e["first_shot_t"], e["score_at_first_unload"]
            if ce and ce > 0 and sc:
                full_rate.append(sc / ce)
                if fs is not None and ce - fs > 0:
                    shoot_rate.append(sc / (ce - fs))
        shoot_rate = np.asarray(shoot_rate, float)
        full_rate = np.asarray(full_rate, float)
        median_tr = float(np.median(unload_tr)) if unload_tr.size else None
        unload_rate = (len(unloaded) / len(eps)) if eps else 0.0
        if unload_rate < 0.5:
            branch = f"LOW_UNLOAD_RATE({unload_rate:.0%})_fix_first_cycle_reliability"
        elif median_tr >= 40:
            branch = "GO_prefix_takeover(>=40s)"
        elif median_tr >= 30:
            branch = "GO_with_caveat_smaller_load(30-40s)"
        else:
            branch = "NO_GO_first_cycle_cadence(<30s)"
        summary = {
            "checkpoint": str(args.checkpoint), "train_steps": int(agent.train_steps),
            "seed_set": args.seed_set, "seed_base": base, "episodes": len(eps),
            "num_envs": args.num_envs, "one_click_one_shot": True, "spawn_under_trench": True,
            "read_only": True,
            "score": {"mean": round(float(scored.mean()), 2), "median": float(np.median(scored)),
                      "min": int(scored.min()), "max": int(scored.max()),
                      "pct_scored": round(100 * float((scored >= 1).mean()), 1),
                      "per_episode": scored.astype(int).tolist()},
            "unload": {"n_unloaded": len(unloaded), "n_no_unload": len(eps) - len(unloaded),
                       "unload_rate": round(unload_rate, 3),
                       "median_time_remaining_at_first_unload_s": median_tr,
                       "p25_time_remaining_s": round(float(np.percentile(unload_tr, 25)), 1) if unload_tr.size else None,
                       "p75_time_remaining_s": round(float(np.percentile(unload_tr, 75)), 1) if unload_tr.size else None,
                       "median_score_at_first_unload": float(np.median([e["score_at_first_unload"] for e in unloaded])) if unloaded else None},
            "required_2nd_cycle": {
                "median_additional_score_needed": float(np.median([e["additional_score_needed"] for e in unloaded])) if unloaded else None,
                "median_required_rate_per_s": round(float(np.median(req_rate)), 3) if req_rate.size else None,
                "measured_shooting_rate_per_s_median": round(float(np.median(shoot_rate)), 3) if shoot_rate.size else None,
                "measured_full_first_cycle_rate_per_s_median": round(float(np.median(full_rate)), 3) if full_rate.size else None,
                "feasible_if_shooting_limited": (bool(np.median(shoot_rate) >= np.median(req_rate))
                                                 if (shoot_rate.size and req_rate.size) else None),
                "feasible_if_full_cycle_rate": (bool(np.median(full_rate) >= np.median(req_rate))
                                                if (full_rate.size and req_rate.size) else None),
            },
            "cycle_2": {
                "n_completed_ordered_cycle2": sum(1 for e in eps if e.get("completed_cycle_2_t") is not None),
                "median_recycling_post_unload_score": float(np.median([e["recycling_post_unload_score"] for e in eps])),
                "note": "completed_cycle_2 = ordered out->neutral-collect->inbound-w/inv->score; "
                        "recycling_post_unload = post-unload score NOT via a field trip (design note)",
            },
            "branch_decision": branch,
        }
        out = {"summary": summary, "episodes": eps}
        args.out.write_text(json.dumps(out, indent=1))
        if args.dump_anchors:
            (anchor_dir / "manifest.json").write_text(json.dumps(
                {"seed_set": args.seed_set, "seed_base": base, "n_chunks": len(manifest),
                 "total_anchors": sum(m["n_anchors"] for m in manifest), "chunks": manifest}, indent=1))
        print("PHASE_EVAL_DONE " + json.dumps(summary["score"]) + " branch=" + branch, flush=True)
        print("PHASE_UNLOAD " + json.dumps(summary["unload"]), flush=True)
        print("PHASE_2ND " + json.dumps(summary["required_2nd_cycle"]), flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
