"""Render the top 5-10 highest-SCORING episodes of a checkpoint as trace HTML.

Fast, GPU-3-friendly. Two modes:

  1. CAPTURE (needs Isaac):  --checkpoint <ck>
     Boots Isaac ONCE, runs a pool of episodes across --num-envs parallel envs
     with auto-reset (single-boot loop modelled on eval_phase_timing.py), logging
     the SAME 10-col route row eval_route.py produces, for EVERY env (not just
     env 0). Writes the full pool to <out-dir>/pool.json, then ranks + renders.

  2. RE-RENDER (Isaac-free, instant):  --from-pool <pool.json>
     Skips all simulation. Ranks an existing pool JSON (from this script OR from
     eval_route.py) by `scored` and renders the top-K. No GPU touched.

Ranking = episodes sorted by `scored` desc; top --top kept (default 8, i.e. the
5-10 range). Each is written as a single-episode route JSON (eval_route schema)
and rendered by the existing scripts/rl/build_trace_html.py (Isaac-free). Also
writes an index.html gallery so the user opens ONE file to see all K traces.

GPU-3 sharing: pin with the env var, e.g.
  CUDA_VISIBLE_DEVICES=3 nice -n 15 python scripts/rl/render_top_traces.py \
    --checkpoint runs/stageC_champion_998753.pt --out-dir runs/traces/topk
That remaps physical GPU 3 -> cuda:0 for both torch and Isaac, co-locating it
with learner_finetune.py on GPU 3 (same pattern the dist_trace loop uses).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROUTE_COLS = ["x", "y", "yaw", "storage_ext", "score_seen",
              "collected_seen", "drive_x", "drive_y", "turn", "shoot"]


# --------------------------------------------------------------------------
# Mode 1: capture a pool of episodes in Isaac (single boot, all envs logged)
# --------------------------------------------------------------------------
def capture_pool(args) -> Path:
    import numpy as np
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_drqv2", PROJECT_ROOT / "scripts" / "rl" / "train_drqv2.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        to_frames = module.to_policy_frames
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

        n = int(args.num_envs)
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=n,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                preload_prob=args.preload_prob,
                mask_illegal_fire=args.mask_illegal_fire,
                seed=args.seed,
                spawn_under_trench=args.spawn_under_trench,
                lock_storage_extended=not args.spawn_under_trench,
            )
        )
        agent = DrQV2Agent(DrQConfig())
        agent.load(args.checkpoint)
        print(f"TOPK_LOADED {args.checkpoint} steps={agent.train_steps} "
              f"envs={n} pool={args.pool}", flush=True)

        fuel_home = np.asarray(env._fuel_home)[:, :2].round(3).tolist()

        def state(i: int):
            pos, quat = env.slots[i].articulation.get_world_pose()
            w, x, y, z = (float(v) for v in quat)
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            ext = float(getattr(env.slots[i].controller, "container_extension", 1.0))
            return float(pos[0]), float(pos[1]), yaw, ext

        episodes: list[dict] = []
        routes: list[list[list]] = [[] for _ in range(n)]
        env.reset_all()
        obs, *_ = env.step(np.zeros((n, 7), np.float32))
        while len(episodes) < args.pool:
            frames = to_frames(obs["rgb"])
            actions = agent.act(frames, obs["proprio"], explore=False).astype(np.float32)
            for i in range(n):
                x, y, yaw, ext = state(i)
                s = env.slots[i]
                routes[i].append(
                    [round(x, 3), round(y, 3), round(yaw, 3), round(ext, 2),
                     int(s.score_seen), int(s.collected_seen),
                     round(float(actions[i, 0]), 2), round(float(actions[i, 1]), 2),
                     round(float(actions[i, 2]), 2), int(float(actions[i, 5]) > 0.25)]
                )
            obs, rewards, dones, info = env.step(actions)
            for i in np.flatnonzero(dones):
                i = int(i)
                if len(episodes) >= args.pool:
                    break
                st = info["episode_stats"][i]
                episodes.append({
                    "scored": st["scored"], "collected": st["collected"],
                    "shots_fired": st.get("shots_fired", 0), "route": routes[i],
                })
                print(f"TOPK_EP {len(episodes)}/{args.pool} env={i} "
                      f"scored={st['scored']} collected={st['collected']} "
                      f"len={len(routes[i])}", flush=True)
                routes[i] = []

        pool = {
            "label": args.label,
            "checkpoint": args.checkpoint,
            "train_steps": agent.train_steps,
            "spawn_under_trench": bool(args.spawn_under_trench),
            "field": {
                "fuel_home": fuel_home,
                "hub": [-0.0199, -3.6874],
                "trench": {"x_min": 2.7189, "x_max": 3.9975,
                           "y_neutral": -3.05, "y_alliance": -4.24,
                           "start": [3.3582, -3.8850]},
            },
            "route_cols": ROUTE_COLS,
            "episodes": episodes,
        }
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pool_path = out_dir / "pool.json"
        pool_path.write_text(json.dumps(pool))
        print(f"TOPK_POOL {pool_path} eps={len(episodes)} "
              f"scored={sorted((e['scored'] for e in episodes), reverse=True)}", flush=True)
        env.close()
        return pool_path
    finally:
        app.close()


# --------------------------------------------------------------------------
# Mode 2 (and tail of mode 1): rank by score + render top-K (Isaac-free)
# --------------------------------------------------------------------------
def rank_and_render(pool_path: Path, out_dir: Path, top: int,
                    dashboard_html: "Path | None" = None, refresh_s: int = 70) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path(pool_path).read_text())
    eps = d.get("episodes", [])
    if not eps:
        raise RuntimeError(f"{pool_path} has no episodes")
    ranked = sorted(eps, key=lambda e: int(e.get("scored", 0)), reverse=True)[:top]
    renderer = str(PROJECT_ROOT / "scripts" / "rl" / "build_trace_html.py")

    tiles = []
    for rank, ep in enumerate(ranked, 1):
        base = f"top{rank:02d}_s{int(ep.get('scored', 0))}"
        split = dict(d)
        split["label"] = f"{d.get('label', 'topk')}-{base}"
        split["episodes"] = [ep]
        jpath = out_dir / f"{base}.json"
        hpath = out_dir / f"{base}.html"
        jpath.write_text(json.dumps(split))
        # build_trace_html imports only json/sys/datetime -> Isaac-free, instant.
        # refresh_s=0: inner tile carries NO <meta refresh> -> only the gallery reloads
        # (once, every browser_refresh s), so the 8 SVG tiles aren't independently
        # reloaded mid-render (was 30s each, compounding the render churn).
        subprocess.run([sys.executable, renderer, str(jpath), str(hpath), "0"], check=True)
        tiles.append((rank, int(ep.get("scored", 0)), int(ep.get("collected", 0)),
                      hpath, int(ep.get("train_steps", d.get("train_steps", 0))),
                      int(ep.get("cycles_completed", 0))))
        print(f"TOPK_RENDER rank={rank} scored={ep.get('scored')} "
              f"cyc2={ep.get('cycles_completed', 0)} -> {hpath}", flush=True)

    # 2nd-cycle tracking over the FULL window (not just the rendered top-K)
    n_eps = len(eps)
    n_cyc2 = sum(1 for e in eps if int(e.get("cycles_completed", 0)) >= 1)
    max_cyc = max((int(e.get("cycles_completed", 0)) for e in eps), default=0)
    print(f"TOPK_CYCLE2 window={n_eps} with_2nd_cycle={n_cyc2} max_cycles={max_cyc}", flush=True)
    # ONE self-contained gallery (each trace inlined via iframe srcdoc) so it can be
    # served by the dashboard's /trace route (single file, no external refs).
    gallery = _build_gallery(d, tiles, refresh_s, n_eps, n_cyc2, max_cyc)
    (out_dir / "index.html").write_text(gallery, encoding="utf-8")
    if dashboard_html is not None:
        dashboard_html.parent.mkdir(parents=True, exist_ok=True)
        dashboard_html.write_text(gallery, encoding="utf-8")
        print(f"TOPK_DASHBOARD served at /trace?run={dashboard_html.stem.replace('live_trace_','')}"
              f" -> {dashboard_html}", flush=True)
    print(f"TOPK_GALLERY {out_dir / 'index.html'} ({len(tiles)} traces)", flush=True)
    return tiles


def _build_gallery(d: dict, tiles: list, refresh_s: int,
                   n_eps: int = 0, n_cyc2: int = 0, max_cyc: int = 0) -> str:
    import html as _html
    cards = []
    for (r, sc, co, hpath, steps_ep, cyc2) in tiles:
        inner = Path(hpath).read_text(encoding="utf-8")
        srcdoc = _html.escape(inner, quote=True)               # inline -> self-contained
        badge = (f'<span class="cyc2">&#10227; 2nd-cycle &times;{cyc2}</span>'
                 if cyc2 >= 1 else '<span class="cyc0">1st cycle only</span>')
        cards.append(
            f'<figure class="{ "hit" if cyc2 >= 1 else "" }">'
            f'<figcaption>#{r} &middot; scored <b>{sc}</b> &middot; collected {co}'
            f' &middot; @{steps_ep} steps {badge}</figcaption>'
            f'<iframe srcdoc="{srcdoc}"></iframe></figure>')
    label = d.get("label", "topk")
    steps = d.get("train_steps", "?")
    rate = (100.0 * n_cyc2 / n_eps) if n_eps else 0.0
    hot = n_cyc2 > 0
    banner = (f'&#10227; SECOND CYCLE: <b>{n_cyc2}/{n_eps}</b> recent episodes '
              f'({rate:.0f}%) &middot; best = {max_cyc} extra cycle(s)') if hot else (
              f'SECOND CYCLE: <b>0/{n_eps}</b> recent episodes &mdash; not yet cycling '
              f'(champion baseline)')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{int(refresh_s)}">
<title>top {len(tiles)} traces &mdash; {label}</title>
<style>
:root {{ --bg:#0b0f14; --ink:#e6edf3; --muted:#8b98a5; --line:#1e2a36; --ok:#34d399; --hot:#f59e0b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:system-ui,-apple-system,Segoe UI,sans-serif; padding:18px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
.banner {{ font:14px/1.4 ui-monospace,monospace; padding:10px 14px; border-radius:8px;
  margin:10px 0 14px; border:1px solid var(--line); }}
.banner.hot {{ background:rgba(52,211,153,.12); border-color:var(--ok); color:var(--ok); }}
.banner.cold {{ background:#111820; color:var(--muted); }}
.banner b {{ font-size:16px; }}
.meta {{ color:var(--muted); font:12px/1.5 ui-monospace,monospace; margin-bottom:16px; }}
.grid {{ display:grid; gap:16px;
  grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); }}
figure {{ margin:0; border:1px solid var(--line); border-radius:10px;
  overflow:hidden; background:#111820; }}
figure.hit {{ border-color:var(--ok); box-shadow:0 0 0 1px var(--ok); }}
figcaption {{ font:12px/1.4 ui-monospace,monospace; color:var(--muted);
  padding:8px 12px; border-bottom:1px solid var(--line); }}
figcaption b {{ color:var(--ok); font-size:14px; }}
.cyc2 {{ color:var(--ok); font-weight:700; }}
.cyc0 {{ color:var(--muted); }}
iframe {{ width:100%; height:560px; border:0; display:block; background:#0d141c; }}
</style></head><body>
<h1>Top {len(tiles)} scoring episodes &mdash; {label}</h1>
<div class="banner {'hot' if hot else 'cold'}">{banner}</div>
<div class="meta">latest @ <b>{steps}</b> steps &middot; ranked by episode score (desc)
&middot; green tiles completed a 2nd field-trip cycle &middot; auto-refresh {int(refresh_s)}&nbsp;s</div>
<div class="grid">
{chr(10).join(cards)}
</div></body></html>"""


def _reload_live(agent, wdir: str, loaded_step: int) -> int:
    """Load the learner's newest published weights (encoder+actor) so the loop traces the
    CURRENT training policy. Returns the loaded step (unchanged if nothing newer)."""
    import torch
    from frc_rebuilt.rl import distributed as D
    got = D.latest_weights(wdir)
    if not got:
        return loaded_step
    path, step = got
    if step == loaded_step:
        return loaded_step
    try:
        blob = torch.load(path, map_location=agent.device)
        agent.encoder.load_state_dict(blob["encoder"])
        agent.actor.load_state_dict(blob["actor"])
        agent.train_steps = int(blob.get("train_steps", agent.train_steps))
        return step
    except Exception as exc:                     # partial write mid-publish -> keep old
        print(f"TOPK_RELOAD_SKIP {exc}", flush=True)
        return loaded_step


# --------------------------------------------------------------------------
# Mode 3: PERSISTENT loop -- one Isaac boot, rolling window, re-render every
# --render-interval s, tracking the learner's LIVE weights (GPU-3 co-tenant).
# --------------------------------------------------------------------------
def capture_loop(args) -> None:
    import collections
    import time
    import numpy as np
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_drqv2", PROJECT_ROOT / "scripts" / "rl" / "train_drqv2.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        to_frames = module.to_policy_frames
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.prefix_takeover import AggressiveCycleShaper

        n = int(args.num_envs)
        env = VecCompetitionEnv(VecEnvCfg(
            num_envs=n, template_usd=args.template, cameras=True,
            episode_len_s=args.episode_len_s, preload_prob=args.preload_prob,
            mask_illegal_fire=args.mask_illegal_fire, seed=args.seed,
            spawn_under_trench=args.spawn_under_trench,
            lock_storage_extended=not args.spawn_under_trench))
        agent = DrQV2Agent(DrQConfig())

        wdir = str(args.live_weights) if args.live_weights else None
        loaded_step = -1
        if wdir:                                  # wait for the learner's first publish
            for _ in range(180):
                if _reload_live(agent, wdir, loaded_step) != loaded_step:
                    loaded_step = _reload_live(agent, wdir, loaded_step); break
                time.sleep(1.0)
            loaded_step = _reload_live(agent, wdir, loaded_step)
        else:
            agent.load(args.checkpoint)
        print(f"TOPK_LOOP envs={n} window={args.window} interval={args.render_interval}s "
              f"explore={bool(args.explore)} "
              f"weights={'live:'+wdir if wdir else args.checkpoint} steps={agent.train_steps}",
              flush=True)

        fuel_home = np.asarray(env._fuel_home)[:, :2].round(3).tolist()

        def state(i: int):
            pos, quat = env.slots[i].articulation.get_world_pose()
            w, x, y, z = (float(v) for v in quat)
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            ext = float(getattr(env.slots[i].controller, "container_extension", 1.0))
            return float(pos[0]), float(pos[1]), yaw, ext

        window: collections.deque = collections.deque(maxlen=int(args.window))
        routes: list[list[list]] = [[] for _ in range(n)]
        cyc = [AggressiveCycleShaper() for _ in range(n)]   # per-env 2nd-cycle detector
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        env.reset_all()
        obs, *_ = env.step(np.zeros((n, 7), np.float32))
        # black-frame guard: num_envs>2 can race the camera warm-up (envs go black); abort
        # loudly rather than serve garbage traces so the caller can fall back to num_envs=2.
        cam_std = obs["rgb"].std(axis=(2, 3, 4))
        if bool((cam_std <= 1.0).any()):
            raise RuntimeError(f"TOPK black camera frames at startup (num_envs={n}): "
                               f"{np.argwhere(cam_std <= 1.0).tolist()} -- retry with --num-envs 2")
        last_render = time.time()
        total_eps = 0
        eps_since_render = 0
        while True:
            frames = to_frames(obs["rgb"])
            actions = agent.act(frames, obs["proprio"], explore=bool(args.explore)).astype(np.float32)
            for i in range(n):
                x, y, yaw, ext = state(i)
                s = env.slots[i]
                routes[i].append(
                    [round(x, 3), round(y, 3), round(yaw, 3), round(ext, 2),
                     int(s.score_seen), int(s.collected_seen),
                     round(float(actions[i, 0]), 2), round(float(actions[i, 1]), 2),
                     round(float(actions[i, 2]), 2), int(float(actions[i, 5]) > 0.25)])
            obs, rewards, dones, info = env.step(actions)
            # advance the 2nd-cycle detector on the post-step state (same logic as training)
            for i in range(n):
                slot = env.slots[i]
                pos, _ = slot.articulation.get_world_pose()
                cyc[i].update(len(slot.controller.magazine),
                              int(slot.router.scored["blue"]),
                              int(slot.controller.balls_collected),
                              int(slot.custody.fresh_score), float(pos[1]), done=False)
            for i in np.flatnonzero(dones):
                i = int(i)
                st = info["episode_stats"][i]
                window.append({"scored": st["scored"], "collected": st["collected"],
                               "shots_fired": st.get("shots_fired", 0), "route": routes[i],
                               "cycles_completed": int(cyc[i].cycles_completed),
                               "train_steps": int(agent.train_steps)})
                cyc[i].reset()
                total_eps += 1
                eps_since_render += 1
                routes[i] = []
            # render the INSTANT a new episode completes (responsive) or on the fallback
            # interval; the render itself is Isaac-free and <1 s so this never throttles the sim.
            if window and (eps_since_render > 0 or (time.time() - last_render) >= args.render_interval):
                if wdir:
                    loaded_step = _reload_live(agent, wdir, loaded_step)
                pool = {
                    "label": args.label, "checkpoint": (wdir or args.checkpoint),
                    "train_steps": int(agent.train_steps),
                    "spawn_under_trench": bool(args.spawn_under_trench),
                    "field": {"fuel_home": fuel_home, "hub": [-0.0199, -3.6874],
                              "trench": {"x_min": 2.7189, "x_max": 3.9975, "y_neutral": -3.05,
                                         "y_alliance": -4.24, "start": [3.3582, -3.8850]}},
                    "route_cols": ROUTE_COLS, "episodes": list(window)}
                pool_path = out_dir / "pool.json"
                pool_path.write_text(json.dumps(pool))
                _dash = (PROJECT_ROOT / "runs" / f"live_trace_{args.dashboard_slot}.html"
                         if args.dashboard_slot else None)
                try:
                    rank_and_render(pool_path, out_dir, args.top, _dash, 25)  # 25s browser reload
                except Exception as exc:
                    print(f"TOPK_RENDER_ERR {exc}", flush=True)
                last_render = time.time()
                eps_since_render = 0
                top_sc = sorted((int(e["scored"]) for e in window), reverse=True)[:args.top]
                print(f"TOPK_ROUND eps_total={total_eps} window={len(window)} "
                      f"steps={agent.train_steps} top={top_sc}", flush=True)
    finally:
        app.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="capture mode: policy to evaluate in Isaac")
    ap.add_argument("--from-pool", type=Path,
                    help="re-render mode: rank+render an existing pool JSON, no Isaac")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "traces" / "topk")
    ap.add_argument("--top", type=int, default=8, help="how many top scorers to render (5-10)")
    ap.add_argument("--pool", type=int, default=24, help="episodes to simulate before ranking")
    ap.add_argument("--num-envs", type=int, default=2,
                    help="parallel envs; 2 is race-free (4 races the intake/shooter cameras)")
    # persistent loop mode (eval round every ~render-interval s, tracking live weights)
    ap.add_argument("--loop", action="store_true",
                    help="persistent: one Isaac boot, rolling window, re-render every "
                         "--render-interval s; with --live-weights it tracks the training policy")
    ap.add_argument("--live-weights", type=Path,
                    help="learner weights dir (e.g. /dev/shm/frc_dist_ft/weights) to reload each "
                         "round so traces follow the CURRENT policy; omit to hold --checkpoint")
    ap.add_argument("--window", type=int, default=16,
                    help="rolling window of recent episodes to rank the top-K from (loop mode)")
    ap.add_argument("--render-interval", type=float, default=70.0,
                    help="seconds between re-renders / weight reloads (loop mode)")
    ap.add_argument("--explore", action="store_true",
                    help="add exploration noise (default deterministic = clean best behaviour)")
    ap.add_argument("--dashboard-slot", default="a",
                    help="also write the gallery to runs/live_trace_<slot>.html so the dashboard "
                         "serves it at /trace?run=<slot> (default 'a'); empty '' to disable")
    ap.add_argument("--episode-len-s", type=float, default=90.0)
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"))
    ap.add_argument("--preload-prob", type=float, default=0.0)
    ap.add_argument("--spawn-under-trench", action="store_true",
                    help="required to eval a Stage-C policy from its true match start")
    ap.add_argument("--mask-illegal-fire", action="store_true")
    ap.add_argument("--label", default="topk")
    ap.add_argument("--seed", type=int, default=424242)
    args = ap.parse_args()

    dash = (PROJECT_ROOT / "runs" / f"live_trace_{args.dashboard_slot}.html"
            if args.dashboard_slot else None)
    if args.from_pool:
        rank_and_render(args.from_pool, Path(args.out_dir), args.top, dash, int(args.render_interval))
        return
    if args.loop:
        if not (args.live_weights or args.checkpoint):
            ap.error("--loop needs --live-weights <dir> (track training) or --checkpoint (fixed)")
        capture_loop(args)
        return
    if not args.checkpoint:
        ap.error("give --checkpoint (capture), --from-pool (re-render), or --loop (persistent)")
    pool_path = capture_pool(args)
    rank_and_render(pool_path, Path(args.out_dir), args.top, dash, int(args.render_interval))


if __name__ == "__main__":
    main()
