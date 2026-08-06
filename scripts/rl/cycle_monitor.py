"""Live SECOND-CYCLE monitor (Isaac-free -> true real-time).

Reads the collectors' append-only ``runs/cycle_telemetry.jsonl`` (one record per finished
episode, emitted by the AggressiveCycleShaper) and renders an auto-refreshing HTML view of
the 2nd-cycle behaviour the policy is (or isn't) doing:

  * FUNNEL over episodes that unloaded: unload -> LEFT the field -> COLLECTED in neutral ->
    RETURNED loaded -> SCORED again (cycle complete).
  * TIMINGS: median seconds for each leg (unload->leave, leave->collect, collect->return,
    return->score) and the whole 2nd cycle.
  * STREAM: the most recent episodes that left the field, each as a timeline showing exactly
    how far it got and when.

Served on the training dashboard at /trace?run=<slot> (default 'b'); refreshes every few
seconds. No GPU, no Isaac -- runs on any CPU next to training.
Usage: python scripts/rl/cycle_monitor.py [--slot b] [--window 400] [--interval 4]
"""
from __future__ import annotations

import argparse
import html as _html
import json
import statistics
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEL_PATH = PROJECT_ROOT / "runs" / "cycle_telemetry.jsonl"
DT = 0.1   # seconds per policy step

# reuse the Isaac-free renderer/ranker from render_top_traces (no Isaac imported at module load)
import importlib.util  # noqa: E402
_rt_spec = importlib.util.spec_from_file_location(
    "render_top_traces", PROJECT_ROOT / "scripts" / "rl" / "render_top_traces.py")
_rt = importlib.util.module_from_spec(_rt_spec)
_rt_spec.loader.exec_module(_rt)


def _load_field() -> dict:
    """Field geometry (fuel positions/hub/trench) for the trace SVG; grab it from a prior
    pool.json if present, else a minimal static field (path still renders, just no ball dots)."""
    try:
        return json.loads((PROJECT_ROOT / "runs" / "traces" / "topk" / "pool.json").read_text())["field"]
    except Exception:
        return {"fuel_home": [], "hub": [-0.0199, -3.6874],
                "trench": {"x_min": 2.7189, "x_max": 3.9975, "y_neutral": -3.05,
                           "y_alliance": -4.24, "start": [3.3582, -3.8850]}}


def render_traces(recs: list[dict], field: dict, dashboard_html: Path, refresh_s: int) -> int:
    """Render the top-8 SCORING route-bearing episodes to the trace gallery (Isaac-free)."""
    routed = [r for r in recs if r.get("route")]
    if not routed:
        return
    ranked = sorted(routed, key=lambda r: int(r.get("scored", 0)), reverse=True)[:8]
    episodes = []
    for r in ranked:
        # expand compact [x,y,score] -> eval_route 10-col schema build_trace_html expects
        route10 = [[p[0], p[1], 0, 1, (p[2] if len(p) > 2 else 0), 0, 0, 0, 0, 0] for p in r["route"]]
        episodes.append({"scored": int(r.get("scored", 0)), "collected": int(r.get("collected", 0)),
                         "route": route10, "cycles_completed": int(r.get("cycles_completed", 0)),
                         "train_steps": int(r.get("train_steps", 0))})
    steps = max((int(r.get("train_steps", 0)) for r in recs), default=0)
    pool = {"label": "live-top", "train_steps": steps, "spawn_under_trench": True,
            "field": field, "route_cols": _rt.ROUTE_COLS, "episodes": episodes}
    out_dir = PROJECT_ROOT / "runs" / "traces" / "topk"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "live_pool.json"
    pool_path.write_text(json.dumps(pool))
    try:
        _rt.rank_and_render(pool_path, out_dir, 8, dashboard_html, int(refresh_s))
    except Exception as exc:
        print("TRACE_RENDER_ERR", exc, flush=True)
    return len(episodes)


def tail_records(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= n + 1:
                step = min(65536, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        out = []
        for ln in data.split(b"\n")[-(n + 1):]:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
        return out
    except Exception:
        return []


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _dt(r: dict, a: str, b: str):
    ta, tb = r.get(a), r.get(b)
    return (tb - ta) * DT if (ta is not None and tb is not None) else None


_STAGE_LABEL = {0: "no leave", 1: "LEFT — didn't collect", 2: "collected — didn't return",
                3: "returned — didn't score", 4: "FULL 2ND CYCLE"}
_STAGE_COLOR = {1: "#f59e0b", 2: "#eab308", 3: "#38bdf8", 4: "#34d399"}


def render(recs: list[dict], refresh_s: int) -> str:
    unloaded = [r for r in recs if r.get("t_unload") is not None]
    n_un = len(unloaded)
    def ge(k):
        return sum(1 for r in unloaded if int(r.get("max_stage", 0)) >= k)
    n_left, n_col, n_ret, n_sc = ge(1), ge(2), ge(3), ge(4)
    def pc(x):
        return (100.0 * x / n_un) if n_un else 0.0

    leave_d = _med([_dt(r, "t_unload", "t_leave") for r in unloaded])
    coll_d = _med([_dt(r, "t_leave", "t_collect") for r in unloaded])
    ret_d = _med([_dt(r, "t_collect", "t_return") for r in unloaded])
    sc_d = _med([_dt(r, "t_return", "t_score2") for r in unloaded])
    tot_d = _med([_dt(r, "t_unload", "t_score2") for r in unloaded if r.get("t_score2") is not None])
    steps = max((int(r.get("train_steps", 0)) for r in recs), default=0)

    # funnel bars
    def bar(name, cnt, color):
        p = pc(cnt)
        return (f'<div class="frow"><div class="flab">{name}</div>'
                f'<div class="ftrack"><div class="ffill" style="width:{max(p,1.5):.1f}%;'
                f'background:{color}"></div></div>'
                f'<div class="fnum">{cnt}<span>&nbsp;{p:.0f}%</span></div></div>')
    funnel = "\n".join([
        bar("unloaded (1st cycle)", n_un, "#64748b"),
        bar("&#8594; LEFT the field", n_left, _STAGE_COLOR[1]),
        bar("&#8594; collected in neutral", n_col, _STAGE_COLOR[2]),
        bar("&#8594; returned loaded", n_ret, _STAGE_COLOR[3]),
        bar("&#8594; SCORED again (2nd cycle)", n_sc, _STAGE_COLOR[4]),
    ])

    def sec(v):
        return f"{v:.1f}s" if v is not None else "&mdash;"
    timing = (f'<tr><td>unload &#8594; leave</td><td>{sec(leave_d)}</td></tr>'
              f'<tr><td>leave &#8594; collect</td><td>{sec(coll_d)}</td></tr>'
              f'<tr><td>collect &#8594; return</td><td>{sec(ret_d)}</td></tr>'
              f'<tr><td>return &#8594; score</td><td>{sec(sc_d)}</td></tr>'
              f'<tr class="tot"><td>whole 2nd cycle</td><td>{sec(tot_d)}</td></tr>')

    # recent episodes that LEFT (most interesting), newest first
    left_eps = [r for r in recs if int(r.get("max_stage", 0)) >= 1][-24:][::-1]
    rows = []
    for r in left_eps:
        ms = int(r.get("max_stage", 0))
        col = _STAGE_COLOR.get(ms, "#8b98a5")
        def at(k):
            t = r.get(k)
            return f'{t*DT:.1f}s' if t is not None else '&mdash;'
        chain = (f'unload@{at("t_unload")} &#8594; <b style="color:{_STAGE_COLOR[1]}">left@{at("t_leave")}</b>')
        if r.get("t_collect") is not None:
            chain += f' &#8594; collect@{at("t_collect")}'
        if r.get("t_return") is not None:
            chain += f' &#8594; return@{at("t_return")}'
        if r.get("t_score2") is not None:
            chain += f' &#8594; <b style="color:{_STAGE_COLOR[4]}">SCORE@{at("t_score2")}</b>'
        rows.append(
            f'<div class="ev" style="border-left-color:{col}">'
            f'<div class="evhead"><span class="evtag" style="color:{col}">{_STAGE_LABEL[ms]}</span>'
            f'<span class="evmeta">scored {int(r.get("scored",0))} &middot; '
            f'{int(r.get("n_leaves",0))} leave(s) &middot; c{r.get("collector","?")} @{int(r.get("train_steps",0))//1000}k</span></div>'
            f'<div class="evchain">{chain}</div></div>')
    stream = "\n".join(rows) if rows else '<div class="ev empty">no leaves recorded yet in this window</div>'

    hot = n_left > 0
    head_cls = "hot" if hot else "cold"
    headline = (f'BOT LEFT THE FIELD: <b>{n_left}/{n_un}</b> recent unloads ({pc(n_left):.0f}%) '
                f'&middot; completed 2nd cycle: <b>{n_sc}</b> ({pc(n_sc):.0f}%)') if hot else (
                f'BOT NOT LEAVING YET &mdash; <b>0/{n_un}</b> recent unloads left the field')

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{int(refresh_s)}">
<title>2nd-cycle live monitor</title>
<style>
:root {{ --bg:#0b0f14; --ink:#e6edf3; --muted:#8b98a5; --line:#1e2a36; --ok:#34d399; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,sans-serif;padding:18px}}
h1{{font-size:17px;margin:0 0 10px}}
.head{{font:14px/1.4 ui-monospace,monospace;padding:11px 14px;border-radius:8px;border:1px solid var(--line);margin-bottom:16px}}
.head.hot{{background:rgba(52,211,153,.12);border-color:var(--ok);color:var(--ok)}}
.head.cold{{background:#111820;color:var(--muted)}}
.head b{{font-size:17px}}
.wrap{{display:grid;grid-template-columns:1.3fr .7fr;gap:18px;align-items:start}}
@media(max-width:820px){{.wrap{{grid-template-columns:1fr}}}}
.card{{border:1px solid var(--line);border-radius:10px;background:#111820;padding:14px 16px}}
.card h2{{font:12px/1 ui-monospace,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:0 0 12px}}
.frow{{display:flex;align-items:center;gap:10px;margin:7px 0;font:12px/1 ui-monospace,monospace}}
.flab{{width:190px;color:var(--ink);text-align:right}}
.ftrack{{flex:1;height:20px;background:#0d141c;border-radius:5px;overflow:hidden}}
.ffill{{height:100%;border-radius:5px;transition:width .4s}}
.fnum{{width:78px;color:var(--muted)}} .fnum span{{color:var(--muted);opacity:.7}}
table{{width:100%;border-collapse:collapse;font:12px/1.5 ui-monospace,monospace}}
td{{padding:5px 4px;border-bottom:1px solid var(--line);color:var(--muted)}}
td:last-child{{text-align:right;color:var(--ink)}}
tr.tot td{{color:var(--ok);font-weight:700;border-bottom:0}}
.ev{{border:1px solid var(--line);border-left-width:4px;border-radius:7px;padding:8px 11px;margin:8px 0;background:#0d141c}}
.ev.empty{{color:var(--muted);border-left-color:var(--line)}}
.evhead{{display:flex;justify-content:space-between;gap:8px;font:11px/1.3 ui-monospace,monospace;margin-bottom:4px}}
.evtag{{font-weight:700}} .evmeta{{color:var(--muted)}}
.evchain{{font:12px/1.5 ui-monospace,monospace;color:var(--ink)}}
.foot{{color:var(--muted);font:11px/1.5 ui-monospace,monospace;margin-top:14px}}
</style></head><body>
<h1>Second-cycle live monitor</h1>
<div class="head {head_cls}">{headline}</div>
<div class="wrap">
  <div class="card"><h2>Funnel &mdash; last {len(recs)} episodes ({n_un} unloaded)</h2>{funnel}
    <div class="foot">how far the best 2nd cycle got in each episode, as a share of episodes
    that completed the 1st cycle (unloaded). Widening the lower bars = the reward is working.</div>
  </div>
  <div class="card"><h2>Median leg timing</h2><table>{timing}</table></div>
</div>
<div class="card" style="margin-top:18px"><h2>Recent field trips (newest first)</h2>{stream}</div>
<div class="foot">latest policy @ {steps//1000}k steps &middot; source: 18 live collectors &middot; auto-refresh {int(refresh_s)}s</div>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="b", help="funnel dashboard slot (a|b|c) -> /trace?run=<slot>")
    ap.add_argument("--trace-slot", default="a", help="top-8 trace-gallery slot; '' to disable")
    ap.add_argument("--window", type=int, default=400, help="recent episodes to summarise")
    ap.add_argument("--interval", type=float, default=4.0,
                    help="seconds between HTML REGENERATIONS on disk (cheap, Isaac-free)")
    ap.add_argument("--browser-refresh", type=float, default=45.0,
                    help="seconds for the page's <meta refresh> (browser reload). Kept LONG and "
                         "decoupled from --interval so the heavy 8-tile SVG gallery finishes "
                         "rendering before the browser reloads it mid-paint.")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "cycle_monitor.html")
    args = ap.parse_args()
    browser_refresh = int(args.browser_refresh)
    slot_path = PROJECT_ROOT / "runs" / f"live_trace_{args.slot}.html"
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    trace_slot_path = (PROJECT_ROOT / "runs" / f"live_trace_{args.trace_slot}.html"
                       if args.trace_slot else None)
    field = _load_field()
    print(f"CYCLE_MONITOR funnel=/trace?run={args.slot} traces=/trace?run={args.trace_slot} "
          f"window={args.window} interval={args.interval}s feed={TEL_PATH}", flush=True)
    while True:
        recs = tail_records(TEL_PATH, args.window)
        html = render(recs, browser_refresh)
        slot_path.write_text(html, encoding="utf-8")
        args.out.write_text(html, encoding="utf-8")
        if trace_slot_path is not None:
            render_traces(recs, field, trace_slot_path, browser_refresh)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
