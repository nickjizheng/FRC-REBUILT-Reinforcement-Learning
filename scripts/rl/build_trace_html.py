"""Render a deterministic route JSON (from eval_route.py) into a live,
auto-refreshing trace page for the dashboard's /trace route.

Zoomed to the blue-half crossing region (the robot + the board line + the hub),
path coloured by zone (neutral gray / own-court green), with an a/b/c run selector.
Usage: build_trace_html.py <route.json> <out.html> [refresh_s]
  refresh_s: <meta refresh> seconds; 0 omits it (used when this page is inlined as an
  iframe tile in a gallery that already handles reloads -> avoids double-refresh churn).
"""
import json
import sys
from datetime import datetime

BOARD_Y = -2.775  # blue own-court starts past this (NEUTRAL_ZONE_HALF_Y_M)


def main() -> None:
    route_path, out_path = sys.argv[1], sys.argv[2]
    refresh_s = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    meta_refresh = f'<meta http-equiv="refresh" content="{refresh_s}">' if refresh_s > 0 else ""
    refresh_note = (f" (page auto-reloads every {refresh_s}&nbsp;s)" if refresh_s > 0 else "")
    d = json.load(open(route_path))
    ep = d["episodes"][0]
    route = ep["route"]
    fuel = d["field"]["fuel_home"]
    tr = d["field"]["trench"]
    hub = d["field"]["hub"]
    run = (d.get("label", "sB_?") or "").replace("sB_", "") or "?"

    rx = [p[0] for p in route]
    ry = [p[1] for p in route]
    # Zoom to the blue-half crossing region: the path + the board + the hub + a
    # margin, full field width. Excludes the far red half so the path isn't a
    # speck in a 17 m field; expands automatically if the path goes deep.
    miny = min(ry + [hub[1], tr["y_alliance"], BOARD_Y]) - 1.3
    maxy = max(ry + [BOARD_Y + 0.6]) + 1.3
    minx, maxx = -4.2, 4.2
    W = 440.0
    H = round(W * (maxy - miny) / (maxx - minx), 1)

    def tx(x):
        return round((x - minx) / (maxx - minx) * W, 1)

    def ty(y):
        return round((maxy - y) / (maxy - miny) * H, 1)

    S = [f'<svg viewBox="0 0 {W} {H}" class="field" role="img">']
    S.append(f'<rect x="0" y="{ty(BOARD_Y)}" width="{W}" height="{max(0, H - ty(BOARD_Y))}" class="court"/>')
    S.append(f'<line x1="0" y1="{ty(BOARD_Y)}" x2="{W}" y2="{ty(BOARD_Y)}" class="board"/>')
    S.append(f'<text x="7" y="{ty(BOARD_Y) - 6}" class="zlabel">&#9650; NEUTRAL ZONE (spawn/collect)</text>')
    S.append(f'<text x="7" y="{ty(BOARD_Y) + 14}" class="zlabel">&#9660; OWN COURT &mdash; cross the board to score</text>')
    S.append(f'<rect x="{tx(tr["x_min"])}" y="{ty(tr["y_neutral"])}" width="{tx(tr["x_max"])-tx(tr["x_min"])}" '
             f'height="{ty(tr["y_alliance"])-ty(tr["y_neutral"])}" class="trench"/>')
    for p in fuel:
        if miny <= p[1] <= maxy:
            S.append(f'<circle cx="{tx(p[0])}" cy="{ty(p[1])}" r="2" class="fuel"/>')
    S.append(f'<circle cx="{tx(hub[0])}" cy="{ty(hub[1])}" r="7" class="hub"/>')
    S.append(f'<circle cx="{tx(hub[0])}" cy="{ty(hub[1])}" r="12" class="hubring"/>')
    S.append(f'<text x="{tx(hub[0])}" y="{ty(hub[1])+24}" class="flabel" text-anchor="middle">HUB</text>')
    run_pts = []
    prev = None
    def flush(pts, own):
        if len(pts) < 2:
            return ""
        pl = " ".join(f"{tx(p[0])},{ty(p[1])}" for p in pts)
        return f'<polyline points="{pl}" class="path {"own" if own else "neutral"}"/>'
    for p in route:
        own = p[1] < BOARD_Y
        if prev is None or own == prev:
            run_pts.append(p)
        else:
            S.append(flush(run_pts, prev)); run_pts = [run_pts[-1], p]
        prev = own
    S.append(flush(run_pts, prev))
    for i in range(1, len(route)):
        if route[i][4] > route[i-1][4]:
            S.append(f'<circle cx="{tx(route[i][0])}" cy="{ty(route[i][1])}" r="3" class="score"/>')
    s, e = route[0], route[-1]
    S.append(f'<circle cx="{tx(s[0])}" cy="{ty(s[1])}" r="5" class="start"/>')
    S.append(f'<rect x="{tx(e[0])-4}" y="{ty(e[1])-4}" width="8" height="8" class="end"/>')
    S.append("</svg>")
    svg = "\n".join(S)

    crossed = any(p[1] < BOARD_Y for p in route)
    now = datetime.now().astimezone().strftime("%H:%M:%S")
    tabs = "".join(
        f'<a href="/trace?run={r}" class="tab{" on" if r == run else ""}">{r}</a>'
        for r in ("a", "b", "c")
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
{meta_refresh}<title>trace {run}</title>
<style>
:root {{ --bg:#0b0f14; --panel:#111820; --field:#0d141c; --ink:#e6edf3; --muted:#8b98a5;
  --line:#1e2a36; --fuel:#f0b429; --hub:#38bdf8; --trench:#ef4444; --board:#f472b6;
  --ok:#34d399; --neutral:#94a3b8; --own:#34d399; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:system-ui,-apple-system,Segoe UI,sans-serif; }}
.wrap {{ max-width:720px; margin:0 auto; padding:18px 16px 40px; }}
.top {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:4px; }}
h1 {{ font-size:18px; margin:0; }}
.tabs {{ display:flex; gap:4px; }}
.tab {{ font:700 13px/1 ui-monospace,monospace; text-decoration:none; color:var(--muted);
  border:1px solid var(--line); border-radius:6px; padding:6px 12px; }}
.tab.on {{ color:var(--bg); background:var(--ok); border-color:var(--ok); }}
.meta {{ color:var(--muted); font:12px/1.5 ui-monospace,monospace; margin:6px 0 12px; }}
.meta b {{ color:var(--ink); }}
.stats {{ display:flex; gap:12px; flex-wrap:wrap; margin:0 0 12px; font:12px/1 ui-monospace,monospace; }}
.stats div {{ background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:8px 12px; }}
.stats b {{ color:var(--ink); font-size:16px; font-variant-numeric:tabular-nums; }} .stats span {{ color:var(--muted); }}
.field {{ width:100%; height:auto; background:var(--field); border:1px solid var(--line); border-radius:9px; display:block; }}
.court {{ fill:color-mix(in srgb, var(--own) 8%, transparent); }}
.board {{ stroke:var(--board); stroke-width:2.2; stroke-dasharray:7 4; }}
.trench {{ fill:color-mix(in srgb,var(--trench) 14%,transparent); stroke:var(--trench); stroke-width:1; stroke-dasharray:3 3; }}
.fuel {{ fill:var(--fuel); opacity:.5; }}
.hub {{ fill:none; stroke:var(--hub); stroke-width:2.5; }} .hubring {{ fill:none; stroke:var(--hub); stroke-width:1; opacity:.4; }}
.flabel {{ fill:var(--muted); font:600 9px/1 ui-monospace,monospace; letter-spacing:.1em; }}
.zlabel {{ fill:var(--board); font:600 9px/1 ui-monospace,monospace; letter-spacing:.06em; }}
.path {{ fill:none; stroke-width:2.4; stroke-linejoin:round; stroke-linecap:round; }}
.path.neutral {{ stroke:var(--neutral); }} .path.own {{ stroke:var(--own); }}
.score {{ fill:var(--ok); stroke:var(--bg); stroke-width:.6; }}
.start {{ fill:var(--ok); stroke:var(--field); stroke-width:1.5; }} .end {{ fill:#fb7185; }}
.legend {{ margin-top:11px; font:11.5px/1.9 ui-monospace,monospace; color:var(--muted); display:flex; gap:15px; flex-wrap:wrap; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.sw {{ width:15px; height:0; border-top-width:2.4px; border-top-style:solid; display:inline-block; }}
</style></head><body><div class="wrap">
  <div class="top"><h1>Live trace</h1><div class="tabs">{tabs}</div></div>
  <div class="meta">run <b>sB_{run}</b> @ <b>{d.get('train_steps','?')}</b> steps &middot; updated <b>{now}</b>{refresh_note}</div>
  <div class="stats">
    <div><span>scored</span><br><b>{ep['scored']}</b></div>
    <div><span>collected</span><br><b>{ep['collected']}</b></div>
    <div><span>reached own court</span><br><b style="color:{'var(--own)' if crossed else 'var(--muted)'}">{'yes' if crossed else 'not yet'}</b></div>
  </div>
  {svg}
  <div class="legend">
    <span><i class="sw" style="border-color:var(--neutral)"></i> path in neutral</span>
    <span><i class="sw" style="border-color:var(--own)"></i> path in own court</span>
    <span><i class="sw" style="border-color:var(--board);border-top-style:dashed"></i> board</span>
    <span style="color:var(--ok)">&#9679; scored</span>
    <span>&#9679; start &middot; &#9632; end</span>
  </div>
</div></body></html>"""
    open(out_path, "w", encoding="utf-8").write(html)
    print(f"wrote {out_path} run={run} scored={ep['scored']} crossed={crossed}")


if __name__ == "__main__":
    main()
