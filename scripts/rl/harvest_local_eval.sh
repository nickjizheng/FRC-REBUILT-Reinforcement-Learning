#!/usr/bin/env bash
# LOCAL champion-harvest loop (run in Git Bash on the laptop).
#
# The server trains and drops a snapshot into runs/distB/eval_queue every 5000
# updates. Server-side evals are SLOW and disturb training, so the gate runs
# HERE: pull new snapshots -> deterministic 8-episode eval on the laptop GPU ->
# leaderboard vs the promoted champion (998753 @ 41.5 mean / 48 median).
# Promote a snapshot only when it beats the champion on mean_scored.
#
# Usage:  bash scripts/rl/harvest_local_eval.sh          # one pass
#         bash scripts/rl/harvest_local_eval.sh --loop   # poll every 10 min
set -u
cd "$(dirname "$0")/../.."   # repo root

PY=${PY:-/c/il/venv/Scripts/python.exe}
SSH_PORT=${SSH_PORT:-22}
HOST=${HOST:?Set HOST to the SSH destination (for example, user@training-host)}
KEY=${KEY:-~/.ssh/id_ed25519}
SNAPDIR=runs/snapshots
CHAMPION_MEAN=41.5           # stageC_champion_998753_eval.json
mkdir -p "$SNAPDIR"

one_pass() {
  echo "== pull new eval_queue snapshots =="
  remote=$(ssh -p $SSH_PORT -i $KEY -o BatchMode=yes -o ConnectTimeout=15 $HOST \
    'ls /root/runs/distB/eval_queue/step_*.pt 2>/dev/null' | tr -d "\r")
  for rp in $remote; do
    f=$(basename "$rp")
    [ -f "$SNAPDIR/$f" ] && continue
    echo "  pulling $f"
    scp -P $SSH_PORT -i $KEY -o BatchMode=yes "$HOST:$rp" "$SNAPDIR/$f" || continue
  done

  echo "== eval new snapshots locally (deterministic, 8 eps, under-trench) =="
  for ck in "$SNAPDIR"/step_*.pt; do
    [ -e "$ck" ] || continue
    out="${ck%.pt}_eval.json"
    [ -f "$out" ] && continue
    echo "  eval $(basename "$ck") ..."
    "$PY" scripts/rl/eval_checkpoint.py --checkpoint "$ck" \
      --spawn-under-trench --episodes 8 --num-envs 2 --episode-len-s 90 \
      --template assets/rl/env_template_200.usd --policies checkpoint \
      --out "$out" > "${ck%.pt}_eval.log" 2>&1
    if [ ! -f "$out" ]; then
      # Isaac RTX cold-render crash is a dice roll on this laptop -- retry once.
      echo "    no result (likely cold-render crash) -- retrying once"
      "$PY" scripts/rl/eval_checkpoint.py --checkpoint "$ck" \
        --spawn-under-trench --episodes 8 --num-envs 2 --episode-len-s 90 \
        --template assets/rl/env_template_200.usd --policies checkpoint \
        --out "$out" >> "${ck%.pt}_eval.log" 2>&1
    fi
  done

  echo "== leaderboard (champion 998753 = ${CHAMPION_MEAN}) =="
  "$PY" - <<'PYEOF'
import json, glob, re
rows = []
for p in sorted(glob.glob("runs/snapshots/step_*_eval.json")):
    try:
        d = json.load(open(p))["checkpoint"]
        step = int(re.search(r"step_(\d+)", p).group(1))
        rows.append((step, d["mean_scored"], d["max_scored"],
                     d["pct_episodes_scored"], [e["scored"] for e in d["per_episode"]]))
    except Exception:
        pass
rows.sort()
champ = 41.5
for step, mean, mx, pct, per in rows:
    mark = "  <-- BEATS CHAMPION, promote" if mean > champ else ""
    print(f"  step {step:>9,}  mean {mean:5.1f}  max {mx:2d}  scored% {pct:5.1f}  {per}{mark}")
if not rows:
    print("  (no snapshot evals yet)")
PYEOF
}

if [ "${1:-}" = "--loop" ]; then
  while true; do one_pass; echo "-- sleeping 600s --"; sleep 600; done
else
  one_pass
fi
