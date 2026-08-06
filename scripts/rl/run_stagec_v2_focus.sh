#!/bin/bash
# Focused Stage C v2 consolidation: one learner + one two-env RETURN collector.
#
# Usage:
#   run_stagec_v2_focus.sh CANDIDATE.pt PREFIX.pt ANCHOR_DIR CURATED_ELITES \
#       [MINUTES] [TEMPLATE.usd] [OUT]
set -euo pipefail

CANDIDATE=${1:?candidate Stage C v2 checkpoint}
PREFIX=${2:?immutable 22-proprio prefix checkpoint}
ANCHOR_DIR=${3:?champion anchor directory}
SEEDMINE_ELITES=${4:?checkpoint-specific curated seed-mine directory}
MINUTES=${5:-180}
TEMPLATE=${6:-/root/frc-rl/assets/rl/env_template_200.usd}
RUN_ID=${STAGEC_V2_FOCUS_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${7:-/root/autodl-tmp/runs/stagec_v2_focus_${RUN_ID}}
ROOT=${STAGEC_V2_FOCUS_ROOT:-/dev/shm/frc_stagec_v2_focus_${RUN_ID}}
STOP=${STAGEC_V2_FOCUS_STOP:-/root/STOP_STAGEC_V2_FOCUS_${RUN_ID}}
CODE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DASHBOARD_RUNS=${STAGEC_V2_DASHBOARD_RUNS:-$CODE_ROOT/runs}
LEARNER_GPU=${STAGEC_V2_FOCUS_LEARNER_GPU:-3}
COLLECTOR_GPU=${STAGEC_V2_FOCUS_COLLECTOR_GPU:-2}

if [[ ! "$MINUTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MINUTES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$LEARNER_GPU" =~ ^[0-9]+$ || ! "$COLLECTOR_GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU ids must be non-negative integers" >&2
  exit 2
fi
if [ "$LEARNER_GPU" = "$COLLECTOR_GPU" ]; then
  echo "focused learner and collector must use separate GPUs" >&2
  exit 2
fi
for path in "$CANDIDATE" "$PREFIX" "$TEMPLATE"; do
  [ -f "$path" ] || { echo "missing file: $path" >&2; exit 2; }
done
[ -d "$ANCHOR_DIR" ] || { echo "missing anchor directory: $ANCHOR_DIR" >&2; exit 2; }
[ -d "$SEEDMINE_ELITES" ] || { echo "missing curated elite directory: $SEEDMINE_ELITES" >&2; exit 2; }
[ -f "$SEEDMINE_ELITES/manifest.json" ] || {
  echo "curated elite directory has no manifest.json" >&2
  exit 2
}
[ ! -e "$OUT" ] || { echo "refusing existing output: $OUT" >&2; exit 2; }
[ ! -e "$ROOT" ] || { echo "refusing existing transport root: $ROOT" >&2; exit 2; }
if pgrep -af '[l]earner_cycle_v2|[c]ollector_cycle_v2'; then
  echo "refusing to overlap another Stage C v2 train process" >&2
  exit 2
fi

# A fresh seed is recorded once and then held stable across watchdog restarts.
# It is deliberately unrelated to the seed-mining environment/action seeds.
if [ -n "${STAGEC_V2_FOCUS_SEED:-}" ]; then
  FOCUS_SEED=$STAGEC_V2_FOCUS_SEED
else
  RAW_SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
  FOCUS_SEED=$((RAW_SEED % 2000000000 + 1))
fi
if [[ ! "$FOCUS_SEED" =~ ^[0-9]+$ ]] || [ "$FOCUS_SEED" -le 0 ]; then
  echo "STAGEC_V2_FOCUS_SEED must be a positive integer" >&2
  exit 2
fi
LEARNER_SEED=$FOCUS_SEED
COLLECTOR_SEED=$((FOCUS_SEED + 1))

TARGET_LOAD=15
RESERVE_COUNT=18
RESERVE_BATCHES=1
MAX_DUMP_TICKS=180
STDDEV_START=1.0
STDDEV_END=0.30
STDDEV_STEPS=150000
CYCLE_SCORE_FRACTION=0.75
CYCLE_SCORE_FLOOR=6
COLLECT_WEIGHT=0.30
PROGRESS_PER_M=5.0
PROGRESS_STEP_CAP=0.75
RAMP_BONUS=6.0
LEAVE_GRACE_STEPS=5
LEAVE_PENALTY_PER_STEP=0.03
LEAVE_PENALTY_CAP=5.0
RETURN_GRACE_STEPS=10
RETURN_PENALTY_PER_STEP=0.02
RETURN_PENALTY_CAP=5.0
SHOOT_GRACE_S=2.0
SHOOT_PENALTY_PER_STEP=0.05
SHOOT_PENALTY_CAP=5.0
DUMP_LOST_AIM_GRACE_TICKS=15
PARTIAL_DUMP_PENALTY_PER_BALL=0.5
PARTIAL_DUMP_PENALTY_CAP=15.0

cd "$CODE_ROOT"
source /root/venv/bin/activate
source setup_render_env.sh >/dev/null 2>&1
PYTHON=${STAGEC_V2_FOCUS_PYTHON:-python}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1

mkdir -p "$ROOT" "$OUT" "$DASHBOARD_RUNS"
echo $$ > "$OUT/launcher.pid"
rm -f "$STOP"

CANDIDATE_SHA=$(sha256sum "$CANDIDATE" | awk '{print $1}')
PREFIX_SHA=$(sha256sum "$PREFIX" | awk '{print $1}')
"$PYTHON" - "$SEEDMINE_ELITES" "$CANDIDATE_SHA" "$PREFIX_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

elite_dir = Path(sys.argv[1]).resolve()
candidate_sha, prefix_sha = sys.argv[2:]
manifest_path = elite_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "stagec_seedmine_curated_v1":
    raise SystemExit("curated manifest schema mismatch")
if manifest.get("checkpoint_sha256") != candidate_sha:
    raise SystemExit("curated manifest checkpoint mismatch")
if manifest.get("prefix_sha256") != prefix_sha:
    raise SystemExit("curated manifest prefix mismatch")
accepted_count = manifest.get("accepted_count")
accepted = manifest.get("accepted")
if (
    isinstance(accepted_count, bool)
    or not isinstance(accepted_count, int)
    or accepted_count < 1
    or not isinstance(accepted, list)
    or len(accepted) != accepted_count
):
    raise SystemExit("curated manifest contains no clean cycles")
listed = set()
for item in accepted:
    if not isinstance(item, dict):
        raise SystemExit("curated manifest accepted entry is invalid")
    name = item.get("filename")
    wanted_sha = item.get("sha256")
    provenance = item.get("provenance")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".npz")
        or name in listed
    ):
        raise SystemExit("curated manifest filename is invalid or duplicated")
    if (
        not isinstance(wanted_sha, str)
        or len(wanted_sha) != 64
        or any(char not in "0123456789abcdef" for char in wanted_sha)
    ):
        raise SystemExit(f"curated manifest hash is invalid: {name}")
    if not isinstance(provenance, dict) or provenance.get(
        "checkpoint_sha256"
    ) != candidate_sha:
        raise SystemExit(f"curated manifest provenance mismatch: {name}")
    archive = elite_dir / name
    if not archive.is_file():
        raise SystemExit(f"curated archive is missing: {name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != wanted_sha:
        raise SystemExit(f"curated archive hash mismatch: {name}")
    listed.add(name)
actual = {path.name for path in elite_dir.glob("*.npz")}
if actual != listed:
    raise SystemExit("curated directory archive set disagrees with manifest")
PY

cp "$SEEDMINE_ELITES/manifest.json" "$OUT/seedmine_manifest.json"
"$PYTHON" - "$OUT/launcher_config.json" <<PY
import json
import sys

json.dump({
    "run_id": ${RUN_ID@Q},
    "code_root": ${CODE_ROOT@Q},
    "root": ${ROOT@Q},
    "out": ${OUT@Q},
    "candidate": ${CANDIDATE@Q},
    "candidate_sha256": ${CANDIDATE_SHA@Q},
    "prefix_checkpoint": ${PREFIX@Q},
    "prefix_sha256": ${PREFIX_SHA@Q},
    "anchor_dir": ${ANCHOR_DIR@Q},
    "seedmine_elites": ${SEEDMINE_ELITES@Q},
    "template": ${TEMPLATE@Q},
    "minutes": int(${MINUTES@Q}),
    "learner_gpu": int(${LEARNER_GPU@Q}),
    "collector_gpu": int(${COLLECTOR_GPU@Q}),
    "learner_seed": int(${LEARNER_SEED@Q}),
    "collector_seed": int(${COLLECTOR_SEED@Q}),
    "stream_groups": ["return", "return"],
    "learning_rate": 1e-5,
    "updates_per_tx": 0.5,
    "elite_replay_fraction": 0.20,
    "elite_consolidation_updates": 10000,
}, open(sys.argv[1], "w", encoding="utf-8"), indent=2, sort_keys=True)
PY

LPID=""
CPID=""
LSTART=0
CSTART=0
LFAIL=0
CFAIL=0

launch_learner() {
  local resume="$CANDIDATE"
  if [ -s "$OUT/latest.pt" ]; then resume="$OUT/latest.pt"; fi
  CUDA_VISIBLE_DEVICES="$LEARNER_GPU" setsid "$PYTHON" scripts/rl/learner_cycle_v2.py \
    --root "$ROOT" --num-collectors 1 --collector-envs 2 \
    --stream-groups return,return --group-weights return=1.0 \
    --resume "$resume" --prefix-checkpoint "$PREFIX" \
    --anchor-dir "$ANCHOR_DIR" --out "$OUT" \
    --minutes "$MINUTES" --batch-size 256 --learning-rate 1e-5 \
    --updates-per-tx .5 --replay-capacity 100000 \
    --gamma 0.999 --n-step 3 --critic-only-updates 5000 \
    --stddev-start "$STDDEV_START" --stddev-end "$STDDEV_END" \
    --stddev-steps "$STDDEV_STEPS" --initial-stddev .50 \
    --suffix-alpha 1.0 --seed "$LEARNER_SEED" \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" --collect-weight "$COLLECT_WEIGHT" \
    --progress-per-m "$PROGRESS_PER_M" --progress-step-cap "$PROGRESS_STEP_CAP" \
    --ramp-bonus "$RAMP_BONUS" --leave-grace-steps "$LEAVE_GRACE_STEPS" \
    --leave-penalty-per-step "$LEAVE_PENALTY_PER_STEP" \
    --leave-penalty-cap "$LEAVE_PENALTY_CAP" \
    --return-grace-steps "$RETURN_GRACE_STEPS" \
    --return-penalty-per-step "$RETURN_PENALTY_PER_STEP" \
    --return-penalty-cap "$RETURN_PENALTY_CAP" \
    --shoot-grace-s "$SHOOT_GRACE_S" \
    --shoot-penalty-per-step "$SHOOT_PENALTY_PER_STEP" \
    --shoot-penalty-cap "$SHOOT_PENALTY_CAP" \
    --dump-lost-aim-grace-ticks "$DUMP_LOST_AIM_GRACE_TICKS" \
    --partial-dump-penalty-per-ball "$PARTIAL_DUMP_PENALTY_PER_BALL" \
    --partial-dump-penalty-cap "$PARTIAL_DUMP_PENALTY_CAP" \
    --elite-dir "$OUT/live_elite_episodes" \
    --elite-replay-fraction .20 --elite-replay-capacity 40000 \
    --elite-consolidation-updates 10000 \
    --seedmine-elite-dir "$SEEDMINE_ELITES" \
    --seedmine-source-checkpoint "$CANDIDATE" \
    >> "$OUT/learner.log" 2>&1 &
  LPID=$!
  LSTART=$(date +%s)
  echo "$(date '+%F %T') focus learner GPU$LEARNER_GPU pid=$LPID resume=$resume"
}

launch_collector() {
  sleep 5
  CUDA_VISIBLE_DEVICES="$COLLECTOR_GPU" setsid "$PYTHON" scripts/rl/collector_cycle_v2.py \
    --collector-id 0 --root "$ROOT" --num-envs 2 \
    --stagec-v2-prefix-checkpoint "$PREFIX" --reset-modes return,return \
    --template "$TEMPLATE" --episode-len-s 75 \
    --target-load "$TARGET_LOAD" --reserve-count "$RESERVE_COUNT" \
    --reserve-batches "$RESERVE_BATCHES" --collect-weight "$COLLECT_WEIGHT" \
    --dump-on-press --max-dump-ticks "$MAX_DUMP_TICKS" \
    --cycle-score-fraction "$CYCLE_SCORE_FRACTION" \
    --cycle-score-floor "$CYCLE_SCORE_FLOOR" \
    --progress-per-m "$PROGRESS_PER_M" --progress-step-cap "$PROGRESS_STEP_CAP" \
    --ramp-bonus "$RAMP_BONUS" --leave-grace-steps "$LEAVE_GRACE_STEPS" \
    --leave-penalty-per-step "$LEAVE_PENALTY_PER_STEP" \
    --leave-penalty-cap "$LEAVE_PENALTY_CAP" \
    --return-grace-steps "$RETURN_GRACE_STEPS" \
    --return-penalty-per-step "$RETURN_PENALTY_PER_STEP" \
    --return-penalty-cap "$RETURN_PENALTY_CAP" \
    --shoot-grace-s "$SHOOT_GRACE_S" \
    --shoot-penalty-per-step "$SHOOT_PENALTY_PER_STEP" \
    --shoot-penalty-cap "$SHOOT_PENALTY_CAP" \
    --dump-lost-aim-grace-ticks "$DUMP_LOST_AIM_GRACE_TICKS" \
    --partial-dump-penalty-per-ball "$PARTIAL_DUMP_PENALTY_PER_BALL" \
    --partial-dump-penalty-cap "$PARTIAL_DUMP_PENALTY_CAP" \
    --stddev-start "$STDDEV_START" --stddev-end "$STDDEV_END" \
    --stddev-steps "$STDDEV_STEPS" --seed "$COLLECTOR_SEED" \
    --minutes "$MINUTES" --telemetry "$OUT/cycle_telemetry.jsonl" \
    >> "$OUT/collector0.log" 2>&1 &
  CPID=$!
  CSTART=$(date +%s)
  echo "$(date '+%F %T') focus collector GPU$COLLECTOR_GPU pid=$CPID modes=return,return"
}

cleanup() {
  echo "$(date '+%F %T') Stage C v2 focus cleanup"
  [ -n "${LPID:-}" ] && kill "$LPID" 2>/dev/null || true
  [ -n "${CPID:-}" ] && kill "$CPID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

launch_learner
for _ in $(seq 1 180); do
  if grep -q 'LEARNER_V2_READY' "$OUT/learner.log" 2>/dev/null; then break; fi
  kill -0 "$LPID" 2>/dev/null || { echo "focus learner failed during startup" >&2; exit 1; }
  sleep 1
done
grep -q 'LEARNER_V2_READY' "$OUT/learner.log" || {
  echo "focus learner did not become ready in 180 seconds" >&2
  exit 1
}
launch_collector

ln -sfn "$OUT" "$DASHBOARD_RUNS/stagec_v2_focus_latest"
echo "$(date '+%F %T') Stage C v2 focus launched; watchdog active"

END=$(( $(date +%s) + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if [ -f "$STOP" ]; then
    echo "$(date '+%F %T') focus stop file detected"
    exit 0
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    now=$(date +%s)
    if [ $((now - LSTART)) -lt 600 ]; then LFAIL=$((LFAIL + 1)); else LFAIL=1; fi
    if [ "$LFAIL" -ge 3 ]; then
      echo "focus learner failed 3 times inside ten-minute windows" | tee "$OUT/LAUNCH_FAILED"
      exit 1
    fi
    sleep 5
    launch_learner
  fi
  if ! kill -0 "$CPID" 2>/dev/null; then
    now=$(date +%s)
    if [ $((now - CSTART)) -lt 600 ]; then CFAIL=$((CFAIL + 1)); else CFAIL=1; fi
    if [ "$CFAIL" -ge 3 ]; then
      echo "focus collector failed 3 times inside ten-minute windows" | tee "$OUT/LAUNCH_FAILED"
      exit 1
    fi
    launch_collector
  fi
  sleep 20
done

echo "$(date '+%F %T') Stage C v2 focus duration complete"
