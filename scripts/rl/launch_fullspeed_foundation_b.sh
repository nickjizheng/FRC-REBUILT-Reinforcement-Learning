#!/bin/bash
# Stage-D full-speed scoring-foundation run.
#
# This deliberately removes the incomplete post-dump lane and the currently
# unlearned ferry/own-court subsystem.  It retains the integrated four-window
# cycle program with exact-source behavior custody, slows actor drift, and
# restores the validated frozen-prefix rescue.
set -eu

RESUME=${1:?verified frozen checkpoint}
TEACHERS=${2:?exact-source four-window teacher directory}
REPORT=${3:?teacher/evaluation provenance report}
RID=${4:-fsfoundation_$(date +%Y%m%d_%H%M%S)}

CODE_ROOT=${CODE_ROOT:-/root/autodl-tmp/frc_staged_v1_20260722}
PREFIX=/root/preserved/stageC_highest_1163753.pt
ANCHORS=/root/autodl-tmp/archives/stagec_v23_ft165k_20260714_1932_final_20260714_214228/inputs/anchors
TEMPLATE=$CODE_ROOT/assets/rl/env_template_456.usd
OUT=/root/autodl-tmp/runs/stage_blue2_${RID}
TEACHER_MIN_COLLECTED=${TEACHER_MIN_COLLECTED:-180}
TEACHER_MIN_REPEAT_SCORED_LOAD_MAX=${TEACHER_MIN_REPEAT_SCORED_LOAD_MAX:-32}

case "$TEACHER_MIN_COLLECTED:$TEACHER_MIN_REPEAT_SCORED_LOAD_MAX" in
  *[!0-9:]*) echo "teacher minimums must be non-negative integers" >&2; exit 2 ;;
esac
export TEACHER_MIN_COLLECTED TEACHER_MIN_REPEAT_SCORED_LOAD_MAX

test -s "$RESUME"
test -s "$PREFIX"
test -s "$TEMPLATE"
test -d "$TEACHERS"
test -s "$REPORT"
test "$(find "$TEACHERS" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 5
test "$(tr -d '[:space:]' < /root/policy_speed_scale.txt)" = "1.0"
if pgrep -f '[l]earner_cycle_v2.py|[c]ollector_cycle_v2.py|[e]val_stagec_seedmine.py' >/dev/null; then
  echo "refusing launch: training or frozen evaluation is active" >&2
  exit 4
fi
if [ "$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')" -lt 5242880 ]; then
  echo "refusing launch: less than 5 GiB free on /root/autodl-tmp" >&2
  exit 5
fi

# Fail closed on exact source and all-window teacher custody.  These checks use
# the same embedded metadata that the learner validates before publishing.
/root/venv/bin/python - "$RESUME" "$TEACHERS" <<'PY'
import glob, hashlib, json, math, os, sys
from pathlib import Path

import numpy as np

checkpoint = Path(sys.argv[1]).resolve()
teachers = Path(sys.argv[2]).resolve()
source_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
rows = []
for path_text in sorted(glob.glob(str(teachers / "*.npz"))):
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["metadata"]).decode("utf-8"))
    episode = meta.get("episode", {})
    if episode.get("checkpoint_sha256") != source_sha:
        raise SystemExit(f"teacher source mismatch: {path}")
    if episode.get("action_mode") != "deterministic" or episode.get("mode") != "full":
        raise SystemExit(f"teacher custody mismatch: {path}")
    if episode.get("terminal_reason") != "horizon":
        raise SystemExit(f"teacher is not a clean horizon: {path}")
    if float(episode.get("episode_len_s", 0.0) or 0.0) < 159.0:
        raise SystemExit(f"teacher is not a full 160-second match: {path}")
    contract = episode.get("stage_d_contract")
    expected_contract = {
        "stage_d": True,
        "first_inactive": "blue",
        "ferry": False,
        "return_when_live": False,
        "owncourt_loop": False,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) != value for key, value in expected_contract.items()
    ):
        raise SystemExit(f"teacher environment contract mismatch: {path} {contract}")
    if not math.isclose(float(contract.get("prefix_rescue_s", -1)), 35.0):
        raise SystemExit(f"teacher prefix-rescue contract mismatch: {path} {contract}")
    if not math.isclose(float(contract.get("policy_speed_scale", -1)), 1.0):
        raise SystemExit(f"teacher speed contract mismatch: {path} {contract}")
    if any(
        int(episode.get(key, 0) or 0) != 0
        for key in (
            "ferried_balls",
            "owncourt_score_entries",
            "owncourt_shots",
            "owncourt_scored",
        )
    ):
        raise SystemExit(f"teacher used deferred ferry/own-court mechanics: {path}")
    timeline = [
        event for event in episode.get("timeline", [])
        if isinstance(event, dict)
    ]
    score_times = [
        float(event.get("t")) for event in timeline
        if event.get("ev") == "score" and event.get("t") is not None
    ]
    first_score_t = min(score_times) if score_times else None
    # The exact frozen opener can press its legacy ferry head before its first
    # score even with suffix ferry disabled.  Reject actual auxiliary counters,
    # every own-court entry, and ferry presses once scoring (hence the handoff)
    # has begun; do not misclassify immutable prefix actions as suffix usage.
    if any(
        event.get("ev") == "oc_entry"
        or (
            event.get("ev") == "ferry"
            and (
                first_score_t is None
                or float(event.get("t", -1.0)) >= first_score_t
            )
        )
        for event in timeline
    ):
        raise SystemExit(f"teacher timeline used deferred mechanics: {path}")
    windows = [0, 0, 0, 0]
    for event in episode.get("timeline", []):
        if event.get("ev") != "score" or not event.get("elig", True):
            continue
        t = float(event.get("t", -1))
        value = int(event.get("q", 0) or 0) + int(event.get("u", 0) or 0)
        if 0 <= t < 33:
            windows[0] += value
        elif 55 <= t < 83:
            windows[1] += value
        elif 105 <= t < 130:
            windows[2] += value
        elif 130 <= t <= 160:
            windows[3] += value
    metrics = {
        "score": int(episode.get("scored", 0) or 0),
        "collected": int(episode.get("collected", 0) or 0),
        "cycles": int(episode.get("cycles_completed", 0) or 0),
        "repeat_scored_load_max": int(
            episode.get("repeat_scored_load_max", 0) or 0
        ),
    }
    minimums = {
        "score": 160,
        "collected": int(os.environ["TEACHER_MIN_COLLECTED"]),
        "cycles": 4,
        "repeat_scored_load_max": int(
            os.environ["TEACHER_MIN_REPEAT_SCORED_LOAD_MAX"]
        ),
    }
    if any(metrics[name] < minimum for name, minimum in minimums.items()):
        raise SystemExit(
            f"teacher misses high-throughput gate: {path} {metrics}"
        )
    if any(value < floor for value, floor in zip(windows, (55, 40, 15, 15))):
        raise SystemExit(f"teacher lacks four-window coverage: {path} {windows}")
    rows.append(
        (
            int(episode.get("env_seed", -1)),
            int(episode.get("env_index", -1)),
            windows,
            metrics,
        )
    )
if (
    len(rows) != 5
    or len({seed for seed, _, _, _ in rows}) < 4
    or len({env for _, env, _, _ in rows}) < 2
):
    raise SystemExit(f"teacher diversity check failed: {rows}")
print(json.dumps({"source_sha256": source_sha, "teachers": rows}, sort_keys=True))
PY

mkdir -p /root/preserved/configs
if test -f /root/blue2_env.sh; then
  cp -p /root/blue2_env.sh "/root/preserved/configs/blue2_env_pre_${RID}.sh"
fi

ALL_FULL=full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full,full
FULL_MODES='full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full;full,full'
ACTION_MODES='mean;explore;mean;explore;mean;explore;mean;explore;mean;explore;mean;explore;mean;explore;mean;explore'
cat > /root/blue2_env.sh <<EOF
export FRC_POLICY_SPEED_FILE=/root/policy_speed_scale.txt
export STAGEC_V2_NCOLL=16
export STAGEC_V2_COLLECTORS_PER_GPU=4
export STAGEC_V2_COLLECTOR_ACTION_MODES='$ACTION_MODES'
export STAGEC_V2_RESET_SCHEDULES=1
export STAGEC_V2_RESET_OPTIMIZER_STATE=0
export STAGEC_V2_ALLOW_STDDEV_SCHEDULE_MIGRATION=1
export STAGEC_V2_ALLOW_TARGET_LOAD_MIGRATION=0
export STAGEC_V2_ALLOW_SUFFIX_ALPHA_MIGRATION=0
export STAGEC_V2_ALLOW_COLLECT_STALL_MIGRATION=0
export STAGEC_V2_CRITIC_ONLY_UPDATES=8000
export STAGEC_V2_LEARNING_RATE=0.000005
export STAGEC_V2_GAMMA=0.9997
export STAGEC_V2_STDDEV_START=1.0
export STAGEC_V2_STDDEV_END=0.30
export STAGEC_V2_INITIAL_STDDEV=0.30
export STAGEC_V2_STDDEV_STEPS=150000
export STAGEC_V2_ANCHOR_BETA_START=0.15
export STAGEC_V2_ANCHOR_BETA_FLOOR=0.10
export STAGEC_V2_ANCHOR_DECAY_UPDATES=80000
export STAGEC_V2_ACTOR_UPDATE_INTERVAL=4
export STAGEC_V2_ACTOR_PHASES=leave,collect,return,score
export STAGEC_V2_STREAM_GROUPS=$ALL_FULL
export STAGEC_V2_GROUP_WEIGHTS=full=1.0
export STAGEC_V2_COLLECTOR_MODES='$FULL_MODES'
export STAGEC_V2_FULL_EPISODE_S=160
export STAGEC_V2_STAGE_D=1
export STAGEC_V2_STAGE_D_FIRST_INACTIVE=blue
export STAGEC_V2_STAGE_D_PRELOAD=0
export STAGEC_V2_STAGE_D_FERRY=0
export STAGEC_V2_STAGE_D_RETURN_WHEN_LIVE=0
export STAGEC_V2_STAGE_D_OWNCOURT_LOOP=0
export STAGEC_V2_STAGE_D_OWNCOURT_INTAKE_REWARD=0
export STAGEC_V2_STAGE_D_ACTIVE_FERRY_PENALTY=0
export STAGEC_V2_POSTDUMP_REQUIRE_TARGET_LOAD=0
export STAGEC_V2_POSTDUMP_COMPLETE_CYCLE=0
export STAGEC_V2_POSTDUMP_DEPLETED_COUNT=0
export STAGEC_V2_POSTDUMP_DEPLETED_PROB=0.0
export STAGEC_V2_SEEDMINE_ELITE_DIR=$TEACHERS
export STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT=$RESUME
export STAGEC_V2_ELITE_BEHAVIOR_SEEDMINE_ONLY=1
export STAGEC_V2_SEEDMINE_BEHAVIOR_ONLY=1
export STAGEC_V2_ELITE_BEHAVIOR_WEIGHT=0.30
export STAGEC_V2_ELITE_BEHAVIOR_BATCH_SIZE=32
export STAGEC_V2_ELITE_BEHAVIOR_SCORE_CAPACITY=5000
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_CAPACITY=2200
export STAGEC_V2_ELITE_BEHAVIOR_TRIGGER_FRACTION=0.25
export STAGEC_V2_ELITE_ARCHIVE_MAX_FILES=32
export STAGEC_V2_TARGET_LOAD=15
export STAGEC_V2_PREFERRED_REPEAT_LOAD=44
export STAGEC_V2_COLLECT_STALL_STEPS=45
export STAGEC_V2_RETURN_TIME_GUARD=0.11
export STAGEC_V2_INTAKE_DURING_RETURN=0
export LEAVE_PENALTY_PER_STEP=0.10
export LEAVE_PENALTY_CAP=40.0
export RETURN_PENALTY_PER_STEP=0.04
export RETURN_PENALTY_CAP=20.0
export STAGEC_V2_REQUIRE_RAMP_OUT=0
export STAGEC_V2_RAMP_OUT_HALF_WIDTH=0.90
export STAGEC_V2_RAMP_OUT_BONUS=24.0
export STAGEC_V2_OFF_RAMP_EXIT_PENALTY=12.0
export STAGEC_V2_OUTER_RAIL_ENTER_X=2.55
export STAGEC_V2_OUTER_RAIL_EXIT_X=2.20
export STAGEC_V2_OUTER_RAIL_MAX_X=3.60
export STAGEC_V2_OUTER_RAIL_GRACE_STEPS=10
export STAGEC_V2_OUTER_RAIL_PENALTY_PER_STEP=0.22
export STAGEC_V2_OUTER_RAIL_PENALTY_CAP=50.0
export STAGEC_V2_OUTER_RAIL_MIN_SCALE=0.35
export STAGEC_V2_OUTER_RAIL_ESCALATION_STEPS=120
export STAGEC_V2_OUTER_RAIL_MAX_MULTIPLIER=3.0
export STAGEC_V2_STAGE_D_LIVE_DUMP_REWARD=0
export STAGEC_V2_STAGE_D_HOME_ARRIVAL_REWARD=0
export STAGEC_V2_STAGE_D_OPENER_TIME_PENALTY=0
export STAGEC_V2_STAGE_D_DEEP_RED_PENALTY=0
export STAGEC_V2_STAGE_D_IDLE_PENALTY=0
export STAGEC_V2_STAGE_D_PREFIX_RESCUE_S=35
export STAGEC_V2_WEIGHT_PUBLISH_UPDATES=400
export STAGEC_V2_FREEZE_COLLECTOR_WEIGHTS=0
export STAGEC_V2_EVAL_SNAPSHOT_UPDATES=5000
export STAGEC_V2_EVAL_QUEUE_MAX_FILES=60
BLUE2_TEMPLATE=$TEMPLATE
BLUE2_CHAMP=$PREFIX
BLUE2_ANCH=$ANCHORS
EOF

mkdir -p "$(dirname "$OUT")"
cp -p /root/blue2_env.sh "/root/preserved/configs/blue2_env_${RID}.sh"
cp -p "$REPORT" "$OUT.teacher_report.json"
printf '%s\n' "$OUT" > /root/blue2_current_out.txt
printf '%s\n' "$OUT" > /root/stage_blue2.outdir
printf '%s\n' "$OUT" > /root/stagec_v2_cycle3.outdir
rm -f /root/STOP_STAGE_BLUE2

cd "$CODE_ROOT"
. /root/blue2_env.sh
export STAGEC_V2_RUN_ID="$RID"
nohup setsid scripts/rl/run_stage_blue2.sh \
  "$PREFIX" "$RESUME" "$ANCHORS" 720 "$TEMPLATE" \
  </dev/null >> "/root/autodl-tmp/runs/sup_${RID}.log" 2>&1 &
echo $! > /root/stage_d_retention_supervisor.pid
echo "FOUNDATION_LAUNCHED out=$OUT supervisor=$(cat /root/stage_d_retention_supervisor.pid)"
