from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "rl" / "run_stagec_v2_cycle3_efficiency.sh"
PILOT = ROOT / "scripts" / "rl" / "run_stagec_v2_score_efficiency_v10_pilot.sh"
LEARNER = ROOT / "scripts" / "rl" / "learner_cycle_v2.py"


def test_guarded_pilot_freezes_candidate_publication_and_binds_capture_source():
    text = PILOT.read_text(encoding="utf-8")
    assert "export STAGEC_V2_FREEZE_COLLECTOR_WEIGHTS=1" in text
    assert "V10 seed-mine source must be the exact initial resume checkpoint" in text
    assert 'sha256sum "$RESUME"' in text
    assert 'sha256sum "$STAGEC_V2_SEEDMINE_SOURCE_CHECKPOINT"' in text


def test_guarded_watchdog_never_restarts_over_a_completed_candidate():
    text = BASE.read_text(encoding="utf-8")
    deadline = text.index('END=$(( $(date +%s) + ${MINUTES%.*} * 60 ))')
    first_launch = text.index("launch_learner", deadline)
    assert deadline < first_launch
    guarded = text.index('if [ "$FREEZE_COLLECTOR_WEIGHTS" = "1" ]; then', first_launch)
    clean_stop = text.index("guarded learner completed cleanly", guarded)
    abort = text.index("guarded learner failed", guarded)
    restart = text.index("restarting from latest", guarded)
    assert guarded < clean_stop < restart
    assert guarded < abort < restart


def test_learner_publishes_validated_resume_once_but_not_guarded_candidates():
    text = LEARNER.read_text(encoding="utf-8")
    startup = text.index(
        "# Publish only after every opt-in archive and provenance check has passed."
    )
    startup_publish = text.index("publish()", startup)
    loop_guard = text.index("not bool(args.freeze_collector_weights)", startup_publish)
    final_guard = text.index(
        "if not bool(args.freeze_collector_weights):", loop_guard + 1
    )
    frozen_message = text.index("LEARNER_V2_COLLECTOR_WEIGHTS_FROZEN", final_guard)
    assert startup < startup_publish < loop_guard < final_guard < frozen_message


def test_camera_contract_is_fail_closed_and_wired_to_every_process():
    text = BASE.read_text(encoding="utf-8")
    assert 'CAMERA_RIG_REVISION=${STAGEC_V2_CAMERA_RIG_REVISION:-unversioned_front_center}' in text
    assert 'TEMPLATE_SHA256=$(sha256sum "$TEMPLATE"' in text
    assert 'CODE_CAMERA_RIG_REVISION=$(PYTHONPATH="$CODE_ROOT/src"' in text
    assert text.count('--camera-rig-revision "$CAMERA_RIG_REVISION"') == 2
    assert text.count('--template-sha256 "$TEMPLATE_SHA256"') == 2
    assert text.count('$TRAIN_ENCODER_FLAG') >= 2
    assert 'camera_migration_flag="--allow-camera-rig-migration"' in text


def test_camera_migration_is_one_shot_on_watchdog_restart():
    text = BASE.read_text(encoding="utf-8")
    latest = text.index('resume="$OUT/latest.pt"')
    cleared = text.index('camera_migration_flag=""', latest)
    learner = text.index('python scripts/rl/learner_cycle_v2.py', cleared)
    assert latest < cleared < learner


def test_schedule_aware_return_does_not_require_ferry_mode():
    text = BASE.read_text(encoding="utf-8")
    ferry_start = text.index('if [ "${STAGEC_V2_STAGE_D_FERRY:-0}" = "1" ]', text.index("launch_collector()"))
    return_start = text.index('if [ "$STAGE_D_RETURN_WHEN_LIVE" = "1" ]', ferry_start)
    section_lanes = text.index("# SECTION LANES:", return_start)
    # The return block appears after the ferry block closes and before the
    # general Stage-D lane options.
    between = text[ferry_start:return_start]
    assert 'stage_d_flags="$stage_d_flags --stage-d-ferry-target-y' in between
    assert return_start < section_lanes
