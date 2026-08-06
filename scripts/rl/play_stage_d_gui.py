"""Drive the RICH interactive match GUI with the STAGE-D COMPOSITE policy.

play_full_gui.py plays a single Stage-B/C network against a 22-D observation.
A Stage-D policy is different in three ways, all handled here:

  1. COMPOSITE ACTION POLICY -- a frozen Stage-C prefix owns the protected
     FIRST cycle, the trainable suffix owns every phase after it.  The split is
     decided by the phase one-hot, exactly as collector_cycle_v2 does it, via
     policy_v2.compose_phase_actions + apply_executed_action_policy.
  2. 30-D OBSERVATION -- the legacy 22 values plus cycle_v2's 8 phase features.
     The prefix is fed pin_prefix_view() (its Stage-C 90 s clock and constant
     hub-eligibility restored); the suffix sees the true Stage-D values.
  3. STAGE-D MATCH CLOCK -- proprio idx 12 carries real blue-hub eligibility
     (blackout windows), and the episode is the full 160 s / 456-ball match.

The cycle FSM is the SAME CycleV2State the trainer uses (pure Python, no Isaac
dependency), advanced once per policy tick from controller state.  Score events
are not reachable from the controller in this GUI (the HubRouter is a local in
isaac_scene.main), and are not needed: every PHASE transition is driven by dump
start/completion, chamber contents, and field region.  Cycle counting shown in
the HUD line is therefore dump-based, matching the trainer's cycles_attempted.

Usage (Windows, Isaac venv):
  C:/il/venv/Scripts/python.exe scripts/rl/play_stage_d_gui.py \
      --checkpoint runs/preserved/stageD_peak_119.pt \
      --prefix-checkpoint runs/preserved/stageC_highest_1163753.pt \
      --dump-on-press --mask-illegal-fire --gui-intake-substeps 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Reuse every proven piece of the Stage-B/C GUI player: camera attach, frame
# reader, readiness gate.  Only the observation and action policy change.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_full_gui as base  # noqa: E402


class _StageD:
    def __init__(self) -> None:
        self.prefix_agent = None
        self.suffix_agent = None
        self.cycle = None
        self.first_inactive = "blue"
        self.dumping = False
        self.dump_mode = "score"
        self.dump_ticks = 0
        self.prev_mag_empty = True
        self.last_phase = None
        self.cycles_dumped = 0
        self.ticks = 0


_SD = _StageD()


def _setup_stage_d(controller) -> None:
    """Attach cameras + spec (base._lazy_setup cannot be reused: it loads the
    --checkpoint into a 22-D agent, which a 30-D Stage-D suffix cannot fill),
    then build BOTH networks and the cycle FSM."""
    from isaacsim.sensors.camera import Camera
    from frc_rebuilt.competition_robot import (
        CAMERA_BASELINE_NAMES,
        CAMERA_PRIM_PATHS,
        CAMERA_RESOLUTION,
    )
    from frc_rebuilt.rl.spec import CompetitionRLSpec
    from frc_rebuilt.rl.cycle_v2 import CycleV2Config, CycleV2State
    from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

    args = base._STATE.args
    n_cam = len(CAMERA_BASELINE_NAMES)

    base._STATE.spec = CompetitionRLSpec()
    base._STATE.spec.validate()
    base._STATE.camera_order = tuple(CAMERA_BASELINE_NAMES)
    for name in base._STATE.camera_order:
        cam = Camera(prim_path=CAMERA_PRIM_PATHS[name], resolution=CAMERA_RESOLUTION)
        cam.initialize()
        base._STATE.cameras[name] = cam

    def _mk(proprio_dim: int, ckpt: str):
        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=3 * n_cam,
                frame_h=CAMERA_RESOLUTION[1] // 4,
                frame_w=CAMERA_RESOLUTION[0] // 4,
                proprio_dim=proprio_dim,
                privileged_dim=26,
            )
        )
        agent.load(ckpt)
        return agent

    _SD.prefix_agent = _mk(22, args.prefix_checkpoint)
    _SD.suffix_agent = _mk(30, args.checkpoint)
    # base._STATE.agent is only used by the parent module's readiness print.
    base._STATE.agent = _SD.suffix_agent

    _SD.cycle = CycleV2State(
        config=CycleV2Config(
            target_load=int(args.target_load),
            chamber_capacity=int(args.chamber_capacity),
        )
    )
    _SD.cycle.reset(initial_magazine_ids=tuple(int(i) for i in controller.magazine))
    print(
        f"STAGE_D_GUI_READY suffix={args.checkpoint} "
        f"steps={_SD.suffix_agent.train_steps} prefix={args.prefix_checkpoint} "
        f"first_inactive={_SD.first_inactive} episode={args.episode_len_s}s",
        flush=True,
    )


def _build_proprio30(controller, now_s: float) -> np.ndarray:
    """Legacy 22 (with the REAL Stage-D hub bit at idx 12) + 8 phase features,
    matching vec_env._observe for stagec_v2 + stage_d."""
    from frc_rebuilt.rl import stage_d as _stage_d

    position, _quat = controller.chassis_pose()
    yaw = controller.chassis_yaw()
    linear, yaw_rate = controller.chassis_velocity()
    mag = len(controller.magazine)
    state = controller.state_machine.state.value
    ep_len = max(1.0, float(base._STATE.args.episode_len_s))
    legacy = np.concatenate(
        [
            np.asarray(
                [
                    position[0] / 8.0,
                    position[1] / 8.0,
                    math.sin(yaw),
                    math.cos(yaw),
                    float(linear[0]) / 4.0,
                    float(linear[1]) / 4.0,
                    float(yaw_rate) / 6.0,
                    float(now_s) / ep_len,
                    mag / 8.0,
                    1.0 if controller.intake_on else 0.0,
                    controller.storage_position,
                    1.0 if state in ("READY", "FEEDING") else 0.0,
                    _stage_d.blue_hub_obs(float(now_s), _SD.first_inactive),
                    float(controller.shots_fired) / 20.0,
                    0.0,  # blue score/20 -- HubRouter unreachable (see base header)
                ],
                np.float32,
            ),
            base._STATE.prev_action.astype(np.float32),
        ]
    )
    remaining = 1.0 - min(1.0, float(now_s) / ep_len)
    phase = np.asarray(
        _SD.cycle.feature_vector(controller.magazine, time_remaining=remaining),
        np.float32,
    )
    return np.concatenate([legacy, phase])[None, :]


def _advance_cycle(controller, now_s: float, started: bool, completed: bool, mode: str) -> None:
    """Advance the trainer's FSM one policy tick from controller state."""
    from frc_rebuilt.rl import stage_d as _stage_d

    ep_len = max(1.0, float(base._STATE.args.episode_len_s))
    remaining = 1.0 - min(1.0, float(now_s) / ep_len)
    position, _q = controller.chassis_pose()
    hub_live = bool(_stage_d.blue_hub_eligible(float(now_s), _SD.first_inactive))
    _SD.cycle.update(
        magazine_ids=tuple(int(i) for i in controller.magazine),
        score_event_ids=(),
        position=(float(position[0]), float(position[1])),
        score=0,
        done=False,
        time_remaining=remaining,
        score_dump_started=bool(started and mode == "score"),
        score_dump_completed=bool(completed and mode == "score"),
        owncourt_score_ready=False,
        hub_live=hub_live,
    )


def _patched_update(
    controller,
    fuel_view,
    now_s: float,
    alliance=None,
    hub_active: bool = True,
    allow_drive: bool = True,
    fire_mode: str = "score",
):
    from frc_rebuilt.rl import stage_d as _stage_d
    from frc_rebuilt.rl.policy_v2 import (
        LEGACY_PROPRIO_DIM,
        apply_executed_action_policy,
        compose_phase_actions,
    )
    from frc_rebuilt.rl.spec import decode_policy_actions

    st = base._STATE
    if st.agent is None:
        _setup_stage_d(controller)

    st.calls += 1

    frames, min_std = base._read_frames()
    if not st.ready:
        if min_std > 1.0:
            st.ready = True
            print(f"STAGE_D_GUI_CAMERAS_READY call={st.calls} std={min_std:.1f}", flush=True)
        else:
            st.black_calls += 1
            if st.black_calls > 150 and not st.warned_black:
                st.warned_black = True
                print("CAMERA_NEVER_FED (see play_full_gui header)", flush=True)
            return st.orig_update(
                controller, fuel_view, now_s=now_s, alliance="blue",
                hub_active=True, allow_drive=True, fire_mode="score",
            )

    # ---- 10 Hz composite policy ------------------------------------------
    if st.calls % 3 == 1:
        pf = base.to_policy_frames(frames)
        proprio = _build_proprio30(controller, now_s)
        prefix_view = _stage_d.pin_prefix_view(
            proprio,
            episode_len_s=float(st.args.episode_len_s),
            legacy_dim=int(LEGACY_PROPRIO_DIM),
        )
        prefix_a = _SD.prefix_agent.act(pf, prefix_view, explore=False).astype(np.float32)
        suffix_a = _SD.suffix_agent.act(pf, proprio, explore=False).astype(np.float32)
        composed = compose_phase_actions(prefix_a, suffix_a, proprio)
        st.cached_action = apply_executed_action_policy(
            composed, proprio, intake_during_return=False, stage_d_ferry=True
        ).astype(np.float32)
        st.prev_action[:] = np.clip(st.cached_action[0], -1.0, 1.0)

        phase = _SD.cycle.phase.value
        if phase != _SD.last_phase:
            _SD.last_phase = phase
            print(
                f"[t={now_s:6.1f}] phase={phase:<12s} mag={len(controller.magazine):2d} "
                f"hub={'LIVE' if _stage_d.blue_hub_eligible(now_s, _SD.first_inactive) else 'dark'} "
                f"dumps={_SD.cycles_dumped}",
                flush=True,
            )

    decoded = decode_policy_actions(st.cached_action, st.spec)

    controller.intake_on = bool(decoded.intake_on[0])
    controller.set_storage_extended(bool(decoded.storage_extended[0]))
    sm = controller.state_machine
    sm.set_continuous(False)

    has_ammo = bool(controller.magazine)
    shoot_ok = bool(decoded.shoot_blue[0]) and has_ammo and bool(
        controller.solve_auto_aim("blue").get("valid", False)
    )
    ferry_ok = bool(decoded.ferry[0]) and has_ammo and bool(
        controller.solve_ferry("blue").get("valid", False)
    )
    fire = shoot_ok or ferry_ok
    fire_mode_use = "ferry" if (ferry_ok and not shoot_ok) else "score"

    # ---- dump-on-press, mirroring vec_env.step ---------------------------
    started = completed = False
    if not _SD.dumping and fire:
        _SD.dumping = True
        _SD.dump_mode = fire_mode_use
        _SD.dump_ticks = 0
        started = True
    if _SD.dumping:
        _SD.dump_ticks += 1
        if not controller.magazine:
            _SD.dumping = False
            completed = True
            if _SD.dump_mode == "score":
                _SD.cycles_dumped += 1
        elif _SD.dump_ticks > int(st.args.max_dump_ticks):
            _SD.dumping = False
    firing = _SD.dumping
    if firing:
        fire_mode_use = _SD.dump_mode
    sm.set_continuous(firing)
    sm.set_emergency_stop(False)
    sm.auto_align = firing

    _advance_cycle(controller, now_s, started, completed, _SD.dump_mode)

    driver = decoded.driver[0]
    moving = (not firing) and bool(np.any(np.abs(driver) > 0.03))
    if moving:
        controller.drive(float(driver[0]), float(driver[2]), strafe=float(driver[1]))

    return st.orig_update(
        controller, fuel_view, now_s=now_s, alliance="blue",
        hub_active=True, allow_drive=not moving, fire_mode=fire_mode_use,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Stage-D suffix policy (.pt, 30-D)")
    ap.add_argument("--prefix-checkpoint", required=True, help="frozen Stage-C prefix (.pt, 22-D)")
    ap.add_argument("--episode-len-s", type=float, default=160.0)
    ap.add_argument("--max-fuel", type=int, default=456)
    ap.add_argument("--target-load", type=int, default=15)
    ap.add_argument("--chamber-capacity", type=int, default=60)
    ap.add_argument("--max-dump-ticks", type=int, default=180)
    ap.add_argument("--first-inactive", default="blue", choices=("blue", "red"))
    ap.add_argument("--gui-intake-substeps", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--gui-camera-views", type=int, choices=(0, 1, 2, 3), default=3)
    ap.add_argument("--gui-render-hz", type=int, choices=(15, 20, 30, 60), default=60)
    args = ap.parse_args()

    # base's helpers read flags off its own _STATE.args namespace.
    args.start_extended = False
    args.lock_storage_extended = False
    args.mask_illegal_fire = True
    args.dump_on_press = True
    base._STATE.args = args
    _SD.first_inactive = args.first_inactive

    import frc_rebuilt.competition_robot as cr
    import frc_rebuilt.isaac_scene as scene

    base._STATE.orig_update = cr.CompetitionRobotController.update
    cr.CompetitionRobotController.update = _patched_update

    sys.argv = [
        sys.argv[0],
        "--max-fuel", str(args.max_fuel),
        "--gui-intake-substeps", str(args.gui_intake_substeps),
        "--gui-camera-views", str(args.gui_camera_views),
        "--gui-render-hz", str(args.gui_render_hz),
    ]
    scene.main()


if __name__ == "__main__":
    main()
