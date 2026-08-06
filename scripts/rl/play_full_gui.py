"""Drive the RICH interactive match GUI (frc_rebuilt.isaac_scene.main) with a
TRAINED DrQ-v2 policy instead of manual keyboard teleop -- WITHOUT editing
isaac_scene.py.

HOW IT WORKS (Option A: monkeypatch a per-frame controller method)
------------------------------------------------------------------
isaac_scene.main() is one monolithic function: it BUILDS the field (SceneBuilder)
and RUNS the loop inline, with no separable "build scene / return handles" entry
point, so there is nothing to import and step ourselves (Option B is impossible).
It also exposes NO external-controller hook or callback (only --no-autopilot,
which just freezes the robot -- see isaac_scene.parse_args:907-930).

The one clean seam is that every frame the articulated robot is advanced by a
METHOD call on the controller:

    isaac_scene.py:1441   last_status = controller.update(fuel_view, now_s=...,
                              alliance=..., hub_active=..., allow_drive=..., ...)

`controller` is a `CompetitionRobotController` (isaac_scene.py:1180). Neither
isaac_scene nor competition_robot import Isaac at module top level, so we can
import the class and REPLACE `CompetitionRobotController.update` with our own
wrapper BEFORE `isaac_scene.main()` creates the SimulationApp. Our wrapper:
  * reads the 3 onboard cameras, builds the EXACT training observation,
  * runs the deterministic policy at 10 Hz (main calls update at 30 Hz; we
    recompute the action every 3rd call and cache it in between, exactly the
    action-repeat cadence vec_env uses),
  * applies the decoded action by mirroring vec_env.step's inner control block
    VERBATIM (drive + shooter FSM), then
  * delegates to the ORIGINAL update() for the shooter/aim tick.

============================ HONEST HEALTH WARNING ============================
This wrapper is UNPROVEN and must be validated on the GPU box. The load-bearing
risk is the OBSERVATION, not the control:

  isaac_scene renders with Fabric + /physics/updateToUsd=False and advances the
  view with `sim.step(render=False)` + a bare `sim.render()` every other frame
  (isaac_scene.py:1347,1351). It NEVER calls `sim.step(render=True)` and it does
  NOT create RGB `isaacsim.sensors.camera.Camera` annotators -- it only opens
  display-only GPU *viewport windows* at 320x180 (isaac_scene.py:1197-1245).
  vec_env's own note is explicit that headless Camera annotators are fed ONLY by
  full Kit updates (`sim.step(render=True)`) and that "a bare sim.render() leaves
  them black" (vec_env.py:461-462). We CANNOT force render=True from here without
  editing isaac_scene.py.

  Therefore the Camera annotators this wrapper attaches MAY read black/stale in
  isaac_scene's loop. The wrapper GATES on this: until all three cameras show
  real content (std > 1.0, the same gate as vec_env.py:475) it drives NOTHING,
  and after ~5 s of black frames it prints CAMERA_NEVER_FED and keeps idling.
  If you see CAMERA_NEVER_FED, the rich GUI cannot faithfully feed a vision
  policy without an edit to isaac_scene.py -- use scripts/rl/play_policy.py
  instead (a bare Isaac viewer running the vec_env obs pipeline VERBATIM, which
  is the guaranteed-correct way to watch the policy drive).

Secondary fidelity gaps vs. the training obs (documented, minor):
  * proprio[14] = blue score / 20: the HubRouter that owns the score is a local
    in isaac_scene.main() and is unreachable from the controller; we send 0.0.
  * a 1-tick (~33 ms) lag on intake actuation, because main() runs step_intake()
    with the keyboard intake state just before our update() sets the policy's.
  * proprio observation noise is omitted (deterministic eval).

Run (from repo root, with Isaac Sim's bundled python):
    isaac-sim/python.bat scripts/rl/play_full_gui.py \
        --checkpoint runs/drqv2_stageB/final.pt \
        --start-extended            # Stage-B locked-extended policies
    isaac-sim/python.bat scripts/rl/play_full_gui.py \
        --checkpoint runs/drqv2_stageC/final.pt   # trench/storage policies
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


def to_policy_frames(rgb: np.ndarray) -> np.ndarray:
    """(N, C_cam, 360, 640, 3) uint8 -> (N, 9, 90, 160) uint8 (4x downsample).

    Copied VERBATIM from scripts/rl/train_drqv2.py:to_policy_frames /
    play_policy.py so the policy sees byte-identical inputs to training.
    """
    small = rgb[:, :, ::4, ::4, :]
    n, cams, h, w, c = small.shape
    return small.transpose(0, 1, 4, 2, 3).reshape(n, cams * c, h, w).copy()


# --------------------------------------------------------------------------- #
# module-level injection state (there is exactly ONE robot in isaac_scene, so a
# single shared block is sufficient; populated lazily on the first update call
# once the SimulationApp / scene / cameras exist).
# --------------------------------------------------------------------------- #
class _Inject:
    def __init__(self) -> None:
        self.args = None
        self.orig_update = None
        self.spec = None
        self.agent = None
        self.cameras: dict[str, object] = {}
        self.camera_order: tuple[str, ...] = ()
        self.ready = False
        self.calls = 0            # counts update() invocations (30 Hz)
        self.black_calls = 0      # consecutive not-ready calls
        self.warned_black = False
        self.cached_action = np.zeros((1, 7), np.float32)
        self.prev_action = np.zeros(7, np.float32)
        self.storage_snapped = False


_STATE = _Inject()


def _lazy_setup(controller) -> None:
    """First-call initialisation: attach the 3 camera annotators, build+load the
    agent. Runs after isaac_scene.main() has created the app, built the robot,
    and called sim.reset() -- the same order vec_env uses (cameras initialised
    after the scene's sim.reset())."""
    from isaacsim.sensors.camera import Camera
    from frc_rebuilt.competition_robot import (
        CAMERA_BASELINE_NAMES,
        CAMERA_PRIM_PATHS,
        CAMERA_RESOLUTION,
    )
    from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
    from frc_rebuilt.rl.spec import CompetitionRLSpec

    _STATE.spec = CompetitionRLSpec()
    _STATE.spec.validate()
    _STATE.camera_order = tuple(CAMERA_BASELINE_NAMES)  # (intake, shooter, navigation)

    # isaac_scene builds a SINGLE robot at ROBOT_ROOT_PATH, so CAMERA_PRIM_PATHS
    # are the exact prim paths of its cameras (no env_i prefix, unlike vec_env).
    for name in _STATE.camera_order:
        cam = Camera(prim_path=CAMERA_PRIM_PATHS[name], resolution=CAMERA_RESOLUTION)
        cam.initialize()
        _STATE.cameras[name] = cam

    # Fixed baseline obs shapes: 3 cameras x 3 channels = 9, 360/4 x 640/4, and
    # the vec_env proprio(22)/privileged(26) widths. Deriving from constants (not
    # a live env) is safe because isaac_scene uses the identical baseline rig.
    _STATE.agent = DrQV2Agent(
        DrQConfig(
            frame_channels=3 * len(_STATE.camera_order),
            frame_h=CAMERA_RESOLUTION[1] // 4,
            frame_w=CAMERA_RESOLUTION[0] // 4,
            proprio_dim=22,
            privileged_dim=26,
        )
    )
    _STATE.agent.load(_STATE.args.checkpoint)
    print(
        f"PLAY_FULL_GUI_READY {_STATE.args.checkpoint} steps={_STATE.agent.train_steps} "
        f"device={_STATE.agent.device} cams={_STATE.camera_order}",
        flush=True,
    )

    # Stage-B locked-extended policies were trained starting EXTENDED, but
    # isaac_scene starts the robot COMPACT (isaac_scene.py:1184). Snap once here
    # so the policy sees the posture it expects. Trench/storage policies leave
    # this off and drive the storage bit themselves.
    if _STATE.args.start_extended:
        controller.snap_storage_state(True)
        _STATE.storage_snapped = True


def _read_frames() -> tuple[np.ndarray, float]:
    """Stack the 3 cameras into (1, 3, 360, 640, 3) uint8 in training order and
    return the frames plus the minimum per-camera colour std (the readiness
    signal, same threshold as vec_env.py:475)."""
    tiles = []
    min_std = np.inf
    for name in _STATE.camera_order:
        rgba = np.asarray(_STATE.cameras[name].get_rgba())
        if rgba.size and rgba.ndim == 3:
            rgb = rgba[..., :3].astype(np.uint8)
            min_std = min(min_std, float(rgb.std()))
        else:
            rgb = np.zeros((360, 640, 3), np.uint8)
            min_std = 0.0
        tiles.append(rgb)
    frames = np.stack(tiles, axis=0)[None, ...]  # (1, 3, 360, 640, 3)
    return frames, (0.0 if not np.isfinite(min_std) else min_std)


def _build_proprio(controller, now_s: float) -> np.ndarray:
    """Reconstruct the 22-D non-privileged proprio EXACTLY as vec_env._observe
    (vec_env.py:833-868), from controller state. Noise is omitted (deterministic
    eval) and blue score is 0.0 (unreachable from the controller -- see header)."""
    position, _quat = controller.chassis_pose()
    yaw = controller.chassis_yaw()
    linear, yaw_rate = controller.chassis_velocity()
    mag = len(controller.magazine)
    state = controller.state_machine.state.value
    ep_len = max(1.0, float(_STATE.args.episode_len_s))
    proprio = np.concatenate(
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
                    1.0,  # blue hub eligible (sandbox, as in training stage A/B)
                    float(controller.shots_fired) / 20.0,
                    0.0,  # blue score / 20 -- unreachable from controller (see header)
                ],
                np.float32,
            ),
            _STATE.prev_action.astype(np.float32),
        ]
    )
    return proprio[None, :]


def _patched_update(
    controller,
    fuel_view,
    now_s: float,
    alliance=None,
    hub_active: bool = True,
    allow_drive: bool = True,
    fire_mode: str = "score",
):
    """Replacement for CompetitionRobotController.update: inject the policy, then
    delegate to the original update() for the shooter/aim tick."""
    from frc_rebuilt.rl.spec import decode_policy_actions

    st = _STATE
    if st.agent is None:
        _lazy_setup(controller)

    st.calls += 1

    # ---- observation-readiness gate ------------------------------------------
    frames, min_std = _read_frames()
    if not st.ready:
        if min_std > 1.0:
            st.ready = True
            print(f"PLAY_FULL_GUI_CAMERAS_READY call={st.calls} std={min_std:.1f}", flush=True)
        else:
            st.black_calls += 1
            if st.black_calls > 150 and not st.warned_black:  # ~5 s at 30 Hz
                st.warned_black = True
                print(
                    "CAMERA_NEVER_FED: onboard Camera annotators are still black after "
                    f"{st.black_calls} frames. isaac_scene's render path (sim.render() "
                    "only, Fabric, updateToUsd=False) does not feed RGB annotators, so a "
                    "vision policy CANNOT see in this GUI without editing isaac_scene.py. "
                    "Use scripts/rl/play_policy.py instead.",
                    flush=True,
                )
            # do NOT act on black observations; let the robot idle (alive + interactive)
            return st.orig_update(
                controller, fuel_view, now_s=now_s, alliance="blue",
                hub_active=True, allow_drive=True, fire_mode="score",
            )

    # ---- 10 Hz policy (main calls update at 30 Hz -> recompute every 3rd) -----
    if st.calls % 3 == 1:  # first ready-ish call recomputes; then every 3rd
        pf = to_policy_frames(frames)
        proprio = _build_proprio(controller, now_s)
        st.cached_action = st.agent.act(pf, proprio, explore=False).astype(np.float32)
        st.prev_action[:] = np.clip(st.cached_action[0], -1.0, 1.0)

    decoded = decode_policy_actions(st.cached_action, st.spec)

    # ---- apply control: mirror vec_env.step's inner block (vec_env.py:669-746) -
    controller.intake_on = bool(decoded.intake_on[0])
    if not st.args.lock_storage_extended:
        controller.set_storage_extended(bool(decoded.storage_extended[0]))
    sm = controller.state_machine
    sm.set_continuous(False)

    if st.args.mask_illegal_fire:
        has_ammo = bool(controller.magazine)
        shoot_ok = bool(decoded.shoot_blue[0]) and has_ammo and bool(
            controller.solve_auto_aim("blue").get("valid", False)
        )
        ferry_ok = bool(decoded.ferry[0]) and has_ammo and bool(
            controller.solve_ferry("blue").get("valid", False)
        )
        fire = shoot_ok or ferry_ok
        fire_mode_use = "ferry" if (ferry_ok and not shoot_ok) else "score"
    else:
        fire = bool(decoded.shoot_blue[0] or decoded.ferry[0])
        fire_mode_use = "ferry" if bool(decoded.ferry[0]) else "score"

    driver = decoded.driver[0]
    if st.args.dump_on_press:
        if not getattr(controller, "_wrap_dumping", False) and fire:
            controller._wrap_dumping = True
            controller._wrap_dump_mode = fire_mode_use
        if getattr(controller, "_wrap_dumping", False) and not controller.magazine:
            controller._wrap_dumping = False
        firing = bool(getattr(controller, "_wrap_dumping", False))
        if firing:
            fire_mode_use = controller._wrap_dump_mode
        sm.set_continuous(firing)
        sm.set_emergency_stop(False)
        sm.auto_align = firing
        moving = (not firing) and bool(np.any(np.abs(driver) > 0.03))
    else:
        if fire:
            sm.request_single()
        sm.set_emergency_stop(False)
        sm.auto_align = fire
        moving = bool(np.any(np.abs(driver) > 0.03)) and not fire

    if moving:
        controller.drive(float(driver[0]), float(driver[2]), strafe=float(driver[1]))

    # delegate to the real update; allow_drive=not moving so its idle/auto-align
    # branch does not stomp our drive (vec_env.py:739-746).
    return st.orig_update(
        controller, fuel_view, now_s=now_s, alliance="blue",
        hub_active=True, allow_drive=not moving, fire_mode=fire_mode_use,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to a saved DrQV2Agent .pt")
    ap.add_argument("--episode-len-s", type=float, default=90.0,
                    help="only normalises proprio[7] (the match-clock feature)")
    ap.add_argument("--start-extended", action="store_true",
                    help="snap storage EXTENDED at start (Stage-B locked-extended policies; "
                    "isaac_scene otherwise starts COMPACT)")
    ap.add_argument("--lock-storage-extended", action="store_true",
                    help="ignore the policy's storage bit and keep the snapped posture "
                    "(pair with --start-extended to match Stage-B lock_storage_extended)")
    ap.add_argument("--mask-illegal-fire", action="store_true",
                    help="match Stage-B: a fire press is a no-op unless a legal shot/ferry "
                    "exists this tick (required to faithfully play a masked policy)")
    ap.add_argument("--dump-on-press", action="store_true",
                    help="one press empties the whole magazine (only for models trained "
                    "with that mechanic)")
    ap.add_argument("--max-fuel", type=int, default=456,
                    help="passed through to isaac_scene (dynamic FUEL bodies)")
    ap.add_argument(
        "--gui-intake-substeps",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="local rich-GUI intake speedup; 1 preserves training-physics parity, "
        "2 is the recommended faster showcase setting",
    )
    ap.add_argument(
        "--gui-camera-views",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
        help="onboard camera windows; default shows intake, shooter, and navigation",
    )
    ap.add_argument(
        "--gui-render-hz",
        type=int,
        choices=(15, 20, 30, 60),
        default=60,
        help="GUI display target; physics remains fixed at 60 Hz",
    )
    args = ap.parse_args()
    _STATE.args = args

    # Patch the class BEFORE isaac_scene.main() creates the app. Safe: neither
    # module imports Isaac at top level, so importing them needs no SimulationApp.
    import frc_rebuilt.competition_robot as cr
    import frc_rebuilt.isaac_scene as scene

    _STATE.orig_update = cr.CompetitionRobotController.update
    cr.CompetitionRobotController.update = _patched_update

    # Hand isaac_scene.main() only the flags IT understands. No --headless -> the
    # full interactive match GUI opens; the articulated robot (default) is driven
    # every frame via controller.update(), which is now our injected policy.
    sys.argv = [
        sys.argv[0],
        "--max-fuel",
        str(args.max_fuel),
        "--gui-intake-substeps",
        str(args.gui_intake_substeps),
        "--gui-camera-views",
        str(args.gui_camera_views),
        "--gui-render-hz",
        str(args.gui_render_hz),
    ]
    scene.main()


if __name__ == "__main__":
    main()
