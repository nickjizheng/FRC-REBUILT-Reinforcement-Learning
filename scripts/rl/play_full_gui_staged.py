"""Play a STAGE-D COMPOSITE policy in the FULL interactive match GUI.

`play_full_gui.py` drives a single 22-D Stage-B/C checkpoint.  A Stage-D policy
is different in three ways, all of which this launcher handles:

  1. It is a COMPOSITE.  A frozen Stage-C prefix owns the protected FIRST cycle;
     the trainable 30-input suffix owns LEAVE/COLLECT/RETURN/SCORE.  The handoff
     is `policy_v2.compose_phase_actions`, and the action that reaches the
     simulator is `policy_v2.apply_executed_action_policy` -- the same pure-NumPy
     module the collector and the deterministic evaluator use, so GUI playback
     executes the identical action contract.

  2. Its proprio is 30-D: the legacy 22 with index 12 replaced by the real blue
     hub state (`stage_d.blue_hub_obs`: 1.0 active/grace, 0.5 deactivation
     warning, 0.0 inactive) plus the 8 cycle-FSM features from
     `CycleV2State.feature_vector`.  Index 14 is the real blue router score.

  3. It needs a live cycle FSM.  `CycleV2State` is driven here from the same
     inputs vec_env feeds it (magazine ids, newly-emitted blue score events,
     chassis position, score, hub liveness), so the phase one-hot the policy
     reads is produced by the real state machine rather than an approximation.

Why a separate launcher instead of editing `play_full_gui.py`: that script drives
the robot through a monkey-patched `CompetitionRobotController.update` while the
scene is in MANUAL mode, and in manual mode `isaac_scene` reassigns
`controller.intake_on` from the GUI toggle every frame (isaac_scene.py:1862)
before `step_mechanisms` runs -- so an injected policy's intake bit is discarded.
Stage-D policies live or die on intake, so this launcher instead enters the
scene's real POLICY MODE by substituting `_RLPolicyDriver`, where the policy
genuinely owns intake, storage and the shooter FSM.  Nothing in `isaac_scene.py`
or `play_full_gui.py` is modified.

Example:
    python scripts/rl/play_full_gui_staged.py \
        --checkpoint runs/stageD_gui_best_20260726.pt \
        --prefix-checkpoint runs/preserved/stageC_highest_1163753.pt \
        --stage-d-ferry --mask-illegal-fire --dump-on-press
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Non-interactive EULA acceptance, exactly as run_sim.py does.  Without it the
# inner Kit kernel blocks on "Do you accept the EULA? (Yes/No):" with no console
# to answer and dies with "Unable to bootstrap inner kit kernel: EOF".
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Populated by main() before isaac_scene builds the driver.
_CFG: argparse.Namespace | None = None
# The live HubRouter instance (a main() local in isaac_scene, captured via a
# patched __init__ -- the same monkey-patch technique play_full_gui.py uses).
_ROUTER: dict[str, object] = {}


class _CompositeAgent:
    """Adapts the two-network composite to the single `.act()` the driver calls.

    Presenting the composite as one agent lets the base `_RLPolicyDriver.step`
    run byte-unchanged: it still calls `self.agent.act(frames, proprio)` at 10 Hz
    and applies its own drive/shooter block to the result.
    """

    def __init__(self, prefix_agent, suffix_agent, *, stage_d_ferry: bool,
                 intake_during_return: bool) -> None:
        from frc_rebuilt.rl import policy_v2

        self._policy_v2 = policy_v2
        self.prefix_agent = prefix_agent
        self.suffix_agent = suffix_agent
        self.stage_d_ferry = bool(stage_d_ferry)
        self.intake_during_return = bool(intake_during_return)
        # surfaced for the driver's POLICY_READY banner
        self.train_steps = suffix_agent.train_steps
        self.device = suffix_agent.device
        self.last_phase = 0

    def act(self, frames, proprio, explore: bool = False):
        pv = self._policy_v2
        proprio = np.asarray(proprio, np.float32)
        legacy = proprio[:, : pv.LEGACY_PROPRIO_DIM]
        prefix = np.asarray(
            self.prefix_agent.act(frames, legacy, explore=explore), np.float32
        )
        candidate = np.asarray(
            self.suffix_agent.act(frames, proprio, explore=explore), np.float32
        )
        composed = pv.compose_phase_actions(prefix, candidate, proprio)
        executed = pv.apply_executed_action_policy(
            composed,
            proprio,
            intake_during_return=self.intake_during_return,
            stage_d_ferry=self.stage_d_ferry,
        )
        self.last_phase = int(np.asarray(pv.phase_from_proprio(proprio)).reshape(-1)[0])
        return np.asarray(executed, np.float32)


def _make_driver_class(scene):
    """Subclass the scene's own driver so the control block stays identical."""

    base = scene._RLPolicyDriver

    class _StageDDriver(base):
        def __init__(self, checkpoint, episode_len_s, mask_illegal_fire,
                     dump_on_press, start_extended):
            super().__init__(checkpoint, episode_len_s, mask_illegal_fire,
                             dump_on_press, start_extended)
            self.cycle = None
            self.score_cursor = 0
            self.prev_dumping = False
            self.prev_region_home = True
            self.phase_names = ("FIRST", "LEAVE", "COLLECT", "RETURN", "SCORE")
            self.last_reported_phase = None
            self._annots = None       # None = not resolved yet, {} = unavailable
            self._annot_tries = 0

        # -- vision: read the viewport render products, not fresh Camera prims --
        #
        # A standalone `Camera` annotator stays BLACK in this GUI (verified: no
        # run on this machine has ever printed POLICY_CAMERAS_READY).  The scene
        # runs Fabric with /physics/updateToUsd=False to keep 456 FUEL transforms
        # off USD, and the extra render products never receive frames.  The three
        # onboard viewport windows the GUI itself opens DO render every frame, so
        # we attach rgb annotators to THEIR render products instead.  Forcing the
        # viewport resolution to the trained 640x360 keeps the stride-4 downsample
        # byte-identical to training (480x270 would give 120x67, not 160x90).
        #
        # Lazy, because isaac_scene creates the viewports (line ~1635) AFTER it
        # constructs and attaches the policy driver (line ~1602).
        def _ensure_annotators(self):
            if self._annots:
                return self._annots
            if self._annots == {} and self._annot_tries > 300:
                return {}
            self._annot_tries += 1
            try:
                from omni.kit.viewport.utility import get_viewport_from_window_name
                import omni.replicator.core as rep
            except Exception as exc:
                if self._annot_tries == 1:
                    print(f"STAGE_D_VIEWPORT_IMPORT_FAILED {exc}", flush=True)
                self._annots = {}
                return {}
            found = {}
            for name in self.camera_order:
                try:
                    vp = get_viewport_from_window_name(f"Viewport {name.title()}")
                    if vp is None:
                        continue
                    if tuple(vp.resolution) != tuple(self.resolution):
                        vp.resolution = self.resolution
                    annot = rep.AnnotatorRegistry.get_annotator("rgb")
                    annot.attach([vp.render_product_path])
                    found[name] = annot
                except Exception as exc:
                    if self._annot_tries % 100 == 1:
                        print(f"STAGE_D_VIEWPORT_ATTACH_RETRY {name}: {exc}", flush=True)
            if len(found) == len(self.camera_order):
                self._annots = found
                print(
                    f"STAGE_D_VIEWPORT_ANNOTATORS attached cams={self.camera_order} "
                    f"res={self.resolution}",
                    flush=True,
                )
                return found
            self._annots = {}
            return {}

        @staticmethod
        def _fit(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
            """Nearest-neighbour fit to the trained frame size (no cv2/scipy)."""
            h, w = rgb.shape[:2]
            if h == height and w == width:
                return rgb
            ys = (np.arange(height) * (h / height)).astype(np.int32).clip(0, h - 1)
            xs = (np.arange(width) * (w / width)).astype(np.int32).clip(0, w - 1)
            return rgb[ys][:, xs]

        def _read_frames(self):
            annots = self._ensure_annotators()
            if not annots:
                return super()._read_frames()
            width, height = self.resolution
            tiles = []
            min_std = np.inf
            for name in self.camera_order:
                raw = annots[name].get_data()
                arr = np.asarray(raw)
                if not self.ready and self._annot_tries % 120 == 0:
                    print(
                        f"STAGE_D_ANNOT_DEBUG {name} type={type(raw).__name__} "
                        f"shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)} "
                        f"size={arr.size} "
                        f"min={arr.min() if arr.size else '-'} "
                        f"max={arr.max() if arr.size else '-'}",
                        flush=True,
                    )
                self._annot_tries += 1
                if arr.size and arr.ndim == 3 and arr.shape[0] > 0:
                    rgb = self._fit(arr[..., :3].astype(np.uint8), height, width)
                    min_std = min(min_std, float(rgb.std()))
                else:
                    rgb = np.zeros((height, width, 3), np.uint8)
                    min_std = 0.0
                tiles.append(rgb)
            frames = np.stack(tiles, axis=0)[None, ...]
            return frames, (0.0 if not np.isfinite(min_std) else min_std)

        # -- one-time attach: cameras from the base, then BOTH networks --------
        def attach(self, controller) -> None:
            from isaacsim.sensors.camera import Camera
            from frc_rebuilt.competition_robot import (
                CAMERA_BASELINE_NAMES,
                CAMERA_PRIM_PATHS,
                CAMERA_RESOLUTION,
            )
            from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
            from frc_rebuilt.rl.spec import CompetitionRLSpec, decode_policy_actions
            from frc_rebuilt.rl.cycle_v2 import CycleV2Config, CycleV2State

            cfg = _CFG
            self.spec = CompetitionRLSpec()
            self.spec.validate()
            self._decode = decode_policy_actions
            self.camera_order = tuple(CAMERA_BASELINE_NAMES)
            self.resolution = CAMERA_RESOLUTION
            for name in self.camera_order:
                cam = Camera(prim_path=CAMERA_PRIM_PATHS[name],
                             resolution=CAMERA_RESOLUTION)
                cam.initialize()
                self.cameras[name] = cam

            def _mk(proprio_dim: int):
                return DrQV2Agent(
                    DrQConfig(
                        frame_channels=3 * len(self.camera_order),
                        frame_h=CAMERA_RESOLUTION[1] // 4,
                        frame_w=CAMERA_RESOLUTION[0] // 4,
                        proprio_dim=proprio_dim,
                        privileged_dim=26,
                    )
                )

            suffix = _mk(30)
            suffix.load(self.checkpoint)
            prefix = _mk(22)
            prefix.load(cfg.prefix_checkpoint)

            self._verify_prefix(cfg)

            self.agent = _CompositeAgent(
                prefix, suffix,
                stage_d_ferry=cfg.stage_d_ferry,
                intake_during_return=cfg.intake_during_return,
            )

            self.cycle = CycleV2State(
                CycleV2Config(
                    target_load=cfg.target_load,
                    preferred_load=cfg.preferred_load,
                    collect_stall_steps=cfg.collect_stall_steps,
                    return_time_guard=cfg.return_time_guard,
                    live_return_load=cfg.live_return_load,
                    chamber_capacity=cfg.chamber_capacity,
                    cycle_score_fraction=cfg.cycle_score_fraction,
                    cycle_score_floor=cfg.cycle_score_floor,
                )
            )
            self.cycle.reset()

            print(
                f"POLICY_READY composite suffix={self.checkpoint} "
                f"steps={suffix.train_steps} prefix={cfg.prefix_checkpoint} "
                f"device={suffix.device} stage_d_ferry={cfg.stage_d_ferry} "
                f"target_load={cfg.target_load} live_return_load={cfg.live_return_load}",
                flush=True,
            )
            if self.start_extended:
                controller.snap_storage_state(True)

        def _verify_prefix(self, cfg) -> None:
            """The composite contract pins the prefix by sha256; a mismatch means
            the FIRST cycle would be driven by a different network than the one
            the suffix was trained against.  Warn loudly rather than fail, so an
            intentional experiment is still possible."""
            import hashlib

            from frc_rebuilt.rl import policy_v2

            try:
                import torch

                meta = torch.load(
                    self.checkpoint, map_location="cpu", weights_only=False
                ).get("stagec_v2")
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"PREFIX_CHECK_SKIPPED unable to read metadata: {exc}", flush=True)
                return
            if not isinstance(meta, dict):
                print("PREFIX_CHECK_SKIPPED checkpoint has no stagec_v2 metadata",
                      flush=True)
                return
            digest = hashlib.sha256(
                Path(cfg.prefix_checkpoint).read_bytes()
            ).hexdigest()
            try:
                policy_v2.validate_composite_metadata(meta, digest)
                print(f"PREFIX_CHECK_OK sha256={digest[:16]}", flush=True)
            except ValueError as exc:
                print(f"PREFIX_CHECK_MISMATCH {exc}", flush=True)

        # -- Stage-D 30-D observation (drives the cycle FSM on the same tick) ---
        def _build_proprio(self, controller, now_s: float) -> np.ndarray:
            from frc_rebuilt.rl import stage_d

            cfg = _CFG
            router = _ROUTER.get("r")
            position, _quat = controller.chassis_pose()
            yaw = controller.chassis_yaw()
            linear, yaw_rate = controller.chassis_velocity()
            mag = len(controller.magazine)
            state = controller.state_machine.state.value
            ep_len = max(1.0, float(self.episode_len_s))

            if cfg.stage_d_first_inactive != "auto":
                first_inactive = cfg.stage_d_first_inactive
            else:
                first_inactive = getattr(router, "match_first_inactive", None)
            # The match picks first_inactive from AUTO fuel at ~23 s, so it is
            # None for the whole AUTO/TRANSITION block -- fine, both hubs are
            # active there.  From SHIFT 1 onward rules.hub_is_active REQUIRES it;
            # if the decision somehow has not landed, fall back to the alliance
            # training pinned rather than letting the observation raise.
            if first_inactive is None and now_s >= 30.0:
                first_inactive = "blue"
                if not getattr(self, "_warned_first_inactive", False):
                    self._warned_first_inactive = True
                    print(
                        "STAGE_D_WARN first_inactive unset at SHIFT 1; assuming 'blue'",
                        flush=True,
                    )

            blue_score = int(getattr(router, "scored", {}).get("blue", 0)) if router else 0

            # newly-emitted blue score events since the last policy tick
            new_ids: list[int] = []
            if router is not None:
                events = getattr(router, "score_events", [])
                if self.score_cursor > len(events):  # router was reset
                    self.score_cursor = 0
                for alliance, index in events[self.score_cursor:]:
                    if alliance == "blue":
                        new_ids.append(int(index))
                self.score_cursor = len(events)

            hub_live = bool(stage_d.blue_hub_eligible(now_s, first_inactive))
            dumping = bool(getattr(controller, "_wrap_dumping", False))
            dump_started = dumping and not self.prev_dumping
            dump_completed = self.prev_dumping and not dumping
            self.prev_dumping = dumping

            # STAGE-D1C own-court short loop.  vec_env owns a richer gate; here we
            # assert it on the observable conditions: the loop is enabled, the hub
            # is live, we are home, and we hold at least the minimum entitled load.
            owncourt_ready = bool(
                cfg.stage_d_owncourt_loop
                and hub_live
                and self.prev_region_home
                and mag >= cfg.stage_d_owncourt_min_balls
            )

            remaining = 1.0 - min(1.0, float(now_s) / ep_len)
            step = self.cycle.update(
                controller.magazine,
                new_ids,
                position,
                blue_score,
                done=False,
                time_remaining=remaining,
                score_dump_started=dump_started,
                score_dump_completed=dump_completed,
                hub_live=hub_live,
                owncourt_score_ready=owncourt_ready,
            )
            self.prev_region_home = (
                getattr(getattr(step, "region", None), "value", "") == "home"
            )

            phase_features = np.asarray(
                self.cycle.feature_vector(controller.magazine, time_remaining=remaining),
                np.float32,
            )
            if phase_features.shape != (8,):
                raise RuntimeError(
                    f"cycle feature contract changed: {phase_features.shape}"
                )

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
                            float(stage_d.blue_hub_obs(now_s, first_inactive)),
                            float(controller.shots_fired) / 20.0,
                            float(blue_score) / 20.0,
                        ],
                        np.float32,
                    ),
                    self.prev_action.astype(np.float32),
                ]
            )
            self._report_phase(now_s, hub_live, mag, blue_score)
            return np.concatenate([legacy, phase_features])[None, :]

        def _report_phase(self, now_s, hub_live, mag, blue_score) -> None:
            phase = getattr(self.cycle, "phase", None)
            name = getattr(phase, "value", str(phase))
            if name != self.last_reported_phase:
                self.last_reported_phase = name
                print(
                    f"STAGE_D t={now_s:6.1f}s phase={name:<12} hub={'LIVE' if hub_live else 'DARK'} "
                    f"mag={mag:2d} blue_score={blue_score} "
                    f"cycles={self.cycle.cycles_completed}",
                    flush=True,
                )

    return _StageDDriver


def _patch_router_capture(scene) -> None:
    """Capture the live HubRouter, which isaac_scene keeps as a main() local."""
    orig_init = scene.HubRouter.__init__

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _ROUTER["r"] = self

    scene.HubRouter.__init__ = _init


def main() -> None:
    global _CFG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Stage-D SUFFIX checkpoint (30-D proprio)")
    ap.add_argument("--prefix-checkpoint", required=True,
                    help="frozen Stage-C prefix (22-D); pinned by the suffix's "
                         "stagec_v2.prefix_sha256")
    ap.add_argument("--episode-len-s", type=float, default=160.0,
                    help="official match length; normalises proprio[7] and the "
                         "cycle time_remaining feature")
    ap.add_argument("--max-fuel", type=int, default=456,
                    help="official field is 456 FUEL")
    ap.add_argument("--stage-d-first-inactive", choices=("auto", "blue", "red"),
                    default="auto",
                    help="'auto' uses the hub the match itself deactivates first "
                         "(chosen from AUTO fuel); training forced 'blue'")
    ap.add_argument("--stage-d-ferry", action="store_true",
                    help="keep ferry policy-controlled in the suffix (forced off "
                         "only in SCORE), matching the trained ferry contract")
    ap.add_argument("--intake-during-return", action="store_true",
                    help="the run trained with this OFF; leave unset to match")
    ap.add_argument("--stage-d-owncourt-loop", action="store_true", default=True)
    ap.add_argument("--stage-d-owncourt-min-balls", type=int, default=2)
    # cycle FSM config -- defaults mirror the live training run
    ap.add_argument("--target-load", type=int, default=15)
    ap.add_argument("--preferred-load", type=int, default=0)
    ap.add_argument("--collect-stall-steps", type=int, default=45)
    ap.add_argument("--return-time-guard", type=float, default=0.11)
    ap.add_argument("--live-return-load", type=int, default=26)
    ap.add_argument("--chamber-capacity", type=int, default=60)
    ap.add_argument("--cycle-score-fraction", type=float, default=0.75)
    ap.add_argument("--cycle-score-floor", type=int, default=6)
    # passthrough
    ap.add_argument("--mask-illegal-fire", action="store_true")
    ap.add_argument("--dump-on-press", action="store_true")
    ap.add_argument("--start-extended", action="store_true")
    ap.add_argument("--gui-intake-substeps", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--gui-camera-views", type=int, choices=(0, 1, 2, 3), default=3)
    ap.add_argument("--gui-render-hz", type=int, choices=(15, 20, 30, 60), default=60)
    ap.add_argument("--render-width", type=int, default=1600)
    ap.add_argument("--render-height", type=int, default=900)
    args = ap.parse_args()
    _CFG = args

    import frc_rebuilt.isaac_scene as scene

    _patch_router_capture(scene)
    scene._RLPolicyDriver = _make_driver_class(scene)

    argv = [
        sys.argv[0],
        "--policy", args.checkpoint,
        "--policy-episode-len-s", str(args.episode_len_s),
        "--max-fuel", str(args.max_fuel),
        "--gui-intake-substeps", str(args.gui_intake_substeps),
        "--gui-camera-views", str(args.gui_camera_views),
        "--gui-render-hz", str(args.gui_render_hz),
        "--render-width", str(args.render_width),
        "--render-height", str(args.render_height),
    ]
    if args.mask_illegal_fire:
        argv.append("--mask-illegal-fire")
    if args.dump_on_press:
        argv.append("--dump-on-press")
    if args.start_extended:
        argv.append("--start-extended")
    sys.argv = argv
    scene.main()


if __name__ == "__main__":
    main()
