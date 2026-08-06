"""Watch a trained DrQ-v2 policy drive the REBUILT robot LIVE in the Isaac Sim GUI.

This is the interactive counterpart to scripts/rl/eval_checkpoint.py and
eval_route.py: instead of a headless batch eval, it opens the Isaac viewer
(headless=False) and steps a SINGLE cloned field with the trained policy in the
loop, so you can watch the robot play in real time.

Hard constraint honoured: the policy only works on the EXACT observations it was
trained on, so this script reuses vec_env's observation pipeline VERBATIM
(VecCompetitionEnv._observe -> rgb/proprio) and the SAME to_policy_frames
downsample used by the trainer. Nothing about the obs is rebuilt here.

Architecture (Option B - zero training-code risk):
  - The GUI/headless choice lives entirely in the ENTRYPOINT: vec_env never
    creates the SimulationApp (see its module docstring: "This module must be
    imported only after SimulationApp is created"). eval_route.py:44 and
    eval_checkpoint.py:39 pass {"headless": True}; we pass {"headless": False}.
    That single dict is the whole GUI switch - no edit to vec_env.py is required.
  - vec_env.step() already renders once per policy step when cameras are on
    (vec_env.py:657 -> sim.step(render=(k==last_k))), so with a window open the
    viewport refreshes at the 10 Hz policy rate. vec_env keeps updateToUsd at its
    default (True), so USD transforms flow to the viewport and you see motion
    without needing Fabric.

Run (from repo root, using Isaac Sim's bundled python):
    isaac-sim/python.bat scripts/rl/play_policy.py \
        --checkpoint runs/drqv2_stageB/final.pt \
        --template   assets/rl/env_template_96.usd \
        --spawn-under-trench          # for a Stage-C trench-start demo
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from frc_rebuilt.rl.policy_v2 import (
    ACTION_POLICY,
    LEGACY_PROPRIO_DIM,
    SCHEMA_VERSION,
    apply_executed_action_policy,
    compose_phase_actions,
    validate_composite_metadata,
)
from frc_rebuilt.rl.cycle_v2 import (
    COLLECT_UNTIL_PREFERRED_REVISIONS,
    POSTDUMP_COMPLETE_CYCLE_REVISIONS,
    POSTDUMP_TARGET_REVISIONS,
    RAMP_OUT_REVISIONS,
    RETURN_INTAKE_REVISIONS,
    SCORE_EFFICIENCY_REVISIONS,
    SUPPORTED_ROUTE_EFFICIENCY_REVISIONS,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _MatchClockOverlay:
    """On-screen match clock for the GUI.

    The Isaac viewer has no scoreboard, so without this there is no way to tell
    which part of the match you are watching -- and for Stage D the interesting
    question is always "is the hub live right now?".  Shows match time, the
    official phase, blue hub state, score, and the policy's cycle phase.

    Built with omni.ui after the SimulationApp exists.  Every call is wrapped:
    a UI failure must never take down playback.
    """

    def __init__(self) -> None:
        self.ok = False
        try:
            import omni.ui as ui

            self._ui = ui
            self.window = ui.Window(
                "MATCH CLOCK", width=340, height=190,
                flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
            )
            with self.window.frame:
                with ui.VStack(spacing=6, height=0):
                    self.l_time = ui.Label(
                        "0:00 / 2:40", height=48,
                        style={"font_size": 40, "color": 0xFFFFFFFF},
                    )
                    self.l_phase = ui.Label("--", style={"font_size": 18})
                    self.l_hub = ui.Label("--", style={"font_size": 20})
                    self.l_score = ui.Label("--", style={"font_size": 18})
                    self.l_cycle = ui.Label("--", style={"font_size": 16})
            self.ok = True
        except Exception as exc:  # pragma: no cover - GUI only
            print(f"CLOCK_OVERLAY_UNAVAILABLE {exc}", flush=True)

    @staticmethod
    def _phase_name(t: float) -> str:
        if t < 20.0:
            return "AUTO"
        if t < 30.0:
            return "TRANSITION"
        if t < 130.0:
            return f"SHIFT {int((t - 30.0) // 25.0) + 1}"
        return "ENDGAME (both hubs live)"

    def update(self, clock_s, match_len, first_inactive, blue_live, blue_score,
               cycle_phase, magazine) -> None:
        if not self.ok:
            return
        try:
            t = max(0.0, float(clock_s))
            self.l_time.text = (
                f"{int(t) // 60}:{int(t) % 60:02d} / "
                f"{int(match_len) // 60}:{int(match_len) % 60:02d}"
            )
            self.l_phase.text = f"{self._phase_name(t)}   (first dark: {first_inactive})"
            if blue_live:
                self.l_hub.text = "BLUE HUB: LIVE - shots score"
                self.l_hub.style = {"font_size": 20, "color": 0xFF44FF44}
            else:
                self.l_hub.text = "BLUE HUB: DARK - shots score 0"
                self.l_hub.style = {"font_size": 20, "color": 0xFF4444FF}
            self.l_score.text = f"BLUE SCORE: {int(blue_score)}"
            self.l_cycle.text = f"policy phase: {cycle_phase}   magazine: {magazine}"
        except Exception:
            self.ok = False


def to_policy_frames(rgb: np.ndarray) -> np.ndarray:
    """(N, C_cam, 360, 640, 3) uint8 -> (N, 9, 90, 160) uint8 (4x downsample).

    Copied VERBATIM from scripts/rl/train_drqv2.py:to_policy_frames so the policy
    sees byte-identical inputs to training. Do NOT "improve" this - the encoder
    only works on this exact stride-4 decimation + channel-stack.
    """
    small = rgb[:, :, ::4, ::4, :]                        # (N, cams, 90, 160, 3)
    n, cams, h, w, c = small.shape
    return small.transpose(0, 1, 4, 2, 3).reshape(n, cams * c, h, w).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to a saved DrQV2Agent .pt")
    ap.add_argument(
        "--template",
        default=str(PROJECT_ROOT / "assets/rl/env_template_96.usd"),
        help="exported single-field USD template (local options: env_template.usd, "
        "env_template_32.usd, env_template_96.usd)",
    )
    ap.add_argument("--episode-len-s", type=float, default=90.0)
    ap.add_argument(
        "--spawn-under-trench",
        action="store_true",
        help="Stage-C match start: compact, fully beneath the blue trench "
        "(also unlocks storage, matching eval_route.py)",
    )
    ap.add_argument(
        "--preload-prob",
        type=float,
        default=0.0,
        help="fraction of episodes that start already holding FUEL at a legal "
        "shooting pose (Stage-B style); 0 = always collect first",
    )
    ap.add_argument(
        "--mask-illegal-fire",
        action="store_true",
        help="match Stage-B training: an illegal fire is a no-op instead of "
        "freezing the chassis (required to faithfully play a masked policy)",
    )
    ap.add_argument(
        "--dump-on-press",
        action="store_true",
        help="one press freezes + empties the whole magazine (only for models "
        "trained with that mechanic); default is the one-click single mechanic",
    )
    ap.add_argument(
        "--max-dump-ticks",
        type=int,
        default=None,
        help="dump safety cap at 30 Hz; Stage C v2 defaults to 180 (six seconds)",
    )
    ap.add_argument(
        "--stagec-v2",
        action="store_true",
        help="load the opt-in 30-value Stage C v2 observation/reward contract",
    )
    ap.add_argument(
        "--stagec-v2-prefix-checkpoint",
        default=None,
        help="required with --stagec-v2: frozen legacy 22-proprio first-cycle policy",
    )
    ap.add_argument(
        "--stagec-v2-reset-mode",
        choices=("full", "postdump", "collect", "return"),
        default="full",
    )
    ap.add_argument("--stagec-v2-target-load", type=int, default=None)
    ap.add_argument("--stagec-v2-reserve-count", type=int, default=None)
    ap.add_argument("--stagec-v2-reserve-batches", type=int, default=None)
    # ---- Stage D: the official 160 s match (hub deactivation shifts + ferry) --
    # Without --stage-d the env keeps both hubs live for the whole episode, so a
    # Stage-D policy is watched under rules it was not trained on: proprio[12]
    # would sit at a constant 1.0 instead of the real hub state.
    ap.add_argument("--stage-d", action="store_true",
                    help="run the official match: AUTO, then alternating hub "
                         "deactivation shifts, with real score eligibility")
    ap.add_argument("--stage-d-first-inactive", choices=("blue", "red"), default="blue",
                    help="which hub goes dark first (training pinned blue)")
    ap.add_argument("--stage-d-ferry", action="store_true",
                    help="keep ferry policy-controlled in the suffix (forced off "
                         "only in SCORE), matching the trained ferry contract")
    ap.add_argument("--stage-d-ferry-dump-on-press", action="store_true")
    ap.add_argument("--stage-d-ferry-blackout-only", action="store_true")
    ap.add_argument("--stage-d-ferry-entitled-only", action="store_true")
    ap.add_argument("--stage-d-ferry-min-load", type=int, default=10)
    ap.add_argument("--stage-d-return-when-live", action="store_true")
    ap.add_argument("--stage-d-live-return-load", type=int, default=26)
    ap.add_argument("--stage-d-return-lead-s", type=float, default=8.0)
    ap.add_argument("--stage-d-owncourt-loop", action="store_true")
    ap.add_argument("--stage-d-owncourt-min-balls", type=int, default=2)
    ap.add_argument("--stage-d-owncourt-rearm", action="store_true")
    ap.add_argument("--stage-d-owncourt-blackout-intake", action="store_true")
    ap.add_argument("--no-clock-overlay", action="store_true",
                    help="hide the on-screen match clock window")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument(
        "--explore",
        action="store_true",
        help="sample actions with the policy's trained exploration noise "
        "(stddev floor ~0.1) instead of the deterministic mean -- dithers the "
        "chassis free of the deterministic trench-escape freeze that makes a "
        "noise-off replay look far worse than the policy actually is",
    )
    ap.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="extra Gaussian action noise added on top of act() (0.15-0.25 "
        "reliably unsticks a frozen deterministic trajectory); clipped to [-1,1]",
    )
    ap.add_argument(
        "--no-realtime",
        action="store_true",
        help="run as fast as the sim allows instead of pacing to wall-clock time",
    )
    args = ap.parse_args()
    if args.stagec_v2 and not args.dump_on_press:
        ap.error("--stagec-v2 requires --dump-on-press to match training mechanics")
    if args.stagec_v2 and not args.stagec_v2_prefix_checkpoint:
        ap.error("--stagec-v2 requires --stagec-v2-prefix-checkpoint")
    prefix_path = (
        Path(args.stagec_v2_prefix_checkpoint).resolve()
        if args.stagec_v2_prefix_checkpoint
        else None
    )
    if prefix_path is not None and not prefix_path.is_file():
        ap.error(f"Stage C v2 prefix checkpoint does not exist: {prefix_path}")

    # ---- open the INTERACTIVE viewer (the entire GUI switch) --------------
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": False, "width": 1600, "height": 900, "multi_gpu": False}
    )
    print("PLAY_APP_READY", app.is_running(), flush=True)

    try:
        # Imported only AFTER SimulationApp exists (vec_env module docstring).
        from frc_rebuilt.rl.vec_env import (
            SimulationUnstable,
            VecCompetitionEnv,
            VecEnvCfg,
        )
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl import stage_d as stage_d_mod
        import torch

        candidate_payload = None
        stagec_metadata: dict[str, object] = {}
        prefix_sha256 = None
        reward_revision = None
        route_efficiency = False
        route_efficiency_v2 = False
        ramp_out_revision = False
        return_intake_enabled = False
        if args.stagec_v2:
            prefix_sha256 = _sha256_file(prefix_path)
            try:
                candidate_payload = torch.load(
                    args.checkpoint, map_location="cpu", weights_only=True
                )
            except TypeError:
                candidate_payload = torch.load(
                    args.checkpoint, map_location="cpu"
                )
            stagec_metadata = validate_composite_metadata(
                candidate_payload.get("stagec_v2"), prefix_sha256
            )
            reward_revision = stagec_metadata.get("reward_revision")
            if reward_revision not in (
                None,
                *SUPPORTED_ROUTE_EFFICIENCY_REVISIONS,
            ):
                raise ValueError(
                    f"unsupported Stage C reward revision: {reward_revision!r}"
                )
            route_efficiency = (
                reward_revision in SUPPORTED_ROUTE_EFFICIENCY_REVISIONS
            )
            route_efficiency_v2 = reward_revision in (
                "outer_rail_v2",
                "outer_rail_v3",
                *RAMP_OUT_REVISIONS,
            )
            ramp_out_revision = reward_revision in RAMP_OUT_REVISIONS
            return_intake_revision = (
                reward_revision in RETURN_INTAKE_REVISIONS
            )
            return_intake_enabled = (
                bool(stagec_metadata["intake_during_return"])
                if return_intake_revision
                else False
            )
            if return_intake_revision and not return_intake_enabled:
                raise ValueError(
                    f"{reward_revision} requires intake_during_return=true"
                )

        def stagec_int_arg(cli_value, key: str, fallback: int) -> int:
            if not args.stagec_v2:
                return int(fallback if cli_value is None else cli_value)
            checkpoint_value = int(stagec_metadata.get(key, fallback))
            if cli_value is not None and int(cli_value) != checkpoint_value:
                raise ValueError(
                    f"Stage C GUI {key} mismatch: CLI {int(cli_value)} != "
                    f"checkpoint {checkpoint_value}"
                )
            return checkpoint_value

        stagec_target_load = stagec_int_arg(
            args.stagec_v2_target_load, "target_load", 15
        )
        stagec_reserve_count = stagec_int_arg(
            args.stagec_v2_reserve_count, "reserve_count", 18
        )
        stagec_reserve_batches = stagec_int_arg(
            args.stagec_v2_reserve_batches, "reserve_batches", 3
        )

        # ---- build ONE field with the training obs pipeline --------------
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=1,
                template_usd=args.template,
                cameras=True,                       # required: policy is vision-based
                episode_len_s=args.episode_len_s,
                preload_prob=args.preload_prob,
                mask_illegal_fire=args.mask_illegal_fire,
                spawn_under_trench=args.spawn_under_trench,
                # trench start only makes sense with storage unlocked (eval_route.py:67)
                lock_storage_extended=not args.spawn_under_trench,
                dump_on_press=args.dump_on_press,
                max_dump_ticks=(
                    int(args.max_dump_ticks)
                    if args.max_dump_ticks is not None
                    else (
                        int(stagec_metadata.get("max_dump_ticks", 180))
                        if args.stagec_v2
                        else 90
                    )
                ),
                collect_reward_weight=0.3 if args.stagec_v2 else 1.5,
                empty_own_court_penalty=0.0 if args.stagec_v2 else 0.02,
                stagec_v2=bool(args.stagec_v2),
                stage_d=bool(args.stage_d),
                stage_d_first_inactive=args.stage_d_first_inactive,
                stage_d_ferry=bool(args.stage_d_ferry),
                stage_d_ferry_dump_on_press=bool(args.stage_d_ferry_dump_on_press),
                stage_d_ferry_blackout_only=bool(args.stage_d_ferry_blackout_only),
                stage_d_ferry_entitled_only=bool(args.stage_d_ferry_entitled_only),
                stage_d_ferry_min_load=int(args.stage_d_ferry_min_load),
                stage_d_return_when_live=bool(args.stage_d_return_when_live),
                stage_d_live_return_load=int(args.stage_d_live_return_load),
                stage_d_return_lead_s=float(args.stage_d_return_lead_s),
                stage_d_owncourt_loop=bool(args.stage_d_owncourt_loop),
                stage_d_owncourt_min_balls=int(args.stage_d_owncourt_min_balls),
                stage_d_owncourt_rearm=bool(args.stage_d_owncourt_rearm),
                stage_d_owncourt_blackout_intake=bool(
                    args.stage_d_owncourt_blackout_intake
                ),
                cycle_v2_reset_modes=(args.stagec_v2_reset_mode,),
                cycle_v2_target_load=stagec_target_load,
                cycle_v2_reserve_count=stagec_reserve_count,
                cycle_v2_reserve_batches=stagec_reserve_batches,
                cycle_v2_refresh_ramp_side_on_dump=(
                    bool(stagec_metadata["refresh_ramp_side_on_dump"])
                    if route_efficiency
                    else False
                ),
                cycle_v2_ramp_side_deadband_x=(
                    float(stagec_metadata["ramp_side_deadband_x"])
                    if route_efficiency
                    else 0.25
                ),
                cycle_v2_require_ramp_out=(
                    bool(stagec_metadata["require_ramp_out"])
                    if ramp_out_revision
                    else False
                ),
                cycle_v2_ramp_out_half_width=(
                    float(stagec_metadata["ramp_out_half_width"])
                    if ramp_out_revision
                    else 0.90
                ),
                cycle_v2_ramp_out_bonus=(
                    float(stagec_metadata["ramp_out_bonus"])
                    if ramp_out_revision
                    else 0.0
                ),
                cycle_v2_off_ramp_exit_penalty=(
                    float(stagec_metadata["off_ramp_exit_penalty"])
                    if ramp_out_revision
                    else 0.0
                ),
                cycle_v2_postdump_require_target_load=(
                    bool(stagec_metadata["postdump_require_target_load"])
                    if reward_revision in POSTDUMP_TARGET_REVISIONS
                    else False
                ),
                cycle_v2_postdump_complete_cycle=(
                    bool(stagec_metadata["postdump_complete_cycle"])
                    if reward_revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS
                    else False
                ),
                cycle_v2_postdump_depleted_count=(
                    int(stagec_metadata["postdump_depleted_count"])
                    if reward_revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS
                    else 0
                ),
                cycle_v2_postdump_depleted_prob=(
                    float(stagec_metadata["postdump_depleted_prob"])
                    if reward_revision in POSTDUMP_COMPLETE_CYCLE_REVISIONS
                    else 0.0
                ),
                cycle_v2_preferred_repeat_load=(
                    int(stagec_metadata["preferred_repeat_load"])
                    if reward_revision in SCORE_EFFICIENCY_REVISIONS
                    else 0
                ),
                cycle_v2_collect_until_preferred=(
                    reward_revision in COLLECT_UNTIL_PREFERRED_REVISIONS
                ),
                cycle_v2_collect_stall_steps=(
                    int(stagec_metadata["collect_stall_steps"])
                    if reward_revision in COLLECT_UNTIL_PREFERRED_REVISIONS
                    else 0
                ),
                cycle_v2_return_time_guard=(
                    float(stagec_metadata["return_time_guard"])
                    if reward_revision in COLLECT_UNTIL_PREFERRED_REVISIONS
                    else 0.0
                ),
                cycle_v2_intake_during_return=return_intake_enabled,
                cycle_v2_repeat_load_return_bonus=(
                    float(stagec_metadata["repeat_load_return_bonus"])
                    if reward_revision in SCORE_EFFICIENCY_REVISIONS
                    else 0.0
                ),
                cycle_v2_repeat_load_score_bonus=(
                    float(stagec_metadata["repeat_load_score_bonus"])
                    if reward_revision in SCORE_EFFICIENCY_REVISIONS
                    else 0.0
                ),
                cycle_v2_outer_rail_enter_x=(
                    float(stagec_metadata["outer_rail_enter_x"])
                    if route_efficiency
                    else 2.85
                ),
                cycle_v2_outer_rail_exit_x=(
                    float(stagec_metadata["outer_rail_exit_x"])
                    if route_efficiency
                    else 2.55
                ),
                cycle_v2_outer_rail_max_x=(
                    float(stagec_metadata["outer_rail_max_x"])
                    if route_efficiency
                    else 3.60
                ),
                cycle_v2_outer_rail_grace_steps=(
                    int(stagec_metadata["outer_rail_grace_steps"])
                    if route_efficiency
                    else 20
                ),
                cycle_v2_outer_rail_penalty_per_step=(
                    float(stagec_metadata["outer_rail_penalty_per_step"])
                    if route_efficiency
                    else 0.0
                ),
                cycle_v2_outer_rail_penalty_cap=(
                    float(stagec_metadata["outer_rail_penalty_cap"])
                    if route_efficiency
                    else 8.0
                ),
                cycle_v2_outer_rail_min_scale=(
                    float(stagec_metadata["outer_rail_min_scale"])
                    if route_efficiency_v2
                    else 0.0
                ),
                cycle_v2_outer_rail_escalation_steps=(
                    int(stagec_metadata["outer_rail_escalation_steps"])
                    if route_efficiency_v2
                    else 0
                ),
                cycle_v2_outer_rail_max_multiplier=(
                    float(stagec_metadata["outer_rail_max_multiplier"])
                    if route_efficiency_v2
                    else 1.0
                ),
                cycle_v2_intake_substeps=(
                    int(stagec_metadata["intake_substeps"])
                    if route_efficiency_v2
                    else 1
                ),
                seed=args.seed,
            )
        )

        # Nice default camera on the blue court (reuse isaac_scene.py's framing).
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(
                eye=np.array([7.2, -10.2, 4.6]),
                target=np.array([0.0, -3.9, 0.65]),
                camera_prim_path="/OmniverseKit_Persp",
            )
        except Exception as cam_err:  # noqa: BLE001
            print("PLAY_CAMERA_VIEW_FAILED", repr(cam_err), flush=True)

        # ---- first observation (reset then a zero step, exactly like eval) ---
        env.reset_all()
        obs, *_ = env.step(np.zeros((1, 7), np.float32))
        frames = to_policy_frames(obs["rgb"])
        cams, fh, fw = frames.shape[1], frames.shape[2], frames.shape[3]

        # ---- construct the agent from the LIVE obs shapes ----------------
        # Mirrors train_drqv2.py:227 (NOT eval's bare DrQConfig()): deriving the
        # dims from the running env guarantees the encoder/actor tensor shapes
        # match the checkpoint regardless of camera count or proprio width.
        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=cams,
                frame_h=fh,
                frame_w=fw,
                proprio_dim=obs["proprio"].shape[1],
                privileged_dim=obs["privileged"].shape[1],
            )
        )
        prefix_agent = None
        if args.stagec_v2:
            prefix_agent = DrQV2Agent(
                DrQConfig(
                    frame_channels=cams,
                    frame_h=fh,
                    frame_w=fw,
                    proprio_dim=LEGACY_PROPRIO_DIM,
                    privileged_dim=obs["privileged"].shape[1],
                )
            )
            prefix_agent.load(str(prefix_path))
        agent.load(args.checkpoint)
        print(
            f"PLAY_LOADED {args.checkpoint} steps={agent.train_steps} "
            f"device={agent.device} frame_shape={[cams, fh, fw]}"
            + (
                f" schema={SCHEMA_VERSION} action_policy={ACTION_POLICY} "
                f"prefix_sha256={prefix_sha256} "
                f"reward_revision={reward_revision or 'legacy'} "
                f"intake_substeps={stagec_metadata.get('intake_substeps', 1)} "
                f"intake_during_return={return_intake_enabled}"
                if args.stagec_v2
                else ""
            ),
            flush=True,
        )

        # ---- real-time pacing target (6 physics steps @ 60 Hz = 0.1 s) -----
        step_dt = env.spec.physics_steps_per_action / env.spec.physics_hz  # 0.1 s
        realtime = not args.no_realtime

        overlay = None if args.no_clock_overlay else _MatchClockOverlay()
        ep_index = 0
        policy_step = 0
        action_rng = np.random.default_rng(args.seed + 1)  # --noise-std dither
        while app.is_running():
            wall0 = time.perf_counter()

            # OBS -> ACT -> STEP  (identical convention to eval_route.py:88-98)
            frames = to_policy_frames(obs["rgb"])
            candidate_actions = agent.act(
                frames, obs["proprio"], explore=bool(args.explore)
            )
            if args.noise_std > 0.0:
                candidate_actions = np.clip(
                    candidate_actions
                    + action_rng.normal(
                        0.0, args.noise_std, size=candidate_actions.shape
                    ),
                    -1.0, 1.0,
                ).astype(np.float32)
            if args.stagec_v2:
                prefix_actions = prefix_agent.act(
                    frames,
                    obs["proprio"][:, :LEGACY_PROPRIO_DIM],
                    explore=False,
                )
                actions = compose_phase_actions(
                    prefix_actions, candidate_actions, obs["proprio"]
                )
                actions = apply_executed_action_policy(
                    actions,
                    obs["proprio"],
                    intake_during_return=return_intake_enabled,
                    # Under Stage-D ferry the suffix keeps the ferry bit and it is
                    # forced off only in SCORE; without this the GUI would mask
                    # every post-FIRST ferry and you would never see one.
                    stage_d_ferry=bool(args.stage_d_ferry),
                )
            else:
                actions = candidate_actions
            try:
                obs, rewards, dones, info = env.step(actions.astype(np.float32))
            except SimulationUnstable as exc:
                print("PLAY_SIM_UNSTABLE", repr(exc), flush=True)
                break

            policy_step += 1

            if overlay is not None:
                try:
                    slot = env.slots[0]
                    fi = getattr(slot, "stage_d_first_inactive", None) or (
                        args.stage_d_first_inactive if args.stage_d else None
                    )
                    live = (
                        bool(stage_d_mod.blue_hub_eligible(slot.clock_s, fi))
                        if args.stage_d
                        else True
                    )
                    cyc = getattr(getattr(slot, "cycle_v2", None), "phase", None)
                    overlay.update(
                        slot.clock_s,
                        args.episode_len_s,
                        fi or "n/a",
                        live,
                        slot.router.scored["blue"],
                        getattr(cyc, "value", "-"),
                        len(slot.controller.magazine),
                    )
                except Exception:
                    pass

            if bool(dones[0]):
                # step() already auto-reset this env into a fresh episode; its
                # terminal counters are in info["episode_stats"] before the wipe.
                stats = info.get("episode_stats", {}).get(0, {})
                ep_index += 1
                print(
                    f"PLAY_EPISODE {ep_index} scored={stats.get('scored')} "
                    f"collected={stats.get('collected')} "
                    f"shots_fired={stats.get('shots_fired')} "
                    f"cycles={stats.get('cycles_completed', 0)} "
                    f"mode={stats.get('reset_mode', 'legacy')}",
                    flush=True,
                )

            if policy_step % 50 == 0:
                s = env.slots[0]
                print(
                    f"PLAY step={policy_step} reward={float(rewards[0]):+.2f} "
                    f"scored={int(s.score_seen)} collected={int(s.collected_seen)}",
                    flush=True,
                )

            # pace to wall-clock so it looks like a real 10 Hz match
            if realtime:
                remaining = step_dt - (time.perf_counter() - wall0)
                if remaining > 0:
                    time.sleep(remaining)

        env.close()
    except KeyboardInterrupt:
        print("PLAY_INTERRUPTED", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
