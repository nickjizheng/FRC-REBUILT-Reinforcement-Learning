"""Vectorized full-physics competition environment (Isaac Sim, N cloned fields).

Implements the converged plan in ``RL_BRAINSTORM.md`` / ``docs/VECTORIZATION_PLAN.md``:

- N complete fields (robot + FUEL + hub) cloned from the exported USD template
  onto one shared GPU PhysX scene (throughput is FUEL-count-bound; stages A-C
  use small FUEL subsets and clear the 8 policy-tx/s gate with headroom).
- The existing, well-tested single-robot stack (``CompetitionRobotController``,
  ``HubRouter``) runs per env behind thin adapters that translate between the
  batched views and the single-env numpy interface, shifting every world
  position by the env origin so all field math stays in field coordinates.
- Actions are the frozen 7-D contract (``xrc_rebuilt.rl.spec``); the policy
  runs at 10 Hz (action repeat 6 over 60 Hz physics; controller cadence 30 Hz
  exactly like the interactive GUI).
- Observations: the 3-camera 640x360 rig (rendered once per policy step) plus
  a non-privileged proprio vector; a privileged vector is provided separately
  for the asymmetric critic (training-time only, never an actor input).
- Reward: legal blue score under the match scoring rules (router-confirmed),
  plus curriculum shaping terms logged separately.

This module must be imported only after ``SimulationApp`` is created.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xrc_rebuilt.rl.spec import CompetitionRLSpec, decode_policy_actions

FUEL_RADIUS_M = 0.076


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class VecEnvCfg:
    num_envs: int = 2
    template_usd: str = ""            # exported env template (required)
    spacing_m: float = 24.0
    episode_len_s: float = 20.0       # stage A: short acquisition episodes
    cameras: bool = True
    camera_names: tuple[str, ...] = ()  # default: rig baseline from the robot module
    # stage A settings
    lock_storage_extended: bool = True
    sandbox_scoring: bool = True      # both hubs always eligible (stages A/B)
    spawn_xy_range: tuple[float, float, float, float] = (-2.4, 2.4, -1.8, 1.2)
    spawn_yaw_range_deg: tuple[float, float] = (-180.0, 180.0)
    proprio_noise_std: float = 0.02
    seed: int = 2026


@dataclass
class EnvSlot:
    """Everything owned by one cloned environment."""

    index: int
    origin: np.ndarray
    controller: Any
    router: Any
    articulation: Any
    fuel: Any
    clock_s: float = 0.0
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(7, np.float32))
    score_seen: int = 0
    collected_seen: int = 0
    reward_components: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# single-env adapters over the batched views
# --------------------------------------------------------------------------- #
class _GainShim:
    """`get_articulation_controller()` facade: set_gains / set_max_efforts."""

    def __init__(self, batched: Any, env_index: int):
        self._b = batched
        self._idx = [env_index]

    def set_gains(self, kps=None, kds=None) -> None:
        kps = None if kps is None else np.asarray(kps, np.float32)[None, :]
        kds = None if kds is None else np.asarray(kds, np.float32)[None, :]
        self._b.set_gains(kps=kps, kds=kds, indices=self._idx)

    def set_max_efforts(self, efforts) -> None:
        self._b.set_max_efforts(
            np.asarray(efforts, np.float32)[None, :], indices=self._idx
        )


class EnvArticulationAdapter:
    """Single-robot facade over the batched Articulation for one env.

    World poses are shifted by the env origin so the controller keeps working
    in field coordinates (hub targets, zones, trench boxes).
    """

    def __init__(self, batched: Any, env_index: int, origin: np.ndarray):
        self._b = batched
        self._i = env_index
        self._idx = [env_index]
        self._origin = origin.astype(np.float32)

    # -- identity ---------------------------------------------------------
    @property
    def dof_names(self):
        return self._b.dof_names

    def get_articulation_controller(self) -> _GainShim:
        return _GainShim(self._b, self._i)

    def set_max_efforts(self, efforts) -> None:
        self._b.set_max_efforts(
            np.asarray(efforts, np.float32)[None, :], indices=self._idx
        )

    # -- joints -----------------------------------------------------------
    @staticmethod
    def _np(value: Any) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    def get_joint_positions(self) -> np.ndarray:
        return self._np(self._b.get_joint_positions(indices=self._idx))[0]

    def set_joint_positions(self, positions) -> None:
        self._b.set_joint_positions(
            np.asarray(positions, np.float32)[None, :], indices=self._idx
        )

    def get_joint_velocities(self) -> np.ndarray:
        return self._np(self._b.get_joint_velocities(indices=self._idx))[0]

    def set_joint_velocities(self, velocities) -> None:
        self._b.set_joint_velocities(
            np.asarray(velocities, np.float32)[None, :], indices=self._idx
        )

    def apply_action(self, action: Any) -> None:
        """NaN-skip semantics identical to SingleArticulation.apply_action."""
        jp = getattr(action, "joint_positions", None)
        jv = getattr(action, "joint_velocities", None)
        if jp is not None:
            jp = np.asarray(jp, np.float32)
            live = np.flatnonzero(~np.isnan(jp))
            if live.size:
                self._b.set_joint_position_targets(
                    jp[live][None, :], indices=self._idx, joint_indices=live
                )
        if jv is not None:
            jv = np.asarray(jv, np.float32)
            live = np.flatnonzero(~np.isnan(jv))
            if live.size:
                self._b.set_joint_velocity_targets(
                    jv[live][None, :], indices=self._idx, joint_indices=live
                )

    # -- chassis ----------------------------------------------------------
    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self._b.get_world_poses(indices=self._idx)
        position = self._np(positions)[0].astype(np.float32) - self._origin
        return position, self._np(orientations)[0].astype(np.float32)

    def set_world_pose(self, position, orientation) -> None:
        self._b.set_world_poses(
            positions=(np.asarray(position, np.float32) + self._origin)[None, :],
            orientations=np.asarray(orientation, np.float32)[None, :],
            indices=self._idx,
        )

    def get_linear_velocity(self) -> np.ndarray:
        return self._np(self._b.get_linear_velocities(indices=self._idx))[0]

    def get_angular_velocity(self) -> np.ndarray:
        return self._np(self._b.get_angular_velocities(indices=self._idx))[0]

    def set_velocities_zero(self) -> None:
        zeros = np.zeros((1, 6), np.float32)
        try:
            self._b.set_velocities(zeros, indices=self._idx)
        except (AttributeError, TypeError):
            self._b.set_linear_velocities(zeros[:, :3], indices=self._idx)
            self._b.set_angular_velocities(zeros[:, 3:], indices=self._idx)


class EnvFuelAdapter:
    """Per-env slice of the global FUEL RigidPrim, origin-shifted.

    Exposes exactly the surface the controller + HubRouter consume:
    count, get/set_world_poses, get/set linear/angular velocities, with local
    ball indices 0..count-1.
    """

    def __init__(self, batched: Any, env_index: int, fuel_per_env: int, origin: np.ndarray):
        self._b = batched
        self.count = int(fuel_per_env)
        self._base = env_index * self.count
        self._all = np.arange(self._base, self._base + self.count, dtype=np.int32)
        self._origin = origin.astype(np.float32)

    @staticmethod
    def _np(value: Any) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    def _global(self, indices) -> np.ndarray:
        return np.asarray(indices, dtype=np.int32) + self._base

    def get_world_poses(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self._b.get_world_poses(indices=self._all)
        return (
            self._np(positions).astype(np.float32) - self._origin[None, :],
            self._np(orientations).astype(np.float32),
        )

    def set_world_poses(self, positions=None, indices=None, orientations=None) -> None:
        shifted = np.asarray(positions, np.float32) + self._origin[None, :]
        self._b.set_world_poses(
            positions=shifted,
            orientations=orientations,
            indices=self._global(indices if indices is not None else range(self.count)),
        )

    def get_linear_velocities(self) -> np.ndarray:
        return self._np(self._b.get_linear_velocities(indices=self._all)).astype(
            np.float32
        )

    def set_linear_velocities(self, velocities, indices=None) -> None:
        self._b.set_linear_velocities(
            np.asarray(velocities, np.float32),
            indices=self._global(indices if indices is not None else range(self.count)),
        )

    def set_angular_velocities(self, velocities, indices=None) -> None:
        self._b.set_angular_velocities(
            np.asarray(velocities, np.float32),
            indices=self._global(indices if indices is not None else range(self.count)),
        )


# --------------------------------------------------------------------------- #
# the vectorized environment
# --------------------------------------------------------------------------- #
class VecCompetitionEnv:
    """N cloned full-physics fields with the real robot stack per env."""

    def __init__(self, cfg: VecEnvCfg):
        self.cfg = cfg
        self.spec = CompetitionRLSpec()
        self.spec.validate()
        self.rng = np.random.default_rng(cfg.seed)
        self._build_scene()
        self._build_views()
        self._build_cameras()
        self.reset_all()

    # -- construction ------------------------------------------------------
    def _build_scene(self) -> None:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.cloner import GridCloner

        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        self.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateEnableGPUDynamicsAttr(True)
        physx.CreateBroadphaseTypeAttr("GPU")
        physx.CreateTimeStepsPerSecondAttr(60)
        physx.CreateGpuFoundLostPairsCapacityAttr(2097152)
        physx.CreateGpuTotalAggregatePairsCapacityAttr(2097152)
        physx.CreateGpuMaxRigidContactCountAttr(2097152)
        physx.CreateGpuMaxRigidPatchCountAttr(327680)
        UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(850)

        envs_root = "/World/envs"
        UsdGeom.Xform.Define(stage, envs_root)
        env0 = f"{envs_root}/env_0"
        UsdGeom.Xform.Define(stage, env0)
        stage.GetPrimAtPath(env0).GetReferences().AddReference(self.cfg.template_usd)

        cloner = GridCloner(spacing=self.cfg.spacing_m)
        cloner.define_base_env(envs_root)
        paths = cloner.generate_paths(f"{envs_root}/env", self.cfg.num_envs)
        positions = cloner.clone(
            source_prim_path=env0,
            prim_paths=paths,
            replicate_physics=True,
            base_env_path=envs_root,
        )
        self.env_origins = np.asarray(positions, dtype=np.float32)

        self.sim = SimulationContext(
            physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.0
        )
        self.sim.reset()

    def _build_views(self) -> None:
        from pxr import UsdPhysics
        from isaacsim.core.prims import Articulation, RigidPrim
        from xrc_rebuilt.competition_robot import (
            CompetitionRobotController,
        )
        from xrc_rebuilt.isaac_scene import HubRouter

        art_roots = [
            p.GetPath().pathString
            for p in self.stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        fuel_prims = sorted(
            p.GetPath().pathString
            for p in self.stage.Traverse()
            if p.GetName().startswith("Fuel_") and p.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        env0_fuel = [p for p in fuel_prims if "/env_0/" in p]
        self.fuel_per_env = len(env0_fuel)
        robot_expr = art_roots[0].replace("/env_0/", "/env_.*/")
        fuel_expr = env0_fuel[0].rsplit("/", 1)[0].replace("/env_0/", "/env_.*/") + "/Fuel_.*"

        self.robots = Articulation(robot_expr)
        self.robots.initialize()
        self.fuel = RigidPrim(fuel_expr)
        self.fuel.initialize()
        assert int(self.robots.count) == self.cfg.num_envs
        assert int(self.fuel.count) == self.cfg.num_envs * self.fuel_per_env

        # remember the template FUEL layout (field frame) for resets
        first = EnvFuelAdapter(self.fuel, 0, self.fuel_per_env, self.env_origins[0])
        self._fuel_home, self._fuel_home_quat = first.get_world_poses()

        env0_robot_root = art_roots[0]
        self.slots: list[EnvSlot] = []
        for i in range(self.cfg.num_envs):
            origin = self.env_origins[i]
            articulation = EnvArticulationAdapter(self.robots, i, origin)
            fuel_view = EnvFuelAdapter(self.fuel, i, self.fuel_per_env, origin)
            controller = CompetitionRobotController(
                alliance_lock="blue",
                usd_root_path=env0_robot_root.replace("/env_0/", f"/env_{i}/"),
            )
            controller.initialize(articulation)
            router = HubRouter(fuel_view, self.fuel_per_env, seed=self.cfg.seed + i)
            router.sandbox = bool(self.cfg.sandbox_scoring)
            self.slots.append(
                EnvSlot(
                    index=i,
                    origin=origin,
                    controller=controller,
                    router=router,
                    articulation=articulation,
                    fuel=fuel_view,
                )
            )

    def _build_cameras(self) -> None:
        self.cameras: dict[tuple[int, str], Any] = {}
        if not self.cfg.cameras:
            return
        from isaacsim.sensors.camera import Camera
        from xrc_rebuilt.competition_robot import (
            CAMERA_BASELINE_NAMES,
            CAMERA_PRIM_PATHS,
            CAMERA_RESOLUTION,
            ROBOT_ROOT_PATH,
        )

        names = self.cfg.camera_names or CAMERA_BASELINE_NAMES
        self.camera_names = tuple(names)
        self.camera_resolution = CAMERA_RESOLUTION
        for slot in self.slots:
            root = slot.controller.usd_root_path
            for name in names:
                path = CAMERA_PRIM_PATHS[name].replace(ROBOT_ROOT_PATH, root, 1)
                camera = Camera(prim_path=path, resolution=CAMERA_RESOLUTION)
                camera.initialize()
                self.cameras[(slot.index, name)] = camera
        # RTX render products publish allocated all-black buffers while they
        # start up asynchronously (same race the camera-preview tool fixed).
        # Gate on real content once, so the first policy observation is valid.
        # NOTE: headless Camera annotators are only fed by full Kit updates
        # (sim.step(render=True)); a bare sim.render() leaves them black.
        self._camera_ready = False
        stds: dict[tuple[int, str], float] = {}
        for _ in range(300):
            self.sim.step(render=True)
            stds = {
                key: (
                    float(np.asarray(camera.get_rgba())[..., :3].std())
                    if np.asarray(camera.get_rgba()).size
                    else 0.0
                )
                for key, camera in self.cameras.items()
            }
            if all(value > 1.0 for value in stds.values()):
                self._camera_ready = True
                break
        laggards = sorted(
            (key for key, value in stds.items() if value <= 1.0), key=str
        )
        print(
            f"VECENV_CAMERAS_READY {self._camera_ready}"
            + (f" laggards={laggards}" if laggards else ""),
            flush=True,
        )

    # -- reset -------------------------------------------------------------
    def _reset_slot(self, slot: EnvSlot) -> None:
        cfg = self.cfg
        # robot pose: random field spawn inside the configured box
        x = float(self.rng.uniform(cfg.spawn_xy_range[0], cfg.spawn_xy_range[1]))
        y = float(self.rng.uniform(cfg.spawn_xy_range[2], cfg.spawn_xy_range[3]))
        yaw = math.radians(
            float(self.rng.uniform(cfg.spawn_yaw_range_deg[0], cfg.spawn_yaw_range_deg[1]))
        )
        half = yaw * 0.5
        slot.articulation.set_world_pose(
            np.asarray([x, y, 0.02], np.float32),
            np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], np.float32),
        )
        slot.articulation.set_velocities_zero()

        # FUEL back to the template layout, at rest
        slot.fuel.set_world_poses(
            positions=self._fuel_home.copy(),
            orientations=self._fuel_home_quat.copy(),
            indices=np.arange(slot.fuel.count),
        )
        zeros = np.zeros((slot.fuel.count, 3), np.float32)
        slot.fuel.set_linear_velocities(zeros, indices=np.arange(slot.fuel.count))
        slot.fuel.set_angular_velocities(zeros, indices=np.arange(slot.fuel.count))

        # fresh robot logic state: EVERYTHING per-episode is cleared, including
        # drive slew memory and the shooter FSM (reviewer finding: FSM/motion
        # state used to leak across episodes)
        controller = slot.controller
        controller.reset_match_state()
        controller.snap_storage_state(bool(cfg.lock_storage_extended))
        controller.intake_on = False

        router = slot.router
        router.pending.clear()
        router.blocked_until_clear = set()
        router.released_watch.clear()
        router.exit_free_at.clear()
        router.scored = {"red": 0, "blue": 0}
        router.detected = 0
        router.released = 0

        slot.clock_s = 0.0
        slot.prev_action[:] = 0.0
        slot.score_seen = 0
        slot.collected_seen = 0

    def reset_all(self) -> np.ndarray:
        for slot in self.slots:
            self._reset_slot(slot)
        # settle one physics step so poses/velocities take effect
        self.sim.step(render=False)
        return np.arange(self.cfg.num_envs)

    # -- stepping ----------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        """Advance one policy step (six 60 Hz physics steps) for every env."""
        cfg = self.cfg
        decoded = decode_policy_actions(actions, self.spec)
        dones = np.zeros(cfg.num_envs, dtype=bool)
        rewards = np.zeros(cfg.num_envs, dtype=np.float32)

        last_k = self.spec.physics_steps_per_action - 1
        for k in range(self.spec.physics_steps_per_action):
            # the final physics step of the window is a full Kit update so the
            # camera render products are fed exactly once per policy step
            self.sim.step(render=bool(self.cameras) and k == last_k)
            if k % 2 == 1:  # 30 Hz controller cadence, mirroring the GUI loop
                for slot in self.slots:
                    i = slot.index
                    controller = slot.controller
                    slot.clock_s += 2.0 / 60.0
                    controller.intake_on = bool(decoded.intake_on[i])
                    if not cfg.lock_storage_extended:
                        controller.set_storage_extended(bool(decoded.storage_extended[i]))
                    controller.step_mechanisms(dt_s=2.0 / 60.0)
                    controller.step_intake(
                        slot.fuel, set(slot.router.pending), dt_s=2.0 / 60.0
                    )
                    slot.router.step(slot.clock_s)
                    sm = controller.state_machine
                    sm.set_continuous(False)
                    fire = bool(decoded.shoot_blue[i] or decoded.ferry[i])
                    sm.press_hold() if fire else sm.release_hold()
                    sm.set_emergency_stop(False)
                    sm.auto_align = fire
                    driver = decoded.driver[i]
                    moving = bool(np.any(np.abs(driver) > 0.03))
                    if moving:
                        controller.drive(
                            float(driver[0]), float(driver[2]), strafe=float(driver[1])
                        )
                    controller.update(
                        slot.fuel,
                        now_s=slot.clock_s,
                        alliance="blue",
                        hub_active=True,
                        allow_drive=not moving,
                        fire_mode="ferry" if bool(decoded.ferry[i]) else "score",
                    )

        info: dict[str, Any] = {"reward_components": []}
        for slot in self.slots:
            i = slot.index
            scored = int(slot.router.scored["blue"])
            collected = int(slot.controller.balls_collected)
            r_score = 10.0 * (scored - slot.score_seen)
            r_collect = 1.5 * (collected - slot.collected_seen)
            r_action = -0.01 * float(np.square(actions[i, :3]).sum())
            rewards[i] = r_score + r_collect + r_action
            info["reward_components"].append(
                {"score": r_score, "collect": r_collect, "action": r_action}
            )
            slot.score_seen = scored
            slot.collected_seen = collected
            slot.prev_action[:] = np.clip(actions[i], -1.0, 1.0)
            if slot.clock_s >= cfg.episode_len_s:
                dones[i] = True
        # terminal episode stats, captured BEFORE auto-reset wipes the counters
        info["episode_stats"] = {
            int(slot.index): {
                "scored": int(slot.router.scored["blue"]),
                "collected": int(slot.controller.balls_collected),
                "shots_fired": int(slot.controller.shots_fired),
            }
            for slot in self.slots
            if dones[slot.index]
        }
        # Auto-reset BEFORE observing, so the returned observation for a done
        # env is the FIRST frame of its new episode (never a stale terminal
        # frame).  Terminal next-observations are unused by the learner: the
        # stored done flag zeroes the bootstrap on episode-ending transitions.
        if bool(dones.any()):
            for slot in self.slots:
                if dones[slot.index]:
                    self._reset_slot(slot)
            # settle step; rendered so post-reset camera frames are fresh
            self.sim.step(render=bool(self.cameras))
        observations = self._observe(decoded)
        return observations, rewards, dones, info

    # -- observations -------------------------------------------------------
    def _observe(self, decoded) -> dict[str, np.ndarray]:
        n = self.cfg.num_envs
        obs: dict[str, np.ndarray] = {}
        if self.cameras:
            w, h = self.camera_resolution
            frames = np.zeros((n, len(self.camera_names), h, w, 3), np.uint8)
            for (i, name), camera in self.cameras.items():
                rgba = np.asarray(camera.get_rgba())
                if rgba.size and rgba.ndim == 3:
                    frames[i, self.camera_names.index(name)] = rgba[..., :3]
            obs["rgb"] = frames

        proprio = np.zeros((n, 22), np.float32)
        privileged = np.zeros((n, 26), np.float32)
        for slot in self.slots:
            i = slot.index
            controller = slot.controller
            position, quat = controller.chassis_pose()
            yaw = controller.chassis_yaw()
            linear, yaw_rate = controller.chassis_velocity()
            noise = self.rng.normal(0.0, self.cfg.proprio_noise_std, 4).astype(np.float32)
            mag = len(controller.magazine)
            state = controller.state_machine.state.value
            proprio[i] = np.concatenate(
                [
                    np.asarray(
                        [
                            (position[0] + noise[0]) / 8.0,
                            (position[1] + noise[1]) / 8.0,
                            math.sin(yaw) + noise[2] * 0.1,
                            math.cos(yaw) + noise[3] * 0.1,
                            linear[0] / 4.0,
                            linear[1] / 4.0,
                            yaw_rate / 6.0,
                            slot.clock_s / max(1.0, self.cfg.episode_len_s),
                            mag / 8.0,
                            1.0 if controller.intake_on else 0.0,
                            controller.storage_position,
                            1.0 if state in ("READY", "FEEDING") else 0.0,
                            1.0,  # blue hub eligible (sandbox stage A)
                            float(controller.shots_fired) / 20.0,
                            float(slot.router.scored["blue"]) / 20.0,
                        ],
                        np.float32,
                    ),
                    slot.prev_action.astype(np.float32),
                ]
            )
            # privileged (critic-only): true pose + nearest-fuel geometry
            balls, _ = slot.fuel.get_world_poses()
            deltas = balls[:, :2] - position[None, :2]
            distances = np.linalg.norm(deltas, axis=1)
            nearest = np.argsort(distances)[:4]
            privileged[i] = np.concatenate(
                [
                    np.asarray(
                        [
                            position[0] / 8.0,
                            position[1] / 8.0,
                            math.sin(yaw),
                            math.cos(yaw),
                            linear[0] / 4.0,
                            linear[1] / 4.0,
                            yaw_rate / 6.0,
                            mag / 8.0,
                            float(slot.router.scored["blue"]) / 20.0,
                            float((distances < 1.5).sum()) / 16.0,
                        ],
                        np.float32,
                    ),
                    (deltas[nearest] / 8.0).reshape(-1).astype(np.float32),
                    (distances[nearest, None] / 8.0).reshape(-1).astype(np.float32),
                    np.asarray(
                        [
                            1.0 if controller.intake_on else 0.0,
                            controller.storage_position,
                            float(controller.shots_fired) / 20.0,
                            slot.clock_s / max(1.0, self.cfg.episode_len_s),
                        ],
                        np.float32,
                    ),
                ]
            )
        obs["proprio"] = proprio
        obs["privileged"] = privileged
        return obs

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass
