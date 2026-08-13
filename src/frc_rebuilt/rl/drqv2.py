"""DrQ-v2 with an asymmetric privileged critic (converged baseline, Turn 2-4).

Pixel actor (multi-camera frames + proprio), twin critic that additionally
receives the privileged vector (training-time only - never an actor input,
and nothing is distilled).  Random-shift augmentation, n-step TD, EMA target
critic, scheduled exploration noise: the standard DrQ-v2 recipe sized for the
9x90x160 policy view (three 640x360 cameras downsampled 4x, channel-stacked).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_STAGEC_V2_PROPRIO_DIM = 30
_STAGEC_V2_PHASE_OFFSET = 22
_STAGEC_V2_FIRST = 0
_STAGEC_V2_COLLECT = 2
_STAGEC_V2_RETURN = 3
_STAGEC_V2_SCORE = 4


def _stagec_v2_executed_action_policy_torch(
    actions: torch.Tensor,
    proprio: torch.Tensor,
    *,
    intake_during_return: bool = False,
    stage_d_ferry: bool = False,
) -> torch.Tensor:
    """Apply the collector's Stage-C v2 action contract inside learning.

    Replay stores actions after the NumPy execution mask, so Bellman targets and
    actor-Q queries must use the same legal action manifold.  Otherwise the
    critic is trained on fixed intake/storage/fire/ferry values but bootstraps
    and optimizes against combinations that can never reach the simulator.

    Legacy 22-wide agents are returned unchanged; ``update_suffix`` remains
    backwards compatible with its CPU unit tests and older experiments.
    """

    if proprio.ndim != 2 or actions.ndim != 2:
        raise ValueError("actions and proprio must both be rank-2 batches")
    if actions.shape[0] != proprio.shape[0]:
        raise ValueError("actions and proprio must contain the same number of rows")
    if proprio.shape[1] != _STAGEC_V2_PROPRIO_DIM:
        return actions
    if actions.shape[1] != 7:
        raise ValueError("Stage-C v2 actions must have width 7")

    phases = torch.argmax(
        proprio[:, _STAGEC_V2_PHASE_OFFSET : _STAGEC_V2_PHASE_OFFSET + 5], dim=1
    )
    post_first = phases.ne(_STAGEC_V2_FIRST)
    intake_on = phases.eq(_STAGEC_V2_COLLECT)
    if intake_during_return:
        intake_on = intake_on | phases.eq(_STAGEC_V2_RETURN)
    not_score = post_first & phases.ne(_STAGEC_V2_SCORE)

    executed = actions.clone()
    # GATE B (STAGE-D1B): mirror apply_executed_action_policy -- under
    # stage_d_ferry ferry is forced off only in SCORE (post-first), else in
    # every post-first phase, so Bellman/actor-Q targets match the executed
    # action manifold.
    if stage_d_ferry:
        ferry_off = post_first & phases.eq(_STAGEC_V2_SCORE)
    else:
        ferry_off = post_first
    executed[:, 6] = torch.where(
        ferry_off, torch.full_like(executed[:, 6], -1.0), executed[:, 6]
    )
    executed[:, 4] = torch.where(
        post_first, torch.ones_like(executed[:, 4]), executed[:, 4]
    )
    executed[:, 3] = torch.where(
        post_first,
        torch.where(
            intake_on,
            torch.ones_like(executed[:, 3]),
            torch.full_like(executed[:, 3], -1.0),
        ),
        executed[:, 3],
    )
    executed[:, 5] = torch.where(
        not_score, torch.full_like(executed[:, 5], -1.0), executed[:, 5]
    )
    return executed


def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-v2 random shift: replicate-pad, then take a random integer-pixel crop
    back to (H, W).  The base grid lands on exact pixel centres (align_corners
    False: output pixel k -> padded pixel k), and the shift is an integer number
    of pixels, so grid_sample returns exact pixel values with NO interpolation
    blur.  This keeps the encoder's training distribution identical (up to an
    integer translation) to the sharp frames act() feeds it at inference.
    """
    b, _, h, w = images.shape
    x = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    ys = (2.0 * torch.arange(h, device=x.device, dtype=x.dtype) + 1.0) / (h + 2 * pad) - 1.0
    xs = (2.0 * torch.arange(w, device=x.device, dtype=x.dtype) + 1.0) / (w + 2 * pad) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (b, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift = shift * 2.0 / torch.tensor([w + 2 * pad, h + 2 * pad], device=x.device, dtype=x.dtype)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(inplace=True),
        )
        self.out_dim: int | None = None

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        x = obs_uint8.float() / 255.0 - 0.5
        x = self.net(x)
        return x.flatten(1)


class Actor(nn.Module):
    def __init__(self, feat_dim: int, proprio_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim + proprio_dim, hidden), nn.LayerNorm(hidden), nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, feat: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        h = self.trunk(torch.cat([feat, proprio], dim=-1))
        return torch.tanh(self.policy(h))


class Critic(nn.Module):
    """Twin Q; privileged vector is critic-only (asymmetric, not distillation)."""

    def __init__(
        self, feat_dim: int, proprio_dim: int, privileged_dim: int, action_dim: int,
        hidden: int = 512,
    ):
        super().__init__()
        in_dim = feat_dim + proprio_dim + privileged_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim + proprio_dim + privileged_dim, hidden),
            nn.LayerNorm(hidden), nn.Tanh(),
        )
        def q_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden + action_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        self.q1 = q_head()
        self.q2 = q_head()
        _ = in_dim

    def forward(self, feat, proprio, privileged, action):
        h = self.trunk(torch.cat([feat, proprio, privileged], dim=-1))
        ha = torch.cat([h, action], dim=-1)
        return self.q1(ha), self.q2(ha)


@dataclass
class DrQConfig:
    action_dim: int = 7
    proprio_dim: int = 22
    privileged_dim: int = 26
    frame_channels: int = 9      # 3 cameras x RGB
    frame_h: int = 90
    frame_w: int = 160
    lr: float = 1e-4
    critic_tau: float = 0.01
    grad_clip_norm: float = 10.0  # loose safety net vs Q blow-ups; 1.0 throttled learning (audit)
    stddev_start: float = 1.0
    stddev_end: float = 0.1
    stddev_steps: int = 100_000
    stddev_clip: float = 0.3
    device: str = "cuda"


class DrQV2Agent:
    def __init__(self, cfg: DrQConfig):
        self.cfg = cfg
        dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.encoder = Encoder(cfg.frame_channels).to(dev)
        with torch.no_grad():
            probe = torch.zeros(1, cfg.frame_channels, cfg.frame_h, cfg.frame_w, device=dev)
            feat_dim = self.encoder(probe).shape[1]
        self.feat_dim = feat_dim
        self.actor = Actor(feat_dim, cfg.proprio_dim, cfg.action_dim).to(dev)
        self.critic = Critic(feat_dim, cfg.proprio_dim, cfg.privileged_dim, cfg.action_dim).to(dev)
        self.critic_target = Critic(
            feat_dim, cfg.proprio_dim, cfg.privileged_dim, cfg.action_dim
        ).to(dev)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.train_steps = 0
        self.skipped_updates = 0
        # subtracted from train_steps in the stddev schedule; setting it to the
        # current train_steps on --resume re-warms exploration for a new stage
        # (a resumed agent past stddev_steps would otherwise be pinned at floor).
        self.explore_offset = 0

    def reset_optimizers(self, lr: float | None = None) -> None:
        """Discard all Adam momentum while preserving network tensors exactly.

        A reward-contract migration must not inherit first/second moments from
        the previous objective.  Recreating the optimizers is stronger and less
        error-prone than clearing selected state entries in-place.
        """

        learning_rate = float(self.cfg.lr if lr is None else lr)
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("optimizer learning rate must be finite and positive")
        self.cfg.lr = learning_rate
        self.encoder_opt = torch.optim.Adam(
            self.encoder.parameters(), lr=learning_rate
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )

    # -- exploration schedule ----------------------------------------------
    def stddev(self) -> float:
        cfg = self.cfg
        step = max(0, self.train_steps - self.explore_offset)
        mix = min(1.0, step / cfg.stddev_steps)
        return cfg.stddev_start + (cfg.stddev_end - cfg.stddev_start) * mix

    @torch.no_grad()
    def act(self, frames: np.ndarray, proprio: np.ndarray, explore: bool) -> np.ndarray:
        obs = torch.as_tensor(frames, device=self.device)
        pro = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        feat = self.encoder(obs)
        mean = self.actor(feat, pro)
        if explore:
            noise = torch.randn_like(mean) * self.stddev()
            mean = mean + noise
        # never emit a non-finite action: a poisoned network would otherwise
        # drive PhysX with NaN and detonate the shared scene (audit finding).
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(mean, -1.0, 1.0).cpu().numpy()

    # -- one gradient update -------------------------------------------------
    def update(self, batch) -> dict[str, float]:
        cfg = self.cfg
        dev = self.device
        # pixels stay on the 0-255 scale through the shift; Encoder divides by 255
        obs = random_shift(torch.as_tensor(batch.obs, device=dev).float())
        next_obs = random_shift(torch.as_tensor(batch.next_obs, device=dev).float())
        proprio = torch.as_tensor(batch.proprio, device=dev)
        next_proprio = torch.as_tensor(batch.next_proprio, device=dev)
        privileged = torch.as_tensor(batch.privileged, device=dev)
        next_privileged = torch.as_tensor(batch.next_privileged, device=dev)
        action = torch.as_tensor(batch.action, device=dev)
        reward = torch.as_tensor(batch.reward, device=dev).unsqueeze(-1)
        discount = torch.as_tensor(batch.discount, device=dev).unsqueeze(-1)

        feat = self.encoder(obs)
        with torch.no_grad():
            next_feat = self.encoder(next_obs)
            stddev = self.stddev()
            next_mean = self.actor(next_feat, next_proprio)
            noise = torch.clamp(
                torch.randn_like(next_mean) * stddev, -cfg.stddev_clip, cfg.stddev_clip
            )
            next_action = torch.clamp(next_mean + noise, -1.0, 1.0)
            tq1, tq2 = self.critic_target(
                next_feat, next_proprio, next_privileged, next_action
            )
            target_q = reward + discount * torch.min(tq1, tq2)

        q1, q2 = self.critic(feat, proprio, privileged, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        # non-finite guard: a diverged batch must never write NaN into the
        # weights (a 4 h run was destroyed by its final ~200 updates)
        if not torch.isfinite(critic_loss):
            self.skipped_updates += 1
            return {"critic_loss": float("nan"), "skipped": float(self.skipped_updates)}
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        gn_e = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.grad_clip_norm)
        gn_c = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        # clip_grad_norm_ does NOT stop a non-finite grad norm from writing NaN
        # into the weights (clip_coef = max/inf = 0, then inf*0 = NaN); the
        # loss-finite check above cannot catch this. Guard the optimizer step.
        if not (torch.isfinite(gn_e) and torch.isfinite(gn_c)):
            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.skipped_updates += 1
            return {
                "critic_loss": float(critic_loss.item()),
                "skipped": float(self.skipped_updates),
            }
        self.encoder_opt.step()
        self.critic_opt.step()

        # actor on detached features (DrQ-v2: encoder learns from critic only)
        feat_detached = feat.detach()
        mean = self.actor(feat_detached, proprio)
        noise = torch.clamp(
            torch.randn_like(mean) * self.stddev(), -cfg.stddev_clip, cfg.stddev_clip
        )
        sampled = torch.clamp(mean + noise, -1.0, 1.0)
        aq1, aq2 = self.critic(feat_detached, proprio, privileged, sampled)
        actor_loss = -torch.min(aq1, aq2).mean()
        actor_applied = False
        if torch.isfinite(actor_loss):
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            gn_a = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
            if torch.isfinite(gn_a):
                self.actor_opt.step()
                actor_applied = True
            else:
                self.actor_opt.zero_grad(set_to_none=True)
        if not actor_applied:
            self.skipped_updates += 1

        # target EMA + schedule advance ALWAYS run: the critic already stepped,
        # so a skipped actor step must never leave a half-applied update (audit).
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, cfg.critic_tau)

        self.train_steps += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()) if torch.isfinite(actor_loss) else float("nan"),
            "q1": float(q1.mean().item()),
            "stddev": float(self.stddev()),
            "skipped": float(self.skipped_updates),
        }

    # -- prefix-takeover suffix update (design notes Turn 12 + PREFIX_TAKEOVER_SPEC) --
    def update_suffix(self, batch, anchor_obs, anchor_proprio, anchor_action,
                      alpha: float, freeze_encoder: bool = True,
                      actor_mask=None, critic_mask=None, actor_update: bool = True,
                      elite_behavior_batch=None,
                      elite_behavior_weight: float = 0.0,
                      elite_behavior_full_episode_s: float = 160.0,
                      anchor_weight: float = 1.0,
                      intake_during_return: bool = False,
                      stage_d_ferry: bool = False,
                      actor_q_center_fraction: float = 0.0) -> dict[str, float]:
        """One prefix-takeover suffix gradient step (candidate learns cycle 2 only).

        OPT-IN; the audited ``update()`` above is untouched.  Differs from it in
        exactly the three spec-mandated ways:

          * ENCODER FROZEN.  The champion's visual features are preserved: when
            ``freeze_encoder`` the suffix obs are encoded under ``no_grad`` so the
            critic loss cannot move the encoder, and the encoder optimizer never
            steps.  (Actor already trains on detached features, as in DrQ-v2.)
          * ACTOR anchored with correctly-normalized TD3+BC on a configurable
            blend of noisy and deterministic executed policy actions:
                Q_actor    = (1-f) * Q(s, pi(s) + noise) + f * Q(s, pi(s))
                lambda_q   = alpha / mean(|Q_actor|).detach()
                actor_loss = -lambda_q * Q_actor.mean()
                             + anchor_weight * MSE(pi(s_a), a_a)
            The RL term is thus scaled to ~alpha while the champion-action anchor
            term is O(1). ``actor_mask`` optionally selects the replay rows used
            by the actor-Q term. ``critic_mask`` independently restricts critic
            fitting (V9 uses only non-FIRST rows); when omitted, the critic uses
            the full batch.  An explicitly empty critic mask is a safe no-op.
          * The anchor minibatch (champion lossless frames + champion mean actions)
            is UNAUGMENTED and never enters the critic -- it only pins the actor.

        Critic / EMA target / every NaN guard match ``update()`` so a diverged
        suffix batch can never poison the immutable champion.
        """
        cfg = self.cfg
        if not np.isfinite(anchor_weight) or float(anchor_weight) < 0.0:
            raise ValueError("anchor_weight must be finite and non-negative")
        if (
            not np.isfinite(actor_q_center_fraction)
            or not 0.0 <= float(actor_q_center_fraction) <= 1.0
        ):
            raise ValueError("actor_q_center_fraction must be finite and in [0, 1]")
        if (
            not np.isfinite(elite_behavior_full_episode_s)
            or float(elite_behavior_full_episode_s) <= 0.0
        ):
            raise ValueError(
                "elite_behavior_full_episode_s must be finite and positive"
            )
        dev = self.device
        obs = random_shift(torch.as_tensor(batch.obs, device=dev).float())
        next_obs = random_shift(torch.as_tensor(batch.next_obs, device=dev).float())
        proprio = torch.as_tensor(batch.proprio, device=dev)
        next_proprio = torch.as_tensor(batch.next_proprio, device=dev)
        privileged = torch.as_tensor(batch.privileged, device=dev)
        next_privileged = torch.as_tensor(batch.next_privileged, device=dev)
        action = torch.as_tensor(batch.action, device=dev)
        reward = torch.as_tensor(batch.reward, device=dev).unsqueeze(-1)
        discount = torch.as_tensor(batch.discount, device=dev).unsqueeze(-1)

        batch_rows = int(proprio.shape[0])
        if actor_mask is None:
            actor_rows_mask = torch.ones(batch_rows, dtype=torch.bool, device=dev)
        else:
            actor_rows_mask = torch.as_tensor(
                actor_mask, device=dev, dtype=torch.bool
            ).reshape(-1)
            if int(actor_rows_mask.numel()) != batch_rows:
                raise ValueError(
                    f"actor_mask has {actor_rows_mask.numel()} rows; batch has {batch_rows}"
                )
        critic_mask_supplied = critic_mask is not None
        if critic_mask is None:
            critic_rows_mask = torch.ones(
                batch_rows, dtype=torch.bool, device=dev
            )
        else:
            critic_rows_mask = torch.as_tensor(
                critic_mask, device=dev, dtype=torch.bool
            ).reshape(-1)
            if int(critic_rows_mask.numel()) != batch_rows:
                raise ValueError(
                    f"critic_mask has {critic_rows_mask.numel()} rows; batch has {batch_rows}"
                )
        selected_actor_rows = int(actor_rows_mask.sum().item())
        actor_rows = selected_actor_rows if bool(actor_update) else 0
        critic_rows = int(critic_rows_mask.sum().item())
        if critic_mask_supplied and critic_rows == 0:
            return {
                "critic_loss": 0.0,
                "actor_rows": 0.0,
                "critic_rows": 0.0,
                "no_critic_rows": 1.0,
                "skipped": float(self.skipped_updates),
            }

        # ---- critic (identical TD to update(); encoder frozen => no_grad feat) ----
        if freeze_encoder:
            with torch.no_grad():
                feat = self.encoder(obs)
        else:
            feat = self.encoder(obs)
        with torch.no_grad():
            next_feat = self.encoder(next_obs)
            next_mean = self.actor(next_feat, next_proprio)
            noise = torch.clamp(torch.randn_like(next_mean) * self.stddev(),
                                -cfg.stddev_clip, cfg.stddev_clip)
            next_action = torch.clamp(next_mean + noise, -1.0, 1.0)
            next_action = _stagec_v2_executed_action_policy_torch(
                next_action,
                next_proprio,
                intake_during_return=intake_during_return,
                stage_d_ferry=stage_d_ferry,
            )
            tq1, tq2 = self.critic_target(next_feat, next_proprio, next_privileged, next_action)
            target_q = reward + discount * torch.min(tq1, tq2)
        q1, q2 = self.critic(feat, proprio, privileged, action)
        # R3: train the critic ONLY on the explicitly selected (non-FIRST/suffix)
        # rows when a mask is supplied. FIRST-phase rows carry the frozen
        # prefix's large cycle-1 reward
        # and their n-step target queries actor(next_feat) at FIRST next-states the
        # actor is never trained on -> off-distribution bootstrap that pulls the shared
        # critic upward. Episodes are ordered FIRST->suffix, so a non-FIRST row's
        # n-step next-state is never FIRST; masking is therefore safe.
        critic_loss = (
            F.mse_loss(q1[critic_rows_mask], target_q[critic_rows_mask])
            + F.mse_loss(q2[critic_rows_mask], target_q[critic_rows_mask])
        )
        if not torch.isfinite(critic_loss):
            self.skipped_updates += 1
            return {
                "critic_loss": float("nan"),
                "actor_rows": float(actor_rows),
                "critic_rows": float(critic_rows),
                "skipped": float(self.skipped_updates),
            }
        if not freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        gn_c = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        gn_e = None
        if not freeze_encoder:
            gn_e = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.grad_clip_norm)
        if not torch.isfinite(gn_c) or (gn_e is not None and not torch.isfinite(gn_e)):
            self.critic_opt.zero_grad(set_to_none=True)
            if not freeze_encoder:
                self.encoder_opt.zero_grad(set_to_none=True)
            self.skipped_updates += 1
            return {
                "critic_loss": float(critic_loss.item()),
                "actor_rows": float(actor_rows),
                "critic_rows": float(critic_rows),
                "skipped": float(self.skipped_updates),
            }
        self.critic_opt.step()
        if not freeze_encoder:            # frozen: champion encoder never moves
            self.encoder_opt.step()

        # ---- actor: TD3+BC anchor on detached features ----
        actor_metrics: dict[str, float] = {
            "actor_rows": float(actor_rows),
            "critic_rows": float(critic_rows),
        }
        if actor_rows:
            feat_det = feat.detach()[actor_rows_mask]
            actor_proprio = proprio[actor_rows_mask]
            actor_privileged = privileged[actor_rows_mask]
            pi_s = self.actor(feat_det, actor_proprio)             # deterministic policy action
            # Target-policy smoothing on the actor-Q action, matching the proven base
            # update() (drqv2.py:222-228). Optimizing Q of the BARE deterministic
            # action lets the actor exploit critic error at the exact deployed point:
            # the critic is only trained on noisy (exploration) actions, so its value
            # at the un-sampled deterministic action is unreliable, the actor climbs
            # that unreliable ridge, and the deterministic policy rots while noisy
            # exploration still completes cycles (the observed mirage + monotonic
            # deterministic decline). Evaluating Q over a small noise ball around pi_s
            # keeps the optimized quantity on the distribution the critic actually saw.
            pi_noise = torch.clamp(
                torch.randn_like(pi_s) * self.stddev(), -cfg.stddev_clip, cfg.stddev_clip
            )
            pi_sampled = torch.clamp(pi_s + pi_noise, -1.0, 1.0)
            pi_sampled = _stagec_v2_executed_action_policy_torch(
                pi_sampled,
                actor_proprio,
                intake_during_return=intake_during_return,
                stage_d_ferry=stage_d_ferry,
            )
            q1_pi_noisy, q2_pi_noisy = self.critic(
                feat_det, actor_proprio, actor_privileged, pi_sampled
            )
            q_pi_noisy = torch.min(q1_pi_noisy, q2_pi_noisy)
            q_pi = q_pi_noisy
            q_pi_center = None
            center_fraction = float(actor_q_center_fraction)
            # Preserve the historical path exactly at f=0: no extra action-mask
            # call or critic forward is executed unless the blend is opted in.
            if center_fraction > 0.0:
                pi_center = _stagec_v2_executed_action_policy_torch(
                    pi_s,
                    actor_proprio,
                    intake_during_return=intake_during_return,
                    stage_d_ferry=stage_d_ferry,
                )
                q1_pi_center, q2_pi_center = self.critic(
                    feat_det, actor_proprio, actor_privileged, pi_center
                )
                q_pi_center = torch.min(q1_pi_center, q2_pi_center)
                q_pi = (
                    (1.0 - center_fraction) * q_pi_noisy
                    + center_fraction * q_pi_center
                )
            lam = alpha / q_pi.abs().mean().detach().clamp_min(1e-6)
            # champion anchor: lossless, UNAUGMENTED, encoder frozen (no_grad feat)
            a_obs = torch.as_tensor(anchor_obs, device=dev)
            a_pro = torch.as_tensor(anchor_proprio, device=dev, dtype=torch.float32)
            a_act = torch.as_tensor(anchor_action, device=dev, dtype=torch.float32)
            with torch.no_grad():
                a_feat = self.encoder(a_obs)
            pi_a = self.actor(a_feat, a_pro)
            bc = F.mse_loss(pi_a, a_act)
            elite_bc = torch.zeros((), device=dev)
            elite_rows = 0
            elite_window_metrics: dict[str, float] = {}
            if elite_behavior_batch is not None and float(elite_behavior_weight) > 0.0:
                elite_obs = torch.as_tensor(elite_behavior_batch.obs, device=dev)
                elite_proprio = torch.as_tensor(
                    elite_behavior_batch.proprio, device=dev, dtype=torch.float32
                )
                elite_action = torch.as_tensor(
                    elite_behavior_batch.action, device=dev, dtype=torch.float32
                )
                with torch.no_grad():
                    elite_feat = self.encoder(elite_obs)
                elite_pi = self.actor(elite_feat, elite_proprio)
                elite_pi = _stagec_v2_executed_action_policy_torch(
                    elite_pi,
                    elite_proprio,
                    intake_during_return=intake_during_return,
                    stage_d_ferry=stage_d_ferry,
                )
                elite_row_bc = (elite_pi - elite_action).square().mean(dim=1)
                elite_bc = elite_row_bc.mean()
                elite_rows = int(elite_proprio.shape[0])
                elite_clock_s = elite_proprio[:, 7].clamp(0.0, 1.0) * float(
                    elite_behavior_full_episode_s
                )
                for name, start_s, end_s in (
                    ("opener", 0.0, 33.0),
                    ("live1", 55.0, 83.0),
                    ("live2", 105.0, 130.0),
                    ("endgame", 130.0, 160.000001),
                ):
                    mask = (elite_clock_s >= start_s) & (elite_clock_s < end_s)
                    rows = int(mask.sum().item())
                    elite_window_metrics[f"elite_behavior_rows_{name}"] = float(rows)
                    if rows:
                        elite_window_metrics[f"elite_behavior_bc_{name}"] = float(
                            elite_row_bc[mask].mean().item()
                        )
            actor_loss = (
                -lam * q_pi.mean()
                + float(anchor_weight) * bc
                + float(elite_behavior_weight) * elite_bc
            )
            actor_applied = False
            if torch.isfinite(actor_loss):
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                gn_a = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
                if torch.isfinite(gn_a):
                    self.actor_opt.step()
                    actor_applied = True
                else:
                    self.actor_opt.zero_grad(set_to_none=True)
            if not actor_applied:
                self.skipped_updates += 1
            actor_metrics.update({
                "actor_loss": (
                    float(actor_loss.item()) if torch.isfinite(actor_loss) else float("nan")
                ),
                "bc_anchor": float(bc.item()),
                "anchor_weight": float(anchor_weight),
                "elite_behavior_bc": float(elite_bc.item()),
                "elite_behavior_rows": float(elite_rows),
                "elite_behavior_weight": float(elite_behavior_weight),
                "lambda_q": float(lam.item()),
                "q_pi": float(q_pi.mean().item()),
                "q_pi_noisy": float(q_pi_noisy.mean().item()),
                "actor_q_center_fraction": center_fraction,
                "actor_applied": float(actor_applied),
            })
            actor_metrics.update(elite_window_metrics)
            if q_pi_center is not None:
                q_pi_center_mean = float(q_pi_center.mean().item())
                actor_metrics.update({
                    "q_pi_center": q_pi_center_mean,
                    "q_pi_center_minus_noisy": (
                        q_pi_center_mean - float(q_pi_noisy.mean().item())
                    ),
                })

        with torch.no_grad():             # EMA target ALWAYS advances (critic already stepped)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, cfg.critic_tau)
        self.train_steps += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "alpha": float(alpha),
            "skipped": float(self.skipped_updates),
            **actor_metrics,
        }

    # -- reward-first warm-start fine-tune (design notes Turns 29-32) --------
    def update_finetune(self, batch, anchor_obs, anchor_proprio, anchor_action,
                        beta: float, critic_only: bool) -> dict[str, float]:
        """One reward-first fine-tune step. OPT-IN; the audited ``update()`` is untouched.

        Two phases (the learner picks ``critic_only`` by update count):
          * ``critic_only=True`` (phase 1, updates 0..N): freeze ACTOR **and** ENCODER —
            the standard critic step trains the encoder (``encoder_opt.step()``) and
            ``act()`` shares it, so a naive actor-only skip would drift the deterministic
            policy un-anchored (Turn 32). Encode under ``no_grad`` and never step the
            encoder; fit the critic head + EMA on FIXED champion features. The policy is
            bit-identical to the champion -> a phase-1 snapshot must give drive-L2 p50 == 0.
          * ``critic_only=False`` (phase 2): standard DrQ-v2 critic (encoder learns from
            the critic) PLUS an END-TO-END champion BC anchor ``beta*MSE(pi(enc(f_a)),
            a_champ)`` whose gradient flows through actor AND encoder (a second encoder
            step from the anchor); ``beta`` is annealed 0.3 -> 0 by the learner.
        Every update() NaN guard is reproduced so a diverged batch can't poison weights.
        """
        cfg = self.cfg
        dev = self.device
        obs = random_shift(torch.as_tensor(batch.obs, device=dev).float())
        next_obs = random_shift(torch.as_tensor(batch.next_obs, device=dev).float())
        proprio = torch.as_tensor(batch.proprio, device=dev)
        next_proprio = torch.as_tensor(batch.next_proprio, device=dev)
        privileged = torch.as_tensor(batch.privileged, device=dev)
        next_privileged = torch.as_tensor(batch.next_privileged, device=dev)
        action = torch.as_tensor(batch.action, device=dev)
        reward = torch.as_tensor(batch.reward, device=dev).unsqueeze(-1)
        discount = torch.as_tensor(batch.discount, device=dev).unsqueeze(-1)

        # ---- critic (encoder FROZEN iff critic_only) ----
        if critic_only:
            with torch.no_grad():
                feat = self.encoder(obs)
        else:
            feat = self.encoder(obs)
        with torch.no_grad():
            next_feat = self.encoder(next_obs)
            next_mean = self.actor(next_feat, next_proprio)
            noise = torch.clamp(torch.randn_like(next_mean) * self.stddev(),
                                -cfg.stddev_clip, cfg.stddev_clip)
            next_action = torch.clamp(next_mean + noise, -1.0, 1.0)
            tq1, tq2 = self.critic_target(next_feat, next_proprio, next_privileged, next_action)
            target_q = reward + discount * torch.min(tq1, tq2)
        q1, q2 = self.critic(feat, proprio, privileged, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if not torch.isfinite(critic_loss):
            self.skipped_updates += 1
            return {"critic_loss": float("nan"), "skipped": float(self.skipped_updates)}
        self.critic_opt.zero_grad(set_to_none=True)
        if not critic_only:
            self.encoder_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        gn_c = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        gn_e = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.grad_clip_norm) \
            if not critic_only else None
        if not (torch.isfinite(gn_c) and (gn_e is None or torch.isfinite(gn_e))):
            self.critic_opt.zero_grad(set_to_none=True)
            if not critic_only:
                self.encoder_opt.zero_grad(set_to_none=True)
            self.skipped_updates += 1
            return {"critic_loss": float(critic_loss.item()), "skipped": float(self.skipped_updates)}
        self.critic_opt.step()
        if not critic_only:
            self.encoder_opt.step()          # encoder adapts to the new reward, via the critic

        metrics = {"critic_loss": float(critic_loss.item()), "q1": float(q1.mean().item()),
                   "stddev": float(self.stddev()), "critic_only": float(critic_only),
                   "beta": float(beta)}

        # ---- actor + end-to-end champion anchor (phase 2 only) ----
        if not critic_only:
            feat_det = feat.detach()
            aq1, aq2 = self.critic(feat_det, proprio, privileged, self.actor(feat_det, proprio))
            actor_q_loss = -torch.min(aq1, aq2).mean()
            a_obs = torch.as_tensor(anchor_obs, device=dev)
            a_pro = torch.as_tensor(anchor_proprio, device=dev, dtype=torch.float32)
            a_act = torch.as_tensor(anchor_action, device=dev, dtype=torch.float32)
            a_feat = self.encoder(a_obs)     # WITH grad -> anchor pins actor AND encoder
            bc = F.mse_loss(self.actor(a_feat, a_pro), a_act)
            actor_loss = actor_q_loss + float(beta) * bc
            actor_applied = False
            if torch.isfinite(actor_loss):
                self.actor_opt.zero_grad(set_to_none=True)
                self.encoder_opt.zero_grad(set_to_none=True)   # 2nd encoder grad: the anchor only
                actor_loss.backward()                          # q term uses detached feat -> actor only
                gn_a = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
                gn_ea = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.grad_clip_norm)
                if torch.isfinite(gn_a) and torch.isfinite(gn_ea):
                    self.actor_opt.step()
                    self.encoder_opt.step()    # encoder pulled toward champion on anchor states
                    actor_applied = True
                else:
                    self.actor_opt.zero_grad(set_to_none=True)
                    self.encoder_opt.zero_grad(set_to_none=True)
            if not actor_applied:
                self.skipped_updates += 1
            metrics.update({
                "actor_loss": float(actor_loss.item()) if torch.isfinite(actor_loss) else float("nan"),
                "bc_anchor": float(bc.item()), "actor_q_loss": float(actor_q_loss.item()),
            })

        with torch.no_grad():                 # EMA target + schedule always advance
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, cfg.critic_tau)
        self.train_steps += 1
        metrics["skipped"] = float(self.skipped_updates)
        return metrics

    # -- persistence ---------------------------------------------------------
    def weights_finite(self) -> bool:
        # Every persisted tensor must be finite. save() writes the live modules,
        # the EMA target critic (updated in-place by lerp_ every step, even on
        # steps where the optimizer step was skipped for a non-finite grad), and
        # the three Adam optimizer states -- so all of them are checked here, not
        # just encoder/actor/critic. A poisoned target critic or momentum buffer
        # would otherwise pass this gate and be saved/published (audit).
        for module in (self.encoder, self.actor, self.critic, self.critic_target):
            for p in module.parameters():
                if not bool(torch.isfinite(p).all()):
                    return False
        for optimizer in (self.encoder_opt, self.actor_opt, self.critic_opt):
            for state in optimizer.state.values():
                for v in state.values():
                    # Adam state holds exp_avg/exp_avg_sq (float tensors) and a
                    # `step` counter (int or 0-dim tensor); only finiteness-check
                    # floating-point tensors so the int step never false-rejects.
                    if torch.is_tensor(v) and v.is_floating_point():
                        if not bool(torch.isfinite(v).all()):
                            return False
        return True

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "encoder_opt": self.encoder_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "train_steps": self.train_steps,
                "skipped_updates": self.skipped_updates,
                # persist the exploration-schedule anchor so a resume (or a
                # crash-restart) cannot silently re-warm stddev back to stddev_start
                # - the confirmed Stage-C champion-degradation cause.
                "explore_offset": self.explore_offset,
            },
            path,
        )

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(payload["encoder"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        # optimizer state enables exact training resume (older checkpoints
        # without it still load for evaluation)
        for name, optimizer in (
            ("encoder_opt", self.encoder_opt),
            ("actor_opt", self.actor_opt),
            ("critic_opt", self.critic_opt),
        ):
            if name in payload:
                optimizer.load_state_dict(payload[name])
        self.train_steps = int(payload.get("train_steps", 0))
        self.skipped_updates = int(payload.get("skipped_updates", 0))
        # older checkpoints predate explore_offset -> default 0 (schedule anchored
        # at the start, i.e. stddev follows train_steps directly).
        self.explore_offset = int(payload.get("explore_offset", 0))
