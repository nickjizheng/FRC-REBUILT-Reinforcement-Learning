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


def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-v2 random shift augmentation on (B, C, H, W) float images."""
    b, _, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    eps_h = 2.0 * pad / (h + 2 * pad)
    eps_w = 2.0 * pad / (w + 2 * pad)
    arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h, device=images.device)
    arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w, device=images.device)
    grid_y, grid_x = torch.meshgrid(arange_h, arange_w, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (b, 1, 1, 2), device=images.device).float()
    shift = (shift - pad) * 2.0 / torch.tensor(
        [w + 2 * pad, h + 2 * pad], device=images.device
    )
    return F.grid_sample(padded, base + shift, padding_mode="zeros", align_corners=False)


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
    grad_clip_norm: float = 1.0   # converged plan; prevents late-run Q blow-ups
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

    # -- exploration schedule ----------------------------------------------
    def stddev(self) -> float:
        cfg = self.cfg
        mix = min(1.0, self.train_steps / cfg.stddev_steps)
        return cfg.stddev_start + (cfg.stddev_end - cfg.stddev_start) * mix

    @torch.no_grad()
    def act(self, frames: np.ndarray, proprio: np.ndarray, explore: bool) -> np.ndarray:
        obs = torch.as_tensor(frames, device=self.device)
        pro = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        feat = self.encoder(obs)
        mean = self.actor(feat, pro)
        if explore:
            noise = torch.randn_like(mean) * self.stddev()
            mean = torch.clamp(mean + noise, -1.0, 1.0)
        return mean.cpu().numpy()

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
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
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
        if not torch.isfinite(actor_loss):
            self.skipped_updates += 1
            return {"actor_loss": float("nan"), "skipped": float(self.skipped_updates)}
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, cfg.critic_tau)

        self.train_steps += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1": float(q1.mean().item()),
            "stddev": float(self.stddev()),
        }

    # -- persistence ---------------------------------------------------------
    def weights_finite(self) -> bool:
        return all(
            bool(torch.isfinite(p).all())
            for module in (self.encoder, self.actor, self.critic)
            for p in module.parameters()
        )

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
