# latent_diffusion.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    timesteps: (B,) int64/long
    returns:   (B, dim) float
    """
    if timesteps.dim() != 1:
        raise ValueError(f"timesteps must be 1D (B,), got {timesteps.shape}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)   # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.size(0), 1), device=emb.device, dtype=emb.dtype)], dim=1)
    return emb


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    a: (T,) buffer
    t: (B,) long
    returns broadcastable tensor shaped (B, 1, 1, ...)
    """
    out = a.gather(0, t)
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # "linear" | "cosine"
    linear_beta_start: float = 1e-4
    linear_beta_end: float = 2e-2
    cosine_s: float = 0.008
    clip_denoised: bool = False


def make_beta_schedule(cfg: DiffusionConfig) -> torch.Tensor:
    T = cfg.timesteps
    if cfg.beta_schedule == "linear":
        return torch.linspace(cfg.linear_beta_start, cfg.linear_beta_end, T, dtype=torch.float32)
    if cfg.beta_schedule == "cosine":
        # Improved DDPM cosine schedule (Nichol & Dhariwal)
        steps = T + 1
        x = torch.linspace(0, T, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / T) + cfg.cosine_s) / (1 + cfg.cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"Unknown beta_schedule: {cfg.beta_schedule}")


class DenoiserTransformer(nn.Module):
    """
    A simple masked Transformer denoiser for latent tensors z_t: (B, N, D).
    Predicts epsilon noise: eps_hat: (B, N, D).
    """

    def __init__(
        self,
        latent_dim: int,
        model_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
        use_cond: bool = False,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.use_cond = use_cond
        self.cond_dim = cond_dim

        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        if self.use_cond:
            if cond_dim <= 0:
                raise ValueError("use_cond=True requires cond_dim > 0")
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,                # (B, N, D)
        t: torch.Tensor,                  # (B,)
        node_mask: Optional[torch.Tensor] = None,  # (B, N, 1) 1=valid
        cond: Optional[torch.Tensor] = None,       # (B, cond_dim)
    ) -> torch.Tensor:
        if z_t.dim() != 3:
            raise ValueError(f"z_t must be (B,N,D), got {z_t.shape}")
        B, N, D = z_t.shape
        if D != self.latent_dim:
            raise ValueError(f"latent_dim mismatch: expected {self.latent_dim}, got {D}")

        if node_mask is None:
            key_padding_mask = None
        else:
            if node_mask.shape != (B, N, 1):
                raise ValueError(f"node_mask must be (B,N,1), got {node_mask.shape}")
            key_padding_mask = (node_mask.squeeze(-1) <= 0.0)  # True means PAD for transformer

        h = self.in_proj(z_t)  # (B,N,model_dim)

        # time embedding
        te = timestep_embedding(t, self.time_mlp[0].in_features)  # (B,time_embed_dim)
        te = self.time_mlp(te).unsqueeze(1)  # (B,1,model_dim)
        h = h + te

        # optional conditioning as a broadcast bias
        if self.use_cond:
            if cond is None:
                raise ValueError("use_cond=True but cond is None")
            if cond.shape != (B, self.cond_dim):
                raise ValueError(f"cond must be (B,{self.cond_dim}), got {cond.shape}")
            ce = self.cond_proj(cond).unsqueeze(1)  # (B,1,model_dim)
            h = h + ce

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,N,model_dim)
        h = self.out_proj(self.out_norm(h))                         # (B,N,D)

        # keep padded positions strictly 0 (optional but recommended)
        if node_mask is not None:
            h = h * node_mask
        return h


class LatentDiffusion(nn.Module):
    """
    DDPM in latent space with node_mask (padding mask) support.
    Epsilon-parameterization: model predicts eps.
    """

    def __init__(self, denoiser: nn.Module, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.cfg = cfg

        betas = make_beta_schedule(cfg)              # (T,)
        alphas = 1.0 - betas                         # (T,)
        alphas_cumprod = torch.cumprod(alphas, 0)    # (T,)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        # register buffers (moved with .to(device))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # posterior q(x_{t-1} | x_t, x0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(1e-20))

        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @property
    def timesteps(self) -> int:
        return int(self.cfg.timesteps)

    def q_sample(
        self,
        z0: torch.Tensor,                 # (B,N,D)
        t: torch.Tensor,                  # (B,)
        noise: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,   # (B,N,1)
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z0)
        if node_mask is not None:
            z0 = z0 * node_mask
            noise = noise * node_mask

        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        zt = sqrt_ab * z0 + sqrt_1mab * noise
        if node_mask is not None:
            zt = zt * node_mask
        return zt

    def predict_start_from_noise(self, zt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        z0 = (zt - sqrt_1mab * eps) / (sqrt_ab + 1e-12)
        return z0

    def p_losses(
        self,
        z0: torch.Tensor,                 # (B,N,D)
        node_mask: torch.Tensor,          # (B,N,1)
        cond: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z0.size(0)
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=z0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(z0)

        zt = self.q_sample(z0, t, noise=noise, node_mask=node_mask)
        eps_hat = self.denoiser(zt, t, node_mask=node_mask, cond=cond)

        # masked MSE
        mse = (eps_hat - noise) ** 2
        mse = mse * node_mask  # (B,N,D)
        denom = node_mask.sum() * z0.size(-1) + 1e-12
        return mse.sum() / denom

    @torch.no_grad()
    def p_sample(
        self,
        zt: torch.Tensor,                 # (B,N,D)
        t: torch.Tensor,                  # (B,) all same step typically
        node_mask: torch.Tensor,          # (B,N,1)
        cond: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        eps_hat = self.denoiser(zt, t, node_mask=node_mask, cond=cond)
        z0_hat = self.predict_start_from_noise(zt, t, eps_hat)

        if self.cfg.clip_denoised:
            z0_hat = z0_hat.clamp(-1.0, 1.0)

        coef1 = extract(self.posterior_mean_coef1, t, zt.shape)
        coef2 = extract(self.posterior_mean_coef2, t, zt.shape)
        mean = coef1 * z0_hat + coef2 * zt

        if deterministic:
            # Deterministic reverse: use posterior mean only (no sampling noise).
            z_prev = mean
        else:
            var = extract(self.posterior_variance, t, zt.shape)
            noise = torch.randn_like(zt)
            # no noise when t == 0
            nonzero = (t != 0).float()
            while nonzero.dim() < mean.dim():
                nonzero = nonzero.unsqueeze(-1)

            z_prev = mean + nonzero * torch.sqrt(var) * noise
        z_prev = z_prev * node_mask
        return z_prev

    @torch.no_grad()
    def sample(
        self,
        node_mask: torch.Tensor,           # (B,N,1)
        latent_dim: int,
        cond: Optional[torch.Tensor] = None,
        zT: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Returns z0 samples: (B,N,D), masked.
        """
        B, N, _ = node_mask.shape
        device = node_mask.device

        if zT is None:
            zt = torch.randn((B, N, latent_dim), device=device)
        else:
            zt = zT.to(device)

        zt = zt * node_mask
        for step in reversed(range(self.timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            zt = self.p_sample(zt, t, node_mask=node_mask, cond=cond, deterministic=deterministic)
        return zt
