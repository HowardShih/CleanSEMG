"""
SDEMG — Score-Based Diffusion Model for sEMG Denoising
=======================================================
CleanSEMG path: denoising/SDEMG.py

Self-contained implementation. No external SDEMG repo required.
ConditionalModel and GaussianDiffusion1D are embedded directly.

Reference:
    Liu et al., "Score-based Generative Models for EMG Signal Denoising,"
    ICASSP 2024.

Shape contract (matches all other CleanSEMG baseline models):
    noisy / clean : FloatTensor[B, L]   (no channel dim)
    pred          : FloatTensor[B, L]

Training:
    Detected by train_neural.py via HAS_DIFFUSION_LOSS = True.
    Call: loss = model.compute_diffusion_loss(clean, noisy)

Inference:
    Runs T reverse-diffusion steps conditioned on the noisy input.
    Call: pred = model(noisy)
"""

from __future__ import annotations

import math
from collections import namedtuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# tqdm is optional for inference progress; silence it during batch eval
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (inlined from SDEMG/utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def _default(val, d):
    """Return val if not None, else d() if callable else d."""
    if val is not None:
        return val
    return d() if callable(d) else d


def _identity(t, *args, **kwargs):
    return t


# ─────────────────────────────────────────────────────────────────────────────
# ConditionalModel (from SDEMG/deep_filter_model.py)
# ─────────────────────────────────────────────────────────────────────────────

class _Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class _PositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0))
        return torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)


class _FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 use_affine_level: bool = False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Linear(
            in_channels, out_channels * (1 + self.use_affine_level))

    def forward(self, x: torch.Tensor,
                noise_embed: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x


class _HNFBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dilation: int):
        super().__init__()
        self.filters = nn.ModuleList([
            _Conv1d(input_size,  hidden_size // 4, 3,  dilation=dilation,
                    padding=1 * dilation, padding_mode='reflect'),
            _Conv1d(hidden_size, hidden_size // 4, 5,  dilation=dilation,
                    padding=2 * dilation, padding_mode='reflect'),
            _Conv1d(hidden_size, hidden_size // 4, 9,  dilation=dilation,
                    padding=4 * dilation, padding_mode='reflect'),
            _Conv1d(hidden_size, hidden_size // 4, 15, dilation=dilation,
                    padding=7 * dilation, padding_mode='reflect'),
        ])
        self.conv_1 = _Conv1d(hidden_size, hidden_size, 9,
                               padding=4, padding_mode='reflect')
        self.norm   = nn.InstanceNorm1d(hidden_size // 2)
        self.conv_2 = _Conv1d(hidden_size, hidden_size, 9,
                               padding=4, padding_mode='reflect')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        filts = torch.cat([layer(x) for layer in self.filters], dim=1)
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        filts = F.leaky_relu(
            torch.cat([self.norm(nfilts), filts], dim=1), 0.2)
        filts = F.leaky_relu(self.conv_2(filts), 0.2)
        return filts + residual


class _Bridge(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoding    = _FeatureWiseAffine(
            input_size, hidden_size, use_affine_level=True)
        self.input_conv  = _Conv1d(input_size, input_size,  3,
                                    padding=1, padding_mode='reflect')
        self.output_conv = _Conv1d(input_size, hidden_size, 3,
                                    padding=1, padding_mode='reflect')

    def forward(self, x: torch.Tensor,
                noise_embed: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        return self.output_conv(x)


class ConditionalModel(nn.Module):
    """
    Conditional noise-prediction network for SDEMG diffusion model.
    Architecture: dual-stream HNF encoder with bridge fusion.

    Input:
        x           : [B, 1, L]  noisy latent at timestep t
        noise_scale : [B, 1]     sqrt(alpha_bar) at timestep t
        cond        : [B, 1, L]  noisy sEMG observation (condition)
    Output:
        [B, 1, L]  predicted noise
    """

    def __init__(self, feats: int = 64):
        super().__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(
                _Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
            ),
            _HNFBlock(feats, feats, 1),
            _HNFBlock(feats, feats, 2),
            _HNFBlock(feats, feats, 4),
            _HNFBlock(feats, feats, 2),
            _HNFBlock(feats, feats, 1),
        ])
        self.stream_cond = nn.ModuleList([
            nn.Sequential(
                _Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
            ),
            _HNFBlock(feats, feats, 1),
            _HNFBlock(feats, feats, 2),
            _HNFBlock(feats, feats, 4),
            _HNFBlock(feats, feats, 2),
            _HNFBlock(feats, feats, 1),
        ])
        self.embed  = _PositionalEncoding(feats)
        self.bridge = nn.ModuleList([_Bridge(feats, feats)] * 5)
        self.conv_out = _Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

    def forward(self, x: torch.Tensor,
                noise_scale: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))
        for xi, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + xi
        return self.conv_out(cond)


# ─────────────────────────────────────────────────────────────────────────────
# Beta schedules
# ─────────────────────────────────────────────────────────────────────────────

def _linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02,
                          timesteps, dtype=torch.float64)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def _quadratic_beta_schedule(timesteps: int,
                              start: float = 1e-4,
                              end: float = 5e-1) -> torch.Tensor:
    return torch.linspace(start ** 0.5, end ** 0.5, timesteps) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# GaussianDiffusion1D (from SDEMG/ddpm_1d.py, cleaned up)
# ─────────────────────────────────────────────────────────────────────────────

_ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def _extract(a: torch.Tensor, t: torch.Tensor,
             x_shape: tuple) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion1D(nn.Module):
    """
    DDPM-style 1D Gaussian diffusion with conditional denoising.
    Adapted from the SDEMG repository (Liu et al., ICASSP 2024).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        seq_length: int,
        timesteps:  int   = 1000,
        objective:  str   = 'pred_noise',
        beta_schedule: str = 'cosine',
        ddim_sampling_eta: float = 0.,
        auto_normalize: bool = False,
        loss_function:  str  = 'l2',
        condition:      bool = False,
    ):
        super().__init__()
        self.model       = model
        self.channels    = 1
        self.self_condition = condition
        self.seq_length  = seq_length
        self.objective   = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            "objective must be 'pred_noise', 'pred_x0', or 'pred_v'"

        if beta_schedule == 'linear':
            betas = _linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = _cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quad':
            betas = _quadratic_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas          = 1. - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod.numpy()))
        self.num_timesteps = int(betas.shape[0])
        self.ddim_sampling_eta = ddim_sampling_eta

        def _reg(name, val):
            self.register_buffer(name, val.to(torch.float32))

        _reg('betas',                  betas)
        _reg('alphas_cumprod',         alphas_cumprod)
        _reg('alphas_cumprod_prev',    alphas_cumprod_prev)
        _reg('sqrt_alphas_cumprod',    torch.sqrt(alphas_cumprod))
        _reg('sqrt_one_minus_alphas_cumprod',
             torch.sqrt(1. - alphas_cumprod))
        _reg('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        _reg('sqrt_recip_alphas_cumprod',   torch.sqrt(1. / alphas_cumprod))
        _reg('sqrt_recipm1_alphas_cumprod',
             torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = (betas * (1. - alphas_cumprod_prev)
                              / (1. - alphas_cumprod))
        _reg('posterior_variance', posterior_variance)
        _reg('posterior_log_variance_clipped',
             torch.log(posterior_variance.clamp(min=1e-20)))
        _reg('posterior_mean_coef1',
             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        _reg('posterior_mean_coef2',
             (1. - alphas_cumprod_prev) * torch.sqrt(alphas)
             / (1. - alphas_cumprod))

        self.normalize   = (lambda x: x * 2 - 1) if auto_normalize else _identity
        self.unnormalize = (lambda t: (t + 1) * 0.5) if auto_normalize else _identity

        if loss_function == 'l1':
            self._loss_fn = nn.L1Loss()
        elif loss_function == 'l2':
            self._loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'Unknown loss function: {loss_function}')

    # ── Posterior helpers ─────────────────────────────────────────────────────

    def _predict_start_from_noise(self, x_t, t, noise):
        return (_extract(self.sqrt_recip_alphas_cumprod,  t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def _predict_noise_from_start(self, x_t, t, x0):
        return ((_extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)
                / _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def _predict_v(self, x_start, t, noise):
        return (_extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
                - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * x_start)

    def _predict_start_from_v(self, x_t, t, v):
        return (_extract(self.sqrt_alphas_cumprod,           t, x_t.shape) * x_t
                - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def _q_posterior(self, x_start, x_t, t):
        mean = (_extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var     = _extract(self.posterior_variance,            t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def _model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            self.sqrt_alphas_cumprod_prev[t.cpu().numpy() + 1]
        ).reshape(batch_size, 1).to(x.device)

        model_output = self.model(x, noise_level, x_self_cond)

        clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else _identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start    = clip(self._predict_start_from_noise(x, t, pred_noise))
        elif self.objective == 'pred_x0':
            x_start    = clip(model_output)
            pred_noise = self._predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            x_start    = clip(self._predict_start_from_v(x, t, model_output))
            pred_noise = self._predict_noise_from_start(x, t, x_start)

        return _ModelPrediction(pred_noise, x_start)

    def _p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds   = self._model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        mean, var, log_var = self._q_posterior(x_start, x, t)
        return mean, var, log_var, x_start

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _p_sample(self, x, t_int, x_self_cond=None, clip_denoised=True):
        b, device = x.shape[0], x.device
        t_batch = torch.full((b,), t_int, device=device, dtype=torch.long)
        mean, _, log_var, x_start = self._p_mean_variance(
            x, t_batch, x_self_cond, clip_denoised)
        noise = torch.randn_like(x) if t_int > 0 else 0.
        return mean + (0.5 * log_var).exp() * noise, x_start

    # ── Public inference API ──────────────────────────────────────────────────

    @torch.no_grad()
    def denoise(self, img_noisy: torch.Tensor,
                denoise_timesteps: int | None = None) -> torch.Tensor:
        """
        Standard DDPM reverse diffusion conditioned on noisy observation.

        Args:
            img_noisy          : [B, 1, L]
            denoise_timesteps  : number of reverse steps (default: all T)
        Returns:
            denoised           : [B, 1, L]
        """
        device = self.betas.device
        img    = torch.randn(img_noisy.shape, device=device)
        steps  = _default(denoise_timesteps, self.num_timesteps)
        cond   = img_noisy if self.self_condition else None
        x_start = None

        for t in reversed(range(0, steps)):
            img, x_start = self._p_sample(img, t, cond)

        return self.unnormalize(img)

    @torch.no_grad()
    def ddim_denoise(self, img_noisy: torch.Tensor,
                     clip_denoised: bool = True,
                     denoise_timesteps: int | None = None) -> torch.Tensor:
        """
        DDIM accelerated reverse diffusion (fewer steps, deterministic).

        Args:
            img_noisy          : [B, 1, L]
            clip_denoised      : clamp x_start to [-1, 1]
            denoise_timesteps  : sampling steps (fewer than T for speedup)
        Returns:
            denoised           : [B, 1, L]
        """
        batch  = img_noisy.shape[0]
        device = self.betas.device
        total  = self.num_timesteps
        steps  = _default(denoise_timesteps, total)
        eta    = self.ddim_sampling_eta

        times = torch.linspace(-1, total - 1, steps=steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img     = torch.randn(img_noisy.shape, device=device)
        x_start = img_noisy

        for time, time_next in _tqdm(time_pairs, desc='DDIM denoise',
                                      leave=False):
            t_cond    = torch.full((batch,), time, device=device,
                                    dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self._model_predictions(
                img, t_cond, self_cond, clip_x_start=clip_denoised)[:2]

            if time_next < 0:
                img = x_start
                continue

            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next)
                           * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            img = (x_start * alpha_next.sqrt()
                   + c * pred_noise
                   + sigma * torch.randn_like(img))

        return self.unnormalize(img)

    # ── Training ──────────────────────────────────────────────────────────────

    @autocast(enabled=False)
    def _q_sample(self, x_start: torch.Tensor,
                  continuous_sqrt_alpha_cumprod: torch.Tensor,
                  noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return (continuous_sqrt_alpha_cumprod * x_start
                + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise)

    def _p_losses(self, x_start: torch.Tensor, t: torch.Tensor,
                  x_self_cond: torch.Tensor | None,
                  noise: torch.Tensor | None = None) -> torch.Tensor:
        b, c, n = x_start.shape
        noise   = _default(noise, lambda: torch.randn_like(x_start))

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t.cpu().numpy() - 1],
                self.sqrt_alphas_cumprod_prev[t.cpu().numpy()],
                size=b,
            )
        ).to(x_start.device).view(b, -1).unsqueeze(1)

        x = self._q_sample(x_start, continuous_sqrt_alpha_cumprod, noise)
        model_out = self.model(x, continuous_sqrt_alpha_cumprod, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self._predict_v(x_start, t, noise)
        else:
            raise ValueError(f'Unknown objective: {self.objective}')

        return self._loss_fn.to(x.device)(model_out, target)

    def forward(self, clean_img: torch.Tensor,
                noisy_img: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass — returns diffusion loss.

        Args:
            clean_img : [B, 1, L]
            noisy_img : [B, 1, L]
        Returns:
            scalar loss tensor
        """
        b, c, n = clean_img.shape
        assert n == self.seq_length, \
            f'Sequence length must be {self.seq_length}, got {n}'
        device = clean_img.device
        t    = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        cond = self.normalize(noisy_img) if self.self_condition else None
        return self._p_losses(self.normalize(clean_img), t, cond)


# ─────────────────────────────────────────────────────────────────────────────
# SDEMG — CleanSEMG wrapper (standard baseline interface)
# ─────────────────────────────────────────────────────────────────────────────

class SDEMG(nn.Module):
    """
    SDEMG wrapper conforming to the CleanSEMG baseline model interface.

    All baseline models share:
        Input:  FloatTensor[B, L]   (no channel dim)
        Output: FloatTensor[B, L]

    Training:
        Detected by train_neural.py via HAS_DIFFUSION_LOSS = True.
        Call: loss = model.compute_diffusion_loss(clean, noisy)

    Inference:
        Call: pred = model(noisy)      # standard forward()
              pred = model(noisy, ddim=True, steps=20)   # DDIM fast mode
    """

    HAS_DIFFUSION_LOSS: bool = True   # read by training/train_neural.py

    # Defaults matching Liu et al. ICASSP 2024 / original SDEMG codebase
    _SEQ_LEN   = 2000   # 2 s @ 1 kHz
    _TIMESTEPS = 50
    _FEATS     = 64

    def __init__(
        self,
        seq_length:        int   = _SEQ_LEN,
        timesteps:         int   = _TIMESTEPS,
        feats:             int   = _FEATS,
        objective:         str   = 'pred_noise',
        loss_function:     str   = 'l2',
        beta_schedule:     str   = 'cosine',
        ddim:              bool  = False,
        denoise_timesteps: int | None = None,
    ):
        super().__init__()
        net = ConditionalModel(feats=feats)
        self.diffusion = GaussianDiffusion1D(
            net,
            seq_length    = seq_length,
            timesteps     = timesteps,
            objective     = objective,
            loss_function = loss_function,
            beta_schedule = beta_schedule,
            condition     = True,
        )
        self.ddim              = ddim
        self.denoise_timesteps = denoise_timesteps

    # ── Training ──────────────────────────────────────────────────────────────

    def compute_diffusion_loss(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion training / validation loss.

        Internally samples a random timestep t, adds Gaussian noise to
        `clean`, and returns the MSE between predicted and actual noise.

        Args:
            clean : FloatTensor[B, L]
            noisy : FloatTensor[B, L]  (sEMG + ECG noise, used as condition)
        Returns:
            scalar loss tensor
        """
        # GaussianDiffusion1D expects a channel dim → [B, 1, L]
        return self.diffusion(clean.unsqueeze(1), noisy.unsqueeze(1))

    # ── Inference ─────────────────────────────────────────────────────────────

    def forward(self, noisy: torch.Tensor,
                ddim: bool | None = None,
                steps: int | None = None) -> torch.Tensor:
        """
        Denoise via reverse-diffusion conditioned on noisy input.

        Args:
            noisy : FloatTensor[B, L]
            ddim  : use DDIM sampling (overrides constructor arg if given)
            steps : number of sampling steps (overrides constructor default)
        Returns:
            pred  : FloatTensor[B, L]
        """
        use_ddim = self.ddim if ddim is None else ddim
        n_steps  = steps if steps is not None else self.denoise_timesteps
        n        = noisy.unsqueeze(1)   # [B, 1, L]

        if use_ddim:
            pred = self.diffusion.ddim_denoise(n, denoise_timesteps=n_steps)
        else:
            pred = self.diffusion.denoise(n, denoise_timesteps=n_steps)

        return pred.squeeze(1)          # [B, L]