#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSEMG (EMG-MAMBA) baseline model for sEMG denoising.

${CLEANSEMG_ROOT}/baseline_models/MSEMG.py

Shape contract (matches other baseline models):
  Input:  [B, L]   (waveform, same as TrustEMGNet / FCN / CNN_waveform)
  Output: [B, L]

Internal forward uses [B, 1, L] as required by Conv1d(1, feats, ...).
The public forward() handles unsqueeze / squeeze automatically so
train_baseline.py and inference_baseline.py need no changes.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── mamba_ssm imports ─────────────────────────────────────────────────────────
# We intentionally do NOT import Block from mamba_ssm because its signature
# changed across versions (mamba_ssm>=2.0 requires mlp_cls).
# Instead we define _MambaResidualBlock below which is version-independent.
from mamba_ssm.modules.mamba_simple import Mamba

# _init_weights: location varies by version
try:
    from mamba_ssm.models.mixer_seq_simple import _init_weights
except ImportError:
    def _init_weights(module, n_layer, *args, **kwargs):
        pass

# RMSNorm: try triton first, fall back to pure-torch
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except ImportError:
        from torch.nn import LayerNorm as RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# Version-independent Mamba residual block
# ─────────────────────────────────────────────────────────────────────────────

class _MambaResidualBlock(nn.Module):
    """
    Minimal Mamba residual block that works with ALL mamba_ssm versions.

    Equivalent to Block(dim, mixer_cls, norm_cls, fused_add_norm=False)
    from mamba_ssm <2.0.  We avoid importing Block entirely because >=2.0
    changed its signature to require mlp_cls.

    Forward:
        residual = x  (or x + prev_residual)
        out      = mixer(norm(residual))
        returns  out, residual
    """
    def __init__(self, dim, mixer_cls, norm_cls):
        super().__init__()
        self.norm  = norm_cls(dim)
        self.mixer = mixer_cls(dim)

    def forward(self, x, residual=None, inference_params=None):
        r = x if residual is None else x + residual
        h = self.mixer(self.norm(r))
        return h, r


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super().__init__()
        self.forward_blocks  = nn.ModuleList([])
        self.backward_blocks = None

        for i in range(n_layer):
            self.forward_blocks.append(
                _MambaResidualBlock(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                )
            )

        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    _MambaResidualBlock(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x):
        # x: [B, L, C]
        for_residual = None
        forward_f = x.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(x, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f
            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)

        return residual


class HNFBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        self.filters = nn.ModuleList([
            Conv1d(input_size,   hidden_size // 4, 3,  dilation=dilation, padding=1  * dilation, padding_mode='reflect'),
            Conv1d(hidden_size,  hidden_size // 4, 5,  dilation=dilation, padding=2  * dilation, padding_mode='reflect'),
            Conv1d(hidden_size,  hidden_size // 4, 9,  dilation=dilation, padding=4  * dilation, padding_mode='reflect'),
            Conv1d(hidden_size,  hidden_size // 4, 15, dilation=dilation, padding=7  * dilation, padding_mode='reflect'),
        ])
        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        self.norm   = nn.InstanceNorm1d(hidden_size // 2)
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')

    def forward(self, x):
        residual = x
        filts = torch.cat([layer(x) for layer in self.filters], dim=1)
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.2)
        filts = F.leaky_relu(self.conv_2(filts), 0.2)
        return filts + residual


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MSEMG(nn.Module):
    """
    EMG-MAMBA waveform denoising model.

    Public interface (matches all other baselines):
      x:    [B, L]   -> float32 waveform
      out:  [B, L]   -> float32 denoised waveform
    """

    def __init__(self, feats: int = 64, n_layer: int = 1):
        super().__init__()

        self.conv = nn.Sequential(
            Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.hnf_encode = HNFBlock(feats, feats, 1)
        self.mamba      = MambaBlock(in_channels=feats, n_layer=n_layer, bidirectional=False)
        self.hnf_decode = HNFBlock(feats, feats, 1)
        self.conv_out   = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

        self.n_layer = n_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)          # [B, L] -> [B, 1, L]

        x = self.conv(x)                # [B, feats, L]
        x = self.hnf_encode(x)          # [B, feats, L]

        for _ in range(self.n_layer):
            x = self.mamba(x.permute(0, 2, 1)).permute(0, 2, 1)   # [B, feats, L]

        x = self.hnf_decode(x)          # [B, feats, L]
        x = self.conv_out(x)            # [B, 1, L]

        return x.squeeze(1)             # [B, L]