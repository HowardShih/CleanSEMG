"""
TrustEMG-Net: U-Net + Transformer Residual Masking for sEMG Denoising
======================================================================
 
Adapted from:
    Wang, Kuan-Chen, Kai-Chun Liu, Sheng-Yu Peng, and Yu Tsao.
    "TrustEMG-Net: Using Representation-Masking Transformer with U-Net
    for Surface Electromyography Enhancement."
    IEEE Journal of Biomedical and Health Informatics, vol. 29, no. 4,
    pp. 2506–2520, 2025.
    DOI: https://doi.org/10.1109/JBHI.2024.3504378
    arXiv: https://arxiv.org/abs/2410.03843
    Code: https://github.com/eric-wang135/TrustEMG
 
Modifications for CLEANSEMG:
  - All model variants (TrustEMGNet_RM, TrustEMGNet_DM, TrustEMGNet_LSTM,
    TrustEMGNet_UNetonly, TrustEMGNet_Skipall) are merged into a single
    file for self-contained deployment.
  - Unified input/output interface: forward(emg: [B, L]) → [B, L]
    with automatic unsqueeze/squeeze.
  - Attribute names kept identical to the original checkpoints so
    pre-trained weights load without key remapping.
  - Removed internal training utilities not needed for inference.
 
License: see https://github.com/eric-wang135/TrustEMG for the original license.
         CLEANSEMG modifications are released under MIT License.
"""

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class conv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift, padding=0, dilation=1):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, frame_size, shift, padding, dilation),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_1d(x)


class deconv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift,
                 padding=0, out_pad=0, dilation=1):
        super().__init__()
        self.deconv_1d = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, frame_size, shift,
                               padding, output_padding=out_pad, dilation=dilation),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv_1d(x)


class up(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift, padding=0, out_pad=0):
        super().__init__()
        self.deconv = deconv_1d(in_channel, in_channel // 2, frame_size, shift, padding, out_pad)
        self.conv   = conv_1d(in_channel, out_channel, frame_size, 1, frame_size // 2)

    def forward(self, x1, x2):
        x1   = self.deconv(x1)
        diff = x2.size()[-1] - x1.size()[-1]
        x1   = F.pad(x1, [diff // 2, diff - diff // 2])
        out  = torch.cat([x1, x2], dim=1)
        return self.conv(out)


# ── U-Net base ─────────────────────────────────────────────────────────────

class TrustEMGNet_UNetonly(nn.Module):
    def __init__(self):
        super().__init__()
        K = 8; H = 64; S = 2
        feature_dim = 16 * H

        self.down1 = conv_1d(1,           H,           K, S)
        self.down2 = conv_1d(H,           2 * H,       K, S)
        self.down3 = conv_1d(2 * H,       4 * H,       K, S)
        self.down4 = conv_1d(4 * H,       8 * H,       K, S)
        self.down5 = conv_1d(8 * H,       feature_dim, K, S)

        self.up0 = up(feature_dim, 8 * H, K, S)
        self.up1 = up(8 * H,       4 * H, K, S)
        self.up2 = up(4 * H,       2 * H, K, S)
        self.up3 = up(2 * H,           H, K, S)
        self.up4 = nn.ConvTranspose1d(H, 1, K, S)

    def forward(self, emg):
        x1  = self.down1(emg.unsqueeze(1))
        x2  = self.down2(x1)
        x3  = self.down3(x2)
        x4  = self.down4(x3)
        x5  = self.down5(x4)
        out = self.up0(x5, x4)
        out = self.up1(out, x3)
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        out = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)   # [B, L]


# ── DM (Dense Mamba-style Transformer at bottleneck) ───────────────────────

class TrustEMGNet_DM(TrustEMGNet_UNetonly):
    def __init__(self):
        super().__init__()
        H = 64; feature_dim = 16 * H
        d_model = feature_dim
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8,
            dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bn = x5.permute(0, 2, 1)
        bn = self.positional_encoding(bn)
        bn = self.transformer_encoder(bn)
        out = self.up0(bn.permute(0, 2, 1), x4)
        out = self.up1(out, x3)
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        out = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)


# ── RM (masking bottleneck) ─────────────────────────────────────────────────

class TrustEMGNet_RM(TrustEMGNet_DM):
    def __init__(self):
        super().__init__()

    def forward(self, emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bn_pre = x5.permute(0, 2, 1)
        mask   = self.positional_encoding(bn_pre)
        mask   = torch.sigmoid(self.transformer_encoder(mask))
        bn     = bn_pre * mask
        out    = self.up0(bn.permute(0, 2, 1), x4)
        out    = self.up1(out, x3)
        out    = self.up2(out, x2)
        out    = self.up3(out, x1)
        out    = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)


# ── Skip-all DM ─────────────────────────────────────────────────────────────

class TrustEMGNet_skipall_DM(TrustEMGNet_DM):
    def __init__(self):
        super().__init__()
        H = 64
        feature_dims = [H * (2 ** (i)) for i in range(5)]   # [64, 128, 256, 512, 1024]
        self.pe_list  = nn.ModuleList()
        self.transformer_list = nn.ModuleList()
        for d in feature_dims:
            self.pe_list.append(PositionalEncoding(d))
            el = nn.TransformerEncoderLayer(
                d_model=d, nhead=8,
                dim_feedforward=d * 2,
                dropout=0.1, batch_first=True,
            )
            self.transformer_list.append(nn.TransformerEncoder(el, num_layers=1))

    def forward(self, emg):
        xs = [
            self.down1(emg.unsqueeze(1)),
        ]
        xs.append(self.down2(xs[-1]))
        xs.append(self.down3(xs[-1]))
        xs.append(self.down4(xs[-1]))
        xs.append(self.down5(xs[-1]))
        f_list = []
        for i, x in enumerate(xs):
            f = x.permute(0, 2, 1)
            f = self.pe_list[i](f)
            f = self.transformer_list[i](f).permute(0, 2, 1)
            f_list.append(f)
        out = self.up0(f_list[4], f_list[3])
        out = self.up1(out, f_list[2])
        out = self.up2(out, f_list[1])
        out = self.up3(out, f_list[0])
        out = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)


# ── Skip-all RM ─────────────────────────────────────────────────────────────

class TrustEMGNet_skipall_RM(TrustEMGNet_skipall_DM):
    def __init__(self):
        super().__init__()

    def forward(self, emg):
        xs = [self.down1(emg.unsqueeze(1))]
        xs.append(self.down2(xs[-1]))
        xs.append(self.down3(xs[-1]))
        xs.append(self.down4(xs[-1]))
        xs.append(self.down5(xs[-1]))
        f_list = []
        for i, x in enumerate(xs):
            f    = x.permute(0, 2, 1)
            mask = self.pe_list[i](f)
            mask = torch.sigmoid(self.transformer_list[i](mask))
            f    = (f * mask).permute(0, 2, 1)
            f_list.append(f)
        out = self.up0(f_list[4], f_list[3])
        out = self.up1(out, f_list[2])
        out = self.up2(out, f_list[1])
        out = self.up3(out, f_list[0])
        out = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)


# ── LSTM DM ─────────────────────────────────────────────────────────────────

class TrustEMGNet_LSTM_DM(TrustEMGNet_UNetonly):
    def __init__(self):
        super().__init__()
        feature_dim = 16 * 64
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=feature_dim,
            batch_first=True, num_layers=1, dropout=0,
        )

    def forward(self, emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bn, _ = self.lstm(x5.permute(0, 2, 1))
        out = self.up0(bn.permute(0, 2, 1), x4)
        out = self.up1(out, x3)
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        out = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)


# ── LSTM RM ──────────────────────────────────────────────────────────────────

class TrustEMGNet_LSTM_RM(TrustEMGNet_LSTM_DM):
    def __init__(self):
        super().__init__()

    def forward(self, emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bn   = x5.permute(0, 2, 1)
        mask, _ = self.lstm(bn)
        bn   = bn * torch.sigmoid(mask)
        out  = self.up0(bn.permute(0, 2, 1), x4)
        out  = self.up1(out, x3)
        out  = self.up2(out, x2)
        out  = self.up3(out, x1)
        out  = self.up4(out)
        return out[:, :, :emg.shape[1]].squeeze(1)