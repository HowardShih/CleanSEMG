"""
FCN Encoder-Decoder for Single-Channel sEMG Denoising
======================================================
 
Adapted from:
    Wang, Kuan-Chen, Kai-Chun Liu, and Yu Tsao.
    "Removing ECG Artifacts from Surface Electromyogram Signals Using
    Fully Convolutional Networks."
    IEEE Sensors Journal, 2023.
    DOI: https://doi.org/10.1109/JSEN.2023.3234567
    Code: https://github.com/ASUS217/ECG-removal-from-sEMG
 
Modifications for CLEANSEMG:
  - Unified input/output interface: forward(x: [B, L]) → [B, L]
  - Added unsqueeze/squeeze so the same train/eval pipeline works
    for all baseline models without per-model boilerplate.
  - Attribute names (conv_1d, deconv_1d) kept identical to the
    original so pre-trained checkpoints load without key remapping.
 
License: see the original repository for the original license.
         CLEANSEMG modifications are released under MIT License.
"""

import torch
import torch.nn as nn


class _conv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift, padding=0, dilation=1):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, frame_size, shift, padding, dilation),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_1d(x)


class _deconv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift,
                 padding=0, out_pad=0, dilation=1):
        super().__init__()
        self.deconv_1d = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, frame_size, shift,
                               padding, output_padding=out_pad, dilation=dilation),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deconv_1d(x)


class FCN(nn.Module):
    """
    Encoder-decoder Fully Convolutional Network for sEMG denoising.
    Input:  [B, L]  (waveform, e.g. L=2000)
    Output: [B, L]
    """
    def __init__(self):
        super().__init__()
        K = 8; H = 64; S = 2
        feature_dim = 16 * H
        self.encoder = nn.Sequential(
            _conv_1d(1,           H,           K, S),
            _conv_1d(H,           2 * H,       K, S),
            _conv_1d(2 * H,       4 * H,       K, S),
            _conv_1d(4 * H,       8 * H,       K, S),
            _conv_1d(8 * H,       feature_dim, K, S),
        )
        self.decoder = nn.Sequential(
            _deconv_1d(feature_dim, 8 * H, K, S, out_pad=1),
            _deconv_1d(8 * H,       4 * H, K, S, out_pad=1),
            _deconv_1d(4 * H,       2 * H, K, S),
            _deconv_1d(2 * H,           H, K, S),
            nn.ConvTranspose1d(H, 1, K, S),
        )

    def forward(self, emg):
        f   = self.encoder(emg.unsqueeze(1))
        out = self.decoder(f)
        return out[:, :, :emg.shape[1]].squeeze(1)