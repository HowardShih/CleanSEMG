"""
${CLEANSEMG_ROOT}/baseline_models/CNN.py
CNN_waveform for waveform-based sEMG denoising.
Note: expects exactly 2000-sample inputs (2s @ 1000Hz),
which matches the pipeline's segment length.
"""
import torch
import torch.nn as nn


class _Dense_L_b(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dense(x)


class _conv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, frame_size, shift),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_1d(x)


class CNN_waveform(nn.Module):
    """
    CNN that processes the 2000-sample input as 10 × 200-sample chunks.
    Input:  [B, 2000]
    Output: [B, 2000]
    """
    N_CHUNKS = 10
    CHUNK_SIZE = 200

    def __init__(self):
        super().__init__()
        K = 8; H = 64; S = 2
        self.encoder = nn.Sequential(
            _conv_1d(1,       H,     K, S),
            _conv_1d(H,       2 * H, K, S),
            _conv_1d(2 * H,   4 * H, K, S),
            _conv_1d(4 * H,   8 * H, K, S),
            nn.Flatten(),
        )
        # After 4 stride-2 convolutions on 200-sample input:
        # length = ((((200-8)//2+1 - 8)//2+1 - 8)//2+1 - 8)//2+1 = 6  (approx)
        # actual flatten dim = 8*H * 6 = 512 * 6 = 3072
        self.FC = nn.Sequential(
            _Dense_L_b(3072, 400),
            nn.Dropout(0.5),
            nn.Linear(400, self.CHUNK_SIZE, bias=True),
        )

    def forward(self, x):
        # x: [B, 2000]
        B = x.shape[0]
        chunks = x.reshape(B, self.N_CHUNKS, self.CHUNK_SIZE)  # [B, 10, 200]
        outs = []
        for i in range(self.N_CHUNKS):
            chunk = chunks[:, i, :].unsqueeze(1)   # [B, 1, 200]
            enc   = self.encoder(chunk)             # [B, 3072]
            out_i = self.FC(enc)                    # [B, 200]
            outs.append(out_i)
        return torch.cat(outs, dim=-1)              # [B, 2000]