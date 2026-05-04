#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CleanSEMG — Fatigue Classification Downstream (Dilated CNN)
CleanSEMG path: downstream/fatigue/fatigue_cnn.py

Dataset:
    Cerqueira et al. (2024) "Muscular Fatigue Dataset."
    Sensors 24(24): 8081. https://doi.org/10.3390/s24248081
    Available at: https://zenodo.org/records/13860256  (CC BY 4.0)

Experiment:
    Train Dilated CNN on baseline (clean) raw waveforms.
    Test the frozen model on:
        1. Baseline (clean filtered signal)
        2. Noisy    (online noise mixing)
        3. Denoised (one of 8 CleanSEMG denoising methods)

Usage:
    export CLEANSEMG_ROOT=/path/to/CleanSEMG
    export DATA_ROOT=/path/to/your/data

    python downstream/fatigue/fatigue_cnn.py --denoiser trustemg
    python downstream/fatigue/fatigue_cnn.py --denoiser hp

    Supported --denoiser values:
        Classical : hp  emd  vmd  ceemdan
        Neural    : fcn  msemg  sdemg  trustemg
"""

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore")

# ── Project root ──────────────────────────────────────────────────────────────
_THIS_DIR      = Path(__file__).resolve().parent
CLEANSEMG_ROOT = Path(os.environ.get("CLEANSEMG_ROOT", _THIS_DIR.parent.parent))
sys.path.insert(0, str(CLEANSEMG_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
FS                = 1000
WINDOW_SEC        = 4.0
STEP_SEC          = 2.0
N_SUBJECTS        = 13
N_FOLDS           = 3
RANDOM_STATE      = 42
SKIP_BOUNDARY_WIN = True

BATCH_SIZE   = 64
N_EPOCHS     = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4

SNR_GRID   = [-15, -10, -5, 0, 5, 10, 15]
MIXED_SEED = 42 + 9999

CHUNK_SAMPLES = 2000
CLIP_RNG      = (-1.0, 1.0)
Q99_PCT       = 0.99

EXCLUDE = {(11,4),(8,9),(9,9),(5,10),(3,1),(7,5),(11,5),(12,4),(9,6)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Denoiser map (same as fatigue_svm.py) ─────────────────────────────────────
_NEURAL_MODEL_MAP = {
    "fcn":      ("FCN",            "FCN_best.pth"),
    "msemg":    ("MSEMG",          "MSEMG_best.pth"),
    "sdemg":    ("SDEMG",          "SDEMG_best.pth"),
    "trustemg": ("TrustEMGNet_RM", "TrustEMGNet_RM_best.pth"),
}
_CLASSICAL_METHODS = {"hp", "emd", "vmd", "ceemdan"}
_ALL_METHODS = list(_NEURAL_MODEL_MAP.keys()) + sorted(_CLASSICAL_METHODS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dilated CNN classifier
# ─────────────────────────────────────────────────────────────────────────────

class DilatedBlock(nn.Module):
    """1-D dilated convolution block with residual connection and GELU activation."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv  = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn    = nn.BatchNorm1d(out_ch)
        self.act   = nn.GELU()
        self.res   = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch
                      else nn.Identity())
        self._trim = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self._trim > 0:
            out = out[:, :, :-self._trim]
        return self.act(self.bn(out)) + self.res(x)


class DilatedCNN(nn.Module):
    """
    Temporal dilated CNN for fatigue classification.

    Architecture: stacked dilated blocks (1, 2, 4, 8, 16) →
    global average pooling → dropout → linear.

    Input : [B, n_channels, window_samples]
    Output: [B, n_classes]
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 3,
                 base_ch: int = 32,
                 dilations: tuple = (1, 2, 4, 8, 16),
                 kernel: int = 7, dropout: float = 0.4):
        super().__init__()
        layers, in_ch = [], n_channels
        for i, d in enumerate(dilations):
            out_ch = base_ch if i == 0 else base_ch * 2
            layers.append(DilatedBlock(in_ch, out_ch, kernel, d))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.gap(self.encoder(x)).squeeze(-1)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Denoiser factory (identical interface to fatigue_svm.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_neural_denoiser(denoiser_key: str, weights_dir: Path):
    from denoising import BASELINE_MODEL_REGISTRY
    model_name, ckpt_file = _NEURAL_MODEL_MAP[denoiser_key]
    ckpt_path = weights_dir / model_name / ckpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BASELINE_MODEL_REGISTRY[model_name]().to(dev)
    state = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state: state = state[key]; break
    model.load_state_dict(state)
    model.eval()
    print(f"[Denoiser] {model_name} on {dev}: {ckpt_path}")
    return model, dev


def build_denoise_fn(denoiser_key: str, weights_dir: Path,
                     tradition_params_path: Path | None):
    if denoiser_key in _NEURAL_MODEL_MAP:
        model, dev = _load_neural_denoiser(denoiser_key, weights_dir)

        @torch.no_grad()
        def _neural(sig: np.ndarray) -> np.ndarray:
            sig = sig.astype(np.float32); n = len(sig); out = np.zeros(n, np.float32)
            for s in range(0, n, CHUNK_SAMPLES):
                e = min(s + CHUNK_SAMPLES, n); al = e - s
                ch = sig[s:e].copy()
                if al < CHUNK_SAMPLES:
                    ch = np.concatenate([ch, np.zeros(CHUNK_SAMPLES - al, np.float32)])
                scale = float(np.quantile(np.abs(ch), Q99_PCT))
                if scale < 1e-9: scale = 1.0
                norm = np.clip(ch / scale, *CLIP_RNG).astype(np.float32)
                t    = torch.from_numpy(norm).float().to(dev).unsqueeze(0)
                pred = model(t)
                out[s:e] = (pred.squeeze(0).cpu().numpy() * scale)[:al]
            return out
        return _neural

    params = {}
    if tradition_params_path and tradition_params_path.exists():
        with open(tradition_params_path) as f:
            params = json.load(f)
        print(f"[Denoiser] {denoiser_key.upper()} params from {tradition_params_path}")
    from denoising.classical_filters import apply_method as _tap

    def _classical(sig: np.ndarray) -> np.ndarray:
        sig = np.asarray(sig, dtype=np.float64)
        result, ok = _tap(denoiser_key, sig, params, fs=FS, noise_type="")
        return (result if ok else sig).astype(np.float32)
    return _classical


# ─────────────────────────────────────────────────────────────────────────────
# 3. Noise mixer (identical to fatigue_svm.py)
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicNoiseMixer:
    def __init__(self, noise_root: Path, seed: int = 42):
        from glob import glob
        self.base_seed = seed; self.noise_paths = {}; self.noise_cache = {}
        for ntype in ["PLI", "ECG", "MOA", "WGN", "Color"]:
            ndir = noise_root / ntype
            if ndir.is_dir():
                paths = sorted(ndir.glob("*.npy"))
                if paths: self.noise_paths[ntype] = [str(p) for p in paths]
        if not self.noise_paths: raise ValueError(f"No noise in {noise_root}")
        print(f"[NoiseMixer] {noise_root}")
        for nt, ps in self.noise_paths.items(): print(f"  {nt}: {len(ps)} files")
        for paths in self.noise_paths.values():
            for p in paths: self.noise_cache[p] = np.load(p).astype(np.float64)
        print(f"[NoiseMixer] {len(self.noise_cache)} cached")

    def _sample(self, noise, length, rng):
        if len(noise) < length: noise = np.tile(noise, (length // len(noise)) + 2)
        ms = len(noise) - length
        return noise[int(rng.integers(0, ms + 1)) if ms > 0 else 0:][:length].copy()

    def mix(self, clean, snr, k, sample_idx):
        seed  = abs((self.base_seed * 1_000_000 + (sample_idx % 100_000) * 100
                     + int(snr + 20) * 10 + k)) % (2**31 - 1)
        rng   = np.random.default_rng(seed)
        clean = np.asarray(clean, dtype=np.float64).flatten(); L = len(clean)
        avail = list(self.noise_paths.keys())
        if snr < -5 and "WGN" in avail: avail = [t for t in avail if t != "WGN"]
        k     = min(k, len(avail))
        sel   = [avail[i] for i in sorted(rng.choice(len(avail), size=k, replace=False))]
        cpow  = np.dot(clean, clean)
        if cpow < 1e-12: return clean.copy(), sel
        te    = cpow / (10.0 ** (snr / 10.0)) / k
        comb  = np.zeros(L)
        for nt in sel:
            fi   = int(rng.integers(0, len(self.noise_paths[nt])))
            seg  = self._sample(self.noise_cache[self.noise_paths[nt][fi]], L, rng)
            npow = np.dot(seg, seg)
            comb += (np.sqrt(te / npow) if npow > 1e-12 else 0.0) * seg
        return clean + comb, sel


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data loading (raw waveforms for CNN)
# ─────────────────────────────────────────────────────────────────────────────

def align_labels(emg_time, lbl_time, lbl_vals):
    idx  = np.searchsorted(lbl_time, emg_time).clip(0, len(lbl_time) - 1)
    prev = np.maximum(idx - 1, 0)
    closer = np.abs(lbl_time[prev] - emg_time) < np.abs(lbl_time[idx] - emg_time)
    idx[closer] = prev[closer]
    return lbl_vals[idx].astype(np.int8)


def compute_rms_norm(chs, la):
    mask = (la == 0)
    if mask.sum() < 100: mask = np.ones(len(la), bool)
    return [float(np.sqrt(np.mean(ch[mask] ** 2))) or 1.0 for ch in chs]


def segment_raw(chs: list[np.ndarray],
                la: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment into windows of shape [n_channels, window_samples]."""
    win  = int(WINDOW_SEC * FS)
    step = int(STEP_SEC * FS)
    n    = len(chs[0]); X, y = [], []

    for s in range(0, n - win + 1, step):
        e  = s + win
        lc = int(la[s + win // 2])
        ls = int(la[s]); le = int(la[min(e - 1, n - 1)])
        if SKIP_BOUNDARY_WIN and ls != le: continue
        seg = np.stack([ch[s:e] for ch in chs], axis=0)  # [C, win]
        if not np.all(np.isfinite(seg)): continue
        X.append(seg.astype(np.float32)); y.append(lc)

    if not X:
        return np.empty((0, len(chs), win), np.float32), np.empty(0, int)
    return np.stack(X), np.array(y, int)


def load_all_data(cache_dir, label_dir, mixer, denoise_fn):
    rng = np.random.default_rng(MIXED_SEED)
    X_bl_all, X_ny_all, X_dn_all, y_all = [], [], [], []

    subj_iter = range(1, N_SUBJECTS + 1)
    if HAS_TQDM: subj_iter = tqdm(subj_iter, desc="Loading subjects")

    for subj in subj_iter:
        bl_subj   = cache_dir / "baseline" / f"subject_{subj}"
        subj_ldir = label_dir / f"subject_{subj}"
        if not bl_subj.is_dir(): continue

        trial_nums = sorted(set(
            int(re.findall(r"trial_(\d+)_ch", f)[0])
            for f in os.listdir(bl_subj)
            if re.match(r"trial_\d+_ch0\.npy", f)))

        for trial in trial_nums:
            if (subj, trial) in EXCLUDE: continue
            t_path = bl_subj / f"trial_{trial}_time.npy"
            if not t_path.exists(): continue
            emg_time = np.load(t_path).astype(np.float64)

            cands = sorted(
                [f for f in os.listdir(subj_ldir)
                 if f.endswith(".csv") and re.search(rf"(?<!\d){trial}(?!\d)", f)],
                key=lambda x: int(re.findall(r"\d+", x)[-1]))
            if not cands: continue

            ldf = pd.read_csv(subj_ldir / cands[0], header=0)
            lt  = ldf.iloc[:, 0].values.astype(np.float64)
            lv  = ldf.iloc[:, 1].values.astype(np.float64)
            si  = np.argsort(lt); lt, lv = lt[si], lv[si]
            la  = align_labels(emg_time, lt, lv)

            chs_bl, ok = [], True
            for ci in range(4):
                p = bl_subj / f"trial_{trial}_ch{ci}.npy"
                if not p.exists(): ok = False; break
                chs_bl.append(np.load(p).astype(np.float64))
            if not ok: continue

            sl = len(chs_bl[0])
            if len(la) != sl:
                la = (la[:sl] if len(la) > sl
                      else np.concatenate([la, np.full(sl - len(la), la[-1])]))

            trial_snr  = float(rng.choice(SNR_GRID))
            trial_k    = int(rng.integers(1, 6))
            if trial_snr < -5: trial_k = min(trial_k, 4)
            trial_seed = subj * 100_000 + trial

            chs_ny, chs_dn = [], []
            for ci, ch in enumerate(chs_bl):
                noisy, _ = mixer.mix(ch, snr=trial_snr, k=trial_k,
                                     sample_idx=trial_seed + ci)
                denoised  = denoise_fn(noisy.astype(np.float32))
                chs_ny.append(noisy); chs_dn.append(denoised.astype(np.float64))

            bases    = compute_rms_norm(chs_bl, la)
            chs_bl_n = [ch / b for ch, b in zip(chs_bl, bases)]
            chs_ny_n = [ch / b for ch, b in zip(chs_ny, bases)]
            chs_dn_n = [ch / b for ch, b in zip(chs_dn, bases)]

            X_bl_t, y_t = segment_raw(chs_bl_n, la)
            X_ny_t, _   = segment_raw(chs_ny_n, la)
            X_dn_t, _   = segment_raw(chs_dn_n, la)
            if len(y_t) == 0: continue

            X_bl_all.append(X_bl_t); X_ny_all.append(X_ny_t)
            X_dn_all.append(X_dn_t); y_all.append(y_t)

    X_bl = np.concatenate(X_bl_all)
    X_ny = np.concatenate(X_ny_all)
    X_dn = np.concatenate(X_dn_all)
    y    = np.concatenate(y_all)

    win = int(WINDOW_SEC * FS)
    print(f"\n[Data] Total windows: {len(y)}  shape: {X_bl.shape}")
    for c, name in enumerate(["Non-fatigue", "Transition", "Fatigue"]):
        cnt = (y == c).sum()
        print(f"  Class {c} ({name:12s}): {cnt:5d}  ({100*cnt/len(y):.1f}%)")
    return X_bl, X_ny, X_dn, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _class_weights(y_tr: np.ndarray) -> torch.Tensor:
    c = np.bincount(y_tr, minlength=3).astype(float)
    return torch.tensor(c.sum() / (3 * c), dtype=torch.float32).to(DEVICE)


def _make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
        batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True)


def _train_epoch(model, loader, opt, crit):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); crit(model(xb), yb).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def _predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); preds, truths = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        truths.extend(yb.numpy())
    return np.array(preds), np.array(truths)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(X_bl, X_ny, X_dn, y, denoiser_name):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    results = {k: [] for k in ("acc_bl","acc_ny","acc_dn","f1_bl","f1_ny","f1_dn")}
    cm      = {k: np.zeros((3, 3), int) for k in ("bl","ny","dn")}

    print(f"\n{'='*60}")
    print(f"  {N_FOLDS}-Fold CV | Denoiser: {denoiser_name.upper()} | Classifier: Dilated CNN")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    folds = list(enumerate(skf.split(X_bl, y), 1))
    if HAS_TQDM: folds = tqdm(folds, desc="CV folds", total=N_FOLDS)

    for fold, (tr_idx, te_idx) in folds:
        tr_loader = _make_loader(X_bl[tr_idx], y[tr_idx], shuffle=True)
        loaders   = {tag: _make_loader(X[te_idx], y[te_idx])
                     for tag, X in [("bl", X_bl), ("ny", X_ny), ("dn", X_dn)]}

        model = DilatedCNN(n_channels=X_bl.shape[1]).to(DEVICE)
        crit  = nn.CrossEntropyLoss(weight=_class_weights(y[tr_idx]))
        opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

        best_acc, best_state = 0.0, None
        for ep in range(1, N_EPOCHS + 1):
            _train_epoch(model, tr_loader, opt, crit); sch.step()
            if ep % 5 == 0 or ep == N_EPOCHS:
                p, t = _predict(model, loaders["bl"])
                a = accuracy_score(t, p) * 100
                if a > best_acc:
                    best_acc  = a
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        for tag, loader in loaders.items():
            yp, yt = _predict(model, loader)
            acc = accuracy_score(yt, yp) * 100
            f1  = f1_score(yt, yp, average="macro") * 100
            results[f"acc_{tag}"].append(acc)
            results[f"f1_{tag}"].append(f1)
            cm[tag] += confusion_matrix(yt, yp, labels=[0, 1, 2])

        if not HAS_TQDM:
            print(f"  Fold {fold:2d} | BL {results['acc_bl'][-1]:.2f}%"
                  f" | NY {results['acc_ny'][-1]:.2f}%"
                  f" | DN {results['acc_dn'][-1]:.2f}%")

    stats = {k: (float(np.mean(v)), float(np.std(v)))
             for k, v in results.items()}
    bl, ny, dn = stats["acc_bl"][0], stats["acc_ny"][0], stats["acc_dn"][0]
    print(f"\n  Baseline : {bl:.3f} ± {stats['acc_bl'][1]:.3f} %")
    print(f"  Noisy    : {ny:.3f} ± {stats['acc_ny'][1]:.3f} %  (Δ {ny-bl:+.3f} pp)")
    print(f"  Denoised : {dn:.3f} ± {stats['acc_dn'][1]:.3f} %  (Δ {dn-bl:+.3f} pp)")
    return results, cm, stats


# ─────────────────────────────────────────────────────────────────────────────
# 7. Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results, cm, stats, denoiser_name, noise_types, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    bl, ny, dn = stats["acc_bl"][0], stats["acc_ny"][0], stats["acc_dn"][0]

    lines = [
        "=" * 70,
        f"CleanSEMG — Fatigue Downstream  |  Denoiser: {denoiser_name.upper()}"
        f"  |  Classifier: Dilated CNN",
        "Dataset: Cerqueira et al. (2024) Sensors 24(24): 8081",
        "         https://zenodo.org/records/13860256  (CC BY 4.0)", "=" * 70, "",
        "── Experiment ──",
        "  Train Dilated CNN on clean baseline raw waveforms.",
        "  Test frozen model on baseline / noisy / denoised.", "",
        "── Classifier ──",
        "  Dilated CNN  dilations=(1,2,4,8,16)  kernel=7  base_ch=32  dropout=0.4",
        f"  Epochs={N_EPOCHS}  LR={LR}  batch={BATCH_SIZE}  CosineAnnealingLR", "",
        f"── Noise ──",
        f"  Types  : {noise_types}",
        f"  SNR    : {SNR_GRID} dB  (mixed per trial)",
        f"  k      : 1–5  (mixed per trial)", "",
        "── Results ──",
        f"  {'Mode':<28} {'Acc (%)':>16}  {'Macro-F1 (%)':>16}  {'Δ Acc':>10}",
        f"  {'─'*28} {'─'*16}  {'─'*16}  {'─'*10}",
        f"  {'Baseline':<28} {bl:>8.3f} ± {stats['acc_bl'][1]:.3f}  "
        f"{stats['f1_bl'][0]:>8.3f} ± {stats['f1_bl'][1]:.3f}  {'—':>10}",
        f"  {'Noisy':<28} {ny:>8.3f} ± {stats['acc_ny'][1]:.3f}  "
        f"{stats['f1_ny'][0]:>8.3f} ± {stats['f1_ny'][1]:.3f}  {ny-bl:>+10.3f}",
        f"  {f'Denoised ({denoiser_name.upper()})':<28} {dn:>8.3f} ± {stats['acc_dn'][1]:.3f}  "
        f"{stats['f1_dn'][0]:>8.3f} ± {stats['f1_dn'][1]:.3f}  {dn-bl:>+10.3f}", "",
        f"  Noise degradation  : {ny-bl:+.3f} pp",
        f"  Denoising recovery : {dn-ny:+.3f} pp",
        f"  Net vs clean       : {dn-bl:+.3f} pp",
    ]
    for tag, lbl in [("bl","Baseline"), ("ny","Noisy"),
                     ("dn", f"Denoised ({denoiser_name.upper()})")]:
        lines += ["", f"Confusion Matrix — {lbl}:",
                  "              Pred:Non  Pred:Trans  Pred:Fat"]
        for i, row in enumerate(cm[tag]):
            lb = ["True:Non  ", "True:Trans", "True:Fat  "][i]
            lines.append(f"  {lb}  {row[0]:7d}   {row[1]:9d}   {row[2]:8d}")

    text = "\n".join(lines)
    print("\n" + text)
    (output_dir / "summary.txt").write_text(text)

    rows = [{"fold": i + 1,
             "acc_baseline": results["acc_bl"][i],
             "acc_noisy":    results["acc_ny"][i],
             "acc_denoised": results["acc_dn"][i],
             "f1_baseline":  results["f1_bl"][i],
             "f1_noisy":     results["f1_ny"][i],
             "f1_denoised":  results["f1_dn"][i]}
            for i in range(N_FOLDS)]
    pd.DataFrame(rows).to_csv(output_dir / "fold_results.csv", index=False)
    print(f"\n[INFO] Results saved → {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="CleanSEMG fatigue downstream (Dilated CNN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Denoiser choices:
  Classical : {sorted(_CLASSICAL_METHODS)}
  Neural    : {sorted(_NEURAL_MODEL_MAP.keys())}

Example:
  export CLEANSEMG_ROOT=/path/to/CleanSEMG
  python downstream/fatigue/fatigue_cnn.py --denoiser trustemg
        """,
    )
    ap.add_argument("--denoiser",   required=True, choices=_ALL_METHODS)
    ap.add_argument("--cache-dir",  default=None)
    ap.add_argument("--label-dir",  default=None)
    ap.add_argument("--noise-root", default=None)
    ap.add_argument("--weights-dir",default=None)
    ap.add_argument("--tradition-params", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root      = CLEANSEMG_ROOT
    data_root = Path(os.environ.get("DATA_ROOT", root))

    cache_dir   = Path(args.cache_dir)   if args.cache_dir       else root / "outputs/downstream/fatigue/cache"
    label_dir   = Path(args.label_dir)   if args.label_dir       else data_root / "Cerqueira/self_perceived_fatigue_index"
    noise_root  = Path(args.noise_root)  if args.noise_root      else root / "outputs/noise/sEMG_noise_test"
    weights_dir = Path(args.weights_dir) if args.weights_dir     else root / "outputs/weights_baseline"
    trad_params = Path(args.tradition_params) if args.tradition_params else root / "outputs/weights_tradition/tradition_params.json"
    output_dir  = Path(args.output_dir)  if args.output_dir      else root / f"outputs/downstream/fatigue/results/cnn_{args.denoiser}"

    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    print(f"\n[INFO] CleanSEMG Fatigue Downstream (Dilated CNN)")
    print(f"  Denoiser : {args.denoiser.upper()}")
    print(f"  Device   : {DEVICE}")
    print(f"  Cache    : {cache_dir}\n")

    denoise_fn  = build_denoise_fn(args.denoiser, weights_dir, trad_params)
    mixer       = DeterministicNoiseMixer(noise_root, seed=42)
    noise_types = list(mixer.noise_paths.keys())

    X_bl, X_ny, X_dn, y = load_all_data(cache_dir, label_dir, mixer, denoise_fn)
    results, cm, stats   = run_cv(X_bl, X_ny, X_dn, y, args.denoiser)
    save_results(results, cm, stats, args.denoiser, noise_types, output_dir)


if __name__ == "__main__":
    main()