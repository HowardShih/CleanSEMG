#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CleanSEMG — Fatigue Classification Downstream (RBF-SVM)
CleanSEMG path: downstream/fatigue/fatigue_svm.py

Dataset:
    Cerqueira et al. (2024) "Muscular Fatigue Dataset."
    Sensors 24(24): 8081. https://doi.org/10.3390/s24248081
    Available at: https://zenodo.org/records/13860256  (CC BY 4.0)

Experiment:
    Fit RBF-SVM on baseline (clean) features.
    Test the frozen scaler + SVM on:
        1. Baseline (clean filtered signal)
        2. Noisy    (online noise mixing)
        3. Denoised (one of 8 CleanSEMG denoising methods)

Features: 4 ch × 8 EMG features + delta = 64-dim vector
    [Phinyomark et al., IntechOpen 2012]

Usage:
    # Set paths
    export CLEANSEMG_ROOT=/path/to/CleanSEMG
    export DATA_ROOT=/path/to/your/data  # contains Cerqueira/ and DB2/ etc.

    # Run with a neural denoiser
    python downstream/fatigue/fatigue_svm.py \\
        --denoiser trustemg \\
        --cache-dir  outputs/downstream/fatigue/cache \\
        --noise-root outputs/noise/sEMG_noise_test \\
        --weights-dir outputs/weights_baseline \\
        --output-dir  outputs/downstream/fatigue/results

    # Run with a classical method
    python downstream/fatigue/fatigue_svm.py \\
        --denoiser hp \\
        --tradition-params outputs/weights_tradition/tradition_params.json

    Supported --denoiser values:
        Classical : hp  emd  vmd  ceemdan
        Neural    : fcn  msemg  sdemg  trustemg

References:
    TrustEMG-Net : Wang et al., IEEE JBHI 29(4):2506-2520, 2025.
                   doi: 10.1109/JBHI.2024.3475817
    MSEMG        : EMG-MAMBA, ICASSP 2025.
    SDEMG        : Liu et al., ICASSP 2024.
    FCN          : Encoder-decoder FCN (waveform baseline).
    HP / EMD / VMD / CEEMDAN : see configs/config_tradition.yaml for references.
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
from scipy.fft import fft, fftfreq
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.nn as nn

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

# ── Constants (shared across all experiments) ─────────────────────────────────
FS                = 1000    # cache is at 1 kHz
WINDOW_SEC        = 4.0
STEP_SEC          = 2.0
N_SUBJECTS        = 13
N_FOLDS           = 3
RANDOM_STATE      = 42
SKIP_BOUNDARY_WIN = True

SNR_GRID   = [-15, -10, -5, 0, 5, 10, 15]
MIXED_SEED = 42 + 9999

CHUNK_SAMPLES = 2000
CLIP_RNG      = (-1.0, 1.0)
Q99_PCT       = 0.99

# Excluded (subj, trial) pairs confirmed artifact-contaminated during QC
EXCLUDE = {(11,4),(8,9),(9,9),(5,10),(3,1),(7,5),(11,5),(12,4),(9,6)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Denoiser factory
# ─────────────────────────────────────────────────────────────────────────────

# ── Neural model map ──────────────────────────────────────────────────────────
_NEURAL_MODEL_MAP = {
    "fcn":      ("FCN",            "FCN_best.pth"),
    "msemg":    ("MSEMG",          "MSEMG_best.pth"),
    "sdemg":    ("SDEMG",          "SDEMG_best.pth"),
    "trustemg": ("TrustEMGNet_RM", "TrustEMGNet_RM_best.pth"),
}

_CLASSICAL_METHODS = {"hp", "emd", "vmd", "ceemdan"}
_ALL_METHODS = list(_NEURAL_MODEL_MAP.keys()) + sorted(_CLASSICAL_METHODS)


def _load_neural_denoiser(denoiser_key: str, weights_dir: Path):
    """Load a neural denoiser from the CleanSEMG denoising package."""
    from denoising import BASELINE_MODEL_REGISTRY

    model_name, ckpt_file = _NEURAL_MODEL_MAP[denoiser_key]
    ckpt_path = weights_dir / model_name / ckpt_file

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first with: bash scripts/run_train.sh --model {model_name}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BASELINE_MODEL_REGISTRY[model_name]().to(device)
    state  = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Support raw state_dict or checkpoint dict
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]; break

    model.load_state_dict(state)
    model.eval()
    print(f"[Denoiser] {model_name} loaded on {device}: {ckpt_path}")
    return model, device


def build_denoise_fn(denoiser_key: str, weights_dir: Path,
                     tradition_params_path: Path | None):
    """
    Returns a function: sig (np.ndarray float32) → denoised (np.ndarray float32).

    Neural denoisers: chunk-wise (2000 samples, Q99 normalization).
    Classical methods: tradition_filters.apply_method() per full segment.
    """
    if denoiser_key in _NEURAL_MODEL_MAP:
        model, device = _load_neural_denoiser(denoiser_key, weights_dir)

        @torch.no_grad()
        def _neural_denoise(sig: np.ndarray) -> np.ndarray:
            sig = sig.astype(np.float32)
            n   = len(sig)
            out = np.zeros(n, np.float32)
            for s in range(0, n, CHUNK_SAMPLES):
                e  = min(s + CHUNK_SAMPLES, n); al = e - s
                ch = sig[s:e].copy()
                if al < CHUNK_SAMPLES:
                    ch = np.concatenate([ch, np.zeros(CHUNK_SAMPLES - al, np.float32)])
                scale = float(np.quantile(np.abs(ch), Q99_PCT))
                if scale < 1e-9: scale = 1.0
                norm = np.clip(ch / scale, *CLIP_RNG).astype(np.float32)
                t    = torch.from_numpy(norm).float().to(device).unsqueeze(0)
                pred = model(t)
                out[s:e] = (pred.squeeze(0).cpu().numpy() * scale)[:al]
            return out

        return _neural_denoise

    # Classical method
    assert denoiser_key in _CLASSICAL_METHODS
    if tradition_params_path is None or not tradition_params_path.exists():
        print(f"[WARN] tradition-params not found — using built-in defaults for {denoiser_key.upper()}")
        params = {}
    else:
        with open(tradition_params_path) as f:
            params = json.load(f)
        print(f"[Denoiser] {denoiser_key.upper()} loaded params from {tradition_params_path}")

    from denoising.classical_filters import apply_method as _trad_apply

    def _classical_denoise(sig: np.ndarray) -> np.ndarray:
        sig = np.asarray(sig, dtype=np.float64)
        result, ok = _trad_apply(denoiser_key, sig, params, fs=FS, noise_type="")
        return (result if ok else sig).astype(np.float32)

    return _classical_denoise


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise mixer (deterministic, reproducible across experiments)
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicNoiseMixer:
    """
    Mixes clean sEMG with a reproducible combination of noise types and SNR.
    Seed is determined by (base_seed, sample_idx, snr, k) — so the same
    trial always gets the same noise across all denoiser experiments.
    """

    def __init__(self, noise_root: Path, seed: int = 42):
        from glob import glob
        self.base_seed   = seed
        self.noise_paths = {}
        self.noise_cache = {}

        for ntype in ["PLI", "ECG", "MOA", "WGN", "Color"]:
            ndir = noise_root / ntype
            if ndir.is_dir():
                paths = sorted(ndir.glob("*.npy"))
                if paths:
                    self.noise_paths[ntype] = [str(p) for p in paths]

        if not self.noise_paths:
            raise ValueError(f"No noise files found in {noise_root}")

        print(f"[NoiseMixer] {noise_root}")
        for nt, ps in self.noise_paths.items():
            print(f"  {nt}: {len(ps)} files")

        for paths in self.noise_paths.values():
            for p in paths:
                self.noise_cache[p] = np.load(p).astype(np.float64)
        print(f"[NoiseMixer] {len(self.noise_cache)} files cached")

    def _sample_seg(self, noise: np.ndarray, length: int,
                    rng: np.random.Generator) -> np.ndarray:
        if len(noise) < length:
            noise = np.tile(noise, (length // len(noise)) + 2)
        ms = len(noise) - length
        s  = int(rng.integers(0, ms + 1)) if ms > 0 else 0
        return noise[s:s + length].copy()

    def mix(self, clean: np.ndarray, snr: float,
            k: int, sample_idx: int) -> tuple[np.ndarray, list[str]]:
        seed = abs((self.base_seed * 1_000_000
                    + (sample_idx % 100_000) * 100
                    + int(snr + 20) * 10 + k)) % (2**31 - 1)
        rng    = np.random.default_rng(seed)
        clean  = np.asarray(clean, dtype=np.float64).flatten()
        length = len(clean)

        available = list(self.noise_paths.keys())
        if snr < -5 and "WGN" in available:
            available = [t for t in available if t != "WGN"]
        k = min(k, len(available))
        selected = [available[i] for i in
                    sorted(rng.choice(len(available), size=k, replace=False))]

        clean_pow = np.dot(clean, clean)
        if clean_pow < 1e-12:
            return clean.copy(), selected

        target_each = clean_pow / (10.0 ** (snr / 10.0)) / k
        combined = np.zeros(length)
        for ntype in selected:
            fi   = int(rng.integers(0, len(self.noise_paths[ntype])))
            seg  = self._sample_seg(
                self.noise_cache[self.noise_paths[ntype][fi]], length, rng)
            npow = np.dot(seg, seg)
            combined += (np.sqrt(target_each / npow) if npow > 1e-12 else 0.0) * seg

        return (clean + combined), selected


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction (EMG time/frequency domain)
# ─────────────────────────────────────────────────────────────────────────────

def _spectrum(x: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    n    = len(x)
    nfft = 1 << (n - 1).bit_length()
    yf   = fft(x - x.mean(), nfft)
    half = nfft // 2
    freq  = fftfreq(nfft, 1.0 / fs)[:half]
    power = (np.abs(yf[:half]) ** 2) / nfft
    return freq, power


def extract_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    8-dim feature vector per channel:
        RMS, MAV, Waveform Length, ZCR, MNF, MDF, Total Power, Mean Power.

    Reference: Phinyomark et al., IntechOpen 2012.
    """
    N   = len(window)
    rms = float(np.sqrt(np.mean(window ** 2)))
    mav = float(np.mean(np.abs(window)))
    wl  = float(np.sum(np.abs(np.diff(window))))
    zc  = float(np.sum((window[:-1] * window[1:]) < 0) / N)

    freq, power = _spectrum(window, fs)
    psum = power.sum()
    if psum < 1e-30:
        mnf = mdf = tp = mnp = 0.0
    else:
        mnf = float(np.sum(freq * power) / psum)
        cum = np.cumsum(power)
        mdf = float(freq[min(np.searchsorted(cum, psum / 2.0), len(freq) - 1)])
        tp  = float(psum)
        mnp = float(np.mean(power))

    return np.array([rms, mav, wl, zc, mnf, mdf, tp, mnp], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data loading and segmentation
# ─────────────────────────────────────────────────────────────────────────────

def align_labels(emg_time: np.ndarray, lbl_time: np.ndarray,
                 lbl_vals: np.ndarray) -> np.ndarray:
    """Nearest-neighbour label alignment (time-based)."""
    idx  = np.searchsorted(lbl_time, emg_time).clip(0, len(lbl_time) - 1)
    prev = np.maximum(idx - 1, 0)
    closer = np.abs(lbl_time[prev] - emg_time) < np.abs(lbl_time[idx] - emg_time)
    idx[closer] = prev[closer]
    return lbl_vals[idx].astype(np.int8)


def segment_trial(chs: list[np.ndarray],
                  la: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a trial into fixed windows and extract features.

    Returns
    -------
    X : (N_windows, n_channels × 8) float64
    y : (N_windows,) int
    """
    win  = int(WINDOW_SEC * FS)
    step = int(STEP_SEC * FS)
    X, y = [], []

    for s in range(0, len(chs[0]) - win + 1, step):
        e  = s + win
        lc = int(la[s + win // 2])
        ls = int(la[s])
        le = int(la[min(e - 1, len(la) - 1)])
        if SKIP_BOUNDARY_WIN and ls != le:
            continue

        ch_feats, valid = [], True
        for ch in chs:
            w = ch[s:e]
            if not np.all(np.isfinite(w)):
                valid = False; break
            f = extract_features(w)
            if not np.all(np.isfinite(f)):
                valid = False; break
            ch_feats.append(f)
        if not valid:
            continue

        X.append(np.concatenate(ch_feats))
        y.append(lc)

    if not X:
        return np.empty((0, len(chs) * 8)), np.empty(0, int)
    return np.vstack(X), np.array(y, int)


def add_delta(X: np.ndarray) -> np.ndarray:
    """Append first-difference features (doubles feature dimension)."""
    if len(X) == 0:
        return X
    return np.hstack([X, np.diff(X, axis=0, prepend=X[0:1])])


def load_all_data(cache_dir: Path, label_dir: Path,
                  mixer: DeterministicNoiseMixer,
                  denoise_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all subjects/trials and return feature matrices.

    Returns
    -------
    X_bl, X_ny, X_dn : (N, feat_dim) float64   baseline / noisy / denoised
    y                 : (N,) int                fatigue label
    """
    rng = np.random.default_rng(MIXED_SEED)
    X_bl_all, X_ny_all, X_dn_all, y_all = [], [], [], []

    subj_iter = range(1, N_SUBJECTS + 1)
    if HAS_TQDM:
        subj_iter = tqdm(subj_iter, desc="Loading subjects")

    for subj in subj_iter:
        bl_subj   = cache_dir / "baseline" / f"subject_{subj}"
        subj_ldir = label_dir / f"subject_{subj}"
        if not bl_subj.is_dir():
            continue

        trial_nums = sorted(set(
            int(re.findall(r"trial_(\d+)_ch", f)[0])
            for f in os.listdir(bl_subj)
            if re.match(r"trial_\d+_ch0\.npy", f)))

        for trial in trial_nums:
            if (subj, trial) in EXCLUDE:
                continue

            t_path = bl_subj / f"trial_{trial}_time.npy"
            if not t_path.exists():
                continue
            emg_time = np.load(t_path).astype(np.float64)

            # Find matching label CSV
            cands = sorted(
                [f for f in os.listdir(subj_ldir)
                 if f.endswith(".csv")
                 and re.search(rf"(?<!\d){trial}(?!\d)", f)],
                key=lambda x: int(re.findall(r"\d+", x)[-1]))
            if not cands:
                continue

            ldf = pd.read_csv(subj_ldir / cands[0], header=0)
            lt  = ldf.iloc[:, 0].values.astype(np.float64)
            lv  = ldf.iloc[:, 1].values.astype(np.float64)
            si  = np.argsort(lt); lt, lv = lt[si], lv[si]
            la  = align_labels(emg_time, lt, lv)

            # Load 4-channel baseline cache
            chs_bl, ok = [], True
            for ci in range(4):
                p = bl_subj / f"trial_{trial}_ch{ci}.npy"
                if not p.exists(): ok = False; break
                chs_bl.append(np.load(p).astype(np.float64))
            if not ok:
                continue

            sl = len(chs_bl[0])
            if len(la) != sl:
                la = (la[:sl] if len(la) > sl
                      else np.concatenate([la, np.full(sl - len(la), la[-1])]))

            # Online noise mixing (same seed per trial for all denoiser experiments)
            trial_snr  = float(rng.choice(SNR_GRID))
            trial_k    = int(rng.integers(1, 6))
            if trial_snr < -5:
                trial_k = min(trial_k, 4)
            trial_seed = subj * 100_000 + trial

            chs_ny, chs_dn = [], []
            for ci, ch_bl in enumerate(chs_bl):
                noisy, _ = mixer.mix(ch_bl, snr=trial_snr,
                                     k=trial_k, sample_idx=trial_seed + ci)
                denoised  = denoise_fn(noisy.astype(np.float32))
                chs_ny.append(noisy)
                chs_dn.append(denoised.astype(np.float64))

            X_bl_t, y_t = segment_trial(chs_bl, la)
            X_ny_t, _   = segment_trial(chs_ny, la)
            X_dn_t, _   = segment_trial(chs_dn, la)
            if len(y_t) == 0:
                continue

            # Per-trial baseline-relative normalization + delta
            mask0     = (y_t == 0)
            base_feat = (X_bl_t[mask0].mean(0) if mask0.sum() > 0
                         else X_bl_t[:max(1, len(y_t) // 5)].mean(0))
            safe = np.where(np.abs(base_feat) < 1e-12, 1.0, base_feat)

            X_bl_all.append(add_delta(X_bl_t / safe))
            X_ny_all.append(add_delta(X_ny_t / safe))
            X_dn_all.append(add_delta(X_dn_t / safe))
            y_all.append(y_t)

    X_bl = np.vstack(X_bl_all)
    X_ny = np.vstack(X_ny_all)
    X_dn = np.vstack(X_dn_all)
    y    = np.concatenate(y_all)

    print(f"\n[Data] Total windows: {len(y)}  feat_dim: {X_bl.shape[1]}")
    for c, name in enumerate(["Non-fatigue", "Transition", "Fatigue"]):
        cnt = (y == c).sum()
        print(f"  Class {c} ({name:12s}): {cnt:5d}  ({100*cnt/len(y):.1f}%)")
    return X_bl, X_ny, X_dn, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(X_bl: np.ndarray, X_ny: np.ndarray, X_dn: np.ndarray,
           y: np.ndarray, denoiser_name: str) -> tuple[dict, dict, dict]:
    """3-fold stratified CV.  Train SVM on baseline; test on all three conditions."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    results = {k: [] for k in ("acc_bl","acc_ny","acc_dn","f1_bl","f1_ny","f1_dn")}
    cm      = {k: np.zeros((3, 3), int) for k in ("bl","ny","dn")}

    print(f"\n{'='*60}")
    print(f"  {N_FOLDS}-Fold CV | Denoiser: {denoiser_name.upper()} | Classifier: RBF-SVM")
    print(f"{'='*60}")

    folds = list(enumerate(skf.split(X_bl, y), 1))
    if HAS_TQDM:
        folds = tqdm(folds, desc="CV folds", total=N_FOLDS)

    for fold, (tr_idx, te_idx) in folds:
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_bl[tr_idx])
        clf = SVC(kernel="rbf", C=10, gamma="scale",
                  class_weight="balanced",
                  decision_function_shape="ovr",
                  random_state=RANDOM_STATE)
        clf.fit(X_tr_s, y[tr_idx])

        for tag, X_te in [("bl", X_bl[te_idx]),
                           ("ny", X_ny[te_idx]),
                           ("dn", X_dn[te_idx])]:
            yp  = clf.predict(scaler.transform(X_te))
            yt  = y[te_idx]
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
# 6. Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, cm: dict, stats: dict,
                 denoiser_name: str, noise_types: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    bl, ny, dn = stats["acc_bl"][0], stats["acc_ny"][0], stats["acc_dn"][0]

    lines = [
        "=" * 70,
        f"CleanSEMG — Fatigue Downstream  |  Denoiser: {denoiser_name.upper()}"
        f"  |  Classifier: RBF-SVM",
        "Dataset: Cerqueira et al. (2024) Sensors 24(24): 8081",
        "         https://zenodo.org/records/13860256  (CC BY 4.0)", "=" * 70, "",
        "── Experiment ──",
        "  Train RBF-SVM on clean baseline features.",
        "  Test frozen scaler + SVM on baseline / noisy / denoised.", "",
        "── Classifier ──",
        "  RBF-SVM  C=10  gamma=scale  class_weight=balanced",
        "  Features: 4-ch × 8 + Δ = 64 dim  [Phinyomark et al. 2012]", "",
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

    rows = [{"fold":         i + 1,
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
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="CleanSEMG fatigue downstream (RBF-SVM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Denoiser choices:
  Classical : {sorted(_CLASSICAL_METHODS)}
  Neural    : {sorted(_NEURAL_MODEL_MAP.keys())}

Example:
  export CLEANSEMG_ROOT=/path/to/CleanSEMG
  python downstream/fatigue/fatigue_svm.py --denoiser trustemg
        """,
    )
    ap.add_argument("--denoiser",   required=True, choices=_ALL_METHODS)
    ap.add_argument("--cache-dir",
                    default=None,
                    help="Path to preprocessed cache (default: outputs/downstream/fatigue/cache)")
    ap.add_argument("--label-dir",
                    default=None,
                    help="Path to self_perceived_fatigue_index (default: DATA_ROOT/Cerqueira/labels)")
    ap.add_argument("--noise-root",
                    default=None,
                    help="Path to sEMG noise test pool (default: outputs/noise/sEMG_noise_test)")
    ap.add_argument("--weights-dir",
                    default=None,
                    help="Path to neural model weights root (default: outputs/weights_baseline)")
    ap.add_argument("--tradition-params",
                    default=None,
                    help="Path to tradition_params.json (for classical methods)")
    ap.add_argument("--output-dir",
                    default=None,
                    help="Output directory (default: outputs/downstream/fatigue/results/svm_{denoiser})")
    args = ap.parse_args()

    root     = CLEANSEMG_ROOT
    data_root = Path(os.environ.get("DATA_ROOT", root))

    cache_dir  = Path(args.cache_dir)  if args.cache_dir        else root / "outputs/downstream/fatigue/cache"
    label_dir  = Path(args.label_dir)  if args.label_dir        else data_root / "Cerqueira/self_perceived_fatigue_index"
    noise_root = Path(args.noise_root) if args.noise_root       else root / "outputs/noise/sEMG_noise_test"
    weights_dir= Path(args.weights_dir) if args.weights_dir     else root / "outputs/weights_baseline"
    trad_params= Path(args.tradition_params) if args.tradition_params else root / "outputs/weights_tradition/tradition_params.json"
    output_dir = Path(args.output_dir) if args.output_dir       else root / f"outputs/downstream/fatigue/results/svm_{args.denoiser}"

    np.random.seed(RANDOM_STATE)
    print(f"\n[INFO] CleanSEMG Fatigue Downstream (RBF-SVM)")
    print(f"  Denoiser    : {args.denoiser.upper()}")
    print(f"  Cache dir   : {cache_dir}")
    print(f"  Noise root  : {noise_root}")
    print(f"  Output dir  : {output_dir}\n")

    denoise_fn  = build_denoise_fn(args.denoiser, weights_dir, trad_params)
    mixer       = DeterministicNoiseMixer(noise_root, seed=42)
    noise_types = list(mixer.noise_paths.keys())

    X_bl, X_ny, X_dn, y = load_all_data(cache_dir, label_dir, mixer, denoise_fn)
    results, cm, stats   = run_cv(X_bl, X_ny, X_dn, y, args.denoiser)
    save_results(results, cm, stats, args.denoiser, noise_types, output_dir)


if __name__ == "__main__":
    main()