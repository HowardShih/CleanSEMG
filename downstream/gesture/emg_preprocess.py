#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STCNet EMG preprocessing for baseline and denoised downstream evaluation.

Both baseline and denoised modes are aligned to the same downstream format.

Baseline mode:
  1. Bandpass filter at the raw sampling rate.
  2. Resample to the target sampling rate.
  3. Apply per-channel min-max normalization to [0, 1].
  4. Apply max sampling to produce the final STCNet input rate.

Denoised mode:
  1. Load preprocessed denoised EMG at the target sampling rate.
  2. Apply per-channel min-max normalization to [0, 1].
  3. Apply max sampling to produce the final STCNet input rate.

Train/test split by repetition:
  DB2/DB4: Train=(1,3,4,6), Test=(2,5)
  DB1:     Train=(1,3,4,6,8,9,10), Test=(2,5,7)

Usage:
  python emg_preprocess_fixed_v4.py --mode baseline \
      --path /path/to/raw/DB2 --dataset nina2 --output ./pkl_baseline

  python emg_preprocess_fixed_v4.py --mode denoised \
      --path /path/to/denoised_1k/DB2 --dataset nina2 --output ./pkl_denoised
"""

import argparse
import os
import re
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_CFG = {
    "nina1": {
        "num_classes": 52,
        "num_subjects": 27,
        "num_channels": 10,
        "raw_fs": 100,
        "target_fs": 100,
        "out_fs": 100,
        "max_sampling_ratio": 1,
        "test_reps": [2, 5, 7],
        "train_reps": [1, 3, 4, 6, 8, 9, 10],
        "expected_train": 9828,
        "expected_test": 4212,
    },
    "nina2": {
        "num_classes": 49,
        "num_subjects": 40,
        "num_channels": 12,
        "raw_fs": 2000,
        "target_fs": 1000,
        "out_fs": 100,
        "max_sampling_ratio": 10,
        "test_reps": [2, 5],
        "train_reps": [1, 3, 4, 6],
        "expected_train": 7840,
        "expected_test": 3920,
    },
    "nina4": {
        "num_classes": 52,
        "num_subjects": 10,
        "num_channels": 12,
        "raw_fs": 2000,
        "target_fs": 1000,
        "out_fs": 100,
        "max_sampling_ratio": 10,
        "test_reps": [2, 5],
        "train_reps": [1, 3, 4, 6],
        "expected_train": 2080,
        "expected_test": 1040,
    },
}


# =============================================================================
# Signal Processing
# =============================================================================

def apply_bandpass_filter(
    emg: np.ndarray,
    fs: float = 2000.0,
    low: float = 20.0,
    high: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to an [N, C] EMG array."""
    emg = np.asarray(emg, dtype=np.float64)
    N, _C = emg.shape

    if N < 3 * order + 1:
        return emg

    nyq = fs / 2.0
    actual_high = min(high, nyq * 0.99)

    if actual_high <= low:
        return emg

    b, a = butter(order, [low / nyq, actual_high / nyq], btype="band")
    return filtfilt(b, a, emg, axis=0)


def resample_emg(emg: np.ndarray, from_fs: int, to_fs: int) -> np.ndarray:
    """Resample an [N, C] EMG array using polyphase filtering."""
    if from_fs == to_fs:
        return emg.copy()

    emg = np.asarray(emg, dtype=np.float64)
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)

    cols = [
        resample_poly(emg[:, ch], frac.numerator, frac.denominator)
        for ch in range(emg.shape[1])
    ]

    return np.stack(cols, axis=1)


def resample_labels_nn(arr: np.ndarray, from_fs: int, to_fs: int) -> np.ndarray:
    """Resample integer label arrays using nearest-neighbor indexing."""
    if from_fs == to_fs:
        return arr.copy()

    arr = np.asarray(arr).reshape(-1)
    N_in = len(arr)
    N_out = int(round(N_in * to_fs / from_fs))

    indices = np.clip(
        np.round(np.arange(N_out) * (N_in / N_out)).astype(int),
        0,
        N_in - 1,
    )

    return arr[indices]


def normalize_minmax(emg: np.ndarray) -> np.ndarray:
    """Apply per-channel min-max normalization to [0, 1]."""
    emg = np.asarray(emg, dtype=np.float32)

    emg_min = np.min(emg, axis=0, keepdims=True)
    emg_max = np.max(emg, axis=0, keepdims=True)

    denom = emg_max - emg_min
    denom = np.where(denom < 1e-12, 1.0, denom)

    return ((emg - emg_min) / denom).astype(np.float32)


def max_sampling(emg: np.ndarray, ratio: int) -> np.ndarray:
    """
    Downsample by selecting the maximum-absolute-value sample in each window
    while preserving the original sign.
    """
    if ratio == 1:
        return emg

    emg = np.asarray(emg)
    N, C = emg.shape
    N_out = N // ratio

    if N_out <= 0:
        return emg[:1].copy()

    out = np.zeros((N_out, C), dtype=emg.dtype)

    for i in range(N_out):
        s = i * ratio
        e = s + ratio
        win = emg[s:e, :]
        idx = np.argmax(np.abs(win), axis=0)
        out[i, :] = win[idx, np.arange(C)]

    return out


# =============================================================================
# MAT File Loading
# =============================================================================

def load_mat_file(mat_path: str, mode: str = "baseline") -> dict:
    """
    Load one .mat file.

    Parameters
    ----------
    mat_path:
        Path to the input .mat file.
    mode:
        ``baseline`` reads the raw ``emg`` field.
        ``denoised`` prefers ``preprocessed_emg`` when available.

    Returns
    -------
    dict
        Dictionary containing EMG data and label arrays.
    """
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if mode == "denoised" and "preprocessed_emg" in m:
        emg = np.asarray(m["preprocessed_emg"])
    elif "emg" in m:
        emg = np.asarray(m["emg"])
    else:
        raise KeyError(f"No EMG field found in {mat_path}")

    if emg.ndim != 2:
        raise ValueError(f"EMG must be 2D, got {emg.shape}")

    if emg.shape[0] < emg.shape[1]:
        emg = emg.T

    def _get_label(key):
        for k in [key, key.capitalize(), key.lower()]:
            if k in m:
                arr = np.asarray(m[k]).squeeze()

                if arr.ndim == 0:
                    arr = np.array([arr.item()])

                arr = arr.reshape(-1)

                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.int32)

                return arr

        return None

    stimulus = _get_label("stimulus")
    restimulus = _get_label("restimulus")
    repetition = _get_label("repetition")

    N = len(emg)

    for name, arr in [
        ("stimulus", stimulus),
        ("restimulus", restimulus),
        ("repetition", repetition),
    ]:
        if arr is not None and len(arr) != N:
            arr = arr[:N] if len(arr) > N else np.pad(arr, (0, N - len(arr)), mode="edge")

            if name == "stimulus":
                stimulus = arr
            elif name == "restimulus":
                restimulus = arr
            elif name == "repetition":
                repetition = arr

    return {
        "emg": emg,
        "stimulus": stimulus,
        "restimulus": restimulus,
        "repetition": repetition,
    }


# =============================================================================
# Preprocessing Pipelines
# =============================================================================

def preprocess_nina1(emg: np.ndarray) -> tuple:
    """Preprocess DB1 signals, which are already sampled at the output rate."""
    normalized = normalize_minmax(emg)
    return normalized, normalized


def preprocess_baseline_1k(
    emg: np.ndarray,
    raw_fs: int,
    target_fs: int,
    ratio: int,
) -> tuple:
    """
    Preprocess raw baseline EMG.

    Steps
    -----
    1. Bandpass filter at the raw sampling rate.
    2. Resample to the target sampling rate.
    3. Normalize to [0, 1].
    4. Apply max sampling to the output sampling rate.

    Returns
    -------
    tuple
        ``normalized`` at target sampling rate and ``sampled`` at output rate.
    """
    filtered = apply_bandpass_filter(
        emg,
        fs=float(raw_fs),
        low=20.0,
        high=500.0,
        order=4,
    )

    emg_1k = resample_emg(filtered, raw_fs, target_fs)
    normalized = normalize_minmax(emg_1k)
    sampled = max_sampling(normalized, ratio=ratio)

    return normalized, sampled


def preprocess_denoised_1k(emg_1k: np.ndarray, ratio: int) -> tuple:
    """
    Preprocess denoised EMG that is already aligned to the target sampling rate.

    Returns
    -------
    tuple
        ``normalized`` at target sampling rate and ``sampled`` at output rate.
    """
    normalized = normalize_minmax(emg_1k)
    sampled = max_sampling(normalized, ratio=ratio)

    return normalized, sampled


# =============================================================================
# Dataset Processing
# =============================================================================

def process_dataset(dir_path: str, dataset: str, mode: str = "baseline") -> tuple:
    """
    Process a full dataset and return train/test dataframes.

    Parameters
    ----------
    dir_path:
        Root directory containing .mat files.
    dataset:
        Dataset identifier: ``nina1``, ``nina2``, or ``nina4``.
    mode:
        Processing mode: ``baseline`` or ``denoised``.

    Returns
    -------
    tuple
        ``train_df`` and ``test_df``.
    """
    cfg = DATASET_CFG[dataset]

    train_reps = cfg["train_reps"]
    test_reps = cfg["test_reps"]
    raw_fs = cfg["raw_fs"]
    target_fs = cfg["target_fs"]
    ratio = cfg["max_sampling_ratio"]

    print(f"\n{'=' * 60}")
    print(f"Processing {dataset.upper()} — Mode: {mode.upper()}")
    print(f"{'=' * 60}")

    if dataset == "nina1":
        print("  DB1: already at 100 Hz, ratio=1")
    elif mode == "baseline":
        print(
            f"  raw_fs={raw_fs} Hz -> bandpass -> resample {target_fs} Hz "
            f"-> normalize -> max_sampling {ratio}:1 -> {cfg['out_fs']} Hz"
        )
    else:
        print(
            f"  denoised at {target_fs} Hz -> normalize "
            f"-> max_sampling {ratio}:1 -> {cfg['out_fs']} Hz"
        )

    print(f"  Train reps: {train_reps}")
    print(f"  Test reps:  {test_reps}")

    mat_files = sorted(
        [
            os.path.join(root, f)
            for root, _, files in os.walk(dir_path)
            for f in files
            if f.lower().endswith(".mat")
        ]
    )

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {dir_path}")

    print(f"  Found {len(mat_files)} .mat files")

    def _ftype(path):
        b = os.path.basename(path)

        if "E3" in b or "e3" in b:
            return "E3"

        if "E2" in b or "e2" in b:
            return "E2"

        return "E1"

    file_types = [_ftype(p) for p in mat_files]

    if dataset == "nina1":
        preprocess_fn = lambda emg, _fs: preprocess_nina1(emg)
    elif mode == "baseline":
        preprocess_fn = lambda emg, fs: preprocess_baseline_1k(
            emg,
            raw_fs=fs,
            target_fs=target_fs,
            ratio=ratio,
        )
    else:
        preprocess_fn = lambda emg, _fs: preprocess_denoised_1k(
            emg,
            ratio=ratio,
        )

    def _parse_subject_id(mat_path: str) -> int:
        """Parse subject ID from filenames such as S1_E1_A1.mat."""
        b = os.path.basename(mat_path)
        m = re.search(r"[Ss](\d+)", b)
        return int(m.group(1)) if m else -1

    def _process_split(reps_keep, split_name):
        samples = []
        now, start = 100, 0

        for _j, (mat_path, ftype) in enumerate(
            tqdm(zip(mat_files, file_types), total=len(mat_files), desc=split_name)
        ):
            try:
                data = load_mat_file(mat_path, mode=mode)
            except Exception as e:
                print(f"\n  [WARN] {os.path.basename(mat_path)}: {e}")
                continue

            emg = data["emg"]
            stim = data["stimulus"]
            rep = data["repetition"]

            if stim is None or rep is None:
                continue

            fs_for_fn = raw_fs if mode == "baseline" else target_fs
            normalized, sampled = preprocess_fn(emg, fs_for_fn)

            if mode == "baseline" and dataset != "nina1":
                stim = resample_labels_nn(stim, raw_fs, target_fs)
                rep = resample_labels_nn(rep, raw_fs, target_fs)

            L = min(len(normalized), len(stim), len(rep))

            normalized = normalized[:L]
            stim = stim[:L]
            rep = rep[:L]

            for i in range(L - 1):
                if stim[i] != now:
                    now = stim[i]
                    start = i

                is_target_rep = int(rep[i]) in reps_keep
                is_trial_end = stim[i + 1] != now and now != 0
                is_last = i == L - 2 and now != 0

                if (is_trial_end or is_last) and is_target_rep:
                    trial_norm = normalized[start:i + 1]

                    if dataset == "nina1":
                        trial_sampled = trial_norm
                    else:
                        s_start = start // ratio
                        s_end = min((i + 1) // ratio, len(sampled))
                        trial_sampled = (
                            sampled[s_start:s_end]
                            if s_end > s_start
                            else sampled[s_start:s_start + 1]
                        )

                    stim_adj = int(now)

                    if dataset == "nina1":
                        if ftype == "E3":
                            stim_adj += 29
                        elif ftype == "E2":
                            stim_adj += 12

                    samples.append(
                        {
                            "stimulus": stim_adj - 1,
                            "subject": _parse_subject_id(mat_path),
                            "repetition": int(rep[i]),
                            "normalized": trial_norm,
                            "sampled_normalized": trial_sampled,
                        }
                    )

        return pd.DataFrame(samples)

    train_df = _process_split(train_reps, "Train")
    test_df = _process_split(test_reps, "Test")

    print("\n[Summary]")
    print(f"  Train: {len(train_df)} (expected {cfg['expected_train']})")
    print(f"  Test:  {len(test_df)} (expected {cfg['expected_test']})")

    if len(train_df) > 0:
        print(f"  Classes: {train_df['stimulus'].nunique()}")
        print(f"  Class range: [{train_df['stimulus'].min()}, {train_df['stimulus'].max()}]")

        s = train_df.iloc[0]
        print(f"  Sample shape (sampled): {s['sampled_normalized'].shape}")
        print(
            f"  Value range: "
            f"[{s['sampled_normalized'].min():.4f}, {s['sampled_normalized'].max():.4f}]"
        )

    return train_df, test_df


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="STCNet EMG preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python emg_preprocess_fixed_v4.py --mode baseline \
      --path /path/to/raw/DB2 --dataset nina2 --output ./pkl_baseline

  python emg_preprocess_fixed_v4.py --mode denoised \
      --path /path/to/denoised_1k/DB2 --dataset nina2 --output ./pkl_denoised
        """,
    )

    parser.add_argument(
        "--path",
        required=True,
        help="Input folder with .mat files",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nina1", "nina2", "nina4"],
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "denoised"],
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for PKL files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = args.output or os.path.join(os.path.dirname(__file__), "pkl")
    os.makedirs(out_dir, exist_ok=True)

    train_df, test_df = process_dataset(args.path, args.dataset, args.mode)

    train_pkl = os.path.join(out_dir, f"train_{args.dataset}.pkl")
    test_pkl = os.path.join(out_dir, f"test_{args.dataset}.pkl")

    train_df.to_pickle(train_pkl)
    test_df.to_pickle(test_pkl)

    print("\n[Saved]")
    print(f"  {train_pkl}  ({len(train_df)} samples)")
    print(f"  {test_pkl}   ({len(test_df)} samples)")

    print(f"\n{'=' * 60}")
    print(f"Mode:    {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()