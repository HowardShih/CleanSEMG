#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CleanSEMG — Cerqueira Dataset Preprocessing
downstream/fatigue/preprocess_and_denoise.py

Preprocesses the Cerqueira sEMG fatigue dataset:
  1. Bandpass filter 20–450 Hz (4th-order Butterworth @ 1259 Hz)
  2. Resample 1259 Hz → 1000 Hz (polyphase)
  3. Save as baseline cache (float32 .npy, 1 kHz)

The fatigue_svm.py and fatigue_cnn.py scripts then apply online noise
mixing + denoising at experiment time — no separate denoised cache needed.

Dataset:
    Cerqueira et al. (2024) "Muscular Fatigue Dataset."
    Sensors 24(24): 8081. https://doi.org/10.3390/s24248081
    Download: https://zenodo.org/records/13860256  (CC BY 4.0)

Expected data structure:
    <cerqueira-data>/
        sEMG_data/
            subject_1/  trial_1.csv  trial_2.csv  ...
            subject_2/  ...
        self_perceived_fatigue_index/
            subject_1/  trial_1.csv  ...

Usage:
    export CLEANSEMG_ROOT=/path/to/CleanSEMG

    python downstream/fatigue/preprocess_and_denoise.py \\
        --cerqueira-data /path/to/Cerqueira \\
        --output-cache   outputs/downstream/fatigue/cache

Cache structure (output):
    <output-cache>/
        baseline/
            subject_1/
                trial_1_ch0.npy    # float32, 1 kHz
                trial_1_ch1.npy
                trial_1_ch2.npy
                trial_1_ch3.npy
                trial_1_time.npy   # float32, time axis in seconds
"""

import os
import re
import sys
import argparse
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import resample_poly

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Constants ─────────────────────────────────────────────────────────────────
FS_RAW    = 1259    # Cerqueira recording sampling rate
FS_TARGET = 1000    # target sampling rate (matches CleanSEMG main pipeline)
BANDPASS  = (20, 450)   # Hz — bandpass range

# EMG channel column indices in the Cerqueira CSV files (0-indexed)
# Columns 1, 3, 5, 7 correspond to the 4 EMG channels
ALL_EMG_IDX = [1, 3, 5, 7]

N_SUBJECTS = 13

# Confirmed artifact-contaminated (subject, trial) pairs
EXCLUDE = {(11,4),(8,9),(9,9),(5,10),(3,1),(7,5),(11,5),(12,4),(9,6)}

# Polyphase resample ratio
_g   = gcd(FS_TARGET, FS_RAW)
_UP  = FS_TARGET // _g
_DN  = FS_RAW    // _g


# ─────────────────────────────────────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(emg: np.ndarray) -> np.ndarray:
    """4th-order Butterworth bandpass at raw sampling rate."""
    b, a = sp_signal.butter(4, BANDPASS, btype="bandpass", fs=FS_RAW)
    return sp_signal.filtfilt(b, a, emg)


def resample_to_1k(sig: np.ndarray) -> np.ndarray:
    """Polyphase resample from FS_RAW to FS_TARGET."""
    return resample_poly(sig, _UP, _DN).astype(np.float32)


def make_time_axis(n_raw: int) -> np.ndarray:
    """Build a time axis (seconds) in 1 kHz space for a trial of n_raw raw samples."""
    n_new = int(round(n_raw * FS_TARGET / FS_RAW))
    return np.linspace(0.0, n_raw / FS_RAW, n_new, endpoint=False).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing loop
# ─────────────────────────────────────────────────────────────────────────────

def run(cerqueira_data: Path, output_cache: Path, force: bool = False):
    emg_dir   = cerqueira_data / "sEMG_data"
    bl_cache  = output_cache / "baseline"

    if not emg_dir.is_dir():
        raise FileNotFoundError(
            f"sEMG_data directory not found: {emg_dir}\n"
            f"Download from https://zenodo.org/records/13860256"
        )

    bl_cache.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CleanSEMG — Cerqueira Preprocessing")
    print(f"  Bandpass  : {BANDPASS[0]}–{BANDPASS[1]} Hz (@ {FS_RAW} Hz)")
    print(f"  Resample  : {FS_RAW} Hz → {FS_TARGET} Hz")
    print(f"  Source    : {emg_dir}")
    print(f"  Cache     : {bl_cache}")
    print("=" * 60)

    skipped, processed = [], []

    subj_iter = range(1, N_SUBJECTS + 1)
    if HAS_TQDM:
        subj_iter = tqdm(subj_iter, desc="Subjects")

    for subj in subj_iter:
        emg_subj_dir  = emg_dir / f"subject_{subj}"
        cache_subj_dir = bl_cache / f"subject_{subj}"
        if not emg_subj_dir.is_dir():
            continue

        cache_subj_dir.mkdir(exist_ok=True)

        emg_files = sorted(
            emg_subj_dir.glob("*.csv"),
            key=lambda p: int(re.findall(r"\d+", p.name)[-1])
        )

        for emg_file in emg_files:
            trial = int(re.findall(r"\d+", emg_file.name)[-1])

            if (subj, trial) in EXCLUDE:
                skipped.append((subj, trial))
                continue

            # Check if all outputs already exist
            outputs_exist = all(
                (cache_subj_dir / f"trial_{trial}_ch{ci}.npy").exists()
                for ci in range(4)
            ) and (cache_subj_dir / f"trial_{trial}_time.npy").exists()

            if outputs_exist and not force:
                processed.append((subj, trial))
                continue

            try:
                df = pd.read_csv(emg_file, header=0)
            except Exception as e:
                print(f"\n[ERROR] S{subj} T{trial}: {e}")
                continue

            n_raw = len(df)

            # Process and cache each of the 4 channels
            for ci in range(4):
                out_path = cache_subj_dir / f"trial_{trial}_ch{ci}.npy"
                if out_path.exists() and not force:
                    continue

                raw = df.iloc[:, ALL_EMG_IDX[ci]].values.astype(np.float64)
                filt = bandpass_filter(raw)
                filt = np.where(np.isfinite(filt), filt, 0.0)
                sig_1k = resample_to_1k(filt)
                np.save(str(out_path), sig_1k)

            # Save time axis
            t_path = cache_subj_dir / f"trial_{trial}_time.npy"
            if not t_path.exists() or force:
                np.save(str(t_path), make_time_axis(n_raw))

            processed.append((subj, trial))

    print(f"\n[Done]")
    print(f"  Processed : {len(processed)} trials")
    print(f"  Excluded  : {len(skipped)} trials  {sorted(skipped)}")
    print(f"  Cache     : {bl_cache}")

    # Sanity check: count files
    npy_files = list(bl_cache.rglob("*.npy"))
    print(f"  Total .npy files in cache: {len(npy_files)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Preprocess Cerqueira fatigue dataset → 1 kHz cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Download the dataset from: https://zenodo.org/records/13860256

Example:
  python downstream/fatigue/preprocess_and_denoise.py \\
      --cerqueira-data /path/to/Cerqueira \\
      --output-cache   outputs/downstream/fatigue/cache
        """,
    )
    ap.add_argument(
        "--cerqueira-data", required=True,
        help="Root directory of the downloaded Cerqueira dataset "
             "(must contain sEMG_data/ and self_perceived_fatigue_index/)")
    ap.add_argument(
        "--output-cache", default=None,
        help="Output cache directory "
             "(default: outputs/downstream/fatigue/cache)")
    ap.add_argument(
        "--force", action="store_true",
        help="Re-process even if cache files already exist")
    args = ap.parse_args()

    _THIS_DIR      = Path(__file__).resolve().parent
    cleansemg_root = Path(os.environ.get("CLEANSEMG_ROOT", _THIS_DIR.parent.parent))

    cerqueira_data = Path(args.cerqueira_data)
    output_cache   = (Path(args.output_cache) if args.output_cache
                      else cleansemg_root / "outputs/downstream/fatigue/cache")

    run(cerqueira_data, output_cache, force=args.force)


if __name__ == "__main__":
    main()