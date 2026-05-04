#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ${CLEANSEMG_ROOT}/downstream_tasks/stcnet/prepare_denoised_mat_traditional.py
prepare_denoised_mat_traditional.py

Traditional denoising (HP / EMD / VMD / CEEMDAN) for STCNet downstream task.
Same canvas architecture as prepare_denoised_mat_baseline_model.py:

  1. Load raw .mat → bandpass + resample to 1000Hz canvas
  2. For each seg_lookup-matched 2s segment:
       noisy_physical = noisy_norm * scale
       denoised_physical = apply_<method>(noisy_physical)
  3. Write noisy / denoised canvas to output .mat

Output MAT files are compatible with emg_preprocess_fixed_v4.py (--mode denoised).

Usage:
  python prepare_denoised_mat_traditional.py \\
      --method       hp \\
      --params-json  /path/to/tradition_params.json \\
      --db2-root     /path/to/DB2 \\
      --test-npz     /path/to/test_combined.npz \\
      --qc-index     /path/to/qc_index.csv \\
      --output-noisy    ./outputs/noisy_data_1k_trad/DB2 \\
      --output-denoised ./outputs/denoised_data_1k_hp/DB2 \\
      [--device cpu]  # ignored for trad methods, kept for CLI compatibility
"""

import os
import sys
import math
import json
import argparse
import time
from glob import glob
from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

# ── path setup ───────────────────────────────────────────────────────────────
SEMG_ROOT = "${CLEANSEMG_ROOT}"
for _p in [SEMG_ROOT,
           os.path.join(SEMG_ROOT, "baseline_models"),
           os.path.dirname(os.path.abspath(__file__))]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from tradition_filters import (
        apply_hp_filter,
        apply_emd_filter,
        apply_vmd_filter,
        apply_ceemdan_filter,
    )
except ImportError as e:
    raise ImportError(
        f"Cannot import tradition_filters: {e}\n"
        f"Ensure tradition_filters.py is in {SEMG_ROOT} or the script directory."
    ) from e

TRADITIONAL_METHODS = ["hp", "emd", "vmd", "ceemdan"]

# ── Default params (used when params_json is absent or method not in JSON) ───
DEFAULT_PARAMS = {
    "hp": {
        "best_cutoff_hz": 40.0,   # Wang et al. JBHI 2025
        "order": 4,
    },
    "emd": {
        "best_f_min_hz": 20.0,
        "f_max_hz": 500.0,
        "max_imfs": 8,
        "fallback": "noisy",
    },
    "vmd": {
        "best_K": 6,
        "alpha": 2000.0,
        "tau": 0.0,
        "tol": 1e-7,
        "f_min_hz": 20.0,
        "f_max_hz": 500.0,
        "fallback": "noisy",
    },
    "ceemdan": {
        "best_f_min_hz": 20.0,
        "f_max_hz": 500.0,
        "trials": 20,
        "epsilon": 0.005,
        "fallback": "noisy",
    },
}


# ============================================================================
# DSP  (identical to prepare_denoised_mat_baseline_model.py)
# ============================================================================

def apply_bandpass_filter(signal, fs, low=20.0, high=500.0, order=4):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[0] < 3 * order + 1:
        return signal
    nyq = fs / 2.0
    actual_high = min(high, nyq * 0.99)
    if actual_high <= low:
        return signal
    b, a = butter(order, [low / nyq, actual_high / nyq], btype="band")
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return filtfilt(b, a, signal, axis=0)


def resample_emg_2d(emg, from_fs, to_fs):
    if from_fs == to_fs:
        return emg.copy()
    emg = np.asarray(emg, dtype=np.float64)
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)
    return np.stack(
        [resample_poly(emg[:, ch], frac.numerator, frac.denominator)
         for ch in range(emg.shape[1])],
        axis=1,
    )


def resample_labels(labels, from_fs, to_fs):
    if labels is None:
        return None
    if from_fs == to_fs:
        return labels.copy()
    labels = np.asarray(labels).reshape(-1)
    N_in = len(labels)
    if N_in == 0:
        return labels
    N_out = int(round(N_in * to_fs / from_fs))
    if N_out <= 0:
        return np.zeros((0,), dtype=labels.dtype)
    idx = np.clip(
        np.round(np.arange(N_out) * (N_in / N_out)).astype(int), 0, N_in - 1
    )
    return labels[idx]


def compute_trial_len_1k(n_raw, from_fs, to_fs):
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)
    return math.ceil(n_raw * frac.numerator / frac.denominator)


# ============================================================================
# MAT I/O  (identical to prepare_denoised_mat_baseline_model.py)
# ============================================================================

def load_emg_mat(mat_path):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def _pick(candidates):
        for k in candidates:
            if k in m:
                return k
        return None

    k_emg  = _pick(["emg", "EMG"])
    k_sti  = _pick(["stimulus", "Stimulus"])
    k_rest = _pick(["restimulus", "restStimulus"])
    k_rep  = _pick(["repetition", "Repetition"])

    if k_emg is None:
        raise KeyError(f"No EMG field in {mat_path}")

    emg = np.asarray(m[k_emg])
    if emg.ndim != 2:
        raise ValueError(f"EMG must be 2D, got {emg.shape}")
    if emg.shape[0] < emg.shape[1]:
        emg = emg.T
    N, C = emg.shape

    def _proc(key):
        if key is None:
            return None
        arr = np.asarray(m[key]).squeeze().reshape(-1)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.int32)
        return arr

    def _align(arr, tlen):
        if arr is None:
            return None
        L = len(arr)
        if L == tlen:
            return arr
        if L < tlen:
            return np.pad(arr, (0, tlen - L), mode="edge")
        return arr[:tlen]

    return {
        "emg": emg, "N": N, "C": C,
        "stimulus":   _align(_proc(k_sti),  N),
        "restimulus": _align(_proc(k_rest), N),
        "repetition": _align(_proc(k_rep),  N),
        "raw_mat": m,
    }


def save_mat(output_path, raw_mat, emg_1k, stim_1k, rest_1k, rep_1k):
    save_dict = {k: v for k, v in raw_mat.items() if not k.startswith("__")}
    save_dict["preprocessed_emg"] = emg_1k.astype(np.float64)
    save_dict["emg"]              = emg_1k.astype(np.float64)
    if stim_1k is not None:
        save_dict["stimulus"] = stim_1k.astype(np.int32)
    if rest_1k is not None:
        save_dict["restimulus"] = rest_1k.astype(np.int32)
    if rep_1k is not None:
        save_dict["repetition"] = rep_1k.astype(np.int32)
    save_dict["fs_preprocessed"] = np.array([1000], dtype=np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sio.savemat(output_path, save_dict, do_compression=False)


# ============================================================================
# QC Index + Segment Lookup  (identical to prepare_denoised_mat_baseline_model.py)
# ============================================================================

def load_qc_index(qc_index_path):
    df = pd.read_csv(qc_index_path)
    required = {"file", "trial_id", "ch", "trial_start_raw", "trial_end_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"qc_index missing columns: {sorted(list(missing))}")

    qc_map = {}
    for _, row in df.iterrows():
        rel = str(row["file"])
        trial_id = int(row["trial_id"])
        ch = int(row["ch"])
        qc_map.setdefault(rel, set()).add((trial_id, ch))

    print(f"[QC Index] {len(qc_map)} files, "
          f"{sum(len(v) for v in qc_map.values())} QC-passed (trial,ch)")
    return qc_map, df


def load_segment_lookup(npz_path, qc_df):
    print(f"\n[Segment Lookup] Building from {npz_path}")
    t0 = time.time()

    npz = np.load(npz_path, allow_pickle=True)
    if "data" not in npz:
        raise KeyError(f"NPZ has no 'data' key: {npz_path}")
    data = npz["data"]
    print(f"  NPZ entries: {len(data)}")

    subj_to_stem = {}
    if "subject_id" not in qc_df.columns:
        raise ValueError("qc_index must contain 'subject_id' column")
    for _, row in qc_df.iterrows():
        subj_id = int(row["subject_id"])
        file_rel = str(row["file"])
        mat_stem = os.path.splitext(os.path.basename(file_rel))[0]
        subj_to_stem.setdefault(subj_id, mat_stem)

    lookup = {}
    n_parsed = n_no_stem = n_parse_fail = 0

    for i in range(len(data)):
        item = data[i]
        segment_id = str(item.get("segment_id", ""))
        parts = segment_id.split("_")
        try:
            subj_id  = int(parts[1])
            ch       = int(parts[2].replace("ch", ""))
            trial_id = int(parts[3].replace("t", ""))
            seg_idx  = int(parts[4].replace("seg", ""))
        except Exception:
            n_parse_fail += 1
            continue

        mat_stem = subj_to_stem.get(subj_id)
        if mat_stem is None:
            n_no_stem += 1
            continue

        key = (mat_stem, ch, trial_id, seg_idx)
        lookup[key] = {
            "noisy": np.asarray(item["noisy"], dtype=np.float32),
            "clean": np.asarray(item["clean"], dtype=np.float32),
            "scale": float(item["scale"]),
            "snr":   float(item.get("snr", np.nan)),
            "k":     int(item.get("k", -1)),
            "segment_id": segment_id,
        }
        n_parsed += 1

    elapsed = time.time() - t0
    print(f"  Parsed OK: {n_parsed}")
    if n_parse_fail:
        print(f"  [WARN] Parse failed: {n_parse_fail}")
    if n_no_stem:
        print(f"  [WARN] No mat_stem for subject_id: {n_no_stem}")
    print(f"  Lookup keys: {len(lookup)} [{elapsed:.1f}s]")
    return lookup


def build_trial_info_from_qc(qc_df, file_rel):
    df_file = qc_df[qc_df["file"] == file_rel]
    trials = {}
    for _, row in df_file.iterrows():
        tid = int(row["trial_id"])
        ch  = int(row["ch"])
        if tid not in trials:
            trials[tid] = {
                "trial_start_raw": int(row["trial_start_raw"]),
                "trial_end_raw":   int(row["trial_end_raw"]),
                "fs_raw": int(row["fs_raw"]) if "fs_raw" in qc_df.columns else 2000,
                "channels": set(),
            }
        trials[tid]["channels"].add(ch)
    return trials


# ============================================================================
# Traditional denoiser dispatch
# ============================================================================

def denoise_segment(method: str, noisy_physical: np.ndarray,
                    params: dict, fs: int = 1000) -> np.ndarray:
    """
    Apply traditional denoising to a single 1D physical-unit segment.

    Args:
        method:         "hp" | "emd" | "vmd" | "ceemdan"
        noisy_physical: [L] float64, physical unit (mV or normalised * scale)
        params:         method-specific params dict
        fs:             sampling rate (1000 Hz)

    Returns:
        denoised [L] float64 (falls back to noisy_physical on failure)
    """
    x = np.asarray(noisy_physical, dtype=np.float64).reshape(-1)

    if method == "hp":
        p = params.get("hp", DEFAULT_PARAMS["hp"])
        try:
            return apply_hp_filter(
                x, fs=fs,
                cutoff_hz=p.get("best_cutoff_hz", 40.0),
                order=int(p.get("order", 4)),
            )
        except Exception:
            return x.copy()

    elif method == "emd":
        p = params.get("emd", DEFAULT_PARAMS["emd"])
        enh, ok = apply_emd_filter(
            x, fs=fs,
            f_min=p.get("best_f_min_hz", p.get("f_min_hz", 20.0)),
            f_max=p.get("f_max_hz", 500.0),
            max_imfs=int(p.get("max_imfs", 8)),
            noise_type="",   # unknown — uses WGN noise-index strategy
        )
        return enh if ok else x.copy()

    elif method == "vmd":
        p = params.get("vmd", DEFAULT_PARAMS["vmd"])
        enh, ok = apply_vmd_filter(
            x, fs=fs,
            K=int(p.get("best_K", p.get("K", 6))),
            alpha=float(p.get("alpha", 2000.0)),
            tau=float(p.get("tau", 0.0)),
            f_min=float(p.get("f_min_hz", 20.0)),
            f_max=float(p.get("f_max_hz", 500.0)),
            tol=float(p.get("tol", 1e-7)),
            noise_type="",
        )
        return enh if ok else x.copy()

    elif method == "ceemdan":
        p = params.get("ceemdan", DEFAULT_PARAMS["ceemdan"])
        enh, ok = apply_ceemdan_filter(
            x, fs=fs,
            trials=int(p.get("trials", 20)),
            f_min=p.get("best_f_min_hz", p.get("f_min_hz", 20.0)),
            f_max=p.get("f_max_hz", 500.0),
            max_imfs=int(p.get("max_imfs", 8)),
            noise_type="",
        )
        return enh if ok else x.copy()

    else:
        raise ValueError(f"Unknown traditional method: {method!r}. "
                         f"Choose from: {TRADITIONAL_METHODS}")


# ============================================================================
# Main processing
# ============================================================================

def process_db2(
    method,
    params,
    db2_root,
    test_npz_path,
    qc_index_path,
    output_dir_noisy,
    output_dir_denoised,
    max_files=None,
    skip_existing=True,
    target_fs=1000,
    seg_len_s=2.0,
):
    print(f"\n{'='*70}")
    print(f"STCNet Denoised MAT — Traditional Method: {method.upper()}")
    print(f"{'='*70}")
    print(f"DB2 Root:         {db2_root}")
    print(f"Output Noisy:     {output_dir_noisy}")
    print(f"Output Denoised:  {output_dir_denoised}")
    if method in params:
        print(f"Params ({method}): {params[method]}")
    else:
        print(f"Params ({method}): [defaults]")
    print(f"{'='*70}\n")

    qc_map, qc_df = load_qc_index(qc_index_path)
    seg_lookup = load_segment_lookup(test_npz_path, qc_df)

    mat_files = sorted(glob(os.path.join(db2_root, "**/*.mat"), recursive=True))
    if max_files is not None:
        mat_files = mat_files[:max_files]
    print(f"\nFound {len(mat_files)} .mat files")

    pts = int(round(seg_len_s * target_fs))   # 2000 samples
    fs_raw = 2000

    stats = {
        "processed": 0, "skipped": 0, "errors": 0,
        "segs_placed": 0, "segs_missing": 0,
        "filter_fallbacks": 0,
    }

    for mat_path in tqdm(mat_files, desc=f"Processing [{method.upper()}]"):
        rel = os.path.relpath(mat_path, db2_root)
        mat_stem = os.path.splitext(os.path.basename(rel))[0]
        out_noisy    = os.path.join(output_dir_noisy,    rel)
        out_denoised = os.path.join(output_dir_denoised, rel)

        if skip_existing and os.path.exists(out_noisy) and os.path.exists(out_denoised):
            stats["skipped"] += 1
            continue

        try:
            raw_data = load_emg_mat(mat_path)
        except Exception as e:
            print(f"\n[ERROR] {rel}: {e}")
            stats["errors"] += 1
            continue

        emg_raw = raw_data["emg"]
        N_raw, C = emg_raw.shape

        # Build baseline 1kHz canvas: bandpass + resample
        emg_bp_whole = apply_bandpass_filter(emg_raw, fs=float(fs_raw))
        emg_1k_whole = resample_emg_2d(emg_bp_whole, fs_raw, target_fs)
        N_1k = emg_1k_whole.shape[0]

        canvas_noisy    = emg_1k_whole.copy()
        canvas_denoised = emg_1k_whole.copy()

        trial_info = build_trial_info_from_qc(qc_df, rel)

        for trial_id, tinfo in trial_info.items():
            s_raw = int(tinfo["trial_start_raw"])
            e_raw = int(tinfo["trial_end_raw"])
            fs    = int(tinfo["fs_raw"]) if tinfo.get("fs_raw") is not None else fs_raw

            trial_start_1k = int(round(s_raw * target_fs / fs))
            n_raw_trial    = e_raw - s_raw + 1
            trial_len_1k   = compute_trial_len_1k(n_raw_trial, fs, target_fs)

            for ch in sorted(list(tinfo["channels"])):
                seg_idx = 0
                s0 = 0
                while s0 + pts <= trial_len_1k:
                    key = (mat_stem, ch, trial_id, seg_idx)
                    canvas_seg_start = trial_start_1k + s0

                    if key in seg_lookup:
                        entry = seg_lookup[key]
                        if canvas_seg_start + pts <= N_1k:
                            noisy_norm    = entry["noisy"].astype(np.float64)
                            scale         = float(entry["scale"])
                            noisy_physical = noisy_norm * scale

                            # Denoise in physical domain
                            denoised_physical = denoise_segment(
                                method, noisy_physical, params, fs=target_fs
                            )

                            seg_end = canvas_seg_start + pts
                            canvas_noisy[canvas_seg_start:seg_end, ch]    = noisy_physical
                            canvas_denoised[canvas_seg_start:seg_end, ch] = denoised_physical
                            stats["segs_placed"] += 1
                    else:
                        stats["segs_missing"] += 1

                    seg_idx += 1
                    s0 += pts

        stim_1k = resample_labels(raw_data["stimulus"],   fs_raw, target_fs)
        rest_1k = resample_labels(raw_data["restimulus"], fs_raw, target_fs)
        rep_1k  = resample_labels(raw_data["repetition"], fs_raw, target_fs)

        save_mat(out_noisy,    raw_data["raw_mat"], canvas_noisy,    stim_1k, rest_1k, rep_1k)
        save_mat(out_denoised, raw_data["raw_mat"], canvas_denoised, stim_1k, rest_1k, rep_1k)
        stats["processed"] += 1

    print(f"\n{'='*70}")
    print(f"DONE — {method.upper()}")
    print(f"{'='*70}")
    print(f"  Processed:        {stats['processed']}")
    print(f"  Skipped:          {stats['skipped']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  Segments placed:  {stats['segs_placed']}")
    print(f"  Segments missing: {stats['segs_missing']}")
    print(f"\n  Noisy:    {output_dir_noisy}")
    print(f"  Denoised: {output_dir_denoised}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare denoised .mat for STCNet — Traditional methods "
                    "(HP / EMD / VMD / CEEMDAN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # HP (high-pass filter)
  python prepare_denoised_mat_traditional.py \
      --method hp \
      --db2-root /path/to/DB2 \
      --test-npz /path/to/test_combined.npz \
      --qc-index /path/to/qc_index.csv \
      --output-noisy    ./outputs/noisy_data_1k_trad/DB2 \
      --output-denoised ./outputs/denoised_data_1k_hp/DB2

  # EMD with calibrated params
  python prepare_denoised_mat_traditional.py \
      --method emd \
      --params-json /path/to/tradition_params.json \
      --db2-root /path/to/DB2 ...
        """,
    )
    parser.add_argument("--method", required=True,
                        choices=TRADITIONAL_METHODS,
                        help="Traditional denoising method")
    parser.add_argument("--params-json", default=None,
                        help="Path to tradition_params.json from train_tradition.py "
                             "(uses built-in defaults if not provided)")
    parser.add_argument("--db2-root",    required=True)
    parser.add_argument("--test-npz",    required=True)
    parser.add_argument("--qc-index",    required=True)
    parser.add_argument("--output-noisy",    required=True)
    parser.add_argument("--output-denoised", required=True)
    parser.add_argument("--max-files",   type=int, default=None)
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--target-fs",   type=int, default=1000)
    parser.add_argument("--seg-len-s",   type=float, default=2.0)
    # Kept for CLI compatibility with the neural-model version; not used.
    parser.add_argument("--device",      default="cpu")

    args = parser.parse_args()

    # Load params (merge defaults with JSON if available)
    params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    if args.params_json and os.path.isfile(args.params_json):
        with open(args.params_json) as f:
            loaded = json.load(f)
        # Only override keys that exist in loaded JSON
        for key in TRADITIONAL_METHODS:
            if key in loaded:
                params[key] = loaded[key]
        print(f"[Params] Loaded from {args.params_json}")
    else:
        if args.params_json:
            print(f"[WARN] params-json not found: {args.params_json!r} — using defaults")
        else:
            print("[Params] No params-json provided — using built-in defaults")

    process_db2(
        method             = args.method,
        params             = params,
        db2_root           = args.db2_root,
        test_npz_path      = args.test_npz,
        qc_index_path      = args.qc_index,
        output_dir_noisy   = args.output_noisy,
        output_dir_denoised= args.output_denoised,
        max_files          = args.max_files,
        skip_existing      = not args.force,
        target_fs          = args.target_fs,
        seg_len_s          = args.seg_len_s,
    )


if __name__ == "__main__":
    main()