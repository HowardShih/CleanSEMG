#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#${CLEANSEMG_ROOT}/step4_preproc_and_segment.py
"""
Step 4 (v6.2.8-FINAL): Preproc + Segment from qc_index
CRITICAL CHANGES:
- Rename scale_factor → clean_scale_factor (clarity)
- Still saves RAW segments (no normalization)
- Training will compute noisy-scale dynamically
"""

import os
import re
import csv
import json
import glob
import argparse
from fractions import Fraction
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly


# =============================================================================
# Normalization Functions (只用於計算 scale factor)
# =============================================================================
def compute_scale_factor(x: np.ndarray, method: str = "Q99", percentile: float = 0.99) -> float:
    """
    計算 normalization scale factor
    
    Args:
        x: 輸入信號 [N]
        method: 正規化方法 (Q99, Q95, RMS, MAD, STD)
        percentile: 當 method 為 Q99/Q95 時使用的百分位數
    
    Returns:
        scale: float, 用於正規化的縮放因子
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 1.0
    
    method = str(method).upper()
    
    if method in ["Q99", "Q95", "Q"]:
        q = float(percentile) if method == "Q" else (0.99 if method == "Q99" else 0.95)
        scale = float(np.quantile(np.abs(x), q))
    
    elif method == "RMS":
        scale = float(np.sqrt(np.mean(x ** 2)))
    
    elif method == "MAD":
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        scale = float(mad * 1.4826)
    
    elif method == "STD":
        scale = float(np.std(x))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if scale < 1e-12:
        scale = 1.0
    
    return scale


# =============================================================================
# Config / Paths
# =============================================================================
def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {config_path}")
    return cfg


def _get_nested(cfg: Dict, keys: List[str], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_output_base(config: Dict) -> str:
    root = _get_nested(config, ["paths", "root"], default=".")
    base = _get_nested(config, ["paths", "output", "base"], default="outputs")
    return base if os.path.isabs(base) else os.path.join(root, base)


def get_db_out_root(config: Dict, db_name: str) -> str:
    out = get_output_base(config)
    sub = _get_nested(config, ["paths", "output", "preprocessed"], default="preprocessed")
    return os.path.join(out, sub, db_name)


def get_splits_root(config: Dict) -> str:
    out = get_output_base(config)
    sub = _get_nested(config, ["paths", "output", "splits"], default="splits")
    return os.path.join(out, sub)


def get_segments_root(config: Dict) -> str:
    out = get_output_base(config)
    sub = _get_nested(config, ["paths", "output", "segments"], default="segments")
    return os.path.join(out, sub)


def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# DSP helpers
# =============================================================================
def apply_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 20.0,
    high: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    min_len = 3 * order + 1
    if signal.shape[0] < min_len:
        return signal

    nyq = fs / 2.0
    actual_high = min(high, nyq * 0.99)
    if actual_high <= low:
        return signal

    low_n = low / nyq
    high_n = actual_high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return filtfilt(b, a, signal, axis=0)


def resample_signal_poly_1d(x: np.ndarray, from_fs: int, to_fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    if from_fs == to_fs:
        return x
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float64)


def load_emg_mat(mat_path: str, db_name: str, fs: int) -> Dict:
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    
    def _pick_key(m, candidates):
        for k in candidates:
            if k in m: return k
        return None

    k_emg = _pick_key(m, ["emg", "EMG", "emg_signals"])
    k_rest = _pick_key(m, ["regrasp", "restimulus", "restim", "restStimulus"])

    if k_emg is None:
        raise KeyError(f"Missing EMG key in: {mat_path}")

    emg = np.asarray(m[k_emg])
    if emg.shape[0] < emg.shape[1]: emg = emg.T
    
    restim = np.asarray(m[k_rest]).reshape(-1) if k_rest else None
    
    return {
        "emg": emg,
        "restimulus": restim,
        "N": emg.shape[0],
        "C": emg.shape[1],
        "fs": fs,
    }


def parse_subject_exercise_from_path(p: str) -> Tuple[int, int]:
    base = os.path.basename(p)
    m = re.search(r"[Ss](\d+)[^0-9A-Za-z]+[Ee](\d+)", base)
    if m: return int(m.group(1)), int(m.group(2))
    return -1, -1


# =============================================================================
# Step 4 - v6.2.8 (Semantic Clarity)
# =============================================================================
def step4_preproc_and_segment(config: Dict, force: bool = False):
    print(f"\n{'='*70}\nStep 4 (v6.2.8-FINAL): Preproc + Segment\n{'='*70}")
    print("✅ Saving RAW segments (no normalization)")
    print("✅ Computing clean_scale_factor (reference only)")
    print("⚠️  Training will compute noisy-scale dynamically")

    split_json = os.path.join(get_splits_root(config), "subject_split_crossDB.json")
    if not os.path.exists(split_json):
        print(f"[ERROR] Split file not found: {split_json}")
        return

    with open(split_json, "r") as f:
        split_data = json.load(f)

    split_map = {sid: "train" for sid in split_data.get("train", [])}
    split_map.update({sid: "val" for sid in split_data.get("valid", [])})
    split_map.update({sid: "test" for sid in split_data.get("test", [])})

    seg_cfg = _get_nested(config, ["preprocessing", "segmentation"], default={}) or {}
    seg_len_s = float(seg_cfg.get("length_s", 2.0))
    target_fs = int(seg_cfg.get("target_fs", 1000))
    overlap = float(seg_cfg.get("overlap", 0.0))
    
    # Bandpass config
    bp_cfg = _get_nested(config, ["preprocessing", "bandpass"], default={}) or {}
    bp_enabled = bool(bp_cfg.get("enabled", True))
    bp_low = float(bp_cfg.get("low_cutoff", 20.0))
    bp_high = float(bp_cfg.get("high_cutoff", 500.0))
    bp_order = int(bp_cfg.get("order", 4))
    
    # Normalization config (只用於計算 clean-scale reference)
    norm_cfg = _get_nested(config, ["normalization"], default={}) or {}
    norm_enabled = bool(norm_cfg.get("enabled", True))
    norm_method = str(norm_cfg.get("method", "Q99"))
    norm_pct = float(norm_cfg.get("percentile", 0.99))
    save_scale_files = bool(norm_cfg.get("save_scale", True))

    print(f"\n[Bandpass] {bp_low}-{bp_high} Hz (order={bp_order})")
    print(f"[Clean Scale] method={norm_method}, percentile={norm_pct}")
    print(f"  Note: This is for reference only. Training uses noisy-scale.")

    pts = int(round(seg_len_s * target_fs))
    step_pts = max(1, int(round(pts * (1.0 - overlap))))

    seg_root = get_segments_root(config)
    manifest_csv = os.path.join(seg_root, "manifests", "segment_manifest.csv")
    
    if os.path.exists(manifest_csv) and not force:
        print(f"[SKIP] manifest exists: {manifest_csv}")
        return

    ensure_dirs(
        os.path.join(seg_root, "train", "raw"), os.path.join(seg_root, "train", "scale"),
        os.path.join(seg_root, "val", "raw"), os.path.join(seg_root, "val", "scale"),
        os.path.join(seg_root, "test", "raw"), os.path.join(seg_root, "test", "scale"),
        os.path.join(seg_root, "manifests")
    )

    # ===== 重要：CSV header 改名 =====
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "split", "dataset", "raw_path", 
            "clean_scale_path", "clean_scale_factor"  # ← 明確標示來自 clean
        ])

    test_db = str(_get_nested(config, ["datasets", "test_db"], default="DB2"))
    train_valid_dbs = list(_get_nested(config, ["datasets", "train_valid_dbs"], default=[]))
    
    stats = Counter()

    for db in [test_db] + train_valid_dbs:
        input_root = _get_nested(config, ["paths", "datasets", db])
        db_out = get_db_out_root(config, db)
        qc_idx_path = os.path.join(db_out, "logs", "qc_index.csv")
        if not os.path.exists(qc_idx_path): continue
        
        dfi = pd.read_csv(qc_idx_path)
        fs_raw = int(_get_nested(config, ["datasets", "sampling_rates", db], default=1000))

        print(f"\n[Processing {db}] FS={fs_raw} Hz")

        file_groups = dfi.groupby("file", sort=False)

        buffer = []
        def _flush():
            nonlocal buffer
            if not buffer: return
            with open(manifest_csv, "a", newline="") as fman:
                wr = csv.writer(fman)
                wr.writerows(buffer)
            buffer = []

        for file_rel, df_file in tqdm(file_groups, desc=f"Step4 {db}"):
            mat_path = os.path.join(input_root, str(file_rel))
            try:
                data = load_emg_mat(mat_path, db, fs_raw)
            except: continue

            emg_raw = data["emg"]
            sid_guess, eid_guess = parse_subject_exercise_from_path(str(file_rel))

            for _, r in df_file.iterrows():
                ch = int(r.get("ch", 0))
                trial_id = int(r.get("trial_id", 0))
                s_trial, e_trial = int(r.get("trial_start_raw", 0)), int(r.get("trial_end_raw", -1))
                cross_sid = str(r.get("cross_subject_id", "UNK"))
                split = split_map.get(cross_sid, "test" if db == test_db else None)
                if not split: continue

                x_raw = emg_raw[s_trial:e_trial+1, ch]
                
                # DSP: Bandpass
                if bp_enabled:
                    x_proc = apply_bandpass_filter(x_raw, fs_raw, low=bp_low, high=bp_high, order=bp_order)
                else:
                    x_proc = x_raw.copy()
                
                # Resample
                if fs_raw != target_fs:
                    x_proc = resample_signal_poly_1d(x_proc, fs_raw, target_fs)
                    trial_len_proc = int(x_proc.size)
                else:
                    trial_len_proc = x_raw.size

                seg_idx = 0
                s0 = 0
                while s0 + pts <= trial_len_proc:
                    e0 = s0 + pts
                    seg_y = x_proc[s0:e0]

                    # 計算 clean-scale (reference only)
                    if norm_enabled:
                        clean_scale = compute_scale_factor(seg_y, method=norm_method, percentile=norm_pct)
                    else:
                        clean_scale = 1.0

                    # 保存原始信號（未 normalize）
                    segment_id = f"{cross_sid}_ch{ch}_t{trial_id}_seg{seg_idx}"
                    raw_path = f"{split}/raw/{segment_id}.npy"
                    clean_scale_path = f"{split}/scale/{segment_id}_clean_scale.npy"

                    np.save(os.path.join(seg_root, raw_path), seg_y.astype(np.float32))
                    
                    if save_scale_files:
                        np.save(os.path.join(seg_root, clean_scale_path), 
                                np.array([clean_scale], dtype=np.float32))
                    else:
                        clean_scale_path = ""

                    buffer.append([
                        segment_id, split, db, raw_path, 
                        clean_scale_path, float(clean_scale)
                    ])
                    
                    if len(buffer) >= 2000:
                        _flush()

                    s0 += step_pts
                    seg_idx += 1
                    stats[split] += 1
        
        _flush()

    print(f"\n✓ Step 4 complete. Total: {sum(stats.values())} segments.")
    print(f"  Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    print(f"\n✅ Segments saved in RAW form")
    print(f"✅ clean_scale_factor saved (reference only)")
    print(f"⚠️  Training will compute noisy-scale dynamically!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    step4_preproc_and_segment(cfg, force=args.force)


if __name__ == "__main__":
    main()