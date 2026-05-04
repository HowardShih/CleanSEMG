#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ${CLEANSEMG_ROOT}/step1_qc_raw.py
"""
Step 1 (v6.2): RAW trial parsing + QC metrics extraction
Output:
  - outputs/.../preprocessed/<DB>/logs/trial_manifest.csv
  - outputs/.../preprocessed/<DB>/logs/qc_metrics_raw.csv
Design:
  - QC computed on RAW (no bandpass).
  - Output index/metrics only (no masking, no editing labels).

v6.2-debug: Added verbose per-file logging for DB2 E1/E2/E3 diagnosis.
"""

import os
import glob
import re
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from scipy.io import loadmat
from scipy.signal import welch
from scipy.ndimage import median_filter
from numpy import trapz


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


def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# Identity parsing
# =============================================================================

def parse_subject_exercise_from_path(p: str) -> Tuple[int, int]:
    base = os.path.basename(p)

    m = re.search(r"[Ss](\d+)[^0-9A-Za-z]+[Ee](\d+)", base)
    if m:
        return int(m.group(1)), int(m.group(2))

    ms = re.search(r"[Ss](\d+)", base)
    me = re.search(r"[Ee](\d+)", base)
    if ms and me:
        return int(ms.group(1)), int(me.group(1))

    m2 = re.search(r"[\\/][Ss](\d+)[\\/]", p)
    sid = int(m2.group(1)) if m2 else -1
    m3 = re.search(r"[\\/][Ee](\d+)[\\/]", p)
    eid = int(m3.group(1)) if m3 else -1
    return sid, eid


# =============================================================================
# EMG I/O (RAW load)
# =============================================================================

def _pick_key(m: Dict, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in m:
            return k
    return None


def load_emg_mat(mat_path: str, db_name: str, fs: int, verbose: bool = False) -> Dict:
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    dbu = db_name.upper()

    # --- DEBUG: show all keys in the mat file ---
    all_keys = [k for k in m.keys() if not k.startswith('__')]
    if verbose:
        print(f"    [MAT keys] {os.path.basename(mat_path)}: {all_keys}")

    if dbu == "DB10":
        k_emg = _pick_key(m, ["emg", "EMG", "emg_signals"])
        k_rest = _pick_key(m, ["regrasp", "restimulus", "restim", "restStimulus"])
        k_rep = _pick_key(m, ["repetition", "Repetition"])
        k_sti = _pick_key(m, ["stimulus", "Stimulus"])
    else:
        k_emg = _pick_key(m, ["emg", "EMG", "emg_signals"])
        k_sti = _pick_key(m, ["stimulus", "Stimulus"])
        k_rest = _pick_key(m, ["restimulus", "restStimulus", "restim"])
        k_rep = _pick_key(m, ["repetition", "Repetition"])

    if verbose:
        print(f"    [Key match] emg={k_emg}, restimulus={k_rest}, stimulus={k_sti}, repetition={k_rep}")

    if k_emg is None:
        raise KeyError(f"Missing EMG key in: {mat_path}")

    emg = np.asarray(m[k_emg])
    if emg.ndim != 2:
        raise ValueError(f"EMG must be 2D, got {emg.shape} in {mat_path}")

    # enforce [N, C]
    if emg.shape[0] < emg.shape[1]:
        emg = emg.T
    N, C = emg.shape

    def _proc_label(key: Optional[str]) -> Optional[np.ndarray]:
        if key is None:
            return None
        arr = np.asarray(m[key]).squeeze().reshape(-1)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.int32)
        return arr

    stimulus = _proc_label(k_sti)
    restimulus = _proc_label(k_rest)
    repetition = _proc_label(k_rep)

    if verbose:
        def _label_summary(arr, name):
            if arr is None:
                return f"{name}=None"
            uniq = np.unique(arr)
            return f"{name}: len={len(arr)}, unique={uniq[:10].tolist()}{'...' if len(uniq)>10 else ''}"
        print(f"    [Labels] {_label_summary(restimulus, 'restimulus')}")
        print(f"    [Labels] {_label_summary(stimulus, 'stimulus')}")
        print(f"    [Labels] {_label_summary(repetition, 'repetition')}")
        print(f"    [EMG]    shape={emg.shape} (N={N}, C={C})")

    def _align(arr: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
        if arr is None:
            return None
        L = len(arr)
        if L == target_len:
            return arr
        if L == target_len - 1:
            return np.pad(arr, (0, 1), mode="edge")
        if L == target_len + 1:
            return arr[:target_len]
        raise ValueError(f"Length mismatch: {L} vs {target_len} in {mat_path}")

    stimulus = _align(stimulus, N)
    restimulus = _align(restimulus, N)
    repetition = _align(repetition, N)

    return {
        "emg": emg,
        "stimulus": stimulus,
        "restimulus": restimulus,
        "repetition": repetition,
        "N": N,
        "C": C,
        "fs": fs,
    }


# =============================================================================
# Trial detection
# =============================================================================

@dataclass
class Trial:
    start: int
    end: int
    gesture: int
    repetition: Optional[int]


def detect_trials(restim: Optional[np.ndarray], repetition: Optional[np.ndarray],
                  verbose: bool = False) -> List[Trial]:
    if restim is None:
        if verbose:
            print("    [detect_trials] restimulus is None → 0 trials")
        return []

    restim = np.asarray(restim).reshape(-1)
    active = (restim > 0).astype(np.int32)
    edges = np.diff(active, prepend=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    if active.size > 0 and active[-1] == 1:
        ends = np.append(ends, len(active) - 1)

    trials: List[Trial] = []
    for s, e in zip(starts, ends):
        if e < s:
            continue
        seg = restim[s:e + 1]
        seg_pos = seg[seg > 0]
        gesture = int(np.bincount(seg_pos).argmax()) if seg_pos.size else 0

        rep = None
        if repetition is not None:
            rep_arr = np.asarray(repetition).reshape(-1)
            rep_seg = rep_arr[s:e + 1]
            rep_pos = rep_seg[rep_seg > 0]
            if rep_pos.size:
                rep = int(np.bincount(rep_pos).argmax())

        trials.append(Trial(s, e, gesture, rep))

    if verbose:
        print(f"    [detect_trials] Found {len(trials)} trials "
              f"(active_pct={100.0*active.mean():.1f}%)")

    return trials


# =============================================================================
# QC metrics (RAW)
# =============================================================================

def hampel_filter_1d(x: np.ndarray, half_window: int, threshold: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or half_window <= 0:
        return x

    win = 2 * half_window + 1
    med = median_filter(x, size=win, mode="nearest")
    abs_dev = np.abs(x - med)
    mad = median_filter(abs_dev, size=win, mode="nearest")

    scale = 1.4286
    S = scale * mad

    y = x.copy()
    mask = (S > 1e-12) & (abs_dev > threshold * S)
    y[mask] = med[mask]
    return y


def hampel_energy_removed(x: np.ndarray, threshold: float = 5.0, half_window_bins: int = 40) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 64:
        return np.nan

    X = np.fft.rfft(x)
    re = X.real.copy()
    im = X.imag.copy()
    nb = re.size
    if nb < 5:
        return np.nan

    hw = int(min(half_window_bins, (nb - 1) // 2))
    if hw <= 0:
        return 0.0

    re2 = hampel_filter_1d(re, half_window=hw, threshold=threshold)
    im2 = hampel_filter_1d(im, half_window=hw, threshold=threshold)
    X2 = re2 + 1j * im2

    E0 = float(np.sum(np.abs(X) ** 2))
    E1 = float(np.sum(np.abs(X2) ** 2))
    if E0 <= 1e-12:
        return 0.0
    return float(np.clip((E0 - E1) / E0, 0.0, 1.0))


def hampel_energy_removed_fixedwin(
    x: np.ndarray,
    fs: int,
    win_s: float = 2.0,
    hop_s: float = 1.0,
    threshold: float = 5.0,
    half_window_bins: int = 40,
    agg: str = "max",
) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < int(max(0.5, win_s) * fs):
        return np.nan
    L = int(round(win_s * fs))
    H = int(round(hop_s * fs))
    H = max(1, H)
    ers = []
    for s in range(0, max(1, x.size - L + 1), H):
        seg = x[s:s+L]
        if seg.size < L:
            break
        er = hampel_energy_removed(seg, threshold=threshold, half_window_bins=half_window_bins)
        if np.isfinite(er):
            ers.append(er)
    if not ers:
        return np.nan
    if agg.lower() == "mean":
        return float(np.mean(ers))
    return float(np.max(ers))


def calculate_spectral_metrics(
    xa: np.ndarray,
    fs: int,
    signal_lo: float = 20.0,
    signal_hi_ref: float = 500.0,
) -> Dict[str, float]:
    xa = np.asarray(xa, dtype=np.float64).reshape(-1)
    T = xa.size
    if T < 128:
        return {"SMR_dB": np.nan, "SPR_dB": np.nan, "SHR_dB": np.nan, "OHM": np.nan, "DPR_dB": np.nan}

    nperseg = min(T, 1024)
    f, Pxx = welch(xa, fs=fs, nperseg=nperseg)
    if Pxx.size < 32:
        return {"SMR_dB": np.nan, "SPR_dB": np.nan, "SHR_dB": np.nan, "OHM": np.nan, "DPR_dB": np.nan}

    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    eps = 1e-12

    k_mpd = 13
    if Pxx.size < k_mpd:
        mpd = Pxx
        f_mpd = f
    else:
        mpd = np.convolve(Pxx, np.ones(k_mpd) / k_mpd, mode="valid")
        idxs = np.arange(mpd.size)
        f_mpd = (idxs + (k_mpd / 2.0)) * df

    f_nyq = fs / 2.0
    signal_hi = min(float(signal_hi_ref), f_nyq * 0.99)
    f_upper_start = f_nyq * 0.8

    sig_mask = (f_mpd >= float(signal_lo)) & (f_mpd <= signal_hi)
    noise_mask = (f_mpd >= f_upper_start) & (f_mpd <= f_nyq)
    if not sig_mask.any():
        sig_mask = np.ones_like(mpd, dtype=bool)
    if not noise_mask.any():
        noise_mask = np.ones_like(mpd, dtype=bool)

    mpd_highest_sig = float(np.max(mpd[sig_mask]))
    mpd_lowest_noise = float(np.min(mpd[noise_mask]))
    dpr = 10.0 * np.log10((mpd_highest_sig + eps) / (mpd_lowest_noise + eps))

    M0 = float(trapz(Pxx, f))
    M1 = float(trapz(Pxx * f, f))
    M2 = float(trapz(Pxx * (f ** 2), f))

    mask_0_20 = (f >= 0) & (f <= 20.0)
    P_0_20 = float(trapz(Pxx[mask_0_20], f[mask_0_20])) if mask_0_20.any() else 0.0
    smr = 10.0 * np.log10((M0 + eps) / (P_0_20 + eps))

    mask_upper = (f >= f_upper_start) & (f <= f_nyq)
    P_upper = float(trapz(Pxx[mask_upper], f[mask_upper])) if mask_upper.any() else 0.0
    shr = 10.0 * np.log10((M0 + eps) / (P_upper + eps))

    if df > 0:
        harmonics = np.arange(50, f_nyq + 1e-6, 50)
        idxs = [int(np.argmin(np.abs(f - h))) for h in harmonics]
        idxs = sorted(set(idxs))
        P_harm = float(np.sum(Pxx[idxs]) * df)
    else:
        P_harm = 0.0
    spr = 10.0 * np.log10((M0 + eps) / (P_harm + eps))

    term1 = np.sqrt(M2 / (M0 + eps))
    term2 = M1 / (M0 + eps)
    ohm = term1 / (term2 + eps)

    return {
        "SMR_dB": float(smr),
        "SPR_dB": float(spr),
        "SHR_dB": float(shr),
        "OHM": float(ohm),
        "DPR_dB": float(dpr),
    }


def calculate_snr(active: np.ndarray, rest_rms: float) -> float:
    active = np.asarray(active, dtype=np.float64).reshape(-1)
    active_rms = float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0
    rr = float(rest_rms) if np.isfinite(rest_rms) else 0.0
    return float(10.0 * np.log10((active_rms ** 2) / (rr ** 2 + 1e-12) + 1e-12))


def compute_rest_rms_per_ch(emg: np.ndarray, restim: np.ndarray) -> np.ndarray:
    emg = np.asarray(emg, dtype=np.float64)
    restim = np.asarray(restim).reshape(-1)
    N = emg.shape[0]
    rest_mask = (restim == 0) if (restim == 0).any() else np.ones(N, dtype=bool)
    rest_emg = emg[rest_mask]
    if rest_emg.size == 0:
        return np.sqrt(np.mean(emg ** 2, axis=0) + 1e-12).astype(np.float64)
    return np.sqrt(np.mean(rest_emg ** 2, axis=0) + 1e-12).astype(np.float64)


# =============================================================================
# Step 1
# =============================================================================

def step1_qc_raw(config: Dict, db_name: str, force: bool = False):
    print(f"\n{'='*70}\nStep 1 (v6.2): QC metrics on RAW  [{db_name}]\n{'='*70}")

    # DB2 診斷模式：對前幾個檔案開啟 verbose
    is_db2 = (db_name.upper() == "DB2")
    VERBOSE_N_FILES = 6  # 每個 exercise 至少 2 個 subject → 印出前 6 個

    input_root = _get_nested(config, ["paths", "datasets", db_name])
    if input_root is None:
        print(f"[ERROR] config.paths.datasets.{db_name} not found")
        return

    out_root = get_db_out_root(config, db_name)
    logs_dir = os.path.join(out_root, "logs")
    reports_dir = os.path.join(out_root, "reports")
    ensure_dirs(logs_dir, reports_dir)

    fs = int(_get_nested(config, ["datasets", "sampling_rates", db_name], default=0))
    if fs <= 0:
        print(f"[ERROR] sampling rate not found for {db_name}")
        return

    bp_high_ref = float(_get_nested(config, ["preprocessing", "bandpass", "high_cutoff"], default=500.0))

    hamp_cfg = _get_nested(config, ["quality_control", "hampel"], default={}) or {}
    hamp_thr = float(hamp_cfg.get("threshold", 5.0))
    hamp_hw = int(hamp_cfg.get("half_window_bins", 40))
    hamp_win_s = float(hamp_cfg.get("window_s", 2.0))
    hamp_hop_s = float(hamp_cfg.get("hop_s", 1.0))
    hamp_agg = str(hamp_cfg.get("aggregate", "max"))

    trial_manifest_csv = os.path.join(logs_dir, "trial_manifest.csv")
    qc_raw_csv = os.path.join(logs_dir, "qc_metrics_raw.csv")

    if (os.path.exists(trial_manifest_csv) and os.path.exists(qc_raw_csv)) and not force:
        print(f"[SKIP] Step1 outputs exist (use --force to rerun)")
        return

    mat_files = sorted(glob.glob(os.path.join(input_root, "**/*.mat"), recursive=True))
    if not mat_files:
        print(f"[ERROR] No .mat files found in {input_root}")
        return

    print(f"Input: {input_root}")
    print(f"Found {len(mat_files)} .mat files")
    print(f"FS(raw): {fs} Hz")

    if is_db2:
        # 印出找到的 E1/E2/E3 檔案數量分佈
        e1 = sum(1 for f in mat_files if re.search(r'[Ee]1', os.path.basename(f)))
        e2 = sum(1 for f in mat_files if re.search(r'[Ee]2', os.path.basename(f)))
        e3 = sum(1 for f in mat_files if re.search(r'[Ee]3', os.path.basename(f)))
        print(f"\n[DB2 FILE SCAN] E1={e1}, E2={e2}, E3={e3} files")
        print(f"[DB2 DIAG] Will print verbose info for first {VERBOSE_N_FILES} files\n")

    trial_manifest_header = [
        "dataset", "file", "subject_id", "exercise_id",
        "trial_id", "trial_start_raw", "trial_end_raw", "duration_s",
        "gesture", "repetition",
        "fs_raw", "n_ch"
    ]
    qc_header = [
        "dataset", "file", "subject_id", "exercise_id",
        "trial_id", "trial_start_raw", "trial_end_raw",
        "gesture", "repetition",
        "fs_raw", "ch",
        "SNR_dB", "SMR_dB", "SPR_dB", "SHR_dB", "OHM", "DPR_dB", "HAMP_ER",
        "RMS_active", "RMS_rest"
    ]

    with open(trial_manifest_csv, "w", newline="") as f:
        csv.writer(f).writerow(trial_manifest_header)
    with open(qc_raw_csv, "w", newline="") as f:
        csv.writer(f).writerow(qc_header)

    total_files_processed = 0
    total_files_skipped = 0
    total_trials = 0
    total_trial_ch = 0

    # Skip counters for summary
    skip_reasons = {"load_error": 0, "restim_none": 0, "no_trials": 0}

    for file_idx, mat_path in enumerate(tqdm(mat_files, desc=f"Step1 RAW QC {db_name}")):
        rel = os.path.relpath(mat_path, input_root)
        verbose = is_db2 and (file_idx < VERBOSE_N_FILES)

        if verbose:
            print(f"\n  ── File [{file_idx+1}] {rel} ──")

        try:
            data = load_emg_mat(mat_path, db_name, fs, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"    [SKIP] load_emg_mat error: {e}")
            skip_reasons["load_error"] += 1
            total_files_skipped += 1
            continue

        emg = data["emg"]
        restim = data["restimulus"]
        repetition = data["repetition"]
        C = data["C"]

        if restim is None:
            if verbose:
                print(f"    [SKIP] restimulus is None")
            skip_reasons["restim_none"] += 1
            total_files_skipped += 1
            continue

        trials = detect_trials(restim, repetition, verbose=verbose)
        if not trials:
            if verbose:
                print(f"    [SKIP] detect_trials returned 0 trials")
            skip_reasons["no_trials"] += 1
            total_files_skipped += 1
            continue

        if verbose:
            print(f"    [OK] {len(trials)} trials, C={C}")

        sid, eid = parse_subject_exercise_from_path(rel)
        rest_rms_per_ch = compute_rest_rms_per_ch(emg, restim)

        with open(trial_manifest_csv, "a", newline="") as fman:
            wrm = csv.writer(fman)
            for ti, tr in enumerate(trials):
                s, e = tr.start, tr.end
                if e <= s: continue
                dur_s = (e - s + 1) / fs
                wrm.writerow([
                    db_name, rel, sid, eid, ti, s, e, dur_s,
                    tr.gesture, tr.repetition if tr.repetition is not None else -1, fs, C
                ])

        with open(qc_raw_csv, "a", newline="") as fqc:
            wrq = csv.writer(fqc)
            for ti, tr in enumerate(trials):
                s, e = tr.start, tr.end
                if e <= s: continue
                seg_rest = np.asarray(restim[s:e + 1])
                act_mask = (seg_rest > 0)
                if not act_mask.any(): continue
                seg_emg = np.asarray(emg[s:e + 1])
                active_emg = seg_emg[act_mask]
                if active_emg.shape[0] < 16: continue

                for ch in range(C):
                    x = active_emg[:, ch]
                    snr = calculate_snr(x, rest_rms_per_ch[ch])

                    try:
                        mtr = calculate_spectral_metrics(x, fs, signal_lo=20.0, signal_hi_ref=bp_high_ref)
                    except Exception:
                        mtr = {"SMR_dB": np.nan, "SPR_dB": np.nan, "SHR_dB": np.nan, "OHM": np.nan, "DPR_dB": np.nan}

                    try:
                        hamp = hampel_energy_removed_fixedwin(
                            x, fs=fs,
                            win_s=hamp_win_s, hop_s=hamp_hop_s,
                            threshold=hamp_thr, half_window_bins=hamp_hw,
                            agg=hamp_agg
                        )
                    except Exception:
                        hamp = np.nan

                    rms_act = float(np.sqrt(np.mean(x ** 2) + 1e-12))
                    wrq.writerow([
                        db_name, rel, sid, eid, ti, s, e,
                        tr.gesture, tr.repetition if tr.repetition is not None else -1,
                        fs, ch, snr, mtr["SMR_dB"], mtr["SPR_dB"], mtr["SHR_dB"], mtr["OHM"], mtr["DPR_dB"],
                        hamp, rms_act, float(rest_rms_per_ch[ch]),
                    ])
                    total_trial_ch += 1
                total_trials += 1
        total_files_processed += 1

    print(f"\n{'='*70}")
    print(f"✓ Step 1 complete [{db_name}]")
    print(f"  Processed : {total_files_processed} files")
    print(f"  Skipped   : {total_files_skipped} files")
    print(f"    - load_error  : {skip_reasons['load_error']}")
    print(f"    - restim_none : {skip_reasons['restim_none']}")
    print(f"    - no_trials   : {skip_reasons['no_trials']}")
    print(f"  Total trials    : {total_trials}")
    print(f"  Total trial×ch  : {total_trial_ch}")

    if is_db2:
        # Per-exercise summary from trial_manifest
        try:
            df_man = pd.read_csv(trial_manifest_csv)
            if not df_man.empty:
                print(f"\n[DB2 TRIAL SUMMARY] Per exercise_id:")
                grp = df_man.groupby("exercise_id")["trial_id"].count()
                for eid, cnt in grp.items():
                    print(f"  E{eid}: {cnt} trials across {df_man[df_man['exercise_id']==eid]['file'].nunique()} files")
            else:
                print("[DB2 TRIAL SUMMARY] trial_manifest.csv is EMPTY — all files were skipped!")
        except Exception as e:
            print(f"[DB2 TRIAL SUMMARY] Could not read manifest: {e}")

    print(f"{'='*70}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    force = bool(args.force)

    if args.db:
        dbs = [args.db]
    else:
        test_db = str(_get_nested(cfg, ["datasets", "test_db"], default="DB2"))
        train_valid_dbs = list(_get_nested(cfg, ["datasets", "train_valid_dbs"], default=[]))
        dbs = [test_db] + train_valid_dbs

    for db in dbs:
        step1_qc_raw(cfg, db, force=force)


if __name__ == "__main__":
    main()