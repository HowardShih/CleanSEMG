#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_tradition.py  v2  —  Traditional Baseline Evaluation
                               (HP + TS + EMD + VMD + CEEMDAN)

Metric set synced with inference.py v6.7.0:
  SNRimp, RMSE, PRD, LSD, RMSE_ARV, RMSE_ZCR, RMSE_MNF, RMSE_MDF, RMSE_Kurtosis

Usage:
    python3 inference_tradition.py \\
        --config        config.yaml \\
        --trad-config   tradition_train_config.yaml \\
        --params        outputs/weights_tradition/tradition_params.json \\
        --test-data     outputs/test_data/test_combined.npz \\
        --output        outputs/inference_tradition/baseline \\
        --methods       hp,ts,emd,vmd,ceemdan   # or "all"
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from scipy import signal

# Shared filter implementations
sys.path.insert(0, os.path.dirname(__file__))
from tradition_filters import apply_method, ALL_METHODS   # noqa: E402 – local import


# ============================================================================
# Config helpers
# ============================================================================
def load_yaml(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get(cfg: Dict, keys: List[str], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ============================================================================
# ── Shared metric functions (synced with inference.py v6.7.0) ───────────────
# ============================================================================

def cal_SNR(clean: np.ndarray, test: np.ndarray) -> float:
    clean = clean.reshape(-1).astype(np.float64)
    test  = test.reshape(-1).astype(np.float64)
    n_pw  = np.dot(test - clean, test - clean)
    s_pw  = np.dot(clean, clean)
    return 100.0 if n_pw < 1e-12 else float(10.0 * np.log10(s_pw / (n_pw + 1e-12)))


def cal_SNRimp(clean, denoised, noisy) -> float:
    return float(cal_SNR(clean, denoised) - cal_SNR(clean, noisy))


def cal_RMSE(clean, enhanced) -> float:
    c = clean.reshape(-1).astype(np.float64)
    e = enhanced.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.mean((e - c) ** 2)))


def cal_PRD(clean, enhanced) -> float:
    c = clean.reshape(-1).astype(np.float64)
    e = enhanced.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.sum((e - c) ** 2) / (np.sum(c ** 2) + 1e-12)) * 100)


def cal_ARV(emg: np.ndarray, window_size: int = 200) -> np.ndarray:
    emg = np.abs(emg.reshape(-1).astype(np.float64))
    return np.array([emg[i:i + window_size].mean()
                     for i in range(0, emg.shape[0], window_size)
                     if len(emg[i:i + window_size]) > 0], dtype=np.float64)


def cal_RMSE_ARV(clean, enhanced, window_size: int = 200) -> float:
    a, b = cal_ARV(clean, window_size), cal_ARV(enhanced, window_size)
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2))) if n > 0 else 0.0


def cal_LSD(clean: np.ndarray, enhanced: np.ndarray,
            sr: int = 1000, n_fft: int = 512, hop: int = 128,
            f_min: float = 20.0, f_max: float = 500.0,
            eps: float = 1e-10) -> float:
    clean    = clean.reshape(-1).astype(np.float64)
    enhanced = enhanced.reshape(-1).astype(np.float64)
    if len(clean) < 3 * n_fft or len(enhanced) < 3 * n_fft:
        return np.nan
    win = np.hanning(n_fft)
    freq, _, S_c = signal.stft(clean,    fs=sr, window=win,
                                nperseg=n_fft, noverlap=n_fft - hop)
    _,    _, S_e = signal.stft(enhanced, fs=sr, window=win,
                                nperseg=n_fft, noverlap=n_fft - hop)
    mask = (freq >= f_min) & (freq <= f_max)
    if not mask.any():
        mask = np.ones(len(freq), dtype=bool)
    log_c = 10.0 * np.log10(np.abs(S_c[mask, :]) ** 2 + eps)
    log_e = 10.0 * np.log10(np.abs(S_e[mask, :]) ** 2 + eps)
    return float(np.mean(np.sqrt(np.mean((log_c - log_e) ** 2, axis=0))))


def _zcr_per_window(x: np.ndarray,
                    window_size: int = 200, sr: int = 1000) -> np.ndarray:
    x = x.reshape(-1).astype(np.float64)
    return np.array(
        [float(np.sum(np.diff(np.sign(x[i:i + window_size])) != 0))
         / (window_size / sr)
         for i in range(0, len(x) - window_size + 1, window_size)],
        dtype=np.float64)


def cal_RMSE_ZCR(clean, enhanced, window_size: int = 200, sr: int = 1000) -> float:
    z_c = _zcr_per_window(clean,    window_size, sr)
    z_e = _zcr_per_window(enhanced, window_size, sr)
    n = min(len(z_c), len(z_e))
    return float(np.sqrt(np.mean((z_c[:n] - z_e[:n]) ** 2))) if n > 0 else 0.0


def _mnf_per_window(emg: np.ndarray, sr: int = 1000,
                     f_min: float = 20.0, f_max: float = 500.0) -> np.ndarray:
    emg = emg.reshape(-1).astype(np.float64)
    if len(emg) < 200:
        return np.array([100.0], dtype=np.float64)
    freq, _, spec = signal.stft(emg, fs=sr, window='boxcar',
                                nperseg=200, noverlap=0, nfft=1024,
                                boundary='constant')
    spec = np.abs(spec)
    rec_win = signal.get_window('boxcar', 200)
    spec = spec / np.sqrt(1.0 / rec_win.sum() ** 2)
    si = max(0, min(np.searchsorted(freq, f_min), len(freq) - 1))
    ei = max(si + 1, min(np.searchsorted(freq, f_max), len(freq)))
    freq_r, spec_r = freq[si:ei], spec[si:ei, :]
    wf = np.sum(freq_r[:, np.newaxis] * spec_r, axis=0)
    sp = np.sum(spec_r, axis=0)
    valid = sp > 1e-12
    MNF = np.zeros_like(sp, dtype=np.float64)
    if np.any(valid):
        MNF[valid]  = wf[valid] / sp[valid]
        MNF[~valid] = np.median(MNF[valid])
    else:
        MNF[:] = 100.0
    return MNF.astype(np.float64)


def cal_RMSE_MNF(clean, enhanced, sr: int = 1000,
                  f_min: float = 20.0, f_max: float = 500.0) -> float:
    try:
        mc = _mnf_per_window(clean,    sr, f_min, f_max)
        me = _mnf_per_window(enhanced, sr, f_min, f_max)
        n = min(len(mc), len(me))
        return float(np.sqrt(np.mean((mc[:n] - me[:n]) ** 2))) if n > 0 else 0.0
    except Exception as e:
        print(f"[WARN] RMSE_MNF failed: {e}")
        return 0.0


def _mdf_per_window(x: np.ndarray, window_size: int = 200,
                     sr: int = 1000, f_min: float = 20.0,
                     f_max: float = 500.0, nfft: int = 1024) -> np.ndarray:
    x = x.reshape(-1).astype(np.float64)
    freq = np.fft.rfftfreq(nfft, 1.0 / sr)
    mask = (freq >= f_min) & (freq <= f_max)
    freq_r = freq[mask]
    mdf = []
    for i in range(0, len(x) - window_size + 1, window_size):
        seg = x[i:i + window_size]
        pad = np.zeros(nfft); pad[:len(seg)] = seg
        psd_r = np.abs(np.fft.rfft(pad * np.hanning(nfft))) ** 2
        psd_r = psd_r[mask]
        total = psd_r.sum()
        if total < 1e-12:
            mdf.append(np.nan); continue
        cum = np.cumsum(psd_r)
        idx = min(int(np.searchsorted(cum, total / 2.0)), len(freq_r) - 1)
        mdf.append(float(freq_r[idx]))
    arr = np.array(mdf, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.any():
        arr[~valid] = np.nanmedian(arr[valid])
    else:
        arr[:] = (f_min + f_max) / 2.0
    return arr


def cal_RMSE_MDF(clean, enhanced, window_size: int = 200, sr: int = 1000,
                  f_min: float = 20.0, f_max: float = 500.0) -> float:
    try:
        mc = _mdf_per_window(clean,    window_size, sr, f_min, f_max)
        me = _mdf_per_window(enhanced, window_size, sr, f_min, f_max)
        n = min(len(mc), len(me))
        return float(np.sqrt(np.mean((mc[:n] - me[:n]) ** 2))) if n > 0 else 0.0
    except Exception as e:
        print(f"[WARN] RMSE_MDF failed: {e}")
        return 0.0


def _kurtosis_per_window(x: np.ndarray, window_size: int = 200) -> np.ndarray:
    x = x.reshape(-1).astype(np.float64)
    out = []
    for i in range(0, len(x) - window_size + 1, window_size):
        seg = x[i:i + window_size]
        std = seg.std()
        out.append(float(((seg - seg.mean()) ** 4).mean() / std ** 4 - 3.0)
                   if std > 1e-12 else 0.0)
    return np.array(out, dtype=np.float64)


def cal_RMSE_Kurtosis(clean, enhanced, window_size: int = 200) -> float:
    kc = _kurtosis_per_window(clean,    window_size)
    ke = _kurtosis_per_window(enhanced, window_size)
    n = min(len(kc), len(ke))
    return float(np.sqrt(np.mean((kc[:n] - ke[:n]) ** 2))) if n > 0 else 0.0


DEFAULT_METRICS = [
    "SNRimp", "RMSE", "PRD", "LSD",
    "RMSE_ARV", "RMSE_ZCR",
    "RMSE_MNF", "RMSE_MDF",
    "RMSE_Kurtosis",
]


def calculate_all_metrics(
    clean, denoised, noisy,
    sr: int = 1000, arv_window: int = 200,
    f_min: float = 20.0, f_max: float = 500.0,
    lsd_n_fft: int = 512, lsd_hop: int = 128,
) -> dict:
    c = np.asarray(clean,    dtype=np.float64).reshape(-1)
    d = np.asarray(denoised, dtype=np.float64).reshape(-1)
    n = np.asarray(noisy,    dtype=np.float64).reshape(-1)

    fns = [
        ("SNRimp",        lambda: cal_SNRimp(c, d, n)),
        ("RMSE",          lambda: cal_RMSE(c, d)),
        ("PRD",           lambda: cal_PRD(c, d)),
        ("LSD",           lambda: cal_LSD(c, d, sr=sr, n_fft=lsd_n_fft,
                                          hop=lsd_hop, f_min=f_min, f_max=f_max)),
        ("RMSE_ARV",      lambda: cal_RMSE_ARV(c, d, arv_window)),
        ("RMSE_ZCR",      lambda: cal_RMSE_ZCR(c, d, arv_window, sr)),
        ("RMSE_MNF",      lambda: cal_RMSE_MNF(c, d, sr, f_min, f_max)),
        ("RMSE_MDF",      lambda: cal_RMSE_MDF(c, d, arv_window, sr, f_min, f_max)),
        ("RMSE_Kurtosis", lambda: cal_RMSE_Kurtosis(c, d, arv_window)),
    ]
    out = {}
    for name, fn in fns:
        try:
            val = fn()
            out[name] = float(val) if np.isfinite(val) else np.nan
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            out[name] = np.nan
    return out


# ============================================================================
# Noise-type helpers
# ============================================================================
def _infer_color_subtype(noise_paths_str: str) -> Optional[str]:
    if not noise_paths_str or noise_paths_str == "nan":
        return None
    subtypes = set()
    for p in noise_paths_str.split("|"):
        bn = os.path.basename(p.strip()).lower()
        if "color" not in bn:
            continue
        if "_pink_" in bn:
            subtypes.add("Pink")
        elif "_brown_" in bn:
            subtypes.add("Brown")
    return subtypes.pop() if len(subtypes) == 1 else None


def _noise_type_labels(noise_types_str: str, noise_paths_str: str) -> List[str]:
    if not noise_types_str or str(noise_types_str) == "nan":
        return []
    types  = [t.strip() for t in str(noise_types_str).split("+") if t.strip()]
    labels = list(types)
    if "Color" in types:
        sub = _infer_color_subtype(str(noise_paths_str))
        if sub:
            labels.append(sub)
    labels.append(f"k={len(types)}")
    return labels


# ============================================================================
# ResultsCollector + table builders
# ============================================================================
_SINGLE_TYPE_ORDER = ["PLI", "ECG", "MOA", "WGN", "Color", "Pink", "Brown"]
_KCOUNT_ORDER      = ["k=1", "k=2", "k=3", "k=4", "k=5"]


class ResultsCollector:
    def __init__(self):
        self.results          = defaultdict(lambda: defaultdict(
                                    lambda: defaultdict(lambda: defaultdict(list))))
        self.by_noise_type    = defaultdict(lambda: defaultdict(list))
        self.by_snr_noisetype = defaultdict(lambda: defaultdict(list))
        self.n_total = self.n_ok = 0

    def add(self, db, snr, k, metrics, noise_type_labels=None):
        self.n_total += 1
        for name, val in metrics.items():
            if val is None or not np.isfinite(val):
                continue
            fval = float(val)
            self.results[db][snr][k][name].append(fval)
            if noise_type_labels:
                for lbl in noise_type_labels:
                    self.by_noise_type[lbl][name].append(fval)
                    self.by_snr_noisetype[(snr, lbl)][name].append(fval)
        self.n_ok += 1

    def _flatten(self, db=None, snr=None, k=None):
        vals = defaultdict(list)
        for d in (list(self.results) if db is None else [db]):
            if d not in self.results: continue
            for s in (list(self.results[d]) if snr is None else [snr]):
                if s not in self.results[d]: continue
                for kk in (list(self.results[d][s]) if k is None else [k]):
                    if kk not in self.results[d][s]: continue
                    for m, arr in self.results[d][s][kk].items():
                        vals[m].extend(arr)
        return {m: {"mean": float(np.mean(a)),
                    "std":  float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                    "n":    int(len(a))}
                for m, a in vals.items() if a}

    def summary(self, db=None, snr=None, k=None):
        return self._flatten(db, snr, k)

    def noise_type_summary(self, label):
        return {m: {"mean": float(np.mean(a)),
                    "std":  float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                    "n":    int(len(a))}
                for m, a in self.by_noise_type[label].items() if a}

    def snr_noisetype_summary(self, snr, label):
        return {m: {"mean": float(np.mean(a)),
                    "std":  float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                    "n":    int(len(a))}
                for m, a in self.by_snr_noisetype[(snr, label)].items() if a}

    def all_noise_type_labels(self):
        return sorted(self.by_noise_type.keys())

    def all_snr_inputs(self):
        return sorted({s for s, _ in self.by_snr_noisetype}, key=int)


def make_snr_k_table(collector, metric, db=None) -> pd.DataFrame:
    snrs, ks = set(), set()
    for d in collector.results:
        if db and d != db: continue
        for s in collector.results[d]:
            snrs.add(s); ks.update(collector.results[d][s].keys())
    rows = []
    for s in sorted(snrs, key=int):
        row = {"SNR": s}
        for k in sorted(ks):
            sm = collector.summary(db=db, snr=s, k=k)
            row[f"k={k}"] = round(sm[metric]["mean"], 7) if metric in sm else np.nan
        sm_all = collector.summary(db=db, snr=s)
        row["Avg"] = round(sm_all[metric]["mean"], 7) if metric in sm_all else np.nan
        rows.append(row)
    row_avg = {"SNR": "Avg"}
    for k in sorted(ks):
        sm = collector.summary(db=db, k=k)
        row_avg[f"k={k}"] = round(sm[metric]["mean"], 7) if metric in sm else np.nan
    overall = collector.summary(db=db)
    row_avg["Avg"] = round(overall[metric]["mean"], 7) if metric in overall else np.nan
    rows.append(row_avg)
    return pd.DataFrame(rows)


def make_noise_type_table(collector, metrics_list) -> pd.DataFrame:
    all_lbl = set(collector.all_noise_type_labels())
    ordered = [l for l in _SINGLE_TYPE_ORDER + _KCOUNT_ORDER if l in all_lbl]
    ordered += sorted(l for l in all_lbl if l not in ordered)
    rows = []
    for label in ordered:
        sm  = collector.noise_type_summary(label)
        row = {"noise_type": label}
        for m in metrics_list:
            row[f"{m}_mean"] = round(sm[m]["mean"], 7) if m in sm else np.nan
            row[f"{m}_std"]  = round(sm[m]["std"], 7) if m in sm else np.nan
            row[f"{m}_n"]    = sm[m]["n"]              if m in sm else 0
        rows.append(row)
    return pd.DataFrame(rows)


def make_snr_noisetype_table(collector, metric) -> pd.DataFrame:
    all_lbl    = set(collector.all_noise_type_labels())
    snr_inputs = collector.all_snr_inputs()
    ordered    = [l for l in _SINGLE_TYPE_ORDER + _KCOUNT_ORDER if l in all_lbl]
    ordered   += sorted(l for l in all_lbl if l not in ordered)
    rows = []
    for snr in snr_inputs:
        row = {"SNR_input": snr}
        for lbl in ordered:
            sm = collector.snr_noisetype_summary(snr, lbl)
            row[lbl] = round(sm[metric]["mean"], 7) if metric in sm else np.nan
        snr_summ = collector.summary(snr=snr)
        row["Avg"] = round(snr_summ[metric]["mean"], 7) if metric in snr_summ else np.nan
        rows.append(row)
    avg_row = {"SNR_input": "Avg"}
    for lbl in ordered:
        sm = collector.noise_type_summary(lbl)
        avg_row[lbl] = round(sm[metric]["mean"], 7) if metric in sm else np.nan
    overall = collector.summary()
    avg_row["Avg"] = round(overall[metric]["mean"], 7) if metric in overall else np.nan
    rows.append(avg_row)
    return pd.DataFrame(rows)


# ============================================================================
# Core inference loop for one method
# ============================================================================
def _worker_apply(args):
    # Top-level worker for multiprocessing.Pool (must be pickleable)
    item, method_name, params, sampling_rate = args
    scale     = float(item["scale"])
    noisy_raw = np.asarray(item["noisy"], dtype=np.float64) * scale
    noise_type = str(item.get("noise_types", ""))
    try:
        enh_raw, ok = apply_method(
            method=method_name, noisy_raw=noisy_raw,
            params=params, fs=sampling_rate, noise_type=noise_type,
        )
    except Exception:
        enh_raw, ok = noisy_raw.copy(), False
    return enh_raw, ok, scale


def run_one_method(
    method_name: str,
    data: np.ndarray,
    params: Dict,
    metrics_list: List[str],
    sampling_rate: int,
    f_min: float,
    f_max: float,
    lsd_n_fft: int = 512,
    lsd_hop: int = 128,
    n_jobs: int = 1,
) -> Tuple["ResultsCollector", Dict]:

    collector   = ResultsCollector()
    extra_stats = {
        "method":       method_name,
        "total":        0,
        "success":      0,
        "fail":         0,
        "fallback_str": params.get(method_name, {}).get("fallback", "noisy"),
    }

    has_noise_paths = (len(data) > 0 and isinstance(data[0], dict)
                       and "noise_paths" in data[0])

    # ── Parallel path (emd / ceemdan) ────────────────────────────────────────
    if n_jobs > 1 and method_name in ("emd", "ceemdan"):
        import multiprocessing as mp
        n_workers = min(n_jobs, mp.cpu_count())
        print(f"  [MP] {method_name.upper()} using {n_workers} workers")
        worker_args = [(item, method_name, params, sampling_rate) for item in data]
        chunk = max(1, len(data) // 200)
        with mp.Pool(processes=n_workers) as pool:
            mp_results = list(tqdm(
                pool.imap(_worker_apply, worker_args, chunksize=chunk),
                total=len(data), desc=f"  [{method_name.upper():8s}]",
            ))
        for item, (enh_raw, ok, scale) in zip(data, mp_results):
            clean_raw = np.asarray(item["clean"], dtype=np.float64) * scale
            noisy_raw = np.asarray(item["noisy"], dtype=np.float64) * scale
            extra_stats["total"]   += 1
            extra_stats["success"] += int(ok)
            extra_stats["fail"]    += int(not ok)
            m = calculate_all_metrics(
                clean_raw, enh_raw, noisy_raw,
                sr=sampling_rate, arv_window=200, f_min=f_min, f_max=f_max,
                lsd_n_fft=lsd_n_fft, lsd_hop=lsd_hop,
            )
            m = {k: v for k, v in m.items() if k in metrics_list}
            nt_labels = _noise_type_labels(
                str(item.get("noise_types", "")),
                str(item.get("noise_paths", "")) if has_noise_paths else "",
            )
            collector.add(
                db=str(item.get("dataset", "unknown")),
                snr=int(item.get("snr", 0)), k=int(item.get("k", 1)),
                metrics=m, noise_type_labels=nt_labels,
            )
        return collector, extra_stats

    # ── Serial path ───────────────────────────────────────────────────────────
    for item in tqdm(data, desc=f"  [{method_name.upper():8s}]"):
        scale     = float(item["scale"])
        clean_raw = np.asarray(item["clean"], dtype=np.float64) * scale
        noisy_raw = np.asarray(item["noisy"], dtype=np.float64) * scale

        item_noise_type = str(item.get("noise_types", ""))

        extra_stats["total"] += 1
        try:
            enh_raw, ok = apply_method(
                method=method_name,
                noisy_raw=noisy_raw,
                params=params,
                fs=sampling_rate,
                noise_type=item_noise_type,
            )
        except Exception as e:
            print(f"[WARN] {method_name} raised: {e}")
            enh_raw, ok = noisy_raw.copy(), False

        if ok:
            extra_stats["success"] += 1
        else:
            extra_stats["fail"] += 1

        m = calculate_all_metrics(
            clean_raw, enh_raw, noisy_raw,
            sr=sampling_rate, arv_window=200,
            f_min=f_min, f_max=f_max,
            lsd_n_fft=lsd_n_fft, lsd_hop=lsd_hop,
        )
        m = {k: v for k, v in m.items() if k in metrics_list}

        nt_labels = _noise_type_labels(
            str(item.get("noise_types", "")),
            str(item.get("noise_paths", "")) if has_noise_paths else "",
        )
        collector.add(
            db=str(item.get("dataset", "unknown")),
            snr=int(item.get("snr", 0)),
            k=int(item.get("k", 1)),
            metrics=m,
            noise_type_labels=nt_labels,
        )

    return collector, extra_stats


def save_results(collector, method_name, output_dir, metrics_list, extra_stats):
    os.makedirs(output_dir, exist_ok=True)
    prefix  = os.path.join(output_dir, method_name)
    overall = collector.summary()

    pd.DataFrame([{"metric": m, **st} for m, st in overall.items()]).to_csv(
        f"{prefix}_overall_summary.csv", index=False, float_format="%.10g")

    for metric in metrics_list:
        if metric not in overall:
            continue
        make_snr_k_table(collector, metric).to_csv(
            f"{prefix}_table_all_{metric}.csv", index=False, float_format="%.10g")
        for db in sorted(collector.results):
            if any(metric in collector.results[db][s][k]
                   for s in collector.results[db]
                   for k in collector.results[db][s]):
                make_snr_k_table(collector, metric, db=db).to_csv(
                    f"{prefix}_table_{db}_{metric}.csv", index=False, float_format="%.10g")

    nt_df = make_noise_type_table(collector, metrics_list)
    nt_df.to_csv(f"{prefix}_table_noise_type_all_metrics.csv", index=False, float_format="%.10g")
    for m in metrics_list:
        mcol = f"{m}_mean"
        if mcol in nt_df.columns:
            nt_df[["noise_type", mcol, f"{m}_std", f"{m}_n"]].rename(
                columns={mcol: "mean", f"{m}_std": "std", f"{m}_n": "n"}
            ).to_csv(f"{prefix}_table_noisetype_{m}.csv", index=False, float_format="%.10g")

    for m in metrics_list:
        if m not in overall:
            continue
        make_snr_noisetype_table(collector, m).to_csv(
            f"{prefix}_table_snr_x_noisetype_{m}.csv", index=False, float_format="%.10g")

    with open(f"{prefix}_extra_stats.json", "w") as f:
        json.dump(extra_stats, f, indent=2)

    print(f"  ✓ saved → {output_dir}  (prefix={method_name}_*)")


def print_summary(method_name, collector, metrics_list, extra_stats):
    W = 70
    print(f"\n{'='*W}")
    print(f"{method_name.upper()} — OVERALL METRICS")
    print(f"{'='*W}")
    overall = collector.summary()
    groups = [
        ("Signal Quality",              ["SNRimp", "RMSE", "PRD", "LSD"]),
        ("Feature Error — Time",        ["RMSE_ARV", "RMSE_ZCR"]),
        ("Feature Error — Frequency",   ["RMSE_MNF", "RMSE_MDF"]),
        ("Feature Error — Statistical", ["RMSE_Kurtosis"]),
    ]
    for grp, mlist in groups:
        print(f"\n  -- {grp} --")
        for m in mlist:
            if m not in metrics_list or m not in overall:
                continue
            st   = overall[m]
            unit = (" dB" if m in ("SNRimp", "LSD") else
                    " %"  if m == "PRD" else
                    " Hz" if m in ("RMSE_MNF", "RMSE_MDF") else
                    " cross/s" if m == "RMSE_ZCR" else "")
            print(f"    {m:<16s}: {st['mean']:9.4f} +/- {st['std']:7.4f}{unit}"
                  f"  (n={st['n']})")

    tot  = extra_stats.get("total",   0)
    succ = extra_stats.get("success", 0)
    fail = extra_stats.get("fail",    0)
    if tot > 0 and method_name not in ("hp",):
        print(f"\n  Success rate: {succ}/{tot} ({100.0*succ/tot:.1f}%)"
              f"  |  fail/fallback: {fail}")
    print(f"{'='*W}\n")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Traditional Baseline Inference v2 (HP / TS / EMD / VMD / CEEMDAN)")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--trad-config", default="tradition_train_config.yaml")
    parser.add_argument("--params",      required=True)
    parser.add_argument("--test-data",   required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--methods",     default="hp,ts",
                        help="Comma-separated methods to evaluate, or 'all'. "
                             f"Choices: {ALL_METHODS}  (default: hp,ts)")
    parser.add_argument("--metrics",     default=",".join(DEFAULT_METRICS))
    parser.add_argument("--sr",          type=int, default=1000)
    parser.add_argument("--n-jobs",      type=int, default=1,
                        help="Parallel workers for emd/ceemdan. "
                             "Set to CPU core count for speedup (default: 1).")
    args = parser.parse_args()

    # Parse --methods
    if args.methods.strip().lower() == "all":
        methods_to_run = list(ALL_METHODS)
    else:
        methods_to_run = [m.strip().lower() for m in args.methods.split(",")
                          if m.strip()]
    unknown = [m for m in methods_to_run if m not in ALL_METHODS]
    if unknown:
        parser.error(f"Unknown methods: {unknown}. Valid: {ALL_METHODS}")

    main_cfg     = load_yaml(args.config)
    metrics_cfg  = _get(main_cfg, ["metrics"], {}) or {}
    f_min        = float(metrics_cfg.get("f_min", 20.0))
    f_max        = float(metrics_cfg.get("f_max", 500.0))
    lsd_n_fft    = int(metrics_cfg.get("lsd_n_fft", 512))
    lsd_hop      = int(metrics_cfg.get("lsd_hop",   128))
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    with open(args.params) as f:
        params = json.load(f)

    # Validate that required params exist for requested methods
    missing = [m for m in methods_to_run
               if m not in params and m not in ("hp", "ts")]
    if missing:
        print(f"[WARN] No params found for: {missing}. "
              f"Run train_tradition.py --methods {','.join(missing)} first.")
        methods_to_run = [m for m in methods_to_run if m not in missing]

    print(f"[Params loaded]  {args.params}")
    for m in methods_to_run:
        p = params.get(m, {})
        if m == "hp":
            print(f"  HP  cutoff={p.get('best_cutoff_hz')} Hz")
        elif m == "ts":
            print(f"  TS  fallback={p.get('fallback')}")
        elif m == "emd":
            print(f"  EMD f_min={p.get('best_f_min_hz')} Hz")
        elif m == "vmd":
            print(f"  VMD K={p.get('best_K')}  alpha={p.get('alpha')}")
        elif m == "ceemdan":
            print(f"  CEEMDAN f_min={p.get('best_f_min_hz')} Hz  "
                  f"trials={p.get('trials')}")

    print(f"\n[Test Data] {args.test_data}")
    raw = np.load(args.test_data, allow_pickle=True)["data"]
    print(f"  Samples  : {len(raw)}")
    print(f"  Methods  : {methods_to_run}")
    print(f"  Metrics  : {metrics_list}")

    os.makedirs(args.output, exist_ok=True)

    for mname in methods_to_run:
        print(f"\n[Evaluating {mname.upper()}]  n={len(raw)}")
        collector, extra = run_one_method(
            method_name=mname, data=raw, params=params,
            metrics_list=metrics_list, sampling_rate=args.sr,
            f_min=f_min, f_max=f_max,
            lsd_n_fft=lsd_n_fft, lsd_hop=lsd_hop,
            n_jobs=args.n_jobs,
        )
        print_summary(mname, collector, metrics_list, extra)
        save_results(collector, mname, args.output, metrics_list, extra)

    print(f"\n✓ Inference complete → {args.output}")


if __name__ == "__main__":
    main()