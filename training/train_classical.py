#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tradition.py  —  Traditional Baseline Calibration

Calibrates parameters for HP, TS, EMD, VMD, CEEMDAN on the val split.

Method            Calibrated parameter
-----------       ---------------------
HP                best_cutoff_hz   (sweep over [20,30,40,50,60,80,100] Hz)
TS                fixed defaults   (no free calibration parameters)
EMD               best_f_min_hz    (sweep frequency boundary for IMF selection)
VMD               best_K           (sweep number of modes [3-8])
CEEMDAN           best_f_min_hz    (sweep frequency boundary, same as EMD)

Output: {weights_dir}/tradition_params.json

Usage:
    python3 train_tradition.py \\
        --config        config.yaml \\
        --trad-config   tradition_train_config.yaml \\
        --segments-root outputs/segments/data_crossDB_seg2s \\
        --noise-root    outputs/noise/sEMG_noise_train \\
        --weights       outputs/weights_tradition \\
        --methods       hp,ts,emd,vmd,ceemdan   # or "all"
"""

import os
import sys
import json
import argparse
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# Shared filter implementations
sys.path.insert(0, os.path.dirname(__file__))
from tradition_filters import (
    apply_hp_filter,
    apply_emd_filter,
    apply_vmd_filter,
    apply_ceemdan_filter,
    ALL_METHODS,
)


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
# Normalization  (mirrors train.py)
# ============================================================================
def compute_scale_factor(x: np.ndarray, method: str = "Q99",
                          percentile: float = 0.99) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 1.0
    method = method.upper()
    if method in ("Q99", "Q95", "Q"):
        q = percentile if method == "Q" else (0.99 if method == "Q99" else 0.95)
        scale = float(np.quantile(np.abs(x), q))
    elif method == "RMS":
        scale = float(np.sqrt(np.mean(x ** 2)))
    elif method == "MAD":
        med = np.median(x)
        scale = float(np.median(np.abs(x - med)) * 1.4826)
    elif method == "STD":
        scale = float(np.std(x))
    else:
        raise ValueError(f"Unknown norm method: {method}")
    return max(scale, 1e-12)


def normalize_clip(x: np.ndarray, scale: float,
                   clip_range: Tuple[float, float]) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).reshape(-1) / float(scale)
    return np.clip(y, clip_range[0], clip_range[1]).astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================
def cal_snr(clean: np.ndarray, test: np.ndarray) -> float:
    c, t = clean.astype(np.float64).reshape(-1), test.astype(np.float64).reshape(-1)
    n_pw = float(np.dot(t - c, t - c))
    s_pw = float(np.dot(c, c))
    return 100.0 if n_pw < 1e-12 else float(10.0 * np.log10(s_pw / n_pw))


def cal_snrimp(clean, denoised, noisy) -> float:
    return cal_snr(clean, denoised) - cal_snr(clean, noisy)


# ============================================================================
# Val-set loader
# ============================================================================
def load_val_samples(
    segments_root: str,
    noise_root: str,
    trad_cfg: Dict,
    main_cfg: Dict,
    max_samples: int = 2000,
    seed: int = 42,
) -> List[Dict]:
    """Build a deterministic val set by mixing segments with noise."""
    import pandas as pd

    man_path = os.path.join(segments_root, "manifests", "segment_manifest.csv")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"Manifest not found: {man_path}")

    df = pd.read_csv(man_path)
    df_val = df[df["split"] == "val"].copy()
    if df_val.empty:
        raise RuntimeError("No val segments found in manifest.")

    rng = np.random.default_rng(seed)
    if len(df_val) > max_samples:
        idx = rng.choice(len(df_val), size=max_samples, replace=False)
        df_val = df_val.iloc[sorted(idx)].copy()

    noise_files: Dict[str, List[str]] = {}
    for ntype in ["PLI", "ECG", "MOA", "WGN", "Color"]:
        nd = os.path.join(noise_root, ntype)
        if os.path.isdir(nd):
            fs_list = sorted([os.path.join(nd, f)
                              for f in os.listdir(nd) if f.endswith(".npy")])
            if fs_list:
                noise_files[ntype] = fs_list

    if not noise_files:
        raise RuntimeError(f"No noise files found under: {noise_root}")

    snr_grid  = [-10, -5, 0, 5, 10]
    norm_method = _get(main_cfg, ["normalization", "method"], "Q99")
    norm_pct    = float(_get(main_cfg, ["normalization", "percentile"], 0.99))
    clip_range  = tuple(_get(main_cfg, ["normalization", "clip_range"], [-1.0, 1.0]))
    target_fs   = int(_get(main_cfg, ["preprocessing", "segmentation", "target_fs"], 1000))

    samples = []
    noise_cache: Dict[str, np.ndarray] = {}

    for _, row in df_val.iterrows():
        raw_path = row["raw_path"]
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(segments_root, raw_path)
        try:
            clean_raw = np.load(raw_path).astype(np.float64)
        except Exception:
            continue

        L = clean_raw.size
        snr_val = float(rng.choice(snr_grid))
        ntype   = rng.choice(list(noise_files.keys()))
        npath   = rng.choice(noise_files[ntype])

        if npath not in noise_cache:
            noise_cache[npath] = np.load(npath).astype(np.float64)
        nfull = noise_cache[npath]
        if nfull.size < L:
            nfull = np.tile(nfull, (L // nfull.size) + 2)
        start = int(rng.integers(0, nfull.size - L + 1))
        nseg  = nfull[start:start + L].copy()

        s_pw = float(np.dot(clean_raw, clean_raw))
        n_pw = float(np.dot(nseg, nseg))
        if n_pw < 1e-12 or s_pw < 1e-12:
            continue
        scalar   = math.sqrt(s_pw / (n_pw * (10.0 ** (snr_val / 10.0))))
        noisy_raw = clean_raw + scalar * nseg

        scale     = compute_scale_factor(noisy_raw, method=norm_method,
                                          percentile=norm_pct)
        samples.append({
            "clean_raw":  clean_raw.astype(np.float32),
            "noisy_raw":  noisy_raw.astype(np.float32),
            "scale":      float(scale),
            "snr":        snr_val,
            "target_fs":  target_fs,
        })

    print(f"[ValLoader] Built {len(samples)} val samples")
    return samples


# ============================================================================
# HP calibration
# ============================================================================
def calibrate_hp(samples: List[Dict], trad_cfg: Dict, order: int = 4) -> Dict:
    sweep_cfg = _get(trad_cfg, ["hp", "sweep"], {})
    default_co = float(_get(trad_cfg, ["hp", "default_cutoff_hz"], 40.0))

    if not sweep_cfg.get("enabled", True):
        print(f"[HP Sweep] disabled → default {default_co} Hz")
        return {"best_cutoff_hz": default_co, "sweep_results": {}}

    cutoffs     = sweep_cfg.get("cutoffs", [20, 30, 40, 50, 60, 80, 100])
    metric_name = sweep_cfg.get("metric", "SNRimp")
    print(f"\n[HP Sweep] cutoffs={cutoffs} Hz | metric={metric_name} | n={len(samples)}")

    results = {}
    for co in cutoffs:
        vals = []
        for s in samples:
            noisy = s["noisy_raw"].astype(np.float64)
            clean = s["clean_raw"].astype(np.float64)
            fs    = s["target_fs"]
            try:
                enh = apply_hp_filter(noisy, fs=fs, cutoff_hz=co, order=order)
            except Exception:
                continue
            vals.append(cal_snrimp(clean, enh, noisy))
        mean_val = float(np.mean(vals)) if vals else -999.0
        results[co] = mean_val
        print(f"  cutoff={co:4.0f} Hz → {metric_name} = {mean_val:.4f}")

    best = max(results, key=lambda x: results[x])
    print(f"  ✓ Best HP cutoff: {best} Hz ({metric_name}={results[best]:.4f})")
    return {"best_cutoff_hz": float(best), "sweep_results": results}


# ============================================================================
# EMD calibration  (sweep f_min_hz for IMF-frequency selection)
# ============================================================================
def calibrate_emd(samples: List[Dict], trad_cfg: Dict) -> Dict:
    sweep_cfg     = _get(trad_cfg, ["emd", "sweep"], {})
    default_f_min = float(_get(trad_cfg, ["emd", "f_min_hz"], 20.0))
    f_max         = float(_get(trad_cfg, ["emd", "f_max_hz"], 500.0))
    max_imfs      = int(_get(trad_cfg, ["emd", "max_imfs"], 20))

    if not sweep_cfg.get("enabled", True):
        print(f"[EMD Sweep] disabled → default f_min={default_f_min} Hz")
        return {"best_f_min_hz": default_f_min, "sweep_results": {}}

    candidates  = sweep_cfg.get("f_min_candidates", [10, 15, 20, 25, 30, 40])
    metric_name = sweep_cfg.get("metric", "SNRimp")
    max_cal     = int(sweep_cfg.get("max_cal_samples", 200))
    cal_samples = samples[:max_cal]

    print(f"\n[EMD Sweep] f_min_candidates={candidates} | n={len(cal_samples)}")
    print("  (EMD is slow; grab a coffee)")

    results = {}
    for f_min in candidates:
        vals = []
        for s in tqdm(cal_samples, desc=f"  EMD f_min={f_min}", leave=False):
            noisy = s["noisy_raw"].astype(np.float64)
            clean = s["clean_raw"].astype(np.float64)
            fs    = s["target_fs"]
            enh, ok = apply_emd_filter(noisy, fs=fs, f_min=f_min,
                                        f_max=f_max, max_imfs=max_imfs)
            if ok:
                vals.append(cal_snrimp(clean, enh, noisy))
        mean_val = float(np.mean(vals)) if vals else -999.0
        results[f_min] = mean_val
        print(f"  f_min={f_min:4.0f} Hz → {metric_name} = {mean_val:.4f}  (n_ok={len(vals)})")

    best = max(results, key=lambda x: results[x]) if results else default_f_min
    print(f"  ✓ Best EMD f_min: {best} Hz")
    return {"best_f_min_hz": float(best), "sweep_results": results}


# ============================================================================
# VMD calibration  (sweep K)
# ============================================================================
def calibrate_vmd(samples: List[Dict], trad_cfg: Dict) -> Dict:
    sweep_cfg = _get(trad_cfg, ["vmd", "sweep"], {})
    default_K = int(_get(trad_cfg, ["vmd", "K"], 6))
    alpha     = float(_get(trad_cfg, ["vmd", "alpha"], 2000.0))
    tau       = float(_get(trad_cfg, ["vmd", "tau"], 0.0))
    f_min     = float(_get(trad_cfg, ["vmd", "f_min_hz"], 20.0))
    f_max     = float(_get(trad_cfg, ["vmd", "f_max_hz"], 500.0))
    tol       = float(_get(trad_cfg, ["vmd", "tol"], 1e-7))

    if not sweep_cfg.get("enabled", True):
        print(f"[VMD Sweep] disabled → default K={default_K}")
        return {"best_K": default_K, "sweep_results": {}}

    candidates  = sweep_cfg.get("K_candidates", [3, 4, 5, 6, 7, 8])
    metric_name = sweep_cfg.get("metric", "SNRimp")
    max_cal     = int(sweep_cfg.get("max_cal_samples", 100))
    cal_samples = samples[:max_cal]

    print(f"\n[VMD Sweep] K_candidates={candidates} | n={len(cal_samples)}")
    print("  (VMD is expensive; be patient)")

    results = {}
    for K in candidates:
        vals = []
        for s in tqdm(cal_samples, desc=f"  VMD K={K}", leave=False):
            noisy = s["noisy_raw"].astype(np.float64)
            clean = s["clean_raw"].astype(np.float64)
            fs    = s["target_fs"]
            enh, ok = apply_vmd_filter(noisy, fs=fs, K=K, alpha=alpha,
                                        tau=tau, f_min=f_min, f_max=f_max, tol=tol)
            if ok:
                vals.append(cal_snrimp(clean, enh, noisy))
        mean_val = float(np.mean(vals)) if vals else -999.0
        results[K] = mean_val
        print(f"  K={K} → {metric_name} = {mean_val:.4f}  (n_ok={len(vals)})")

    best_K = max(results, key=lambda x: results[x]) if results else default_K
    print(f"  ✓ Best VMD K: {best_K}")
    return {"best_K": int(best_K), "sweep_results": results}


# ============================================================================
# CEEMDAN calibration  (sweep f_min_hz, same strategy as EMD)
# ============================================================================
def calibrate_ceemdan(samples: List[Dict], trad_cfg: Dict) -> Dict:
    sweep_cfg     = _get(trad_cfg, ["ceemdan", "sweep"], {})
    default_f_min = float(_get(trad_cfg, ["ceemdan", "f_min_hz"], 20.0))
    f_max         = float(_get(trad_cfg, ["ceemdan", "f_max_hz"], 500.0))
    trials        = int(_get(trad_cfg, ["ceemdan", "trials"], 100))
    epsilon       = float(_get(trad_cfg, ["ceemdan", "epsilon"], 0.005))

    if not sweep_cfg.get("enabled", True):
        print(f"[CEEMDAN Sweep] disabled → default f_min={default_f_min} Hz")
        return {"best_f_min_hz": default_f_min, "sweep_results": {}}

    candidates  = sweep_cfg.get("f_min_candidates", [10, 15, 20, 25, 30, 40])
    metric_name = sweep_cfg.get("metric", "SNRimp")
    max_cal     = int(sweep_cfg.get("max_cal_samples", 100))
    cal_samples = samples[:max_cal]

    print(f"\n[CEEMDAN Sweep] f_min_candidates={candidates} | n={len(cal_samples)}")
    print("  (CEEMDAN is very slow; this may take a while)")

    results = {}
    for f_min in candidates:
        vals = []
        for s in tqdm(cal_samples, desc=f"  CEEMDAN f_min={f_min}", leave=False):
            noisy = s["noisy_raw"].astype(np.float64)
            clean = s["clean_raw"].astype(np.float64)
            fs    = s["target_fs"]
            enh, ok = apply_ceemdan_filter(noisy, fs=fs, trials=trials,
                                            epsilon=epsilon, f_min=f_min, f_max=f_max)
            if ok:
                vals.append(cal_snrimp(clean, enh, noisy))
        mean_val = float(np.mean(vals)) if vals else -999.0
        results[f_min] = mean_val
        print(f"  f_min={f_min:4.0f} Hz → {metric_name} = {mean_val:.4f}  (n_ok={len(vals)})")

    best = max(results, key=lambda x: results[x]) if results else default_f_min
    print(f"  ✓ Best CEEMDAN f_min: {best} Hz")
    return {"best_f_min_hz": float(best), "sweep_results": results}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Traditional Baseline Calibration (HP / TS / EMD / VMD / CEEMDAN)")
    parser.add_argument("--config",           type=str, required=True)
    parser.add_argument("--trad-config",      type=str,
                        default="tradition_train_config.yaml")
    parser.add_argument("--segments-root",    type=str, required=True)
    parser.add_argument("--noise-root",       type=str, required=True)
    parser.add_argument("--weights",          type=str,
                        default="outputs/weights_tradition")
    parser.add_argument("--max-val-samples",  type=int, default=2000)
    parser.add_argument("--methods",          type=str, default="hp,ts",
                        help="Comma-separated methods to calibrate, or 'all'. "
                             "Choices: hp ts emd vmd ceemdan  (default: hp,ts)")
    parser.add_argument("--force",            action="store_true",
                        help="Re-run even if params JSON already exists")
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

    main_cfg  = load_yaml(args.config)
    trad_cfg  = load_yaml(args.trad_config)
    os.makedirs(args.weights, exist_ok=True)

    out_json = os.path.join(args.weights, "tradition_params.json")

    # Load existing params if present (to merge, not overwrite unselected methods)
    existing_params: Dict = {}
    if os.path.exists(out_json):
        if not args.force:
            print(f"[SKIP] {out_json} already exists (use --force to rerun)")
            return
        with open(out_json) as f:
            existing_params = json.load(f)

    seed = int(_get(main_cfg, ["project", "random_seed"], 12345))
    random.seed(seed); np.random.seed(seed)

    print(f"\n{'='*70}")
    print("Traditional Baseline Calibration")
    print(f"{'='*70}")
    print(f"config:       {args.config}")
    print(f"trad-config:  {args.trad_config}")
    print(f"segments:     {args.segments_root}")
    print(f"noise:        {args.noise_root}")
    print(f"output:       {out_json}")
    print(f"methods:      {methods_to_run}")

    # ── Load val samples ──────────────────────────────────────────────────────
    need_val = [m for m in methods_to_run if m != "ts"]
    val_samples = []
    if need_val:
        print(f"\n[Step 1] Loading val samples (max={args.max_val_samples})…")
        try:
            val_samples = load_val_samples(
                segments_root=args.segments_root,
                noise_root=args.noise_root,
                trad_cfg=trad_cfg,
                main_cfg=main_cfg,
                max_samples=args.max_val_samples,
                seed=seed,
            )
        except Exception as e:
            print(f"[WARN] Could not build val samples: {e}")
            print("       Falling back to paper defaults for all methods.")

    # ── Build params dict (start from existing, override calibrated methods) ──
    target_fs  = int(_get(main_cfg, ["preprocessing", "segmentation", "target_fs"], 1000))
    seg_len_s  = float(_get(main_cfg, ["preprocessing", "segmentation", "length_s"], 2.0))
    hp_order   = int(_get(trad_cfg, ["hp", "order"], 4))

    params = dict(existing_params)  # preserve previously calibrated entries

    # ── HP ────────────────────────────────────────────────────────────────────
    if "hp" in methods_to_run:
        print(f"\n[Calibrating HP]")
        if val_samples:
            hp_result = calibrate_hp(val_samples, trad_cfg, order=hp_order)
        else:
            default_co = float(_get(trad_cfg, ["hp", "default_cutoff_hz"], 40.0))
            hp_result  = {"best_cutoff_hz": default_co, "sweep_results": {}}
        params["hp"] = {
            "best_cutoff_hz": hp_result["best_cutoff_hz"],
            "order":          hp_order,
            "sweep_results":  hp_result["sweep_results"],
        }

    # Ensure HP baseline always exists (needed as fallback by other methods)
    if "hp" not in params:
        default_co = float(_get(trad_cfg, ["hp", "default_cutoff_hz"], 40.0))
        params["hp"] = {"best_cutoff_hz": default_co, "order": hp_order,
                        "sweep_results": {}}

    # ── TS ────────────────────────────────────────────────────────────────────
    if "ts" in methods_to_run:
        print(f"\n[Calibrating TS]  (fixed defaults, no sweep)")
        ts_cfg = _get(trad_cfg, ["ts"], {})
        params["ts"] = {
            "peak_detect_bp_low_hz":  float(ts_cfg.get("peak_detect_bp_low_hz", 2.5)),
            "peak_detect_bp_high_hz": float(ts_cfg.get("peak_detect_bp_high_hz", 50.0)),
            "peak_detect_order":      int(ts_cfg.get("peak_detect_order", 4)),
            "avg_window":             int(ts_cfg.get("avg_window", 11)),
            "min_peaks":              int(ts_cfg.get("min_peaks", 2)),
            "tile_factor":            int(ts_cfg.get("tile_factor", 8)),
            "fallback":               str(ts_cfg.get("fallback", "noisy")),
            "min_beat_gap_samples":   int(ts_cfg.get("min_beat_gap_samples", 140)),
        }
        if seg_len_s < 8.0:
            print(f"  ⚠  SHORT SEGMENT ({seg_len_s}s): TS will often fall back "
                  f"to '{params['ts']['fallback']}' — this is expected.")

    # ── EMD ───────────────────────────────────────────────────────────────────
    if "emd" in methods_to_run:
        print(f"\n[Calibrating EMD]")
        emd_cfg = _get(trad_cfg, ["emd"], {})
        if val_samples:
            emd_result = calibrate_emd(val_samples, trad_cfg)
        else:
            emd_result = {"best_f_min_hz": emd_cfg.get("f_min_hz", 20.0),
                          "sweep_results": {}}
        params["emd"] = {
            "best_f_min_hz": emd_result["best_f_min_hz"],
            "f_max_hz":      float(emd_cfg.get("f_max_hz", 500.0)),
            "max_imfs":      int(emd_cfg.get("max_imfs", 20)),
            "fallback":      str(emd_cfg.get("fallback", "noisy")),
            "sweep_results": emd_result["sweep_results"],
        }

    # ── VMD ───────────────────────────────────────────────────────────────────
    if "vmd" in methods_to_run:
        print(f"\n[Calibrating VMD]")
        vmd_cfg = _get(trad_cfg, ["vmd"], {})
        if val_samples:
            vmd_result = calibrate_vmd(val_samples, trad_cfg)
        else:
            vmd_result = {"best_K": vmd_cfg.get("K", 6), "sweep_results": {}}
        params["vmd"] = {
            "best_K":        vmd_result["best_K"],
            "alpha":         float(vmd_cfg.get("alpha", 2000.0)),
            "tau":           float(vmd_cfg.get("tau", 0.0)),
            "tol":           float(vmd_cfg.get("tol", 1e-7)),
            "f_min_hz":      float(vmd_cfg.get("f_min_hz", 20.0)),
            "f_max_hz":      float(vmd_cfg.get("f_max_hz", 500.0)),
            "fallback":      str(vmd_cfg.get("fallback", "noisy")),
            "sweep_results": vmd_result["sweep_results"],
        }

    # ── CEEMDAN ───────────────────────────────────────────────────────────────
    if "ceemdan" in methods_to_run:
        print(f"\n[Calibrating CEEMDAN]")
        cem_cfg = _get(trad_cfg, ["ceemdan"], {})
        if val_samples:
            cem_result = calibrate_ceemdan(val_samples, trad_cfg)
        else:
            cem_result = {"best_f_min_hz": cem_cfg.get("f_min_hz", 20.0),
                          "sweep_results": {}}
        params["ceemdan"] = {
            "best_f_min_hz": cem_result["best_f_min_hz"],
            "f_max_hz":      float(cem_cfg.get("f_max_hz", 500.0)),
            "trials":        int(cem_cfg.get("trials", 100)),
            "epsilon":       float(cem_cfg.get("epsilon", 0.005)),
            "fallback":      str(cem_cfg.get("fallback", "noisy")),
            "sweep_results": cem_result["sweep_results"],
        }

    # ── Shared meta ───────────────────────────────────────────────────────────
    params["target_fs"] = target_fs
    params["calibration_meta"] = {
        "n_val_samples_used": len(val_samples),
        "seed":               seed,
        "methods_calibrated": methods_to_run,
    }

    with open(out_json, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\n{'='*70}")
    print("✓ Calibration complete.")
    for m in methods_to_run:
        if m == "hp":
            print(f"  HP  best_cutoff_hz : {params['hp']['best_cutoff_hz']} Hz")
        elif m == "ts":
            print(f"  TS  fallback       : {params['ts']['fallback']}")
        elif m == "emd":
            print(f"  EMD best_f_min_hz  : {params['emd']['best_f_min_hz']} Hz")
        elif m == "vmd":
            print(f"  VMD best_K         : {params['vmd']['best_K']}")
        elif m == "ceemdan":
            print(f"  CEM best_f_min_hz  : {params['ceemdan']['best_f_min_hz']} Hz")
    print(f"  Saved → {out_json}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()