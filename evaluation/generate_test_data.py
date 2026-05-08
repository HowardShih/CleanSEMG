#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Offline Test Data (v6.8.0)

═══════════════════════════════════════════════════════════════════
WGN Power Allocation Rule
═══════════════════════════════════════════════════════════════════

Mixing divides total noise power equally among k types:
    per_component_SNR = total_SNR + 10·log₁₀(k)

WGN floor: WGN's component SNR must be >= WGN_COMPONENT_SNR_MIN_DB (-5 dB).

Case 1 — Normal (per-component SNR >= floor):
    All k types receive equal power.  (unchanged from v6.7.0)

Case 2 — WGN would go below floor (per-component SNR < floor), k >= 2:
    WGN  → fixed at floor power  (corresponding to -5 dB component SNR)
    Rest → remaining power shared equally among the other (k-1) types
    Total noise power is preserved → SNR label remains correct.

    Example  total=-15 dB, k=5:
        Equal share: WGN at -8 dB component  (< -5 dB floor → capped)
        WGN  power  = P_clean / 10^(−5/10)   = 3.16·P_clean
        Remaining   = 31.62·P_clean − 3.16·P = 28.46·P_clean
        Other 4     = 28.46 / 4               = 7.12·P_clean each (≈ −8.5 dB)
        Total noise = 31.62·P_clean  ✓  (still −15 dB)

Case 3 — k=1 with WGN, per-component SNR < floor:
    WGN is removed from the available pool; another type is selected.
    (No remaining power to distribute → capping would break SNR label.)

Key consequence:
    ALL (snr, k) combinations are now valid — no combos skipped.
    ➜ fair_snr_set is not needed.
    ➜ Inference v2.0 (simple) is sufficient.

Changes from v6.7.0:
    - _is_valid_combo / _wgn_excluded removed; all combos valid.
    - _compute_noise_targets() implements the capping rule.
    - DeterministicNoiseMixer.mix() uses per-type power targets.
    - Combined file generation: no rejection loop.
═══════════════════════════════════════════════════════════════════
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

WGN_COMPONENT_SNR_MIN_DB = -5.0   # WGN per-component SNR floor


# ============================================================================
# Normalization
# ============================================================================
def compute_scale_factor(x: np.ndarray, method: str = "Q99",
                         percentile: float = 0.99) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 1.0
    m = str(method).upper()
    if m in ("Q99", "Q95", "Q"):
        q = float(percentile) if m == "Q" else (0.99 if m == "Q99" else 0.95)
        scale = float(np.quantile(np.abs(x), q))
    elif m == "RMS":
        scale = float(np.sqrt(np.mean(x ** 2)))
    elif m == "MAD":
        med = np.median(x)
        scale = float(np.median(np.abs(x - med)) * 1.4826)
    elif m == "STD":
        scale = float(np.std(x))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return max(scale, 1e-12)


def normalize_clip(x: np.ndarray, scale: float,
                   clip_range: Tuple[float, float]) -> np.ndarray:
    s = float(scale) if (scale is not None and np.isfinite(scale) and scale > 0) else 1.0
    return np.clip(np.asarray(x, dtype=np.float32).reshape(-1) / s,
                   float(clip_range[0]), float(clip_range[1])).astype(np.float32)


# ============================================================================
# WGN power allocation
# ============================================================================

def _compute_noise_targets(clean_pow: float, total_snr: float,
                            selected: List[str]) -> Dict[str, float]:
    """
    Compute target noise power for each selected noise type.

    WGN rule:
      - Equal split if WGN's equal-share component SNR is >= floor.
      - Cap WGN only if equal-share component SNR would be below floor.
      - k=1 WGN below floor is handled upstream by excluding WGN.
    """
    k = len(selected)
    total_noise = clean_pow / (10.0 ** (total_snr / 10.0))

    if k <= 0:
        return {}

    if "WGN" not in selected or k <= 1:
        return {nt: total_noise / k for nt in selected}

    # WGN present, k >= 2.
    # This is the maximum allowed WGN power corresponding to the -5 dB floor.
    wgn_equal_pow = total_noise / k
    wgn_floor_pow = clean_pow / (10.0 ** (WGN_COMPONENT_SNR_MIN_DB / 10.0))

    # If equal split already keeps WGN at or above the floor, do not cap.
    # In power terms: component SNR >= floor  <=>  WGN power <= floor power.
    if wgn_equal_pow <= wgn_floor_pow:
        return {nt: total_noise / k for nt in selected}

    # Otherwise, WGN would be too strong, so cap WGN and redistribute the rest.
    remaining = total_noise - wgn_floor_pow

    # Numerical guard. This should not be negative after the condition above.
    if remaining < 0:
        remaining = 0.0

    per_other = remaining / (k - 1)

    return {
        nt: (wgn_floor_pow if nt == "WGN" else per_other)
        for nt in selected
    }


# ============================================================================
# Deterministic Noise Mixer
# ============================================================================
class DeterministicNoiseMixer:
    """
    Deterministic, reproducible noise mixer for offline test generation.

    All (total_snr, k) combinations are valid.
    WGN is capped at WGN_COMPONENT_SNR_MIN_DB when needed (see module doc).
    """

    def __init__(self, noise_root: str, config: Dict, seed: int = 42):
        from glob import glob
        self.noise_root  = noise_root
        self.base_seed   = seed
        self.noise_paths: Dict[str, List[str]] = {}
        self.noise_cache: Dict[str, np.ndarray] = {}
        self._load_noise_paths()
        self._cache_all()

    def _load_noise_paths(self):
        from glob import glob
        for ntype in ["PLI", "ECG", "MOA", "WGN", "Color"]:
            ndir = os.path.join(self.noise_root, ntype)
            if os.path.isdir(ndir):
                paths = sorted(glob(os.path.join(ndir, "*.npy")))
                if paths:
                    self.noise_paths[ntype] = paths
                    if ntype == "Color":
                        np_ = sum(1 for p in paths if "_pink_"  in os.path.basename(p))
                        nb  = sum(1 for p in paths if "_brown_" in os.path.basename(p))
                        print(f"[Mixer] {ntype}: {len(paths)} (pink={np_}, brown={nb})")
                    else:
                        print(f"[Mixer] {ntype}: {len(paths)}")
        if not self.noise_paths:
            raise ValueError(f"No noise files in {self.noise_root}")

    def _cache_all(self):
        print("[Mixer] Caching …")
        for ps in self.noise_paths.values():
            for p in ps:
                self.noise_cache[p] = np.load(p).astype(np.float64)
        print(f"[Mixer] Cached {len(self.noise_cache)} files")

    def _sample_seg(self, noise: np.ndarray, length: int, rng) -> np.ndarray:
        if len(noise) < length:
            noise = np.tile(noise, (length // len(noise)) + 2)
        ms = len(noise) - length
        start = 0 if ms <= 0 else int(rng.integers(0, ms + 1))
        return noise[start:start + length].copy()

    def mix(self, clean: np.ndarray, snr: float, k: int,
            sample_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Mix clean signal with k noise types at the specified total SNR.
        WGN power is capped when its per-component SNR would fall below
        WGN_COMPONENT_SNR_MIN_DB (see _compute_noise_targets).
        All (snr, k) combinations are valid — no combos are skipped.
        """
        seed = abs((self.base_seed * 1_000_000
                    + (sample_idx % 100_000) * 100
                    + int(snr + 20) * 10 + k)) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        clean  = np.asarray(clean, dtype=np.float64).flatten()
        length = len(clean)

        available = list(self.noise_paths.keys())

        # k=1 special case: if WGN alone would be below floor, remove it.
        # (No other types to absorb the remaining power, so capping would
        #  silently change the actual SNR without updating the label.)
        if k == 1 and "WGN" in available and snr < WGN_COMPONENT_SNR_MIN_DB:
            available = [t for t in available if t != "WGN"]

        k = min(k, len(available))   # safety guard

        idx      = rng.choice(len(available), size=k, replace=False)
        selected = [available[i] for i in sorted(idx)]

        clean_pow = float(np.dot(clean, clean))
        if clean_pow < 1e-12:
            return clean.copy(), {"snr": snr, "k": k, "noise_types": selected,
                                   "scalars": [0.0] * k, "noise_paths": [],
                                   "wgn_capped": False}

        targets = _compute_noise_targets(clean_pow, snr, selected)
        wgn_capped = (
            "WGN" in selected and k >= 2 and
            (snr + 10.0 * np.log10(k)) < WGN_COMPONENT_SNR_MIN_DB
        )

        combined = np.zeros(length, dtype=np.float64)
        scalars, used_paths = [], []

        for ntype in selected:
            fi    = int(rng.integers(0, len(self.noise_paths[ntype])))
            npath = self.noise_paths[ntype][fi]
            used_paths.append(npath)
            nseg  = self._sample_seg(self.noise_cache[npath], length, rng)
            npow  = float(np.dot(nseg, nseg))
            s     = float(np.sqrt(targets[ntype] / npow)) if npow > 1e-12 else 0.0
            scalars.append(s)
            combined += s * nseg

        return clean + combined, {
            "snr": snr, "k": k,
            "noise_types": selected, "scalars": scalars,
            "noise_paths": used_paths, "wgn_capped": wgn_capped,
        }


# ============================================================================
# Main Generation
# ============================================================================
def generate_test_data(
    config:        Dict,
    segments_root: str,
    noise_root:    str,
    output_dir:    str,
    split:         str                 = "test",
    snr_grid:      Optional[List[int]] = None,
    k_range:       Optional[List[int]] = None,
    seed:          int                 = 42,
    force:         bool                = False,
):
    if snr_grid is None:
        snr_grid = [-15, -10, -5, 0, 5, 10, 15]
    if k_range is None:
        k_range  = [1, 2, 3, 4, 5]

    clip_cfg    = config.get("normalization", {}) or {}
    clip_range  = tuple(clip_cfg.get("clip_range", [-1.0, 1.0]))
    norm_method = str(clip_cfg.get("method",      "Q99"))
    norm_pct    = float(clip_cfg.get("percentile",  0.99))

    W = 70
    print(f"\n{'='*W}")
    print("Generate Test Data  v6.8.0")
    print(f"{'='*W}")
    print(f"WGN floor   : {WGN_COMPONENT_SNR_MIN_DB} dB per-component SNR")
    print(f"WGN capping : k>=2 → cap WGN, remaining to others (label preserved)")
    print(f"k=1 WGN     : excluded from pool if total SNR < {WGN_COMPONENT_SNR_MIN_DB} dB")
    print(f"All combos  : VALID — no skips → inference v2.0 sufficient")
    print(f"Scale from  : NOISY signal")
    print(f"SNR grid    : {snr_grid}")
    print(f"k range     : {k_range}")

    # Show WGN component SNR for each (snr, k) combination
    print(f"\nWGN per-component SNR table (bold = capping applies):")
    header = f"{'':>6}" + "".join(f"  k={k}" for k in k_range)
    print(header)
    for snr in snr_grid:
        row = f"{snr:>5}:"
        for k in k_range:
            comp = snr + 10.0 * np.log10(k)
            if k == 1 and snr < WGN_COMPONENT_SNR_MIN_DB:
                row += f"  excl"   # WGN excluded from pool
            elif comp < WGN_COMPONENT_SNR_MIN_DB:
                row += f" *{comp:+.1f}"  # capped
            else:
                row += f"  {comp:+.1f}"
        print(row)
    print("  * = WGN capped at floor, remaining to other types")
    print("  excl = WGN excluded from pool (k=1 only)")

    manifest_path = os.path.join(segments_root, "manifests", "segment_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df       = pd.read_csv(manifest_path)
    df_split = df[df["split"] == split].copy()
    if df_split.empty:
        raise ValueError(f"No data for split='{split}'")
    print(f"\nSegments : {len(df_split)}")

    os.makedirs(output_dir, exist_ok=True)
    mixer        = DeterministicNoiseMixer(noise_root, config, seed=seed)
    metadata_all = []

    # ── Per-combination files ─────────────────────────────────────────────────
    for snr in snr_grid:
        for k in k_range:
            combo_key   = f"snr{snr}_k{k}"
            output_file = os.path.join(output_dir, f"test_{combo_key}.npz")
            if os.path.exists(output_file) and not force:
                print(f"[SKIP] {combo_key} (exists)")
                continue

            wgn_comp = snr + 10.0 * np.log10(k)
            mode_note = ("WGN excl" if k == 1 and snr < WGN_COMPONENT_SNR_MIN_DB
                         else f"WGN*{wgn_comp:+.1f}dB" if wgn_comp < WGN_COMPONENT_SNR_MIN_DB
                         else f"WGN{wgn_comp:+.1f}dB")
            print(f"[Gen]  {combo_key}  [{mode_note}]")
            combo_data = []

            for idx, row in tqdm(df_split.iterrows(), total=len(df_split),
                                 desc=f"  {combo_key}", leave=False):
                raw_path = row["raw_path"]
                if not os.path.isabs(raw_path):
                    raw_path = os.path.join(segments_root, raw_path)

                clean_raw = np.load(raw_path).astype(np.float64)
                noisy_raw, info = mixer.mix(clean_raw, snr, k, idx)

                scale      = compute_scale_factor(noisy_raw, norm_method, norm_pct)
                clean_norm = normalize_clip(clean_raw, scale, clip_range)
                noisy_norm = normalize_clip(noisy_raw, scale, clip_range)
                pnames     = [os.path.basename(p) for p in info.get("noise_paths", [])]

                combo_data.append({
                    "segment_id":  row["segment_id"],
                    "dataset":     row["dataset"],
                    "clean":       clean_norm,
                    "noisy":       noisy_norm,
                    "scale":       scale,
                    "snr":         snr,
                    "k":           k,
                    "noise_types": "+".join(info["noise_types"]),
                    "noise_paths": "|".join(pnames),
                    "wgn_capped":  info["wgn_capped"],
                })
                metadata_all.append({
                    "segment_id":  row["segment_id"],
                    "dataset":     row["dataset"],
                    "snr": snr, "k": k,
                    "noise_types": "+".join(info["noise_types"]),
                    "noise_paths": "|".join(pnames),
                    "scale":       scale,
                    "wgn_capped":  info["wgn_capped"],
                })

            np.savez_compressed(output_file, data=combo_data)
            print(f"  ✓ {os.path.getsize(output_file)/(1024**2):.1f} MB")

    with open(os.path.join(output_dir, "test_metadata_all.csv"), "w", newline="") as f:
        if metadata_all:
            wr = csv.DictWriter(f, fieldnames=metadata_all[0].keys())
            wr.writeheader(); wr.writerows(metadata_all)

    # ── Combined file ─────────────────────────────────────────────────────────
    print(f"\n[Gen]  test_combined.npz")
    combined_file = os.path.join(output_dir, "test_combined.npz")

    if not os.path.exists(combined_file) or force:
        combined_data, combined_meta = [], []
        rng = np.random.default_rng(seed + 9999)

        for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc="  Combined"):
            snr = int(rng.choice(snr_grid))
            k   = int(rng.integers(1, 6))
            # All combos valid — no rejection loop needed

            raw_path = row["raw_path"]
            if not os.path.isabs(raw_path):
                raw_path = os.path.join(segments_root, raw_path)

            clean_raw = np.load(raw_path).astype(np.float64)
            noisy_raw, info = mixer.mix(clean_raw, snr, k, idx)

            scale      = compute_scale_factor(noisy_raw, norm_method, norm_pct)
            clean_norm = normalize_clip(clean_raw, scale, clip_range)
            noisy_norm = normalize_clip(noisy_raw, scale, clip_range)
            pnames     = [os.path.basename(p) for p in info.get("noise_paths", [])]

            combined_data.append({
                "segment_id":  row["segment_id"], "dataset": row["dataset"],
                "clean": clean_norm, "noisy": noisy_norm, "scale": scale,
                "snr": snr, "k": k,
                "noise_types": "+".join(info["noise_types"]),
                "noise_paths": "|".join(pnames),
                "wgn_capped":  info["wgn_capped"],
            })
            combined_meta.append({
                "segment_id": row["segment_id"], "dataset": row["dataset"],
                "snr": snr, "k": k,
                "noise_types": "+".join(info["noise_types"]),
                "noise_paths": "|".join(pnames),
                "scale": scale, "wgn_capped": info["wgn_capped"],
            })

        np.savez_compressed(combined_file, data=combined_data)
        with open(os.path.join(output_dir, "test_metadata_combined.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=combined_meta[0].keys())
            wr.writeheader(); wr.writerows(combined_meta)

        print(f"  ✓ {os.path.getsize(combined_file)/(1024**2):.1f} MB")
        k_counts = Counter(m["k"] for m in combined_meta)
        print(f"\n  k distribution (all k now cover all {len(snr_grid)} SNR values):")
        for kv in sorted(k_counts):
            print(f"    k={kv}: {k_counts[kv]:5d}")

    print(f"\n{'='*W}")
    print("✓ Done (v6.8.0)")
    print(f"  All {len(snr_grid)*len(k_range)} (snr,k) combos generated — no skips.")
    print(f"  Inference v2.0 is sufficient (no fair_snr_set needed).")
    print(f"{'='*W}\n")


def main():
    ap = argparse.ArgumentParser(description="Generate Test Data v6.8.0")
    ap.add_argument("--config",        required=True)
    ap.add_argument("--segments-root", required=True)
    ap.add_argument("--noise-root",    required=True)
    ap.add_argument("--output-dir",    required=True)
    ap.add_argument("--split",         default="test")
    ap.add_argument("--snr-grid",      type=int, nargs="+", default=None)
    ap.add_argument("--k-range",       type=int, nargs="+", default=None)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--force",         action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    generate_test_data(config, args.segments_root, args.noise_root,
                       args.output_dir, args.split, args.snr_grid,
                       args.k_range, args.seed, args.force)


if __name__ == "__main__":
    main()