#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Offline Test Data (v6.3.0)

Changes from v6.2.7:
- Store noise_paths in each sample so inference.py can determine
  pink vs brown subtype from Color noise filenames.
- DeterministicNoiseMixer also loads Color_pink_*.npy / Color_brown_*.npy
  under key "Color" (framework unchanged).

Critical policy (unchanged from v6.2.7):
- Scale computed from NOISY signal to prevent clipping at low SNR.
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml


# ============================================================================
# Normalization
# ============================================================================
def compute_scale_factor(x: np.ndarray, method: str = "Q99", percentile: float = 0.99) -> float:
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
    return max(scale, 1e-12)


def normalize_clip(x: np.ndarray, scale: float, clip_range: Tuple[float, float]) -> np.ndarray:
    scale = float(scale) if (scale is not None and np.isfinite(scale) and scale > 0) else 1.0
    y = (np.asarray(x, dtype=np.float32).reshape(-1) / scale)
    lo, hi = float(clip_range[0]), float(clip_range[1])
    return np.clip(y, lo, hi).astype(np.float32)


# ============================================================================
# Deterministic Noise Mixer
# ============================================================================
class DeterministicNoiseMixer:
    """
    Deterministic noise mixer for test set generation.
    Color noise files (Color_pink_*.npy / Color_brown_*.npy) are all loaded
    under key "Color" so the 5-type framework is unchanged.
    noise_paths are returned per sample so inference can identify pink/brown.
    """

    def __init__(self, noise_root: str, config: Dict, seed: int = 42):
        from glob import glob

        self.noise_root = noise_root
        self.config = config
        self.base_seed = seed
        self.noise_paths: Dict[str, List[str]] = {}
        self.noise_cache: Dict[str, np.ndarray] = {}

        ncfg = config.get("noise", {}) or {}
        self.k_min = int(ncfg.get("k_types", {}).get("min", 1))
        self.k_max = int(ncfg.get("k_types", {}).get("max", 5))

        self._load_noise_paths()
        self._cache_all_noise()

    def _load_noise_paths(self):
        from glob import glob
        noise_types = ["PLI", "ECG", "MOA", "WGN", "Color"]
        for ntype in noise_types:
            ndir = os.path.join(self.noise_root, ntype)
            if os.path.isdir(ndir):
                paths = sorted(glob(os.path.join(ndir, "*.npy")))
                if paths:
                    self.noise_paths[ntype] = paths
                    if ntype == "Color":
                        n_pink = sum(1 for p in paths if "_pink_" in os.path.basename(p))
                        n_brown = sum(1 for p in paths if "_brown_" in os.path.basename(p))
                        print(f"[NoiseMixer] Loaded {len(paths)} files for Color "
                              f"(pink={n_pink}, brown={n_brown})")
                    else:
                        print(f"[NoiseMixer] Loaded {len(paths)} files for {ntype}")

        if not self.noise_paths:
            raise ValueError(f"No noise files found in {self.noise_root}")

    def _cache_all_noise(self):
        print("[NoiseMixer] Caching noise files...")
        for ntype, paths in self.noise_paths.items():
            for p in paths:
                self.noise_cache[p] = np.load(p).astype(np.float64)
        print(f"[NoiseMixer] Cached {len(self.noise_cache)} files")

    def _sample_noise_segment(self, noise: np.ndarray, length: int, rng) -> np.ndarray:
        if len(noise) < length:
            repeats = (length // len(noise)) + 2
            noise = np.tile(noise, repeats)
        max_start = len(noise) - length
        if max_start <= 0:
            return noise[:length].copy()
        start = rng.integers(0, max_start + 1)
        return noise[start:start + length].copy()

    def mix(self, clean: np.ndarray, snr: float, k: int, sample_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Returns:
            noisy: mixed signal
            info: dict with snr, k, noise_types, scalars, noise_paths
                  noise_paths contains the exact file paths used,
                  allowing inference to distinguish Color_pink vs Color_brown.
        """
        seed = abs((self.base_seed * 1000000 + (sample_idx % 100000) * 100
                    + int(snr + 20) * 10 + k)) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        clean = np.asarray(clean, dtype=np.float64).flatten()
        length = len(clean)

        available = list(self.noise_paths.keys())
        if snr < -5 and "WGN" in available:
            available = [t for t in available if t != "WGN"]

        k = min(k, len(available))
        indices = rng.choice(len(available), size=k, replace=False)
        selected_types = [available[i] for i in sorted(indices)]

        clean_power = np.dot(clean, clean)
        if clean_power < 1e-12:
            return clean.copy(), {
                'snr': snr, 'k': k, 'noise_types': selected_types,
                'scalars': [0.0] * k, 'noise_paths': []
            }

        target_total_noise_power = clean_power / (10.0 ** (snr / 10.0))
        target_per_noise_power = target_total_noise_power / k

        combined_noise = np.zeros(length, dtype=np.float64)
        scalars = []
        used_paths = []

        for ntype in selected_types:
            nfile_idx = rng.integers(0, len(self.noise_paths[ntype]))
            npath = self.noise_paths[ntype][nfile_idx]
            used_paths.append(npath)
            noise_full = self.noise_cache[npath]
            noise_seg = self._sample_noise_segment(noise_full, length, rng)

            noise_power = np.dot(noise_seg, noise_seg)
            scalar = np.sqrt(target_per_noise_power / noise_power) if noise_power > 1e-12 else 0.0
            scalars.append(scalar)
            combined_noise += scalar * noise_seg

        noisy = clean + combined_noise

        return noisy, {
            'snr': snr,
            'k': k,
            'noise_types': selected_types,
            'scalars': scalars,
            'noise_paths': used_paths,   # ← NEW: full paths for pink/brown identification
        }


# ============================================================================
# Main Generation
# ============================================================================
def generate_test_data(
    config: Dict,
    segments_root: str,
    noise_root: str,
    output_dir: str,
    split: str = "test",
    snr_grid: Optional[List[int]] = None,
    k_range: Optional[List[int]] = None,
    seed: int = 42,
    force: bool = False
):
    if snr_grid is None:
        snr_grid = [-15, -10, -5, 0, 5, 10, 15]
    if k_range is None:
        k_range = [1, 2, 3, 4, 5]

    clip_cfg = config.get("normalization", {}) or {}
    clip_range = tuple(clip_cfg.get("clip_range", [-1.0, 1.0]))
    norm_method = str(clip_cfg.get("method", "Q99"))
    norm_pct = float(clip_cfg.get("percentile", 0.99))

    print(f"\n{'='*70}")
    print(f"Generating Test Data (v6.3.0)")
    print(f"{'='*70}")
    print(f"⚠️  CRITICAL: Scale from NOISY (not clean) - prevents clipping")
    print(f"✅  noise_paths stored per sample for pink/brown subtype analysis")
    print(f"Segments: {segments_root}")
    print(f"Noise: {noise_root}")
    print(f"Output: {output_dir}")
    print(f"SNR grid: {snr_grid}")
    print(f"k range: {k_range}")

    manifest_path = os.path.join(segments_root, "manifests", "segment_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df_split = df[df["split"] == split].copy()

    if df_split.empty:
        raise ValueError(f"No data for split='{split}'")

    print(f"Total segments: {len(df_split)}\n")

    os.makedirs(output_dir, exist_ok=True)
    mixer = DeterministicNoiseMixer(noise_root, config, seed=seed)

    metadata_all = []

    # Generate each (SNR, k) combination
    for snr in snr_grid:
        for k in k_range:
            if snr < -5 and k == 5:
                print(f"[SKIP] SNR={snr}, k={k}")
                continue

            combo_key = f"snr{snr}_k{k}"
            output_file = os.path.join(output_dir, f"test_{combo_key}.npz")

            if os.path.exists(output_file) and not force:
                print(f"[SKIP] {combo_key} (exists)")
                continue

            print(f"[Gen] {combo_key}")
            combo_data = []

            for idx, row in tqdm(df_split.iterrows(), total=len(df_split),
                                 desc=f"  {combo_key}", leave=False):
                raw_path = row["raw_path"]
                if not os.path.isabs(raw_path):
                    raw_path = os.path.join(segments_root, raw_path)

                clean_raw = np.load(raw_path).astype(np.float64)
                noisy_raw, info = mixer.mix(clean_raw, snr, k, idx)

                # CRITICAL: Scale from NOISY
                scale = compute_scale_factor(noisy_raw, method=norm_method, percentile=norm_pct)

                clean_norm = normalize_clip(clean_raw, scale, clip_range)
                noisy_norm = normalize_clip(noisy_raw, scale, clip_range)

                # Store relative paths for portability (basename only)
                path_basenames = [os.path.basename(p) for p in info.get("noise_paths", [])]

                combo_data.append({
                    "segment_id": row["segment_id"],
                    "dataset": row["dataset"],
                    "clean": clean_norm,
                    "noisy": noisy_norm,
                    "scale": scale,
                    "snr": snr,
                    "k": k,
                    "noise_types": "+".join(info["noise_types"]),
                    "noise_paths": "|".join(path_basenames),   # ← NEW
                })

                metadata_all.append({
                    "segment_id": row["segment_id"],
                    "dataset": row["dataset"],
                    "snr": snr,
                    "k": k,
                    "noise_types": "+".join(info["noise_types"]),
                    "noise_paths": "|".join(path_basenames),   # ← NEW
                    "scale": scale,
                })

            np.savez_compressed(output_file, data=combo_data)
            print(f"  ✓ {os.path.getsize(output_file) / (1024**2):.1f} MB")

    # Save metadata
    metadata_csv = os.path.join(output_dir, "test_metadata_all.csv")
    with open(metadata_csv, "w", newline="") as f:
        if metadata_all:
            writer = csv.DictWriter(f, fieldnames=metadata_all[0].keys())
            writer.writeheader()
            writer.writerows(metadata_all)

    # Generate combined file
    print(f"\n[Gen] test_combined.npz")
    combined_file = os.path.join(output_dir, "test_combined.npz")

    if not os.path.exists(combined_file) or force:
        combined_data = []
        combined_meta = []
        rng = np.random.default_rng(seed + 9999)

        for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc="  Combined"):
            snr = int(rng.choice(snr_grid))
            k = int(rng.integers(1, 6))
            if snr < -5:
                k = min(k, 4)

            raw_path = row["raw_path"]
            if not os.path.isabs(raw_path):
                raw_path = os.path.join(segments_root, raw_path)

            clean_raw = np.load(raw_path).astype(np.float64)
            noisy_raw, info = mixer.mix(clean_raw, snr, k, idx)

            # CRITICAL: Scale from NOISY
            scale = compute_scale_factor(noisy_raw, method=norm_method, percentile=norm_pct)

            clean_norm = normalize_clip(clean_raw, scale, clip_range)
            noisy_norm = normalize_clip(noisy_raw, scale, clip_range)

            path_basenames = [os.path.basename(p) for p in info.get("noise_paths", [])]

            combined_data.append({
                "segment_id": row["segment_id"],
                "dataset": row["dataset"],
                "clean": clean_norm,
                "noisy": noisy_norm,
                "scale": scale,
                "snr": snr,
                "k": k,
                "noise_types": "+".join(info["noise_types"]),
                "noise_paths": "|".join(path_basenames),   # ← NEW
            })

            combined_meta.append({
                "segment_id": row["segment_id"],
                "dataset": row["dataset"],
                "snr": snr,
                "k": k,
                "noise_types": "+".join(info["noise_types"]),
                "noise_paths": "|".join(path_basenames),   # ← NEW
                "scale": scale,
            })

        np.savez_compressed(combined_file, data=combined_data)

        combined_meta_csv = os.path.join(output_dir, "test_metadata_combined.csv")
        with open(combined_meta_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=combined_meta[0].keys())
            writer.writeheader()
            writer.writerows(combined_meta)

        print(f"  ✓ {os.path.getsize(combined_file) / (1024**2):.1f} MB")

        # Show distribution
        snr_counts = Counter(m["snr"] for m in combined_meta)
        k_counts = Counter(m["k"] for m in combined_meta)
        print(f"\n  SNR distribution:")
        for snr_v in sorted(snr_counts.keys()):
            print(f"    {snr_v:3d} dB: {snr_counts[snr_v]:5d} "
                  f"({100.0 * snr_counts[snr_v] / len(combined_meta):.1f}%)")

    print(f"\n{'='*70}")
    print(f"✓ Test Data Generation Complete")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Test Data (v6.3.0)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--segments-root", type=str, required=True)
    parser.add_argument("--noise-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--snr-grid", type=int, nargs="+", default=None)
    parser.add_argument("--k-range", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    generate_test_data(
        config,
        args.segments_root,
        args.noise_root,
        args.output_dir,
        args.split,
        args.snr_grid,
        args.k_range,
        args.seed,
        args.force
    )


if __name__ == "__main__":
    main()