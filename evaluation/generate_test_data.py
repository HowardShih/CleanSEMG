#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate deterministic offline test data for sEMG denoising evaluation.

This script builds noisy test datasets from clean sEMG segments and a noise
library. For each clean segment, it mixes one or more noise types at a target
total SNR, normalizes the clean/noisy pair using the noisy signal scale, and
exports both per-condition and combined NPZ files.

Inputs
------
- Segment manifest:
    <segments_root>/manifests/segment_manifest.csv
- Clean raw segment files listed in the manifest.
- Noise library root containing subfolders such as:
    PLI, ECG, MOA, WGN, Color
- YAML configuration containing normalization settings.

Outputs
-------
- Per-condition files:
    test_snr<SNR>_k<K>.npz
- Combined mixed-condition file:
    test_combined.npz
- Metadata files:
    test_metadata_all.csv
    test_metadata_combined.csv

Mixing design
-------------
The total noise power is determined by the requested total SNR:

    total_noise_power = clean_power / 10^(total_snr / 10)

When k noise types are selected, the default behavior is equal power allocation
across all selected types. WGN has a configurable minimum per-component SNR
floor. If equal allocation would make WGN stronger than this floor, WGN power is
capped and the remaining noise power is redistributed across the other selected
noise types. This preserves the total SNR label while avoiding overly dominant
WGN components.

For k=1 with WGN below the floor, WGN is excluded from the candidate pool because
there is no other component available to absorb the remaining noise power.
"""

import argparse
import csv
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


WGN_COMPONENT_SNR_MIN_DB = -5.0


# ============================================================================
# Normalization
# ============================================================================

def compute_scale_factor(
    x: np.ndarray,
    method: str = "Q99",
    percentile: float = 0.99,
) -> float:
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


def normalize_clip(
    x: np.ndarray,
    scale: float,
    clip_range: Tuple[float, float],
) -> np.ndarray:
    s = (
        float(scale)
        if scale is not None and np.isfinite(scale) and scale > 0
        else 1.0
    )

    return np.clip(
        np.asarray(x, dtype=np.float32).reshape(-1) / s,
        float(clip_range[0]),
        float(clip_range[1]),
    ).astype(np.float32)


# ============================================================================
# Noise Power Allocation
# ============================================================================

def _compute_noise_targets(
    clean_pow: float,
    total_snr: float,
    selected: List[str],
) -> Dict[str, float]:
    """
    Compute target noise power for each selected noise type.

    WGN allocation rule:
    - Use equal allocation if WGN's equal-share component SNR is above the floor.
    - Cap WGN if equal allocation would place WGN below the floor.
    - Redistribute the remaining noise power to the other selected types.
    """
    k = len(selected)
    total_noise = clean_pow / (10.0 ** (total_snr / 10.0))

    if k <= 0:
        return {}

    if "WGN" not in selected or k <= 1:
        return {nt: total_noise / k for nt in selected}

    wgn_equal_pow = total_noise / k
    wgn_floor_pow = clean_pow / (10.0 ** (WGN_COMPONENT_SNR_MIN_DB / 10.0))

    if wgn_equal_pow <= wgn_floor_pow:
        return {nt: total_noise / k for nt in selected}

    remaining = total_noise - wgn_floor_pow

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
    Deterministic noise mixer for offline test generation.

    The mixer selects noise types and noise segments using a seed derived from
    the base seed, sample index, target SNR, and number of noise types. This
    makes generated test data reproducible across runs.
    """

    def __init__(self, noise_root: str, config: Dict, seed: int = 42):
        from glob import glob

        self.noise_root = noise_root
        self.base_seed = seed
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
                        np_ = sum(
                            1 for p in paths
                            if "_pink_" in os.path.basename(p)
                        )
                        nb = sum(
                            1 for p in paths
                            if "_brown_" in os.path.basename(p)
                        )
                        print(f"[Mixer] {ntype}: {len(paths)} (pink={np_}, brown={nb})")
                    else:
                        print(f"[Mixer] {ntype}: {len(paths)}")

        if not self.noise_paths:
            raise ValueError(f"No noise files in {self.noise_root}")

    def _cache_all(self):
        print("[Mixer] Caching noise files")

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

    def mix(
        self,
        clean: np.ndarray,
        snr: float,
        k: int,
        sample_idx: int,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Mix one clean signal with k noise types at the specified total SNR.

        WGN power is capped when its per-component SNR would fall below
        WGN_COMPONENT_SNR_MIN_DB. For k=1, WGN is excluded from the candidate
        pool when the requested total SNR is below the WGN floor.
        """
        seed = abs(
            self.base_seed * 1_000_000
            + (sample_idx % 100_000) * 100
            + int(snr + 20) * 10
            + k
        ) % (2**31 - 1)

        rng = np.random.default_rng(seed)

        clean = np.asarray(clean, dtype=np.float64).flatten()
        length = len(clean)

        available = list(self.noise_paths.keys())

        if k == 1 and "WGN" in available and snr < WGN_COMPONENT_SNR_MIN_DB:
            available = [t for t in available if t != "WGN"]

        k = min(k, len(available))

        idx = rng.choice(len(available), size=k, replace=False)
        selected = [available[i] for i in sorted(idx)]

        clean_pow = float(np.dot(clean, clean))

        if clean_pow < 1e-12:
            return clean.copy(), {
                "snr": snr,
                "k": k,
                "noise_types": selected,
                "scalars": [0.0] * k,
                "noise_paths": [],
                "wgn_capped": False,
            }

        targets = _compute_noise_targets(clean_pow, snr, selected)

        wgn_capped = (
            "WGN" in selected
            and k >= 2
            and (snr + 10.0 * np.log10(k)) < WGN_COMPONENT_SNR_MIN_DB
        )

        combined = np.zeros(length, dtype=np.float64)
        scalars = []
        used_paths = []

        for ntype in selected:
            fi = int(rng.integers(0, len(self.noise_paths[ntype])))
            npath = self.noise_paths[ntype][fi]
            used_paths.append(npath)

            nseg = self._sample_seg(self.noise_cache[npath], length, rng)
            npow = float(np.dot(nseg, nseg))

            s = float(np.sqrt(targets[ntype] / npow)) if npow > 1e-12 else 0.0
            scalars.append(s)

            combined += s * nseg

        return clean + combined, {
            "snr": snr,
            "k": k,
            "noise_types": selected,
            "scalars": scalars,
            "noise_paths": used_paths,
            "wgn_capped": wgn_capped,
        }


# ============================================================================
# Test Data Generation
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
    force: bool = False,
):
    if snr_grid is None:
        snr_grid = [-15, -10, -5, 0, 5, 10, 15]

    if k_range is None:
        k_range = [1, 2, 3, 4, 5]

    clip_cfg = config.get("normalization", {}) or {}
    clip_range = tuple(clip_cfg.get("clip_range", [-1.0, 1.0]))
    norm_method = str(clip_cfg.get("method", "Q99"))
    norm_pct = float(clip_cfg.get("percentile", 0.99))

    W = 70

    print(f"\n{'=' * W}")
    print("Generate Offline Test Data")
    print(f"{'=' * W}")
    print(f"WGN floor   : {WGN_COMPONENT_SNR_MIN_DB} dB per-component SNR")
    print("WGN capping : k>=2 -> cap WGN and redistribute remaining power")
    print(f"k=1 WGN     : excluded from pool if total SNR < {WGN_COMPONENT_SNR_MIN_DB} dB")
    print("Scale from  : noisy signal")
    print(f"SNR grid    : {snr_grid}")
    print(f"k range     : {k_range}")

    print("\nWGN per-component SNR table:")
    header = f"{'':>6}" + "".join(f"  k={k}" for k in k_range)
    print(header)

    for snr in snr_grid:
        row = f"{snr:>5}:"

        for k in k_range:
            comp = snr + 10.0 * np.log10(k)

            if k == 1 and snr < WGN_COMPONENT_SNR_MIN_DB:
                row += "  excl"
            elif comp < WGN_COMPONENT_SNR_MIN_DB:
                row += f" *{comp:+.1f}"
            else:
                row += f"  {comp:+.1f}"

        print(row)

    print("  * = WGN capped at the floor and remaining power is redistributed")
    print("  excl = WGN excluded from the candidate pool for k=1")

    manifest_path = os.path.join(
        segments_root,
        "manifests",
        "segment_manifest.csv",
    )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df_split = df[df["split"] == split].copy()

    if df_split.empty:
        raise ValueError(f"No data for split='{split}'")

    print(f"\nSegments : {len(df_split)}")

    os.makedirs(output_dir, exist_ok=True)

    mixer = DeterministicNoiseMixer(noise_root, config, seed=seed)
    metadata_all = []

    for snr in snr_grid:
        for k in k_range:
            combo_key = f"snr{snr}_k{k}"
            output_file = os.path.join(output_dir, f"test_{combo_key}.npz")

            if os.path.exists(output_file) and not force:
                print(f"[SKIP] {combo_key} (exists)")
                continue

            wgn_comp = snr + 10.0 * np.log10(k)

            if k == 1 and snr < WGN_COMPONENT_SNR_MIN_DB:
                mode_note = "WGN excluded"
            elif wgn_comp < WGN_COMPONENT_SNR_MIN_DB:
                mode_note = f"WGN capped ({wgn_comp:+.1f} dB equal-share)"
            else:
                mode_note = f"WGN equal-share ({wgn_comp:+.1f} dB)"

            print(f"[Gen]  {combo_key}  [{mode_note}]")

            combo_data = []

            for idx, row in tqdm(
                df_split.iterrows(),
                total=len(df_split),
                desc=f"  {combo_key}",
                leave=False,
            ):
                raw_path = row["raw_path"]

                if not os.path.isabs(raw_path):
                    raw_path = os.path.join(segments_root, raw_path)

                clean_raw = np.load(raw_path).astype(np.float64)
                noisy_raw, info = mixer.mix(clean_raw, snr, k, idx)

                scale = compute_scale_factor(noisy_raw, norm_method, norm_pct)
                clean_norm = normalize_clip(clean_raw, scale, clip_range)
                noisy_norm = normalize_clip(noisy_raw, scale, clip_range)

                pnames = [
                    os.path.basename(p)
                    for p in info.get("noise_paths", [])
                ]

                combo_data.append(
                    {
                        "segment_id": row["segment_id"],
                        "dataset": row["dataset"],
                        "clean": clean_norm,
                        "noisy": noisy_norm,
                        "scale": scale,
                        "snr": snr,
                        "k": k,
                        "noise_types": "+".join(info["noise_types"]),
                        "noise_paths": "|".join(pnames),
                        "wgn_capped": info["wgn_capped"],
                    }
                )

                metadata_all.append(
                    {
                        "segment_id": row["segment_id"],
                        "dataset": row["dataset"],
                        "snr": snr,
                        "k": k,
                        "noise_types": "+".join(info["noise_types"]),
                        "noise_paths": "|".join(pnames),
                        "scale": scale,
                        "wgn_capped": info["wgn_capped"],
                    }
                )

            np.savez_compressed(output_file, data=combo_data)
            print(f"  Saved: {os.path.getsize(output_file) / (1024**2):.1f} MB")

    metadata_all_path = os.path.join(output_dir, "test_metadata_all.csv")

    with open(metadata_all_path, "w", newline="") as f:
        if metadata_all:
            wr = csv.DictWriter(f, fieldnames=metadata_all[0].keys())
            wr.writeheader()
            wr.writerows(metadata_all)

    print("\n[Gen]  test_combined.npz")

    combined_file = os.path.join(output_dir, "test_combined.npz")

    if not os.path.exists(combined_file) or force:
        combined_data = []
        combined_meta = []
        rng = np.random.default_rng(seed + 9999)

        for idx, row in tqdm(
            df_split.iterrows(),
            total=len(df_split),
            desc="  Combined",
        ):
            snr = int(rng.choice(snr_grid))
            k = int(rng.integers(1, 6))

            raw_path = row["raw_path"]

            if not os.path.isabs(raw_path):
                raw_path = os.path.join(segments_root, raw_path)

            clean_raw = np.load(raw_path).astype(np.float64)
            noisy_raw, info = mixer.mix(clean_raw, snr, k, idx)

            scale = compute_scale_factor(noisy_raw, norm_method, norm_pct)
            clean_norm = normalize_clip(clean_raw, scale, clip_range)
            noisy_norm = normalize_clip(noisy_raw, scale, clip_range)

            pnames = [
                os.path.basename(p)
                for p in info.get("noise_paths", [])
            ]

            combined_data.append(
                {
                    "segment_id": row["segment_id"],
                    "dataset": row["dataset"],
                    "clean": clean_norm,
                    "noisy": noisy_norm,
                    "scale": scale,
                    "snr": snr,
                    "k": k,
                    "noise_types": "+".join(info["noise_types"]),
                    "noise_paths": "|".join(pnames),
                    "wgn_capped": info["wgn_capped"],
                }
            )

            combined_meta.append(
                {
                    "segment_id": row["segment_id"],
                    "dataset": row["dataset"],
                    "snr": snr,
                    "k": k,
                    "noise_types": "+".join(info["noise_types"]),
                    "noise_paths": "|".join(pnames),
                    "scale": scale,
                    "wgn_capped": info["wgn_capped"],
                }
            )

        np.savez_compressed(combined_file, data=combined_data)

        combined_meta_path = os.path.join(
            output_dir,
            "test_metadata_combined.csv",
        )

        with open(combined_meta_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=combined_meta[0].keys())
            wr.writeheader()
            wr.writerows(combined_meta)

        print(f"  Saved: {os.path.getsize(combined_file) / (1024**2):.1f} MB")

        k_counts = Counter(m["k"] for m in combined_meta)

        print("\n  k distribution:")
        for kv in sorted(k_counts):
            print(f"    k={kv}: {k_counts[kv]:5d}")

    print(f"\n{'=' * W}")
    print("Done")
    print(f"  Generated {len(snr_grid) * len(k_range)} requested (SNR, k) conditions.")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * W}\n")


def main():
    ap = argparse.ArgumentParser(description="Generate deterministic offline test data")
    ap.add_argument("--config", required=True)
    ap.add_argument("--segments-root", required=True)
    ap.add_argument("--noise-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--snr-grid", type=int, nargs="+", default=None)
    ap.add_argument("--k-range", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    with open(args.config) as f:
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
        args.force,
    )


if __name__ == "__main__":
    main()