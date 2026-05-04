#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ${CLEANSEMG_ROOT}/noise.py
"""
sEMG Noise Generation and Online Mixing Module
Version 6.3.0
${CLEANSEMG_ROOT}/noise.py

Changes from 6.2.1:
- ColorNoiseGenerator: saves files as Color_pink_i.npy / Color_brown_i.npy
  so inference can distinguish pink vs brown in per-type metrics.
- OnlineNoiseMixer: still loads ALL Color/* files under key "Color",
  so the 5-type framework (PLI/ECG/MOA/WGN/Color) is unchanged.
- mix() info dict now always includes "noise_paths" so inference can
  reconstruct pink/brown split from filename.
"""

import os
import random
import argparse
import json
import csv
from typing import List, Tuple, Dict, Optional
from glob import glob
from fractions import Fraction
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample_poly, iirnotch
import scipy.io
import wfdb


# ============================================================================
# Reproducibility
# ============================================================================
def seed_everything(seed: int) -> None:
    seed = int(seed) & 0xffffffff
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# Signal Processing Utilities
# ============================================================================
def apply_bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    min_len = max(64, 3 * order + 1)
    if x.size < min_len:
        return x
    nyq = fs / 2.0
    high_eff = min(high, nyq * 0.99)
    if high_eff <= low:
        return x
    b, a = butter(order, [low / nyq, high_eff / nyq], btype="band")
    return filtfilt(b, a, x)


def apply_lowpass_filter(x: np.ndarray, fs: float, cutoff: float = 200.0, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    min_len = max(64, 3 * order + 1)
    if x.size < min_len:
        return x
    nyq = fs / 2.0
    cutoff_eff = min(cutoff, nyq * 0.99)
    b, a = butter(order, cutoff_eff / nyq, btype="low")
    return filtfilt(b, a, x)


def apply_notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 128:
        return x
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, x)


def resample_signal_poly(x: np.ndarray, from_fs: int, to_fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0 or from_fs == to_fs:
        return x
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float64)


def ensure_length(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((L,), dtype=np.float64)
    if x.size >= L:
        return x[:L]
    rep = (L // x.size) + 2
    y = np.tile(x, rep)
    return y[:L]


# ============================================================================
# Config helpers
# ============================================================================
def get_output_base(config: Dict) -> str:
    root = config["paths"]["root"]
    base = config["paths"]["output"]["base"]
    return base if os.path.isabs(base) else os.path.join(root, base)


def _get_bp_params(config: Dict) -> Tuple[bool, float, float, int]:
    bp = config.get("preprocessing", {}).get("bandpass", {})
    enabled = bool(bp.get("enabled", True))
    low = float(bp.get("low_cutoff", 20.0))
    high = float(bp.get("high_cutoff", 500.0))
    order = int(bp.get("order", 4))
    return enabled, low, high, order


def _get_notch_params(config: Dict) -> Tuple[bool, float, float]:
    nc = config.get("preprocessing", {}).get("notch", {})
    enabled = bool(nc.get("enabled", True))
    freq = float(nc.get("freq_hz", 50.0))
    q = float(nc.get("q", 30.0))
    return enabled, freq, q


# ============================================================================
# ECG split helpers
# ============================================================================
def _ecg_split_file(config: Dict) -> str:
    out_base = get_output_base(config)
    d = os.path.join(out_base, "noise")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "ecg_split.json")


def get_or_create_ecg_test_ids(config: Dict, ecg_ids: List[int]) -> Tuple[List[int], List[int]]:
    split_path = _ecg_split_file(config)
    ecg_cfg = config.get("noise", {}).get("generation", {}).get("ecg", {})
    test_record_count = int(ecg_cfg.get("test_record_count", 4))
    seed = int(config.get("project", {}).get("random_seed", 12345))

    if os.path.exists(split_path):
        try:
            with open(split_path, "r") as f:
                obj = json.load(f)
            saved_test = obj.get("test_ids", None)
            saved_seed = obj.get("seed", None)
            saved_count = obj.get("test_record_count", None)
            if (isinstance(saved_test, list) and
                    isinstance(saved_seed, int) and saved_seed == seed and
                    isinstance(saved_count, int) and saved_count == test_record_count):
                test_ids = [int(x) for x in saved_test]
                train_ids = [rid for rid in ecg_ids if rid not in set(test_ids)]
                print(f"[ECG Split] Loaded from: {split_path}")
                return train_ids, test_ids
        except Exception as e:
            print(f"[WARN] Failed to load ECG split: {e}")

    print(f"[ECG Split] Creating new split (seed={seed}, test_records={test_record_count})")
    rng = random.Random(seed)
    test_ids = rng.sample(list(ecg_ids), int(test_record_count))
    train_ids = [rid for rid in ecg_ids if rid not in set(test_ids)]

    with open(split_path, "w") as f:
        json.dump({
            "seed": seed,
            "test_record_count": int(test_record_count),
            "test_ids": test_ids,
            "train_ids": train_ids,
        }, f, indent=2)

    print(f"  Saved to: {split_path}")
    print(f"  Train records: {len(train_ids)}, Test records: {len(test_ids)}")
    return train_ids, test_ids


# ============================================================================
# Noise Generators
# ============================================================================
class NoiseGenerator:
    def __init__(self, target_fs: int, time_length: int, config: Dict):
        self.target_fs = int(target_fs)
        self.time_length = int(time_length)
        self.L = self.target_fs * self.time_length

        bp_enabled, bp_low, bp_high, bp_order = _get_bp_params(config)
        self.bp_enabled = bp_enabled
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.bp_order = bp_order

    def _bp(self, x: np.ndarray) -> np.ndarray:
        if self.bp_enabled:
            return apply_bandpass_filter(
                x, fs=self.target_fs, low=self.bp_low, high=self.bp_high, order=self.bp_order
            )
        return np.asarray(x, dtype=np.float64).reshape(-1)


class PLIGenerator(NoiseGenerator):
    def generate(self, count: int, config: Dict) -> List[np.ndarray]:
        pli_cfg = config.get("noise", {}).get("generation", {}).get("pli", {})
        fundamentals = pli_cfg.get("fundamental_hz", [50])
        drift_enabled = bool(pli_cfg.get("drift_enabled", True))
        drift_range = float(pli_cfg.get("drift_range", 0.3))
        H = int(pli_cfg.get("harmonics", 5))
        alpha = float(pli_cfg.get("amplitude_decay", 1.0))
        all_harmonics = bool(pli_cfg.get("all_harmonics", True))

        t = np.arange(self.L, dtype=np.float64) / self.target_fs
        noises = []

        for _ in range(int(count)):
            f0 = float(random.choice(fundamentals))
            if drift_enabled:
                f0 += random.uniform(-drift_range, drift_range)

            y = np.zeros_like(t, dtype=np.float64)
            harmonics = list(range(1, H + 1)) if all_harmonics else list(range(1, 2 * H, 2))[:H]

            for k in harmonics:
                A_k = 1.0 / (k ** alpha)
                phi_k = 2 * np.pi * random.random()
                y += A_k * np.sin(2 * np.pi * (k * f0) * t + phi_k)

            if y.std() > 0:
                y = y / y.std()

            y = self._bp(y)
            y = ensure_length(y, self.L)
            noises.append(y)

        return noises


class WGNGenerator(NoiseGenerator):
    def generate(self, count: int) -> List[np.ndarray]:
        noises = []
        for _ in range(int(count)):
            x = np.random.normal(0, 1, self.L).astype(np.float64)
            x = self._bp(x)
            x = ensure_length(x, self.L)
            noises.append(x)
        return noises


class ColorNoiseGenerator(NoiseGenerator):
    def generate(self, count: int, config: Dict) -> List[Tuple[np.ndarray, str]]:
        """
        Returns list of (array, color_type) tuples where color_type is "pink" or "brown".
        This allows the caller to save files with type-specific names for later analysis.
        The noise framework still treats them all as "Color" during mixing.
        """
        color_cfg = config.get("noise", {}).get("generation", {}).get("color", {})
        types = color_cfg.get("types", ["pink", "brown"])
        sample_mode = str(color_cfg.get("sample_mode", "random")).lower()
        pink_alpha = float(color_cfg.get("pink_alpha", 1.0))
        brown_alpha = float(color_cfg.get("brown_alpha", 2.0))

        results = []
        N = self.L
        freqs = np.fft.rfftfreq(N, 1.0 / self.target_fs)
        freqs[0] = 1e-10

        for i in range(int(count)):
            if sample_mode == "alternating":
                noise_type = types[i % len(types)]
            else:
                noise_type = random.choice(types)

            alpha = pink_alpha if noise_type == "pink" else brown_alpha
            white = np.random.randn(N // 2 + 1) + 1j * np.random.randn(N // 2 + 1)
            colored = white / (freqs ** (alpha / 2.0))
            x = np.fft.irfft(colored, n=N).real.astype(np.float64)

            if x.std() > 0:
                x = x / x.std()

            x = self._bp(x)
            x = ensure_length(x, self.L)
            results.append((x, noise_type))

        return results


class ECGGenerator(NoiseGenerator):
    ECG_IDS = [16265, 16272, 16273, 16420, 16483, 16539, 16773, 16786, 16795,
               17052, 17453, 18177, 18184, 19088, 19090, 19093, 19140, 19830]

    def __init__(self, ecg_root: str, target_fs: int, time_length: int, config: Dict):
        super().__init__(target_fs=target_fs, time_length=time_length, config=config)
        self.ecg_root = ecg_root
        self.fs_src = 128
        self.seg_src_len = int(self.time_length * self.fs_src)

        notch_enabled, notch_freq, notch_q = _get_notch_params(config)
        self.notch_enabled = notch_enabled
        self.notch_freq = notch_freq
        self.notch_q = notch_q

    def _read_segment_128(self, rid: int, start_128: int) -> Optional[np.ndarray]:
        rec_path = os.path.join(self.ecg_root, str(rid))
        try:
            rec = wfdb.rdrecord(rec_path, sampfrom=int(start_128), sampto=int(start_128 + self.seg_src_len))
            p = getattr(rec, "p_signal", None)
            if p is None:
                return None
            x = p[:, 0].astype(np.float64)
            if x.size < self.seg_src_len:
                x = ensure_length(x, self.seg_src_len)
            return x
        except Exception:
            return None

    def _record_len_128(self, rid: int) -> Optional[int]:
        rec_path = os.path.join(self.ecg_root, str(rid))
        try:
            hdr = wfdb.rdheader(rec_path)
            return int(hdr.sig_len)
        except Exception:
            return None

    def _process_200s(self, x_128: np.ndarray) -> np.ndarray:
        x = resample_signal_poly(x_128, self.fs_src, self.target_fs)
        if self.notch_enabled:
            x = apply_notch_filter(x, fs=self.target_fs, freq=self.notch_freq, q=self.notch_q)
        x = apply_lowpass_filter(x, fs=self.target_fs, cutoff=200.0, order=4)
        x = self._bp(x)
        x = ensure_length(x, self.L)
        return x

    @staticmethod
    def _quick_stats(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return {
            "mean": float(np.mean(x)) if x.size else 0.0,
            "std": float(np.std(x)) if x.size else 0.0,
            "rms": float(np.sqrt(np.mean(x**2))) if x.size else 0.0,
            "maxabs": float(np.max(np.abs(x))) if x.size else 0.0,
        }

    def generate_cross_hour(
        self,
        ids_to_use: List[int],
        total_segments: int,
        segments_per_hour: int = 1,
        total_hours: int = 24,
        seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        rng = random.Random(int(seed) & 0xffffffff) if seed is not None else random.Random()
        manifest = []
        noises = []

        hours = list(range(int(total_hours)))
        plan = []
        for h in hours:
            for _ in range(int(segments_per_hour)):
                plan.append(h)
        if not plan:
            plan = [0]

        while len(plan) < int(total_segments):
            plan += plan
        plan = plan[:int(total_segments)]

        max_attempts = int(total_segments) * 10

        for attempt in range(max_attempts):
            if len(noises) >= int(total_segments):
                break

            h = plan[len(noises) % len(plan)]
            rid = rng.choice(ids_to_use)
            rec_len = self._record_len_128(rid)
            if rec_len is None:
                continue

            hour_len = int(3600 * self.fs_src)
            hour_start = int(h * hour_len)
            hour_end = int(min(rec_len, hour_start + hour_len))
            if hour_end - hour_start <= self.seg_src_len + 1:
                continue

            start_128 = rng.randint(hour_start, hour_end - self.seg_src_len)
            x_128 = self._read_segment_128(rid, start_128)
            if x_128 is None:
                continue

            x = self._process_200s(x_128)
            st = self._quick_stats(x)
            flagged = (st["std"] < 1e-6) or (st["maxabs"] < 1e-6) or (not np.isfinite(st["std"]))

            manifest.append({
                "index": len(manifest),
                "record_id": rid,
                "hour": int(h),
                "start_128": int(start_128),
                "flagged": 1 if flagged else 0,
                "status": "rejected" if flagged else "accepted",
                **st
            })

            if not flagged:
                noises.append(x)

        if len(noises) < int(total_segments):
            print(f"[WARN] ECG: Only generated {len(noises)}/{total_segments} valid segments")

        return noises, manifest

    def generate_random(
        self,
        ids_to_use: List[int],
        total_segments: int,
        seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        rng = random.Random(int(seed) & 0xffffffff) if seed is not None else random.Random()
        manifest = []
        noises = []

        max_attempts = int(total_segments) * 10

        for attempt in range(max_attempts):
            if len(noises) >= int(total_segments):
                break

            rid = rng.choice(ids_to_use)
            rec_len = self._record_len_128(rid)
            if rec_len is None or rec_len <= self.seg_src_len + 1:
                continue

            start_128 = rng.randint(0, rec_len - self.seg_src_len)
            x_128 = self._read_segment_128(rid, start_128)
            if x_128 is None:
                continue

            x = self._process_200s(x_128)
            st = self._quick_stats(x)
            flagged = (st["std"] < 1e-6) or (st["maxabs"] < 1e-6) or (not np.isfinite(st["std"]))

            manifest.append({
                "index": len(manifest),
                "record_id": rid,
                "hour": -1,
                "start_128": int(start_128),
                "flagged": 1 if flagged else 0,
                "status": "rejected" if flagged else "accepted",
                **st
            })

            if not flagged:
                noises.append(x)

        if len(noises) < int(total_segments):
            print(f"[WARN] ECG: Only generated {len(noises)}/{total_segments} valid segments")

        return noises, manifest


class MOAGenerator(NoiseGenerator):
    def __init__(self, moa_path: str, target_fs: int, time_length: int, config: Dict):
        super().__init__(target_fs=target_fs, time_length=time_length, config=config)
        self.moa_path = moa_path

        notch_enabled, notch_freq, notch_q = _get_notch_params(config)
        self.notch_enabled = notch_enabled
        self.notch_freq = notch_freq
        self.notch_q = notch_q

    def generate(self) -> List[np.ndarray]:
        mat_files = sorted(glob(os.path.join(self.moa_path, "*.mat")))
        if not mat_files:
            print(f"[WARN] No MOA mats in {self.moa_path}")
            return []

        noises = []
        for mp in mat_files:
            try:
                m = scipy.io.loadmat(mp)
                if "a" in m:
                    x = np.asarray(m["a"]).squeeze()
                else:
                    cand = None
                    for k, v in m.items():
                        if k.startswith("__"):
                            continue
                        vv = np.asarray(v)
                        if vv.ndim >= 1 and np.issubdtype(vv.dtype, np.number):
                            cand = vv.squeeze()
                            break
                    if cand is None:
                        continue
                    x = cand

                x = x.astype(np.float64).reshape(-1)
                if x.size < 256:
                    continue

                if self.notch_enabled:
                    x = apply_notch_filter(x, fs=2000, freq=self.notch_freq, q=self.notch_q)

                x = np.convolve(x, np.ones(51) / 51, mode="valid")[::2]

                if self.target_fs != 1000:
                    x = resample_signal_poly(x, 1000, self.target_fs)

                x = self._bp(x)

                if x.size < self.L:
                    seg = ensure_length(x, self.L)
                else:
                    s = random.randint(0, x.size - self.L)
                    seg = x[s:s + self.L]
                noises.append(seg.copy())

            except Exception as e:
                print(f"[WARN] MOA process fail {mp}: {e}")

        return noises


# ============================================================================
# Build noise library
# ============================================================================
def _save_manifest_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def build_noise_library(config: Dict, mode: str) -> None:
    is_train = (mode == "train")

    out_base = get_output_base(config)
    noise_dir = os.path.join(out_base, config["paths"]["output"][f"noise_{mode}"])
    os.makedirs(noise_dir, exist_ok=True)

    seed = int(config.get("project", {}).get("random_seed", 12345)) + (0 if is_train else 999)
    seed_everything(seed)

    src = config["paths"]["noise_sources"]
    target_fs = int(config["noise"]["generation"].get("target_fs", config["preprocessing"]["segmentation"]["target_fs"]))
    time_length = int(config["noise"]["generation"]["time_length"])
    counts = config["noise"]["generation"][f"{mode}_count"]

    bp_enabled, bp_low, bp_high, _ = _get_bp_params(config)
    notch_enabled, notch_freq, notch_q = _get_notch_params(config)

    print(f"\n{'='*80}")
    print(f"Building Noise Library ({mode}) - v6.3.0")
    print(f"{'='*80}")
    print(f"Output: {noise_dir}")
    print(f"seed={seed}")
    print(f"target_fs={target_fs}, time_length={time_length}s")
    print(f"Notch: {'ON' if notch_enabled else 'OFF'} freq={notch_freq}Hz q={notch_q}")
    print(f"Final bandpass: {('%.1f-%.1f Hz' % (bp_low, bp_high)) if bp_enabled else 'DISABLED'}")
    print(f"Noise Types: {config['noise']['types']}")
    print(f"NOTE: Color noise saved as Color_pink_*.npy / Color_brown_*.npy")

    # PLI
    print("\n[1/5] PLI")
    pli = PLIGenerator(target_fs=target_fs, time_length=time_length, config=config)
    pli_list = pli.generate(int(counts.get("PLI", 0)), config=config)
    d = os.path.join(noise_dir, "PLI")
    os.makedirs(d, exist_ok=True)
    for i, x in enumerate(pli_list):
        np.save(os.path.join(d, f"PLI_{i}.npy"), x.astype(np.float32))
    print(f"  Saved {len(pli_list)} files")

    # WGN
    print("\n[2/5] WGN")
    wgn = WGNGenerator(target_fs=target_fs, time_length=time_length, config=config)
    wgn_list = wgn.generate(int(counts.get("WGN", 0)))
    d = os.path.join(noise_dir, "WGN")
    os.makedirs(d, exist_ok=True)
    for i, x in enumerate(wgn_list):
        np.save(os.path.join(d, f"WGN_{i}.npy"), x.astype(np.float32))
    print(f"  Saved {len(wgn_list)} files")

    # Color (pink + brown - saved with type-specific filenames)
    print("\n[3/5] Color (pink/brown - type-specific filenames)")
    color = ColorNoiseGenerator(target_fs=target_fs, time_length=time_length, config=config)
    color_results = color.generate(int(counts.get("Color", 0)), config=config)
    d = os.path.join(noise_dir, "Color")
    os.makedirs(d, exist_ok=True)
    # Track per-type counts for informative output
    pink_count = 0
    brown_count = 0
    for i, (x, color_type) in enumerate(color_results):
        # Save with type in filename: Color_pink_0.npy, Color_brown_1.npy, etc.
        np.save(os.path.join(d, f"Color_{color_type}_{i}.npy"), x.astype(np.float32))
        if color_type == "pink":
            pink_count += 1
        else:
            brown_count += 1
    print(f"  Saved {len(color_results)} files (pink={pink_count}, brown={brown_count})")
    print(f"  OnlineNoiseMixer will load all as key='Color' (framework unchanged)")

    # ECG
    print("\n[4/5] ECG (cross-hour sampling, v6.3.0)")
    d = os.path.join(noise_dir, "ECG")
    os.makedirs(d, exist_ok=True)

    if os.path.exists(src["ecg"]):
        ecg = ECGGenerator(ecg_root=src["ecg"], target_fs=target_fs, time_length=time_length, config=config)

        train_ids, test_ids = get_or_create_ecg_test_ids(config, ecg.ECG_IDS)
        ids_to_use = train_ids if is_train else test_ids

        ecg_cfg = config.get("noise", {}).get("generation", {}).get("ecg", {})
        hourly = bool(ecg_cfg.get("hourly_sampling", True)) and is_train
        segments_per_hour = int(ecg_cfg.get("segments_per_hour", 1))
        total_hours = int(ecg_cfg.get("total_hours", 24))

        if is_train:
            total_segments = int(ecg_cfg.get("train_segments", 24))
        else:
            total_segments = int(ecg_cfg.get("test_segments", 4))

        print(f"  Mode: {'train' if is_train else 'test'}")
        print(f"  Record IDs ({len(ids_to_use)}): {ids_to_use}")
        print(f"  Total segments to generate: {total_segments}")

        if hourly:
            ecg_list, manifest = ecg.generate_cross_hour(
                ids_to_use=ids_to_use,
                total_segments=total_segments,
                segments_per_hour=segments_per_hour,
                total_hours=total_hours,
                seed=seed
            )
        else:
            ecg_list, manifest = ecg.generate_random(
                ids_to_use=ids_to_use,
                total_segments=total_segments,
                seed=seed
            )

        for i, x in enumerate(ecg_list):
            np.save(os.path.join(d, f"ECG_{i}.npy"), x.astype(np.float32))

        man_path = os.path.join(noise_dir, f"ECG_manifest_{mode}.csv")
        _save_manifest_csv(man_path, manifest)

        print(f"  hourly_sampling={hourly}")
        print(f"  Generated: {len(ecg_list)} segments")
        print(f"  Manifest: {man_path}")
    else:
        print(f"  [SKIP] ECG not found: {src['ecg']}")

    # MOA
    print("\n[5/5] MOA")
    d = os.path.join(noise_dir, "MOA")
    os.makedirs(d, exist_ok=True)
    moa_path = src.get(f"moa_{mode}", "")
    if moa_path and os.path.exists(moa_path):
        moa = MOAGenerator(moa_path=moa_path, target_fs=target_fs, time_length=time_length, config=config)
        moa_list = moa.generate()
        for i, x in enumerate(moa_list):
            np.save(os.path.join(d, f"MOA_{i}.npy"), x.astype(np.float32))
        print(f"  Saved {len(moa_list)} files")
    else:
        print(f"  [SKIP] MOA not found: {moa_path}")

    print(f"\n{'='*80}\nNoise Library Complete ({mode})\n{'='*80}")
    for ntype in ["PLI", "WGN", "Color", "ECG", "MOA"]:
        nd = os.path.join(noise_dir, ntype)
        if os.path.isdir(nd):
            nfiles = len(glob(os.path.join(nd, "*.npy")))
            print(f"  {ntype}: {nfiles} files")
            # For Color, show pink/brown breakdown
            if ntype == "Color":
                n_pink = len(glob(os.path.join(nd, "*_pink_*.npy")))
                n_brown = len(glob(os.path.join(nd, "*_brown_*.npy")))
                n_legacy = nfiles - n_pink - n_brown
                if n_pink or n_brown:
                    print(f"    → pink={n_pink}, brown={n_brown}", end="")
                    if n_legacy:
                        print(f", legacy(no-type)={n_legacy}", end="")
                    print()


# ============================================================================
# Online mixer
# ============================================================================
def _infer_color_subtype(path: str) -> str:
    """
    Infer whether a Color noise file is pink or brown from its filename.
    Returns "Pink", "Brown", or "Color" (for legacy files without type in name).
    """
    basename = os.path.basename(path).lower()
    if "_pink_" in basename:
        return "Pink"
    if "_brown_" in basename:
        return "Brown"
    return "Color"


class OnlineNoiseMixer:
    # The 5-type framework is unchanged. Color is still one type for mixing.
    NOISE_TYPES = ["PLI", "ECG", "MOA", "WGN", "Color"]

    def __init__(
        self,
        noise_root: str,
        config: Optional[Dict] = None,
        noise_types: Optional[List[str]] = None,
        cache_noise: bool = True,
        seed: Optional[int] = None
    ):
        self.noise_root = noise_root
        self.cfg = config or {}
        self.noise_types = noise_types or self.cfg.get("noise", {}).get("types", self.NOISE_TYPES)
        self.cache_noise = cache_noise

        self.noise_paths: Dict[str, List[str]] = {}
        self.noise_cache: Dict[str, np.ndarray] = {}

        self.stats = {
            "total_mixed": 0,
            "noise_type_counts": Counter(),
            "k_distribution": Counter(),
            "snr_values": defaultdict(list),
        }

        ncfg = self.cfg.get("noise", {})
        self.k_min = int(ncfg.get("k_types", {}).get("min", 1))
        self.k_max = int(ncfg.get("k_types", {}).get("max", 5))
        self.snr_train_min = float(ncfg.get("snr_train", {}).get("min", -15.0))
        self.snr_train_max = float(ncfg.get("snr_train", {}).get("max", 15.0))
        self.snr_train_wgn_min = float(ncfg.get("snr_train", {}).get("wgn_min", -5.0))
        self.snr_dist = str(ncfg.get("snr_train", {}).get("distribution", "uniform")).lower()

        self.snr_test_grid = list(ncfg.get("snr_test", {}).get("grid", [-15, -10, -5, 0, 5, 10, 15]))
        self.snr_test_grid_wgn = list(ncfg.get("snr_test", {}).get("grid_wgn", [-5, 0, 5, 10, 15]))

        self.base_seed = int(seed) & 0xffffffff if seed is not None else None

        if seed is not None:
            self.rng = random.Random(seed)
            self.rng_np = np.random.default_rng(seed)
        else:
            self.rng = random.Random()
            self.rng_np = np.random.default_rng()

        self._load_noise_paths()
        if cache_noise:
            self._cache_all()

    def _load_noise_paths(self):
        """
        Load noise paths. Color/* files (including Color_pink_*.npy and
        Color_brown_*.npy) are ALL loaded under key "Color" so the 5-type
        mixing framework is unchanged.
        """
        for nt in self.noise_types:
            d = os.path.join(self.noise_root, nt)
            if not os.path.isdir(d):
                continue
            ps = sorted(glob(os.path.join(d, "*.npy")))
            if ps:
                self.noise_paths[nt] = ps
        if not self.noise_paths:
            raise ValueError(f"No noise found under: {self.noise_root}")

    def _cache_all(self):
        print("[OnlineNoiseMixer] Caching noise files...")
        for _, ps in self.noise_paths.items():
            for p in ps:
                if p not in self.noise_cache:
                    self.noise_cache[p] = np.load(p).astype(np.float64)
        print(f"[OnlineNoiseMixer] Cached {len(self.noise_cache)} files")
        print(f"  Available types: {list(self.noise_paths.keys())}")
        for nt, ps in self.noise_paths.items():
            if nt == "Color":
                n_pink = sum(1 for p in ps if "_pink_" in os.path.basename(p))
                n_brown = sum(1 for p in ps if "_brown_" in os.path.basename(p))
                n_legacy = len(ps) - n_pink - n_brown
                extra = f" (pink={n_pink}, brown={n_brown}"
                if n_legacy:
                    extra += f", legacy={n_legacy}"
                extra += ")"
                print(f"    {nt}: {len(ps)} files{extra}")
            else:
                print(f"    {nt}: {len(ps)} files")

    def _get(self, path: str) -> np.ndarray:
        if path in self.noise_cache:
            return self.noise_cache[path]
        return np.load(path).astype(np.float64)

    @staticmethod
    def _sample_seg(x: np.ndarray, L: int, rng: random.Random) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < L:
            return ensure_length(x, L).copy()
        s = rng.randint(0, x.size - L)
        return x[s:s + L].copy()

    def _sample_snr_train(self, has_wgn: bool, rng_np: np.random.Generator) -> float:
        lo = self.snr_train_wgn_min if has_wgn else self.snr_train_min
        hi = self.snr_train_max
        if self.snr_dist == "uniform":
            return float(rng_np.uniform(lo, hi))
        return float(rng_np.uniform(lo, hi))

    def mix(
        self,
        clean: np.ndarray,
        k: Optional[int] = None,
        snr: Optional[float] = None,
        noise_types: Optional[List[str]] = None,
        mode: str = "train",
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            seed_use = int(seed) & 0xffffffff
            rng = random.Random(seed_use)
            rng_np = np.random.default_rng(seed_use)
        else:
            rng = self.rng
            rng_np = self.rng_np
            seed_use = self.base_seed

        clean = np.asarray(clean, dtype=np.float64).reshape(-1)
        L = clean.size
        if L == 0:
            return clean, {"snr": None, "k": 0, "noise_types": [], "scalars": [],
                           "noise_paths": [], "mode": mode, "has_wgn": False}

        available = list(self.noise_paths.keys())
        if not available:
            return clean, {"snr": None, "k": 0, "noise_types": [], "scalars": [],
                           "noise_paths": [], "mode": mode, "has_wgn": False}

        max_k = min(int(self.k_max), len(available))
        min_k = max(1, min(int(self.k_min), max_k))

        if k is None:
            k = rng.randint(min_k, max_k)
        else:
            k = max(min_k, min(int(k), max_k))

        if noise_types is not None:
            selected = [nt for nt in noise_types if nt in available]
            if len(selected) < k:
                rest = [nt for nt in available if nt not in selected]
                if len(rest) >= (k - len(selected)):
                    selected += rng.sample(rest, k - len(selected))
                else:
                    selected += rest
            selected = selected[:k]
        else:
            selected = rng.sample(available, k)

        has_wgn = ("WGN" in selected)

        if mode == "train":
            snr_val = self._sample_snr_train(has_wgn, rng_np) if snr is None else float(snr)
            lo = self.snr_train_wgn_min if has_wgn else self.snr_train_min
            hi = self.snr_train_max
            snr_val = float(np.clip(snr_val, lo, hi))
        else:
            grid = self.snr_test_grid_wgn if has_wgn else self.snr_test_grid
            if snr is None:
                snr_val = float(rng.choice(grid))
            else:
                snr_val = float(snr)
                lo = self.snr_train_wgn_min if has_wgn else self.snr_train_min
                hi = self.snr_train_max
                snr_val = float(np.clip(snr_val, lo, hi))

        clean_pow = float(np.dot(clean, clean))
        target_noise_pow = clean_pow / (10.0 ** (snr_val / 10.0)) if clean_pow > 0 else 0.0
        target_each = target_noise_pow / k if k > 0 else 0.0

        combined = np.zeros(L, dtype=np.float64)
        scalars, used_paths = [], []

        for nt in selected:
            p = rng.choice(self.noise_paths[nt])
            used_paths.append(p)
            nfull = self._get(p)
            nseg = self._sample_seg(nfull, L, rng)

            n_pow = float(np.dot(nseg, nseg))
            s = float(np.sqrt(target_each / n_pow)) if n_pow > 1e-12 else 0.0
            scalars.append(s)
            combined += s * nseg

        noisy = clean + combined

        self.stats["total_mixed"] += 1
        for nt in selected:
            self.stats["noise_type_counts"][nt] += 1
        self.stats["k_distribution"][k] += 1
        self.stats["snr_values"][mode].append(snr_val)

        info = {
            "snr": snr_val,
            "k": k,
            "noise_types": selected,       # e.g. ["Color", "WGN"]
            "scalars": scalars,
            "noise_paths": used_paths,     # e.g. [".../Color/Color_pink_3.npy", ...]
            "mode": mode,
            "has_wgn": has_wgn,
            "seed": seed_use
        }
        return noisy, info

    def get_statistics(self) -> Dict:
        stats_report = {
            "total_mixed": self.stats["total_mixed"],
            "noise_type_distribution": dict(self.stats["noise_type_counts"]),
            "k_distribution": dict(self.stats["k_distribution"]),
            "snr_statistics": {}
        }
        for mode, snr_list in self.stats["snr_values"].items():
            if snr_list:
                stats_report["snr_statistics"][mode] = {
                    "count": len(snr_list),
                    "min": float(np.min(snr_list)),
                    "max": float(np.max(snr_list)),
                    "mean": float(np.mean(snr_list)),
                    "std": float(np.std(snr_list)),
                }
        return stats_report

    def reset_statistics(self):
        self.stats = {
            "total_mixed": 0,
            "noise_type_counts": Counter(),
            "k_distribution": Counter(),
            "snr_values": defaultdict(list),
        }


class OnlineMixingDataset:
    def __init__(self, clean_data, noise_root: str, config: Optional[Dict] = None,
                 mode: str = "train", transform=None, seed: Optional[int] = None):
        self.clean_data = clean_data
        self.mode = mode
        self.transform = transform
        self.mixer = OnlineNoiseMixer(noise_root=noise_root, config=config, cache_noise=True, seed=seed)

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        clean = self.clean_data[idx]
        clean_np = clean.numpy() if hasattr(clean, "numpy") else np.asarray(clean)
        noisy_np, _ = self.mixer.mix(clean_np, mode=self.mode)

        if self.transform:
            return self.transform(noisy_np), self.transform(clean_np)

        try:
            import torch
            return torch.as_tensor(noisy_np, dtype=torch.float32), torch.as_tensor(clean_np, dtype=torch.float32)
        except ImportError:
            return noisy_np, clean_np


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="sEMG Noise Module v6.3.0")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.demo:
        out_base = get_output_base(config)
        noise_root = os.path.join(out_base, config["paths"]["output"]["noise_train"])
        if not os.path.isdir(noise_root):
            print(f"[ERROR] Noise library not found: {noise_root}")
            print("Run: python noise.py --mode both")
            return

        seed = int(config.get("project", {}).get("random_seed", 12345))
        mixer = OnlineNoiseMixer(noise_root=noise_root, config=config, cache_noise=True, seed=seed)
        clean = np.sin(2 * np.pi * 100 * np.linspace(0, 2, 2000, endpoint=False)).astype(np.float64)

        print("\n[Training mode]")
        for i in range(3):
            noisy, info = mixer.mix(clean, mode="train", seed=seed + i)
            print(f"  SNR={info['snr']:.2f} dB, k={info['k']}, types={info['noise_types']}")
            print(f"  paths={[os.path.basename(p) for p in info['noise_paths']]}")

        print("\n[Testing mode]")
        for snr in [-10, 0, 10]:
            noisy, info = mixer.mix(clean, mode="test", snr=snr, k=2, seed=seed + int(snr))
            print(f"  SNR={info['snr']:.2f} dB, k={info['k']}, types={info['noise_types']}")

        print("\n[Mixer Statistics]")
        print(json.dumps(mixer.get_statistics(), indent=2))
        return

    if args.mode in ["train", "both"]:
        build_noise_library(config, mode="train")
    if args.mode in ["test", "both"]:
        build_noise_library(config, mode="test")

    print("\n✓ Noise library generation complete!")


if __name__ == "__main__":
    main()