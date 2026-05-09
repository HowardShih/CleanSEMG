#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ${CLEANSEMG_ROOT}/downstream_tasks/stcnet/prepare_denoised_mat_baseline_model.py

Prepare denoised MAT files for STCNet downstream evaluation using waveform
baseline denoisers.

Supported waveform-in / waveform-out models:
  - CNN_waveform
  - FCN
  - MSEMG
  - TrustEMGNet_UNetonly
  - TrustEMGNet_DM
  - TrustEMGNet_RM
  - TrustEMGNet_skipall_DM
  - TrustEMGNet_skipall_RM
  - TrustEMGNet_LSTM_DM
  - TrustEMGNet_LSTM_RM

Baseline waveform denoisers use the same segment interface:
  Input:  [B, L=2000] noisy_norm
  Output: [B, L=2000] denoised_norm

The script reconstructs noisy and denoised DB2 MAT files by placing
segment-level noisy/denoised predictions back onto a 1000 Hz EMG canvas.
The resulting MAT files can be used by the STCNet preprocessing pipeline.

Usage:
  python prepare_denoised_mat_baseline_model.py \
      --model-name FCN \
      --model-path /path/to/FCN_best.pth \
      --db2-root /path/to/DB2 \
      --test-npz /path/to/test_combined.npz \
      --qc-index /path/to/qc_index.csv \
      --output-noisy ./outputs/noisy_data_baseline/DB2 \
      --output-denoised ./outputs/denoised_data_baseline_FCN/DB2 \
      --device cuda
"""

import argparse
import math
import os
import sys
import time
from fractions import Fraction
from glob import glob

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm


# ============================================================================
# Path Configuration
# ============================================================================

SEMG_ROOT = "${CLEANSEMG_ROOT}"
sys.path.insert(0, SEMG_ROOT)
sys.path.insert(0, os.path.join(SEMG_ROOT, "baseline_models"))


# ============================================================================
# Signal Processing
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
        [
            resample_poly(emg[:, ch], frac.numerator, frac.denominator)
            for ch in range(emg.shape[1])
        ],
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
        np.round(np.arange(N_out) * (N_in / N_out)).astype(int),
        0,
        N_in - 1,
    )

    return labels[idx]


def compute_trial_len_1k(n_raw, from_fs, to_fs):
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)
    return math.ceil(n_raw * frac.numerator / frac.denominator)


# ============================================================================
# MAT I/O
# ============================================================================

def load_emg_mat(mat_path):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def _pick(candidates):
        for k in candidates:
            if k in m:
                return k
        return None

    k_emg = _pick(["emg", "EMG"])
    k_sti = _pick(["stimulus", "Stimulus"])
    k_rest = _pick(["restimulus", "restStimulus"])
    k_rep = _pick(["repetition", "Repetition"])

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
        "emg": emg,
        "N": N,
        "C": C,
        "stimulus": _align(_proc(k_sti), N),
        "restimulus": _align(_proc(k_rest), N),
        "repetition": _align(_proc(k_rep), N),
        "raw_mat": m,
    }


def save_mat(output_path, raw_mat, emg_1k, stim_1k, rest_1k, rep_1k):
    save_dict = {k: v for k, v in raw_mat.items() if not k.startswith("__")}

    save_dict["preprocessed_emg"] = emg_1k.astype(np.float64)
    save_dict["emg"] = emg_1k.astype(np.float64)

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
# QC Index and Segment Lookup
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

    print(
        f"[QC Index] {len(qc_map)} files, "
        f"{sum(len(v) for v in qc_map.values())} QC-passed (trial,ch)"
    )

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
    n_parsed = 0
    n_no_stem = 0
    n_parse_fail = 0

    for i in range(len(data)):
        item = data[i]
        segment_id = str(item.get("segment_id", ""))
        parts = segment_id.split("_")

        try:
            subj_id = int(parts[1])
            ch = int(parts[2].replace("ch", ""))
            trial_id = int(parts[3].replace("t", ""))
            seg_idx = int(parts[4].replace("seg", ""))
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
            "snr": float(item.get("snr", np.nan)),
            "k": int(item.get("k", -1)),
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
        ch = int(row["ch"])

        if tid not in trials:
            trials[tid] = {
                "trial_start_raw": int(row["trial_start_raw"]),
                "trial_end_raw": int(row["trial_end_raw"]),
                "fs_raw": int(row["fs_raw"]) if "fs_raw" in qc_df.columns else 2000,
                "channels": set(),
            }

        trials[tid]["channels"].add(ch)

    return trials


# ============================================================================
# Baseline Waveform Denoiser
# ============================================================================

class BaselineDenoiser:
    """
    Wrapper for waveform baseline denoisers.

    All baseline models share the same interface:
      Input:  [B, L] float32
      Output: [B, L] float32

    The models process one complete segment per forward pass.
    CNN_waveform internally splits a 2000-sample segment into
    10 chunks of 200 samples, so the input length must remain 2000.
    """

    def __init__(self, model_name: str, model_path: str, device: str = "cuda"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        from baseline_models import BASELINE_MODEL_REGISTRY

        if model_name not in BASELINE_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {sorted(BASELINE_MODEL_REGISTRY.keys())}"
            )

        model_cls = BASELINE_MODEL_REGISTRY[model_name]
        self.model = model_cls().to(self.device)

        state = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break

        self.model.load_state_dict(state)
        self.model.eval()

        self.model_name = model_name

        print(f"[BaselineDenoiser] model={model_name}  device={self.device}")
        print(f"  Loaded: {model_path}")

    @torch.no_grad()
    def denoise_batch(self, noisy_batch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        noisy_batch:
            Normalized noisy waveforms with shape [B, L].

        Returns
        -------
        np.ndarray
            Denoised normalized waveforms with shape [B, L].
        """
        noisy_t = torch.from_numpy(noisy_batch).float().to(self.device)
        pred_t = self.model(noisy_t)
        return pred_t.cpu().numpy().astype(np.float32)


# ============================================================================
# Optional Visualization
# ============================================================================

def _maybe_plot(
    vis_output_dir,
    rel,
    plot_id,
    baseline_seg,
    noisy_seg,
    denoised_seg,
    model_name,
    snr=None,
):
    if vis_output_dir is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(vis_output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(baseline_seg, label="baseline", lw=0.8, color="steelblue")
    axes[0].plot(noisy_seg, label="noisy", lw=0.6, color="orange", alpha=0.7)
    axes[0].plot(denoised_seg, label="denoised", lw=0.8, color="crimson")

    title = f"{os.path.basename(rel)} | plot#{plot_id} | {model_name}"

    if snr is not None and not (isinstance(snr, float) and np.isnan(snr)):
        title += f" | SNR={snr:.1f}dB"

    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_xlabel("samples (@1kHz)")

    tail_start = max(0, len(baseline_seg) - 300)

    axes[1].plot(
        baseline_seg[tail_start:],
        label="baseline",
        lw=1.0,
        color="steelblue",
    )
    axes[1].plot(
        noisy_seg[tail_start:],
        label="noisy",
        lw=0.8,
        color="orange",
        alpha=0.7,
    )
    axes[1].plot(
        denoised_seg[tail_start:],
        label="denoised",
        lw=1.0,
        color="crimson",
    )

    axes[1].set_title("Last 300ms")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].set_xlabel("samples (offset)")

    out_png = os.path.join(
        vis_output_dir,
        f"{os.path.basename(rel)}__{plot_id:02d}.png",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================================
# Main Processing
# ============================================================================

def process_db2(
    model_name,
    model_path,
    db2_root,
    test_npz_path,
    qc_index_path,
    output_dir_noisy,
    output_dir_denoised,
    device="cuda",
    max_files=None,
    skip_existing=True,
    vis_output_dir=None,
    vis_n_per_file=2,
    batch_size=16,
    seg_len_s=2.0,
    target_fs=1000,
):
    print(f"\n{'=' * 70}")
    print(f"STCNet Denoised MAT — Baseline Model: {model_name}")
    print(f"{'=' * 70}")
    print(f"DB2 Root:         {db2_root}")
    print(f"Model path:       {model_path}")
    print(f"Output Noisy:     {output_dir_noisy}")
    print(f"Output Denoised:  {output_dir_denoised}")
    print(f"{'=' * 70}\n")

    qc_map, qc_df = load_qc_index(qc_index_path)
    seg_lookup = load_segment_lookup(test_npz_path, qc_df)

    print(f"\nLoading baseline denoiser: {model_name}")
    denoiser = BaselineDenoiser(model_name, model_path, device=device)

    mat_files = sorted(glob(os.path.join(db2_root, "**/*.mat"), recursive=True))

    if max_files is not None:
        mat_files = mat_files[:max_files]

    print(f"\nFound {len(mat_files)} .mat files")

    pts = int(round(seg_len_s * target_fs))

    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "segs_placed": 0,
        "segs_missing": 0,
        "rms_warn": 0,
    }

    all_rms_ratios_denoised = []
    all_rms_ratios_noisy = []

    for mat_path in tqdm(mat_files, desc=f"Processing DB2 [{model_name}]"):
        rel = os.path.relpath(mat_path, db2_root)
        mat_stem = os.path.splitext(os.path.basename(rel))[0]

        out_noisy = os.path.join(output_dir_noisy, rel)
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
        _N_raw, C = emg_raw.shape
        fs_raw = 2000

        emg_bp_whole = apply_bandpass_filter(emg_raw, fs=float(fs_raw))
        emg_1k_whole = resample_emg_2d(emg_bp_whole, fs_raw, target_fs)
        N_1k = emg_1k_whole.shape[0]

        canvas_noisy = emg_1k_whole.copy()
        canvas_denoised = emg_1k_whole.copy()

        trial_info = build_trial_info_from_qc(qc_df, rel)

        pending = []
        plots_done = 0

        for trial_id, tinfo in trial_info.items():
            s_raw = int(tinfo["trial_start_raw"])
            e_raw = int(tinfo["trial_end_raw"])
            fs = int(tinfo["fs_raw"]) if tinfo.get("fs_raw") is not None else fs_raw

            trial_start_1k = int(round(s_raw * target_fs / fs))
            n_raw_trial = e_raw - s_raw + 1
            trial_len_1k = compute_trial_len_1k(n_raw_trial, fs, target_fs)

            for ch in sorted(list(tinfo["channels"])):
                seg_idx = 0
                s0 = 0

                while s0 + pts <= trial_len_1k:
                    key = (mat_stem, ch, trial_id, seg_idx)
                    canvas_seg_start = trial_start_1k + s0

                    if key in seg_lookup:
                        entry = seg_lookup[key]

                        if canvas_seg_start + pts <= N_1k:
                            baseline_seg = emg_1k_whole[
                                canvas_seg_start: canvas_seg_start + pts,
                                ch,
                            ].copy()

                            pending.append(
                                (
                                    canvas_seg_start,
                                    ch,
                                    entry["noisy"],
                                    entry["scale"],
                                    entry.get("snr", np.nan),
                                    baseline_seg,
                                    rel,
                                )
                            )
                    else:
                        stats["segs_missing"] += 1

                    seg_idx += 1
                    s0 += pts

        if pending:
            noisy_norms = np.stack([p[2] for p in pending], axis=0)
            denoised_norms = np.zeros_like(noisy_norms)

            for i in range(0, len(pending), batch_size):
                sub = noisy_norms[i: i + batch_size]
                denoised_norms[i: i + batch_size] = denoiser.denoise_batch(sub)

            for idx, (
                seg_start,
                ch,
                noisy_norm,
                scale,
                snr,
                baseline_seg,
                rel_local,
            ) in enumerate(pending):
                seg_end = seg_start + pts

                noisy_physical = noisy_norm.astype(np.float64) * float(scale)
                denoised_physical = denoised_norms[idx].astype(np.float64) * float(scale)

                canvas_noisy[seg_start:seg_end, ch] = noisy_physical
                canvas_denoised[seg_start:seg_end, ch] = denoised_physical

                stats["segs_placed"] += 1

                rms_b = float(np.sqrt(np.mean(baseline_seg ** 2)))

                if rms_b > 1e-12:
                    rms_d = float(np.sqrt(np.mean(denoised_physical ** 2)))
                    rms_n = float(np.sqrt(np.mean(noisy_physical ** 2)))

                    all_rms_ratios_denoised.append(rms_d / rms_b)
                    all_rms_ratios_noisy.append(rms_n / rms_b)

                    if rms_n / rms_b > 5.0 or rms_n / rms_b < 0.2:
                        stats["rms_warn"] += 1

                if vis_output_dir is not None and plots_done < int(vis_n_per_file):
                    _maybe_plot(
                        vis_output_dir,
                        rel_local,
                        plots_done,
                        baseline_seg,
                        noisy_physical,
                        denoised_physical,
                        model_name=model_name,
                        snr=snr,
                    )
                    plots_done += 1

        stim_1k = resample_labels(raw_data["stimulus"], fs_raw, target_fs)
        rest_1k = resample_labels(raw_data["restimulus"], fs_raw, target_fs)
        rep_1k = resample_labels(raw_data["repetition"], fs_raw, target_fs)

        save_mat(out_noisy, raw_data["raw_mat"], canvas_noisy, stim_1k, rest_1k, rep_1k)
        save_mat(
            out_denoised,
            raw_data["raw_mat"],
            canvas_denoised,
            stim_1k,
            rest_1k,
            rep_1k,
        )

        stats["processed"] += 1

    print(f"\n{'=' * 70}")
    print(f"DONE — {model_name}")
    print(f"{'=' * 70}")
    print(f"  Processed:        {stats['processed']}")
    print(f"  Skipped:          {stats['skipped']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  Segments placed:  {stats['segs_placed']}")
    print(f"  Segments missing: {stats['segs_missing']}")

    if stats["rms_warn"] > 0:
        print(f"  [WARN] RMS ratio outliers: {stats['rms_warn']}")

    if all_rms_ratios_denoised:
        rd = np.array(all_rms_ratios_denoised)
        rn = np.array(all_rms_ratios_noisy)

        print("\n  [Amplitude Diagnostics] Denoised / Baseline RMS ratio:")
        print(
            f"    mean={rd.mean():.4f}  median={np.median(rd):.4f}  "
            f"std={rd.std():.4f}  min={rd.min():.4f}  max={rd.max():.4f}"
        )

        print("  [Amplitude Diagnostics] Noisy / Baseline RMS ratio:")
        print(
            f"    mean={rn.mean():.4f}  median={np.median(rn):.4f}  "
            f"std={rn.std():.4f}  min={rn.min():.4f}  max={rn.max():.4f}"
        )

    print(f"\n  Noisy:    {output_dir_noisy}")
    print(f"  Denoised: {output_dir_denoised}")

    if vis_output_dir:
        print(f"  Plots:    {vis_output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare denoised .mat files for STCNet using waveform baseline models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python prepare_denoised_mat_baseline_model.py \
      --model-name FCN \
      --model-path /data/.../FCN_best.pth \
      --db2-root /data/.../DB2 \
      --test-npz /data/.../test_combined.npz \
      --qc-index /data/.../qc_index.csv \
      --output-noisy ./outputs/noisy_data_baseline/DB2 \
      --output-denoised ./outputs/denoised_data_baseline_FCN/DB2 \
      --device cuda
        """,
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name, e.g. FCN / CNN_waveform / MSEMG / TrustEMGNet_DM",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained .pth checkpoint",
    )
    parser.add_argument("--db2-root", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--qc-index", required=True)
    parser.add_argument("--output-noisy", required=True)
    parser.add_argument("--output-denoised", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument("--vis-output", default=None)
    parser.add_argument("--vis-n-per-file", type=int, default=2)

    args = parser.parse_args()

    process_db2(
        model_name=args.model_name,
        model_path=args.model_path,
        db2_root=args.db2_root,
        test_npz_path=args.test_npz,
        qc_index_path=args.qc_index,
        output_dir_noisy=args.output_noisy,
        output_dir_denoised=args.output_denoised,
        device=args.device,
        max_files=args.max_files,
        skip_existing=not args.force,
        vis_output_dir=args.vis_output,
        vis_n_per_file=args.vis_n_per_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()