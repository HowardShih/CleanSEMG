#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ${CLEANSEMG_ROOT}/step2_quality_filter.py
"""
Step 2 (v6.2): Apply QC rules + grading
Input:
  - outputs/.../preprocessed/<DB>/logs/qc_metrics_raw.csv
Output:
  - outputs/.../preprocessed/<DB>/logs/qc_metrics_labeled.csv
  - outputs/.../preprocessed/<DB>/logs/qc_index.csv
Design:
  - QC output is index/manifest only (no masking/export).
"""

import os
import csv
import argparse
import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


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


def make_cross_subject_id(db_name: str, subject_id: int, file_rel: Optional[str] = None) -> str:
    # Avoid collapsing multiple unknown subjects into the same "DB_UNK"
    if subject_id is None or int(subject_id) < 0:
        token = "UNK"
        if file_rel:
            token = "UNK_" + hashlib.md5(str(file_rel).encode("utf-8")).hexdigest()[:8]
        return f"{db_name}_{token}"
    return f"{db_name}_{int(subject_id)}"


# =============================================================================
# Step 2
# =============================================================================

def step2_quality_filter(config: Dict, db_name: str, force: bool = False):
    print(f"\n{'='*70}\nStep 2 (v6.2): Quality Filter -> qc_index  [{db_name}]\n{'='*70}")

    out_root = get_db_out_root(config, db_name)
    logs_dir = os.path.join(out_root, "logs")
    reports_dir = os.path.join(out_root, "reports")
    ensure_dirs(logs_dir, reports_dir)

    qc_raw_csv = os.path.join(logs_dir, "qc_metrics_raw.csv")
    if not os.path.exists(qc_raw_csv):
        print(f"[ERROR] qc_metrics_raw.csv not found: {qc_raw_csv}")
        return

    labeled_csv = os.path.join(logs_dir, "qc_metrics_labeled.csv")
    qc_index_csv = os.path.join(logs_dir, "qc_index.csv")

    if (os.path.exists(labeled_csv) and os.path.exists(qc_index_csv)) and not force:
        print(f"[SKIP] Step2 outputs exist (use --force to rerun)")
        return

    df = pd.read_csv(qc_raw_csv)
    if df.empty:
        print("[ERROR] qc_metrics_raw.csv is empty")
        return

    qc = _get_nested(config, ["quality_control"], default={}) or {}
    spec = _get_nested(qc, ["spectral"], default={}) or {}
    smr_min = float(spec.get("SMR_min", 0.0))
    shr_min = float(spec.get("SHR_min", 0.0))
    ohm_max = float(spec.get("OHM_max", 1e9))

    snr_gr = _get_nested(qc, ["snr_grading"], default={}) or {}
    snr_A = float(snr_gr.get("A_excellent", 999.0))
    snr_B = float(snr_gr.get("B_good", 999.0))
    snr_C = float(snr_gr.get("C_unacceptable", -999.0))

    hamp = _get_nested(qc, ["hampel"], default={}) or {}
    hamp_enabled = bool(hamp.get("enabled", False))
    apply_to_A = bool(hamp.get("apply_to_A", True))
    apply_to_B = bool(hamp.get("apply_to_B", True))
    nan_is_fail = bool(hamp.get("nan_is_fail", True))
    er_max_A = float(hamp.get("er_max_A", 1e9))
    er_max_B = float(hamp.get("er_max_B", 1e9))

    keep_grades = _get_nested(config, ["pipeline", "keep_labels"], default=["A_excellent", "B_good"])
    keep_grades = set([str(x) for x in keep_grades])

    for col in ["SNR_dB", "SMR_dB", "SHR_dB", "OHM", "HAMP_ER"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # QC Logic
    base_valid = (~df["SNR_dB"].isna()) & (~df["SMR_dB"].isna()) & (~df["SHR_dB"].isna()) & (~df["OHM"].isna())
    base_pass = base_valid & (df["SMR_dB"] > smr_min) & (df["SHR_dB"] > shr_min) & (df["OHM"] < ohm_max)

    labels = np.full(len(df), "rejected", dtype=object)
    snr = df["SNR_dB"].to_numpy()

    labels[base_pass & (snr >= snr_A)] = "A_excellent"
    labels[base_pass & (snr >= snr_B) & (snr < snr_A)] = "B_good"
    labels[base_pass & (snr >= snr_C) & (snr < snr_B)] = "C_unacceptable"

    gate = np.ones(len(df), dtype=bool)
    if hamp_enabled and "HAMP_ER" in df.columns:
        er = df["HAMP_ER"].to_numpy()
        if apply_to_A:
            maskA = (labels == "A_excellent")
            gate[maskA] = (~np.isnan(er[maskA])) & (er[maskA] <= er_max_A) if nan_is_fail else (er[maskA] <= er_max_A) | np.isnan(er[maskA])
        if apply_to_B:
            maskB = (labels == "B_good")
            gate[maskB] = (~np.isnan(er[maskB])) & (er[maskB] <= er_max_B) if nan_is_fail else (er[maskB] <= er_max_B) | np.isnan(er[maskB])

    is_keep = np.array([str(x) in keep_grades for x in labels], dtype=bool)
    final_pass = base_pass & is_keep & gate

    df["pass_base"] = base_pass
    df["qc_grade"] = labels
    df["pass_hampel"] = gate
    df["pass_final"] = final_pass

    def _reason_row(i: int) -> str:
        flags = []
        if not bool(base_valid.iloc[i]): flags.append("nan_metric")
        if bool(base_valid.iloc[i]) and not bool(base_pass.iloc[i]): flags.append("fail_spectral")
        if bool(base_pass.iloc[i]) and (labels[i] == "rejected"): flags.append("fail_snr")
        if bool(base_pass.iloc[i]) and bool(is_keep[i]) and not bool(gate[i]): flags.append("fail_hampel")
        return "|".join(flags) if flags else "OK"

    df["reason_flags"] = [ _reason_row(i) for i in range(len(df)) ]
    df.to_csv(labeled_csv, index=False)

    # Make Index
    idx_df = df[df["pass_final"]].copy()
    idx_df["cross_subject_id"] = [
        make_cross_subject_id(
            db_name,
            int(sid) if np.isfinite(sid) else -1,
            file_rel=str(fr) if isinstance(fr, str) else None
        )
        for sid, fr in zip(idx_df["subject_id"].fillna(-1).to_numpy(), idx_df["file"].astype(str).to_numpy())
    ]

    qc_index_cols = [
        "dataset", "file", "subject_id", "exercise_id", "cross_subject_id",
        "trial_id", "trial_start_raw", "trial_end_raw",
        "gesture", "repetition", "fs_raw", "ch",
        "qc_grade", "reason_flags"
    ]
    idx_df[qc_index_cols].to_csv(qc_index_csv, index=False)

    # Stats for report
    total = len(df)
    pass_n = int(final_pass.sum())
    n_total = int(total)
    n_nan = int((~base_valid).sum())
    n_fail_spec = int((base_valid & (~base_pass)).sum())
    n_fail_snr = int((base_pass & (labels == "rejected")).sum())
    n_fail_hamp = int((base_pass & is_keep & (~gate)).sum()) if hamp_enabled else 0

    report_path = os.path.join(reports_dir, "step2_quality_filter_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Step 2 Quality Filter Report - {db_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("SUMMARY\n")
        f.write(f"  Total trial-channel rows: {total}\n")
        f.write(f"  PASS final: {pass_n} ({100*pass_n/total:.2f}%)\n\n")
        f.write("STAGE COUNTS\n")
        f.write(f"  Stage-0 nan_metric:    {n_nan}\n")
        f.write(f"  Stage-1 fail_spectral: {n_fail_spec}\n")
        f.write(f"  Stage-2 fail_snr:      {n_fail_snr}\n")
        if hamp_enabled:
            f.write(f"  Stage-3 fail_hampel:   {n_fail_hamp}\n")
        f.write("\n")
        f.write("OUTPUT\n")
        f.write(f"  qc_metrics_labeled.csv: {labeled_csv}\n")
        f.write(f"  qc_index.csv:           {qc_index_csv}\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Step 2 complete. PASS(final): {pass_n}/{total} ({100*pass_n/total:.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.db:
        dbs = [args.db]
    else:
        test_db = str(_get_nested(cfg, ["datasets", "test_db"], default="DB2"))
        train_valid_dbs = list(_get_nested(cfg, ["datasets", "train_valid_dbs"], default=[]))
        dbs = [test_db] + train_valid_dbs

    for db in dbs:
        step2_quality_filter(cfg, db, force=args.force)

if __name__ == "__main__":
    main()