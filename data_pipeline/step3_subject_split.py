#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#${CLEANSEMG_ROOT}/step3_subject_split.py
"""
Step 3 (v6.2): Cross-DB subject split using qc_index
Output:
  - outputs/.../splits/subject_split_crossDB.json
"""

import os
import json
import argparse
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


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


def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def step3_subject_split(config: Dict, force: bool = False):
    print(f"\n{'='*70}\nStep 3 (v6.2): Cross-DB Subject Split (Stratified)\n{'='*70}")

    test_db = str(_get_nested(config, ["datasets", "test_db"], default="DB2"))
    train_valid_dbs = list(_get_nested(config, ["datasets", "train_valid_dbs"], default=[]))

    seg_len_s = float(_get_nested(config, ["preprocessing", "segmentation", "length_s"], default=2.0))
    train_ratio = float(_get_nested(config, ["splitting", "train_ratio"], default=0.8))
    min_valid = int(_get_nested(config, ["splitting", "min_valid_subjects"], default=1))
    
    # Random Seed
    seed = int(_get_nested(config, ["project", "random_seed"], default=12345))
    rng = random.Random(seed)

    splits_root = get_splits_root(config)
    ensure_dirs(splits_root)

    split_json = os.path.join(splits_root, "subject_split_crossDB.json")
    report_path = os.path.join(splits_root, "step3_subject_split_report.txt")

    if os.path.exists(split_json) and not force:
        print(f"[SKIP] split json exists (use --force to rerun)")
        return

    all_dbs = [test_db] + train_valid_dbs
    subj_segs = defaultdict(int)
    db_stats = {}

    for db in all_dbs:
        db_out = get_db_out_root(config, db)
        qc_index_csv = os.path.join(db_out, "logs", "qc_index.csv")
        if not os.path.exists(qc_index_csv):
            continue

        dfi = pd.read_csv(qc_index_csv)
        if dfi.empty: continue

        cnt_by_subj = defaultdict(int)
        for _, r in dfi.iterrows():
            fs_raw = float(r.get("fs_raw", 0))
            if fs_raw <= 0: continue
            s, e = int(r.get("trial_start_raw", 0)), int(r.get("trial_end_raw", -1))
            if e <= s: continue
            nseg = int(((e - s + 1) / fs_raw) // seg_len_s)
            if nseg <= 0: continue
            csid = str(r.get("cross_subject_id", f"{db}_UNK"))
            cnt_by_subj[csid] += nseg

        for csid, nseg in cnt_by_subj.items():
            subj_segs[csid] += int(nseg)
        db_stats[db] = {"subjects": len(cnt_by_subj), "est_segments": int(sum(cnt_by_subj.values()))}

    # Split Logic
    test_subjects = sorted([sid for sid in subj_segs.keys() if sid.split("_")[0] == test_db])
    test_segments = int(sum(subj_segs[sid] for sid in test_subjects))

    # Pool by DB (exclude test_db)
    pool_by_db = defaultdict(list)
    for sid, segs in subj_segs.items():
        db = sid.split("_")[0]
        if db == test_db: continue
        pool_by_db[db].append((sid, int(segs)))

    train_subjects, valid_subjects = [], []
    train_segments, valid_segments = 0, 0
    degrade_notes = []

    for db, items in pool_by_db.items():
        items = sorted(items, key=lambda x: x[1])  # Ascending by segment count
        n = len(items)
        if n <= 1:
            for sid, segs in items:
                train_subjects.append(sid); train_segments += segs
            degrade_notes.append(f"{db}: subjects={n} (no valid possible)")
            continue

        # Stratified pick for valid set
        n_valid = int(round((1.0 - train_ratio) * n))
        n_valid = max(min_valid, n_valid)
        n_valid = min(n_valid, n - 1)

        # Evenly spaced indices over sorted subjects
        idxs = np.linspace(0, n - 1, n_valid, dtype=int).tolist()
        idxs = sorted(set(idxs))
        if len(idxs) < n_valid:
            remain = [i for i in range(n) if i not in idxs]
            rng.shuffle(remain)
            idxs += remain[:(n_valid - len(idxs))]

        valid_set = set(items[i][0] for i in idxs)
        for sid, segs in items:
            if sid in valid_set:
                valid_subjects.append(sid); valid_segments += segs
            else:
                train_subjects.append(sid); train_segments += segs

    split_data = {
        "train": sorted(train_subjects),
        "valid": sorted(valid_subjects),
        "test": sorted(test_subjects),
        "metadata": {
            "test_db": test_db,
            "train_valid_dbs": train_valid_dbs,
            "seed": seed,
            "segment_length_s": seg_len_s,
            "train_ratio": train_ratio,
            "train_est_segments": int(train_segments),
            "valid_est_segments": int(valid_segments),
            "test_est_segments": int(test_segments),
            "total_est_segments": int(train_segments + valid_segments + test_segments),
        },
    }

    with open(split_json, "w") as f:
        json.dump(split_data, f, indent=2)

    with open(report_path, "w") as f:
        f.write("Step 3 Report (v6.2) - Stratified Split\n\n")
        f.write("PER-DB STATS\n")
        for db, st in db_stats.items():
            f.write(f"  {db}: subjects={st['subjects']}, segments={st['est_segments']}\n")
        if degrade_notes:
            f.write("\nDEGRADED DBS\n")
            for s in degrade_notes: f.write(f"  - {s}\n")
        f.write(f"\nFINAL: Train={len(train_subjects)}, Valid={len(valid_subjects)}, Test={len(test_subjects)}\n")

    print(f"✓ Step 3 Done. Split ratio preserved per DB where possible.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    step3_subject_split(cfg, force=args.force)

if __name__ == "__main__":
    main()