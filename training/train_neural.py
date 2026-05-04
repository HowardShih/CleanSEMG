#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sEMG Baseline Waveform Model Training (v1.1)
Supports: TrustEMGNet variants, FCN, CNN_waveform, SDEMG

Data pipeline is IDENTICAL to train.py (MECGE):
  - Same segments (outputs/segments/...)
  - Same OnlineNoiseMixer with same config
  - Same noisy-scale normalization (Q99 from noisy_raw)
  - Same clip_range

Difference from v1.0:
  - SDEMG uses diffusion MSE loss (HAS_DIFFUSION_LOSS flag)
    instead of plain L1 waveform loss.
  - All other models unchanged.

Outputs to: outputs/weights_baseline/{model_name}/
"""

import os
import sys
import yaml
import argparse
import random
import csv
from datetime import datetime
from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm

# ── resolve paths ─────────────────────────────────────────────────────────────
SEMG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SEMG_ROOT)

from noise import OnlineNoiseMixer
from baseline_models import BASELINE_MODEL_REGISTRY


# ============================================================================
# Normalization (identical to train.py)
# ============================================================================
def compute_scale_factor(x: np.ndarray, method: str = "Q99",
                         percentile: float = 0.99) -> float:
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
        scale  = float(np.median(np.abs(x - median)) * 1.4826)
    elif method == "STD":
        scale = float(np.std(x))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return max(scale, 1e-12)


def normalize_clip(x: np.ndarray, scale: float, clip_range) -> np.ndarray:
    scale = float(scale) if (scale is not None and np.isfinite(scale) and scale > 0) else 1.0
    y  = np.asarray(x, dtype=np.float32).reshape(-1) / scale
    lo, hi = float(clip_range[0]), float(clip_range[1])
    return np.clip(y, lo, hi).astype(np.float32)


# ============================================================================
# Dataset — waveform output (no STFT)
# ============================================================================
class WaveSegDatasetBaseline(Dataset):
    """
    Same as WaveSegDataset in train.py but returns waveforms instead of
    spectrograms.  Flow is identical:
      1. Load RAW segment
      2. Raw-domain online noise mixing
      3. Compute scale from noisy_raw
      4. Normalize clean & noisy with the SAME scale
    """

    def __init__(
        self,
        segments_root: str,
        split: str,
        mixer: OnlineNoiseMixer,
        norm_cfg: Dict,
        db_filter: Optional[List[str]] = None,
    ):
        import pandas as pd
        man = os.path.join(segments_root, "manifests", "segment_manifest.csv")
        if not os.path.exists(man):
            raise FileNotFoundError(f"Manifest not found: {man}")

        df = pd.read_csv(man)
        df = df[df["split"].astype(str) == split].copy()
        if db_filter:
            df = df[df["dataset"].astype(str).isin(set(db_filter))].copy()
        if df.empty:
            raise RuntimeError(f"No data for split='{split}', db_filter={db_filter}")

        raw_paths = df["raw_path"].astype(str).tolist()
        self.paths = [
            p if os.path.isabs(p) else os.path.join(segments_root, p)
            for p in raw_paths
        ]
        self.mixer          = mixer
        self.norm_method    = str(norm_cfg.get("method", "Q99"))
        self.norm_pct       = float(norm_cfg.get("percentile", 0.99))
        self.clip_range     = tuple(norm_cfg.get("clip_range", [-1.0, 1.0]))

        print(f"[WaveSegDatasetBaseline] split='{split}': {len(self.paths)} segments")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clean_raw = np.load(self.paths[idx]).astype(np.float64).reshape(-1)

        noisy_raw, _ = self.mixer.mix(clean_raw, mode="train")
        noisy_raw    = np.asarray(noisy_raw, dtype=np.float64).reshape(-1)

        scale      = compute_scale_factor(noisy_raw, self.norm_method, self.norm_pct)
        clean_norm = normalize_clip(clean_raw, scale, self.clip_range)
        noisy_norm = normalize_clip(noisy_raw, scale, self.clip_range)

        return (
            torch.from_numpy(clean_norm).float(),   # [L]
            torch.from_numpy(noisy_norm).float(),   # [L]
        )


# ============================================================================
# Helpers
# ============================================================================
def _get_nested(cfg: Dict, keys: List[str], default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def seed_everything(seed: int) -> None:
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_select_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        import subprocess, re
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        gpus = []
        for line in out.strip().splitlines():
            idx, free = map(int, re.split(r",\s*", line))
            gpus.append({"idx": idx, "free": free})
        best = max(gpus, key=lambda g: g["free"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best["idx"])
        return torch.device("cuda")
    except Exception:
        return torch.device("cuda")


def make_worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        s = (int(base_seed) + 1000 * worker_id) & 0xFFFFFFFF
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        wi = get_worker_info()
        if wi is not None and hasattr(wi.dataset, "mixer"):
            mx = wi.dataset.mixer
            if hasattr(mx, "rng"):    mx.rng    = random.Random(s)
            if hasattr(mx, "rng_np"): mx.rng_np = np.random.default_rng(s)
    return _init


# ============================================================================
# Loss helpers
# ============================================================================
def _compute_loss(
    model: nn.Module,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    is_diffusion: bool,
) -> torch.Tensor:
    """
    Unified loss computation for waveform and diffusion models.

    Waveform models (FCN, TrustEMGNet, …): L1(pred, clean)
    Diffusion models (SDEMG):              diffusion MSE loss
    """
    if is_diffusion:
        # SDEMG: loss computed internally by the diffusion process
        return model.compute_diffusion_loss(clean, noisy)
    else:
        pred = model(noisy)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        return F.l1_loss(pred, clean)


# ============================================================================
# Training
# ============================================================================
def train(config: Dict, model_name: str, segments_root: str,
          noise_dir: str, weights_dir: str, train_datasets: str = ""):

    device  = auto_select_gpu()
    seed    = int(_get_nested(config, ["project", "random_seed"], default=12345))
    seed_everything(seed)

    # ── model ────────────────────────────────────────────────────────────────
    if model_name not in BASELINE_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(BASELINE_MODEL_REGISTRY.keys())}"
        )
    model = BASELINE_MODEL_REGISTRY[model_name]().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} | params={n_params:,}")

    # ── diffusion flag ────────────────────────────────────────────────────────
    # SDEMG (and future diffusion models) set HAS_DIFFUSION_LOSS = True.
    # This switches the training / validation loss from plain L1 to the
    # model's own compute_diffusion_loss() method.
    is_diffusion = getattr(model, "HAS_DIFFUSION_LOSS", False)
    if is_diffusion:
        print(f"[Model] {model_name} uses diffusion loss (not L1)")

    # ── training config ───────────────────────────────────────────────────────
    bt_cfg     = config.get("baseline_train", {}) or {}
    defaults   = bt_cfg.get("defaults", bt_cfg) or {}
    per_model  = (bt_cfg.get("models", {}) or {}).get(model_name, {}) or {}
    norm_cfg   = config.get("normalization", {}) or {}

    def _get(key, fallback):
        return per_model.get(key, defaults.get(key, fallback))

    epochs      = int(_get("epochs", 100))
    batch_size  = int(_get("batch_size", 32))
    lr          = float(_get("lr", 1e-3))
    clip_val    = float(_get("clip_grad", 10.0))
    patience    = int(_get("patience", 20))
    num_workers = int(_get("num_workers", 4))
    prefetch    = int(_get("prefetch_factor", 4))
    pin_memory  = bool(_get("pin_memory", True))
    cache_noise = bool(_get("cache_noise", True))
    lr_schedule        = str(_get("lr_schedule", "plateau"))
    lr_milestones      = list(_get("lr_step_milestones", [3, 30]))
    lr_gamma           = float(_get("lr_step_gamma", 0.1))
    optimizer_name     = str(_get("optimizer", "adamw")).lower()
    adam_betas         = tuple(_get("adam_betas", [0.9, 0.999]))
    lr_plateau_factor  = float(_get("lr_plateau_factor", 0.5))
    lr_plateau_patience= int(_get("lr_plateau_patience", 5))

    print(f"[Config] model={model_name} | optimizer={optimizer_name} | "
          f"lr_schedule={lr_schedule} | epochs={epochs} | "
          f"batch={batch_size} | lr={lr:.2e} | patience={patience}")

    # ── mixer ─────────────────────────────────────────────────────────────────
    print(f"\n[Mixer] noise_root={noise_dir} | cache_noise={cache_noise}")
    mixer = OnlineNoiseMixer(
        noise_root=noise_dir, config=config,
        cache_noise=cache_noise, seed=seed,
    )

    # ── datasets ──────────────────────────────────────────────────────────────
    db_filter = [x.strip() for x in train_datasets.split(",")
                 if x.strip()] if train_datasets else None

    train_ds = WaveSegDatasetBaseline(
        segments_root, "train", mixer, norm_cfg, db_filter=db_filter)
    val_ds   = WaveSegDatasetBaseline(
        segments_root, "val",   mixer, norm_cfg, db_filter=db_filter)

    wifn = make_worker_init_fn(seed)
    g    = torch.Generator(); g.manual_seed(seed)

    train_dl_kw = dict(batch_size=batch_size, pin_memory=pin_memory,
                       num_workers=num_workers, worker_init_fn=wifn)
    if num_workers > 0:
        train_dl_kw["prefetch_factor"]    = prefetch
        train_dl_kw["persistent_workers"] = True

    val_workers = min(num_workers, 2)
    val_dl_kw = dict(batch_size=batch_size, pin_memory=False,
                     num_workers=val_workers, worker_init_fn=wifn)
    if val_workers > 0:
        val_dl_kw["prefetch_factor"]    = 2
        val_dl_kw["persistent_workers"] = False

    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **train_dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **val_dl_kw)

    # ── optimizer / scheduler ─────────────────────────────────────────────────
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=adam_betas)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=adam_betas)

    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=lr_gamma)
        _sched_is_val_based = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=lr_plateau_factor, patience=lr_plateau_patience)
        _sched_is_val_based = True

    # ── save paths ────────────────────────────────────────────────────────────
    save_dir = os.path.join(weights_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"{model_name}_best.pth")

    log_dir  = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"{model_name}_train_{ts}.csv")

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                      "lr", "best_val", "improved"]
                       ).writeheader()

    loss_label = "diffusion MSE" if is_diffusion else "L1 waveform"
    print(f"\n[Train] {model_name}")
    print(f"  device={device} | bs={batch_size} | lr={lr} | epochs={epochs} | patience={patience}")
    print(f"  loss={loss_label}")
    print(f"  weights → {save_dir}")
    print(f"  log     → {csv_path}")
    if not is_diffusion:
        print(f"\n✅ Noisy-Scale Policy: raw mix → scale from noisy → same-scale normalize")

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_sum = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}",
                    dynamic_ncols=True)

        optimizer.zero_grad(set_to_none=True)
        for step, (clean, noisy) in enumerate(pbar, 1):
            clean = clean.to(device, non_blocking=True)   # [B, L]
            noisy = noisy.to(device, non_blocking=True)   # [B, L]

            # ── loss (unified for waveform and diffusion models) ──────────
            loss = _compute_loss(model, clean, noisy, is_diffusion)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")

            train_sum += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({
                "avg": f"{train_sum/step:.4f}",
                "lr":  f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        train_avg = train_sum / max(1, len(train_loader))

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for clean, noisy in val_loader:
                clean = clean.to(device, non_blocking=True)
                noisy = noisy.to(device, non_blocking=True)
                # Diffusion val loss: same objective as training (fast, no full denoise)
                val_sum += _compute_loss(model, clean, noisy, is_diffusion).item()

        val_avg  = val_sum / max(1, len(val_loader))
        lr_now   = float(optimizer.param_groups[0]["lr"])
        improved = val_avg < best_val

        print(f"  Epoch {epoch:4d} | train={train_avg:.4f} | "
              f"val={val_avg:.4f} | lr={lr_now:.2e}"
              + (" ✓ best" if improved else ""))

        if _sched_is_val_based:
            scheduler.step(val_avg)
        else:
            scheduler.step()

        if improved:
            best_val   = val_avg
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                          "lr", "best_val", "improved"]
                           ).writerow({
                "epoch": epoch, "train_loss": f"{train_avg:.6f}",
                "val_loss": f"{val_avg:.6f}", "lr": f"{lr_now:.8f}",
                "best_val": f"{best_val:.6f}", "improved": str(improved),
            })

        if patience > 0 and no_improve >= patience:
            print(f"[EarlyStop] {patience} epochs without improvement.")
            break

    print(f"\n{'='*60}")
    print(f"✓ Training complete — {model_name}")
    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Best model:    {best_path}")
    print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Train sEMG baseline waveform model")
    ap.add_argument("--config",          required=True,
                    help="Path to config.yaml (same one used for MECGE)")
    ap.add_argument("--baseline-config", default=None,
                    help="Path to baseline_train_config.yaml (per-model hyperparams). "
                         "If given, its baseline_train section overrides config.yaml.")
    ap.add_argument("--model",           required=True,
                    help=f"Model name. Choices: {list(BASELINE_MODEL_REGISTRY.keys())}")
    ap.add_argument("--segments-root",   required=True,
                    help="Path to segments root (shared with MECGE)")
    ap.add_argument("--noise-root",      required=True,
                    help="Path to train noise pool (shared with MECGE)")
    ap.add_argument("--weights",         default="outputs/weights_baseline",
                    help="Root for baseline weights (default: outputs/weights_baseline)")
    ap.add_argument("--train-datasets",  default="",
                    help="Comma-separated DB names to include (empty = all)")
    ap.add_argument("--epochs",     type=int,   default=None)
    ap.add_argument("--batch-size", type=int,   default=None)
    ap.add_argument("--lr",         type=float, default=None)
    ap.add_argument("--patience",   type=int,   default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.baseline_config:
        with open(args.baseline_config) as f:
            bl_cfg = yaml.safe_load(f) or {}
        if "baseline_train" in bl_cfg:
            config["baseline_train"] = bl_cfg["baseline_train"]
            print(f"[Config] Loaded baseline hyperparams from: {args.baseline_config}")

    bt = config.setdefault("baseline_train", {})
    bt_defaults = bt.setdefault("defaults", {})
    if args.epochs     is not None: bt_defaults["epochs"]     = args.epochs
    if args.batch_size is not None: bt_defaults["batch_size"] = args.batch_size
    if args.lr         is not None: bt_defaults["lr"]         = args.lr
    if args.patience   is not None: bt_defaults["patience"]   = args.patience

    train(
        config       = config,
        model_name   = args.model,
        segments_root= args.segments_root,
        noise_dir    = args.noise_root,
        weights_dir  = args.weights,
        train_datasets = args.train_datasets,
    )


if __name__ == "__main__":
    main()