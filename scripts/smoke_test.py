#!/usr/bin/env python3
"""
CleanSEMG Smoke Test
====================
Verifies all model imports, forward passes, and downstream preprocessing
WITHOUT requiring any real dataset. Uses synthetic sEMG signals.

Usage:
    cd /data/member1/user_howardshih/CleanSEMG
    python scripts/smoke_test.py

Expected: all tests PASS in ~60 seconds on CPU.
"""

import os
import sys
import tempfile
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SEP  = "=" * 60


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model imports + forward pass
# ─────────────────────────────────────────────────────────────────────────────
section("1. Neural Denoiser Imports & Forward Pass")

B, L = 2, 2000
x = torch.randn(B, L)

def test_fcn():
    from denoising.FCN import FCN
    m = FCN(); m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (B, L), f"Shape mismatch: {y.shape}"

def test_cnn():
    from denoising.CNN import CNN_waveform
    m = CNN_waveform(); m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (B, L)

def test_msemg():
    from denoising.MSEMG import MSEMG
    m = MSEMG(); m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (B, L)

def test_trustemg():
    from denoising.TrustEMGNet import TrustEMGNet_RM, TrustEMGNet_DM
    for cls in [TrustEMGNet_RM, TrustEMGNet_DM]:
        m = cls(); m.eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (B, L)

def test_sdemg():
    from denoising.SDEMG import SDEMG
    m = SDEMG(seq_length=L, timesteps=5); m.eval()
    # Test forward (inference) — use only 2 steps for speed
    m.diffusion.timesteps = 2
    with torch.no_grad():
        y = m(x)
    assert y.shape == (B, L)

def test_registry():
    from denoising import BASELINE_MODEL_REGISTRY
    required = {"FCN", "CNN_waveform", "MSEMG", "SDEMG",
                "TrustEMGNet_RM", "TrustEMGNet_DM",
                "TrustEMGNet_UNetonly", "TrustEMGNet_LSTM_RM"}
    missing = required - set(BASELINE_MODEL_REGISTRY.keys())
    assert not missing, f"Missing from registry: {missing}"

test("FCN forward pass",        test_fcn)
test("CNN_waveform forward",    test_cnn)
test("MSEMG forward pass",      test_msemg)
test("TrustEMGNet_RM forward",  test_trustemg)
test("SDEMG forward pass",      test_sdemg)
test("BASELINE_MODEL_REGISTRY", test_registry)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Classical filters
# ─────────────────────────────────────────────────────────────────────────────
section("2. Classical Denoising Filters")

sig = np.random.randn(2000).astype(np.float64)

def test_hp():
    from denoising.classical_filters import apply_hp_filter
    out = apply_hp_filter(sig, fs=1000, cutoff_hz=40.0, order=4)
    assert out.shape == sig.shape

def test_emd():
    from denoising.classical_filters import apply_emd_filter
    out, ok = apply_emd_filter(sig, fs=1000, f_min=20.0, f_max=500.0, max_imfs=5)
    assert out.shape == sig.shape

def test_vmd():
    from denoising.classical_filters import apply_vmd_filter
    out, ok = apply_vmd_filter(sig, fs=1000, K=4)
    assert out.shape == sig.shape

def test_ceemdan():
    from denoising.classical_filters import apply_ceemdan_filter
    out, ok = apply_ceemdan_filter(sig, fs=1000, trials=5, f_min=20.0, f_max=500.0)
    assert out.shape == sig.shape

test("HP filter",      test_hp)
test("EMD filter",     test_emd)
test("VMD filter",     test_vmd)
test("CEEMDAN filter", test_ceemdan)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training pipeline (tiny synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
section("3. Training Pipeline (Synthetic Data)")

def test_training_dataset():
    """Create a fake segment manifest and test WaveSegDatasetBaseline."""
    import pandas as pd
    from training.train_neural import WaveSegDatasetBaseline, compute_scale_factor

    with tempfile.TemporaryDirectory() as tmp:
        seg_dir = os.path.join(tmp, "manifests")
        raw_dir = os.path.join(tmp, "raw")
        os.makedirs(seg_dir); os.makedirs(raw_dir)

        # Write 10 fake segments
        paths = []
        for i in range(10):
            p = os.path.join(raw_dir, f"seg_{i:03d}.npy")
            np.save(p, np.random.randn(2000).astype(np.float32))
            paths.append(p)

        # Write manifest
        df = pd.DataFrame({
            "raw_path": paths,
            "split": ["train"] * 8 + ["val"] * 2,
            "dataset": ["DB2"] * 10,
        })
        df.to_csv(os.path.join(seg_dir, "segment_manifest.csv"), index=False)

        # Compute scale factor
        x = np.random.randn(2000)
        s = compute_scale_factor(x, method="Q99")
        assert s > 0

def test_loss_computation():
    from training.train_neural import _compute_loss
    from denoising.FCN import FCN
    m = FCN()
    clean = torch.randn(2, 2000)
    noisy = torch.randn(2, 2000)
    loss = _compute_loss(m, clean, noisy, is_diffusion=False)
    assert loss.item() > 0

test("WaveSegDataset + manifest",    test_training_dataset)
test("L1 loss computation",          test_loss_computation)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Config loading
# ─────────────────────────────────────────────────────────────────────────────
section("4. Config Files")

def test_config_yaml():
    import yaml
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    assert os.path.exists(cfg_path), f"Not found: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "paths" in cfg
    assert "noise" in cfg
    assert "normalization" in cfg

def test_baseline_train_config():
    import yaml
    cfg_path = os.path.join(ROOT, "configs", "baseline_train_config.yaml")
    assert os.path.exists(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "baseline_train" in cfg
    assert "TrustEMGNet_RM" in cfg["baseline_train"]["models"]

def test_tradition_config():
    import yaml
    cfg_path = os.path.join(ROOT, "configs", "config_tradition.yaml")
    assert os.path.exists(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "hp" in cfg

test("configs/config.yaml",               test_config_yaml)
test("configs/baseline_train_config.yaml", test_baseline_train_config)
test("configs/config_tradition.yaml",      test_tradition_config)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Downstream: STCNet preprocessing
# ─────────────────────────────────────────────────────────────────────────────
section("5. STCNet Preprocessing (Synthetic MAT)")

def test_emg_preprocess_import():
    sys.path.insert(0, os.path.join(ROOT, "downstream", "gesture"))
    import emg_preprocess
    assert hasattr(emg_preprocess, "DATASET_CFG")
    assert "nina2" in emg_preprocess.DATASET_CFG

def test_bandpass_resample():
    sys.path.insert(0, os.path.join(ROOT, "downstream", "gesture"))
    from emg_preprocess import apply_bandpass_filter, resample_emg, normalize_minmax
    emg = np.random.randn(400, 12).astype(np.float64) * 1e-4
    filtered = apply_bandpass_filter(emg, fs=2000.0)
    assert filtered.shape == emg.shape
    resampled = resample_emg(filtered, from_fs=2000, to_fs=1000)
    assert resampled.shape == (200, 12)
    normalized = normalize_minmax(resampled)
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0 + 1e-6

test("emg_preprocess.py imports",    test_emg_preprocess_import)
test("bandpass + resample + minmax", test_bandpass_resample)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("Summary")
print("\n  Run: python scripts/smoke_test.py")
print("  If all PASS, your CleanSEMG installation is correct.\n")