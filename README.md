# CLEANSEMG: A Benchmark for Single-Channel sEMG Denoising

[![Paper](https://img.shields.io/badge/NeurIPS-2026-blue)](https://anonymous.4open.science/...)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/anonymous/cleansemg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CLEANSEMG is the first standardized benchmark for evaluating single-channel sEMG denoising methods. It provides a unified evaluation framework covering signal-level reconstruction quality, feature-level preservation, and downstream task performance.

## Overview

| Feature | Details |
|---|---|
| Datasets | Ninapro DB2–DB10, Cerqueira fatigue |
| Noise types | PLI, ECG, MOA, WGN, Colored (pink/brown) |
| Classical methods | HP, EMD, CEEMDAN, VMD |
| Neural methods | FCN, SDEMG, MSEMG, TrustEMG-Net |
| Downstream tasks | Hand gesture recognition (STCNet), Fatigue classification (SVM + CNN) |
| Evaluation | SNRimp, RMSE, PRD, LSD, feature errors, downstream accuracy |

---

## Installation

```bash
git clone https://github.com/HowardShih/CleanSEMG.git
cd CleanSEMG
pip install -r requirements.txt
```

**For SDEMG (diffusion model):** Clone the SDEMG repo and set the path:
```bash
git clone https://github.com/yutingLiu2024/SDEMG.git
export SDEMG_REPO_PATH=/path/to/SDEMG
```

---

## Dataset Download

### 1. Preprocessed benchmark data (HuggingFace — recommended)

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="anonymous/cleansemg", repo_type="dataset",
                  local_dir="./data/cleansemg")
```

Or download only the fixed test set for quick evaluation:
```python
snapshot_download(repo_id="anonymous/cleansemg-sample", repo_type="dataset",
                  local_dir="./data/cleansemg-sample")
```

### 2. Raw datasets (required to run from scratch)

Download the following and place them under your `DATA_ROOT`:

| Dataset | Source | Directory name |
|---|---|---|
| Ninapro DB2–DB10 | [ninapro.hevs.ch](http://ninapro.hevs.ch) | `DB2/`, `DB3/`, ..., `DB10/` |
| Cerqueira fatigue | [Zenodo](https://zenodo.org/record/...) | `Cerqueira/` |
| MIT-BIH NSRD (ECG) | [PhysioNet](https://physionet.org/content/nsrdb/) | `mit-bih-normal-sinus-rhythm-database-1.0.0/` |
| MIT-BIH NSTDB (MOA) | [PhysioNet](https://physionet.org/content/nstdb/) | `mit-bih-noise-stress-test-database-1.0.0/` |
| Machado MOA | [Link](https://doi.org/10.1016/j.bspc.2021.102752) | `moa_train/`, `moa_test/` |

Set `data_root` in `configs/config.yaml`:
```yaml
paths:
  root: "."
  data_root: "/path/to/your/data"  # ← change this
```

---

## Quick Start

### Using pre-processed data from HuggingFace

If you downloaded the pre-processed dataset, you can skip directly to evaluation:

```bash
# Evaluate TrustEMG-Net (neural)
python evaluation/evaluate_neural.py \
    --config configs/config.yaml \
    --model TrustEMGNet_RM \
    --model-path /path/to/TrustEMGNet_RM_best.pth \
    --test-data data/cleansemg/fixed_test_set/ \
    --output results/trustemg/

# Evaluate HP filter (classical)
python evaluation/evaluate_classical.py \
    --config configs/config.yaml \
    --method hp \
    --test-data data/cleansemg/fixed_test_set/ \
    --output results/hp/
```

### Running from scratch

**Step 1: Data preparation**
```bash
bash scripts/run_pipeline.sh --config configs/config.yaml --stages all
```
This runs the full pipeline:
- Quality control on raw sEMG (Step 1–2)
- Subject split (Step 3)
- Bandpass filtering + segmentation (Step 4)
- Noise pool generation (PLI, WGN, CLN, ECG, MOA)
- Fixed offline test set generation

**Step 2: Train neural denoisers**
```bash
bash scripts/run_train.sh --config configs/config.yaml --model FCN
bash scripts/run_train.sh --config configs/config.yaml --model SDEMG
bash scripts/run_train.sh --config configs/config.yaml --model MSEMG
bash scripts/run_train.sh --config configs/config.yaml --model TrustEMGNet_RM
```

**Step 3: Calibrate classical methods** (optional, defaults are used otherwise)
```bash
python training/train_classical.py --config configs/config.yaml
```

**Step 4: Evaluate all methods**
```bash
# Neural denoisers
for model in FCN SDEMG MSEMG TrustEMGNet_RM; do
    python evaluation/evaluate_neural.py \
        --config configs/config.yaml \
        --model $model \
        --model-path outputs/weights_baseline/${model}/${model}_best.pth \
        --test-data outputs/test_data/ \
        --output results/${model}/
done

# Classical methods
for method in hp emd ceemdan vmd; do
    python evaluation/evaluate_classical.py \
        --config configs/config.yaml \
        --method $method \
        --test-data outputs/test_data/ \
        --output results/${method}/
done
```

---

## Model Weights

Pre-trained weights are available on HuggingFace Model Hub:

```python
from huggingface_hub import hf_hub_download

# Download TrustEMG-Net weights
ckpt = hf_hub_download("anonymous-cleansemg/trustemg-net", "TrustEMGNet_RM_best.pth")

# Other models
# anonymous-cleansemg/fcn
# anonymous-cleansemg/sdemg
# anonymous-cleansemg/msemg
```

---

## Downstream Tasks

### Hand Gesture Recognition (Ninapro DB2, STCNet)

```bash
# Step 1: Generate denoised MAT files (example: TrustEMG-Net)
python downstream/gesture/prepare_denoised_neural.py \
    --model-name TrustEMGNet_RM \
    --model-path outputs/weights_baseline/TrustEMGNet_RM/TrustEMGNet_RM_best.pth \
    --db2-root /path/to/data/DB2 \
    --test-npz outputs/test_data/test_combined.npz \
    --qc-index outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy outputs/downstream/noisy/DB2 \
    --output-denoised outputs/downstream/trustemg/DB2 \
    --device cuda

# For classical methods (example: HP filter)
python downstream/gesture/prepare_denoised_classical.py \
    --method hp \
    --db2-root /path/to/data/DB2 \
    --test-npz outputs/test_data/test_combined.npz \
    --qc-index outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy outputs/downstream/noisy/DB2 \
    --output-denoised outputs/downstream/hp/DB2

# Step 2: Run STCNet evaluation
cd downstream/gesture/stcnet
python test.py \
    --model_path save/CE/.../best_model.pth \
    --dataset nina2 --model STCNet --batch_size 64

# Run all 8 denoisers at once
bash downstream/gesture/run_all_denoisers.sh
```

### Fatigue Classification (Cerqueira dataset)

The benchmark evaluates two classifier families (RBF-SVM and Dilated CNN) across all 8 denoisers. Use `--denoiser` to select the denoising method:

```bash
# Set project root
export CLEANSEMG_ROOT=/path/to/CleanSEMG

# SVM classifier
python downstream/fatigue/fatigue_svm.py \
    --denoiser trustemg \
    --noise-root outputs/noise/sEMG_noise_test \
    --cerqueira-data /path/to/data/Cerqueira

# CNN classifier
python downstream/fatigue/fatigue_cnn.py \
    --denoiser fcn \
    --noise-root outputs/noise/sEMG_noise_test \
    --cerqueira-data /path/to/data/Cerqueira
```

Supported `--denoiser` values:

| Type | Values |
|---|---|
| Classical | `hp`, `emd`, `ceemdan`, `vmd` |
| Neural | `fcn`, `sdemg`, `msemg`, `trustemg` |

---

## Results

Table 1 from the paper (Ninapro DB2, macro-averaged over all SNR × k):

| Method | SNRimp (dB) ↑ | RMSE (×10⁻⁵) ↓ | PRD (%) ↓ | Gesture Acc (%) ↑ | Fatigue Acc (%) ↑ |
|---|---|---|---|---|---|
| HP | 1.27 | 5.44 | 121.80 | 65.82 | 42.78 |
| EMD | 0.34 | 5.45 | 119.58 | 62.24 | 40.93 |
| CEEMDAN | 0.46 | 5.30 | 116.57 | — | 40.89 |
| VMD | 0.31 | 10.00 | 120.02 | 62.73 | 38.94 |
| FCN | 6.95 | 2.41 | 54.07 | 67.24 | **61.60** |
| SDEMG | 5.69 | 2.94 | 65.95 | 68.46 | 57.55 |
| MSEMG | 7.24 | 2.41 | 53.91 | 68.04 | 58.84 |
| TrustEMG-Net | **7.72** | **2.27** | **50.72** | **68.52** | 60.05 |

---

## Project Structure

```
CleanSEMG/
├── configs/           # Configuration files
├── data_pipeline/     # QC and segmentation (Step 1–4)
├── denoising/         # Neural and classical denoiser implementations
├── training/          # Training scripts for all denoisers
├── evaluation/        # Evaluation scripts with all metrics
├── noise/             # Noise generation and online mixing
├── downstream/
│   ├── gesture/       # STCNet gesture recognition pipeline
│   └── fatigue/       # SVM + CNN fatigue classification
└── scripts/           # Convenience shell scripts
```

---

## Citation

If you use CLEANSEMG in your research, please cite:

```bibtex
@inproceedings{cleansemg2026,
  title     = {{CLEANSEMG}: A Benchmark for Single-Channel Surface Electromyography Denoising Algorithms},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

---

## Acknowledgements

This benchmark builds on the following public resources:
- [Ninapro Database](http://ninapro.hevs.ch) (Atzori et al., 2014)
- [Cerqueira fatigue dataset](https://zenodo.org/...) (Cerqueira et al., 2024)
- [MIT-BIH PhysioNet databases](https://physionet.org)
- [TrustEMG-Net](https://github.com/eric-wang135/TrustEMG) (Wang et al., 2025)
- [SDEMG](https://github.com/...) (Liu et al., 2024)