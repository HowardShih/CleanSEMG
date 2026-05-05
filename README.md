# CLEANSEMG: A Benchmark for Single-Channel sEMG Denoising

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)](https://anonymous.4open.science/r/CleanSEMG)
[![Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/AnonResearcher0029/CLEANsEMG)
[![Model Weights](https://img.shields.io/badge/🤗_HuggingFace-Model_Weights-orange)](https://huggingface.co/AnonResearcher0029/CLEANsEMG-weights)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CLEANSEMG is the first standardized benchmark for evaluating single-channel surface EMG (sEMG) denoising methods. It provides a unified evaluation framework covering **signal-level reconstruction quality**, **feature-level preservation**, and **downstream task performance** across multiple datasets, noise types, and denoising approaches.

---

## Overview

| | Details |
|---|---|
| **Datasets** | Ninapro; Muscle Fatigue Analysis |
| **Noise types** | PLI, ECG artifact, MOA (motion artifact), WGN, Colored (pink/brown) |
| **Classical methods** | HP (high-pass filter), EMD, CEEMDAN, VMD |
| **Neural methods** | FCN, SDEMG, MSEMG (EMG-MAMBA), TrustEMG-Net |
| **Signal metrics** | SNRimp, RMSE, PRD, LSD |
| **Feature metrics** | Error in ARV, ZCR, MNF, MDF, Kurtosis |
| **Downstream tasks** | Hand gesture recognition (STCNet), Muscle Fatigue classification (SVM, CNN) |

---

## Quickstart for Reviewers

**No raw data download needed.** We provide the benchmark test set and pre-trained weights on HuggingFace to reproduce Table 1 directly.

```bash
git clone https://github.com/HowardShih/CleanSEMG.git
cd CleanSEMG
pip install -r requirements.txt
```

**Step 1: Download test set + weights**

```python
from huggingface_hub import snapshot_download, hf_hub_download
import os

# Benchmark test set, noise pool, and calibrated classical params
snapshot_download(
    repo_id="AnonResearcher0029/CLEANsEMG",
    repo_type="dataset",
    local_dir="data/sample"
)

# Pre-trained neural denoiser weights (all 4 models in one repo)
for fname in [
    "FCN_best.pth",
    "MSEMG_best.pth",
    "SDEMG_best.pth",
    "TrustEMGNet_RM_best.pth",
]:
    name = fname.replace("_best.pth", "")
    os.makedirs(f"outputs/weights_baseline/{name}", exist_ok=True)
    hf_hub_download(
        repo_id="AnonResearcher0029/CLEANsEMG-weights",
        filename=fname,
        local_dir=f"outputs/weights_baseline/{name}/"
    )
```

**Step 2: Evaluate all methods**

```bash
# Neural denoisers
for model in FCN MSEMG SDEMG TrustEMGNet_RM; do
    python evaluation/evaluate_neural.py \
        --config     configs/config.yaml \
        --model      $model \
        --model-path outputs/weights_baseline/${model}/${model}_best.pth \
        --test-data  data/sample/test_combined.npz \
        --output     results/${model}/
done

# Classical methods (pre-calibrated params included)
for method in hp emd vmd ceemdan; do
    python evaluation/evaluate_classical.py \
        --config      configs/config.yaml \
        --trad-config configs/config_tradition.yaml \
        --params      data/sample/tradition_params.json \
        --test-data   data/sample/test_combined.npz \
        --output      results/${method}/
done
```

---

## Installation

```bash
git clone https://github.com/HowardShih/CleanSEMG.git
cd CleanSEMG
pip install -r requirements.txt
```

**SDEMG** is fully self-contained — the diffusion model (Liu et al., ICASSP 2024) is embedded in `denoising/SDEMG.py`. No external repository is needed.

**MSEMG** requires `mamba-ssm` and a CUDA GPU:

```bash
pip install mamba-ssm   # CUDA >= 11.8 required
```

MSEMG is skipped automatically on CPU-only machines. All other methods (FCN, SDEMG, TrustEMG-Net, all classical) run on CPU.

---

## HuggingFace Artifacts

### Test set and noise pool

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AnonResearcher0029/CLEANsEMG",
    repo_type="dataset",
    local_dir="data/sample"
)
```

| Path | Description |
|---|---|
| `test_combined.npz` | Fixed test set: clean/noisy segment pairs across all 7 SNR levels and 5 noise types |
| `noise_pool/PLI/` | Synthetic 50/60 Hz power-line interference |
| `noise_pool/ECG/` | ECG artifact segments (MIT-BIH NSR Database, ODC-BY) |
| `noise_pool/MOA/` | Motion artifact segments (MIT-BIH NSTDB, ODC-BY) |
| `noise_pool/WGN/` | White Gaussian noise |
| `noise_pool/Color/` | Colored (pink/brown) noise |
| `tradition_params.json` | Pre-calibrated HP/EMD/VMD/CEEMDAN parameters |

> **Note on Ninapro licensing.** Raw Ninapro .mat files are for academic non-commercial use only and cannot be redistributed. The test set contains only anonymized, derived signal segments.

### Pre-trained model weights

```python
from huggingface_hub import hf_hub_download
import os

for fname in [
    "FCN_best.pth",
    "MSEMG_best.pth",
    "SDEMG_best.pth",
    "TrustEMGNet_RM_best.pth",
]:
    name = fname.replace("_best.pth", "")
    os.makedirs(f"outputs/weights_baseline/{name}", exist_ok=True)
    hf_hub_download(
        repo_id="AnonResearcher0029/CLEANsEMG-weights",
        filename=fname,
        local_dir=f"outputs/weights_baseline/{name}/"
    )
```

---

## Full Reproduction from Raw Data

### Required datasets

Download the following and set `data_root` in `configs/config.yaml`:

| Dataset | Source | License | Directory |
|---|---|---|---|
| **Ninapro DB2–DB10** | [ninapro.hevs.ch](http://ninapro.hevs.ch) | Academic non-commercial | `DB2/` … `DB10/` |
| **Cerqueira fatigue** | [Zenodo 13860256](https://zenodo.org/records/13860256) | CC BY 4.0 | `Cerqueira/` |
| **MIT-BIH NSR** | [PhysioNet nsrdb 1.0.0](https://physionet.org/content/nsrdb/1.0.0/) | ODC-BY | `mit-bih-normal-sinus-rhythm-database-1.0.0/` |
| **MIT-BIH NSTDB** | [PhysioNet nstdb 1.0.0](https://physionet.org/content/nstdb/1.0.0/) | ODC-BY | `mit-bih-noise-stress-test-database-1.0.0/` |
| **Machado MOA** | [doi:10.1016/j.bspc.2021.102752](https://doi.org/10.1016/j.bspc.2021.102752) | See paper | `moa_train/`, `moa_test/` |

```yaml
# configs/config.yaml — only this line needs changing per machine
paths:
  root: "."
  data_root: "/path/to/your/data"
```

### Step 1: Data preparation

```bash
export CLEANSEMG_ROOT=/path/to/CleanSEMG
export DATA_ROOT=/path/to/your/data
bash scripts/run_pipeline.sh --config configs/config.yaml --stages all
```

| Step | Script | Description |
|---|---|---|
| 1–2 | `step1_qc_raw.py`, `step2_quality_filter.py` | QC metrics (SMR, SHR, OHM, Hampel); grade A/B/C |
| 3 | `step3_subject_split.py` | Cross-DB stratified split; DB2 held out as test |
| 4 | `step4_preproc_and_segment.py` | Bandpass 20–500 Hz, resample to 1 kHz, 2 s windows |
| Noise | `noise/noise_generator.py` | PLI/ECG/MOA/WGN/Color noise libraries |
| Test set | `evaluation/generate_test_data.py` | Fixed offline mixing at 7 SNR levels |

### Step 2: Train neural denoisers

```bash
for model in TrustEMGNet_RM FCN MSEMG SDEMG; do
    bash scripts/run_train.sh --config configs/config.yaml --model $model
done
```

### Step 3: Calibrate classical methods

```bash
python training/train_classical.py \
    --config        configs/config.yaml \
    --trad-config   configs/config_tradition.yaml \
    --segments-root outputs/segments/data_crossDB_seg2s \
    --noise-root    outputs/noise/sEMG_noise_train \
    --weights       outputs/weights_tradition \
    --methods       all
```

### Step 4: Evaluate all methods

```bash
bash scripts/run_evaluate.sh --test-data outputs/test_data/test_combined.npz
```

---

## Downstream Tasks

### Hand Gesture Recognition (Ninapro DB2 + STCNet)

STCNet (Yang et al., 2025) is used as the frozen gesture recognition classifier.

```bash
# Neural denoiser: generate denoised .mat files
python downstream/gesture/prepare_neural.py \
    --model-name      TrustEMGNet_RM \
    --model-path      outputs/weights_baseline/TrustEMGNet_RM/TrustEMGNet_RM_best.pth \
    --db2-root        $DATA_ROOT/DB2 \
    --test-npz        outputs/test_data/test_combined.npz \
    --qc-index        outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy    outputs/downstream/gesture/noisy/DB2 \
    --output-denoised outputs/downstream/gesture/trustemg/DB2

# Classical denoiser
python downstream/gesture/prepare_classical.py \
    --method hp --db2-root $DATA_ROOT/DB2 \
    --test-npz        outputs/test_data/test_combined.npz \
    --qc-index        outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy    outputs/downstream/gesture/noisy/DB2 \
    --output-denoised outputs/downstream/gesture/hp/DB2

# All 8 denoisers end-to-end
bash downstream/gesture/run_all.sh
```

### Fatigue Classification (Cerqueira Dataset)

```bash
# Step 1: Preprocess Cerqueira → 1 kHz cache (one-time)
python downstream/fatigue/preprocess_and_denoise.py \
    --cerqueira-data $DATA_ROOT/Cerqueira \
    --output-cache   outputs/downstream/fatigue/cache

# Step 2: Run a single experiment
export CLEANSEMG_ROOT=/path/to/CleanSEMG
python downstream/fatigue/fatigue_svm.py --denoiser trustemg
python downstream/fatigue/fatigue_cnn.py --denoiser trustemg

# Step 3: All 8 denoisers × 2 classifiers = 16 experiments
bash downstream/fatigue/run_all.sh
```

Supported `--denoiser`: `hp`, `emd`, `vmd`, `ceemdan`, `fcn`, `msemg`, `sdemg`, `trustemg`

---

## Results

**Table 1.** Signal-level and feature-level metrics on Ninapro DB2 (macro-average over all SNR × noise-type combinations).

| Method | SNRimp (dB) ↑ | RMSE (×10⁻⁵) ↓ | PRD (%) ↓ | LSD (dB) ↓ | RMSE-ZCR ↓ | RMSE-MNF ↓ | RMSE-MDF ↓ |
|---|---|---|---|---|---|---|---|
| HP | 1.27 | 5.44 | 121.80 | 0.662 | 83.64 | 32.78 | 40.48 |
| EMD | 0.34 | 5.45 | 119.58 | 0.593 | 107.91 | 31.90 | 44.38 |
| CEEMDAN | 0.46 | 5.30 | 116.57 | 0.581 | 98.47 | 32.63 | 46.33 |
| VMD | 0.31 | 10.00 | 120.02 | 0.624 | 98.48 | 36.97 | 64.16 |
| FCN | 6.95 | 2.41 | 54.07 | 0.224 | 47.59 | 16.92 | 25.17 |
| SDEMG | 5.69 | 2.94 | 65.95 | 0.272 | 51.09 | 16.58 | 28.31 |
| MSEMG | 7.24 | 2.41 | 53.91 | 0.230 | **45.82** | **14.88** | **25.05** |
| **TrustEMG-Net** | **7.72** | **2.27** | **50.72** | **0.213** | 47.87 | 16.15 | 25.16 |

**Table 2.** Downstream task performance.

| Method | Gesture Acc (%) ↑ | Gesture F1 (%) ↑ | Fatigue Acc (%) ↑ | Fatigue F1 (%) ↑ |
|---|---|---|---|---|
| Noisy | — | — | 51.63 | 51.28 |
| HP | 65.82 | 61.60 | 42.19 | 38.43 |
| EMD | 62.24 | 57.60 | 41.36 | 40.46 |
| CEEMDAN | 61.12 | 57.80 | 41.29 | 40.32 |
| VMD | 62.73 | 59.24 | 39.29 | 35.49 |
| FCN | 67.24 | 63.81 | **61.60** | **61.70** |
| SDEMG | 68.46 | **66.11** | 57.55 | 57.68 |
| MSEMG | 68.04 | 65.31 | 58.84 | 58.94 |
| **TrustEMG-Net** | **68.52** | 65.59 | 60.05 | 60.22 |

Fatigue results use the Dilated CNN classifier. See Appendix for SVM results.

---

## Project Structure

```
CleanSEMG/
├── configs/
│   ├── config.yaml                  # Main config (set data_root here)
│   ├── baseline_train_config.yaml
│   └── config_tradition.yaml
├── data_pipeline/
│   ├── step1_qc_raw.py
│   ├── step2_quality_filter.py
│   ├── step3_subject_split.py
│   └── step4_preproc_and_segment.py
├── denoising/
│   ├── FCN.py
│   ├── CNN.py
│   ├── MSEMG.py
│   ├── SDEMG.py                     # Self-contained (no external repo needed)
│   ├── TrustEMGNet.py
│   ├── classical_filters.py
│   └── __init__.py                  # BASELINE_MODEL_REGISTRY
├── training/
│   ├── train_neural.py
│   └── train_classical.py
├── evaluation/
│   ├── generate_test_data.py
│   ├── evaluate_neural.py
│   └── evaluate_classical.py
├── noise/
│   └── noise_generator.py
├── downstream/
│   ├── gesture/
│   │   ├── prepare_neural.py
│   │   ├── prepare_classical.py
│   │   ├── emg_preprocess.py
│   │   ├── run_all.sh
│   │   └── STCNet/
│   └── fatigue/
│       ├── preprocess_and_denoise.py
│       ├── fatigue_svm.py           # --denoiser {hp,...,trustemg}
│       ├── fatigue_cnn.py
│       └── run_all.sh
└── scripts/
    ├── run_pipeline.sh
    ├── run_train.sh
    └── run_evaluate.sh
```

---

## Citation

```bibtex
@inproceedings{cleansemg2026,
  title     = {{CLEANSEMG}: A Benchmark for Single-Channel
               Surface Electromyography Denoising Algorithms},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

<details>
<summary>All dataset and method citations</summary>

```bibtex
% Ninapro Database
@article{atzori2014ninapro,
  title   = {Electromyography data for non-invasive naturally-controlled
             robotic hand prostheses},
  author  = {Atzori, Manfredo and others},
  journal = {Scientific Data},
  volume  = {1},
  pages   = {140053},
  year    = {2014},
  doi     = {10.1038/sdata.2014.53}
}

% Cerqueira fatigue dataset
@article{cerqueira2024fatigue,
  title   = {Muscular Fatigue Dataset},
  author  = {Cerqueira, Ana Sofia and others},
  journal = {Sensors},
  volume  = {24},
  number  = {24},
  pages   = {8081},
  year    = {2024},
  doi     = {10.3390/s24248081}
}

% MIT-BIH databases (ECG + MOA noise sources)
@article{goldberger2000physiobank,
  title   = {{PhysioBank, PhysioToolkit, and PhysioNet}},
  author  = {Goldberger, Ary L and others},
  journal = {Circulation},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  year    = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}

% Machado MOA noise dataset
@article{machado2021motion,
  title   = {A dataset of surface electromyography signals obtained under
             controlled artifact-contamination conditions},
  author  = {Machado, Andr{\'{e}} Ferreira and others},
  journal = {Biomedical Signal Processing and Control},
  volume  = {68},
  pages   = {102752},
  year    = {2021},
  doi     = {10.1016/j.bspc.2021.102752}
}

% TrustEMG-Net
@article{wang2025trustemg,
  title   = {{TrustEMG-Net}: Trustworthy {EMG} Denoising with Evidential
             Uncertainty Estimation Based on Dynamic Residual Masking},
  author  = {Wang, Eric and others},
  journal = {{IEEE} Journal of Biomedical and Health Informatics},
  volume  = {29},
  number  = {4},
  pages   = {2506--2520},
  year    = {2025},
  doi     = {10.1109/JBHI.2024.3475817}
}

% SDEMG
@inproceedings{liu2024sdemg,
  title     = {{SDEMG}: Score-Based Diffusion Model for Surface
               Electromyographic Signal Denoising},
  author    = {Liu, Yuting and others},
  booktitle = {{IEEE} ICASSP},
  pages     = {1--5},
  year      = {2024},
  doi       = {10.1109/ICASSP48485.2024.10446431}
}

% MSEMG / EMG-MAMBA
@inproceedings{msemg2025,
  title     = {{EMG-MAMBA}: Surface {EMG} Signal Denoising with Selective
               State Space Models},
  booktitle = {{IEEE} ICASSP},
  year      = {2025},
}

% STCNet
@article{yang2025stcnet,
  title   = {{STCNet}: Spatio-Temporal Cross Network with subject-aware
             contrastive learning for hand gesture recognition in surface {EMG}},
  author  = {Yang, Jaemo and Cha, Doheun and Lee, Dong-Gyu and Ahn, Sangtae},
  journal = {Computers in Biology and Medicine},
  volume  = {185},
  pages   = {109525},
  year    = {2025},
  doi     = {10.1016/j.compbiomed.2024.109525}
}

% VMD
@article{dragomiretskiy2014vmd,
  title   = {Variational Mode Decomposition},
  author  = {Dragomiretskiy, Konstantin and Zosso, Dominique},
  journal = {{IEEE} Transactions on Signal Processing},
  volume  = {62},
  number  = {3},
  pages   = {531--544},
  year    = {2014},
  doi     = {10.1109/TSP.2013.2288675}
}

% CEEMDAN
@inproceedings{torres2011ceemdan,
  title     = {A complete ensemble empirical mode decomposition with adaptive noise},
  author    = {Torres, Mar{\'{i}}a E and Colominas, Marcelo A and
               Schlotthauer, Gast{\'{o}}n and Flandrin, Patrick},
  booktitle = {{IEEE} ICASSP},
  pages     = {4144--4147},
  year      = {2011},
  doi       = {10.1109/ICASSP.2011.5947265}
}

% EMG feature extraction
@article{phinyomark2012feature,
  title   = {Feature Reduction and Selection for {EMG} Signal Classification},
  author  = {Phinyomark, Angkoon and Phukpattaranont, Pornchai and
             Limsakul, Chusak},
  journal = {Expert Systems with Applications},
  volume  = {39},
  number  = {8},
  pages   = {7420--7431},
  year    = {2012},
  doi     = {10.1016/j.eswa.2012.01.102}
}
```

</details>

---

## License

Code is released under the **MIT License**. Datasets retain their original licenses: Cerqueira is CC BY 4.0; Ninapro is academic non-commercial; MIT-BIH is ODC-BY.