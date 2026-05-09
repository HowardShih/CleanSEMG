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
| **sEMG datasets** | Ninapro DB2, DB3, DB4, DB7, DB8, DB10; Upper-Limb sEMG Fatigue Dataset |
| **Noise types** | PLI, ECG artifact, Motion artifact (MOA), WGN, Colored noise (pink/brown) |
| **Classical methods** | HP/IIR (TrustEMG-Net contaminant-aware filter chain), EMD, CEEMDAN, VMD |
| **Neural methods** | FCN, SDEMG, MSEMG (EMG-MAMBA), TrustEMG-Net |
| **Signal metrics** | SNRimp, RMSE, PRD, LSD |
| **Feature metrics** | ARV, ZCR, MNF, MDF, Kurtosis (RMSE vs. clean) |
| **Downstream tasks** | Hand gesture recognition (STCNet on Ninapro DB2), Fatigue classification (Dilated CNN on Cerqueira dataset) |

---

## Installation

```bash
git clone https://anonymous.4open.science/r/CleanSEMG
cd CleanSEMG
pip install -r requirements.txt
```

**SDEMG** — the score-based diffusion denoiser (Liu et al., ICASSP 2024) is fully embedded in `denoising/SDEMG.py`. No external repository is needed.

**MSEMG** — requires `mamba-ssm` with a CUDA GPU (Triton kernel; CUDA ≥ 11.8):

```bash
pip install mamba-ssm
```

All other methods (FCN, SDEMG, TrustEMG-Net, all classical) run on CPU.

---

## Datasets

### sEMG signal data

| Dataset | Source | License | Role |
|---|---|---|---|
| **Ninapro DB2** | [ninapro.hevs.ch](http://ninapro.hevs.ch) | Academic non-commercial | Test (gesture recognition) |
| **Ninapro DB3, DB4, DB7, DB8, DB10** | [ninapro.hevs.ch](http://ninapro.hevs.ch) | Academic non-commercial | Train |
| **Upper-Limb sEMG Fatigue Dataset** | [Zenodo 13860256](https://zenodo.org/records/13860256) | CC BY 4.0 | Test (fatigue classification) |

> Ninapro raw `.mat` files cannot be redistributed. Download them directly from ninapro.hevs.ch.

### Contaminant sources

| Source | Type | License |
|---|---|---|
| Synthetic | PLI, WGN, Colored noise | — (generated) |
| MIT-BIH NSR Database ([PhysioNet](https://physionet.org/content/nsrdb/1.0.0/)) | ECG artifact | ODC-BY |
| MIT-BIH NSTDB ([PhysioNet](https://physionet.org/content/nstdb/1.0.0/)) | Motion artifact (electrode-motion channel) | ODC-BY |
| Machado et al., BSPC 2021 ([doi](https://doi.org/10.1016/j.bspc.2021.102752)) | Motion artifact | See paper — **not redistributable** |

> The Machado et al. motion artifact dataset is used for training but cannot be redistributed. It is therefore not included in the HuggingFace release. Users wishing to reproduce the full MOA noise pool should download the dataset from the original authors and rerun `noise/noise_generator.py`.

### HuggingFace artifacts

We provide a pre-built evaluation package on HuggingFace sufficient to reproduce the main Table 1 results without downloading raw Ninapro data.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AnonResearcher0029/CLEANsEMG",
    repo_type="dataset",
    local_dir="data/cleansemg"
)
```

| Path | Description |
|---|---|
| `test/test_combined.npz` | Fixed test set: Ninapro DB2 clean/noisy pairs, 7 SNR levels × 5 contaminant counts |
| `test/test_metadata_combined.csv` | Per-segment metadata for `test_combined.npz` |
| `test/test_metadata_all.csv` | Metadata for all 35 (SNR, k) conditions (per-condition files regenerable via `evaluation/generate_test_data.py`) |
| `noise_pool/test/{PLI,ECG,WGN,Color}/` | Test noise library (MOA excluded; see note above) |
| `metadata/tradition_params.json` | Pre-calibrated HP/IIR, EMD, VMD, CEEMDAN parameters |
| `metadata/ecg_split.json` | Train/test ECG subject split for contaminant-source mismatch |
| `train/manifests/segment_manifest.csv` | Index of all clean training segments |
| `fatigue/baseline/subject_N/` | Preprocessed Cerqueira fatigue sEMG at 1 kHz (used by fatigue downstream scripts) |
| `fatigue/labels/subject_N/` | Self-perceived fatigue labels (0=non-fatigue, 1=transition, 2=fatigue) |

---

## Quick Start

### Evaluate with pre-built data and weights

**1. Download weights**

```python
from huggingface_hub import hf_hub_download
import os

for fname in ["FCN_best.pth", "MSEMG_best.pth",
              "SDEMG_best.pth", "TrustEMGNet_RM_best.pth"]:
    model_name = fname.replace("_best.pth", "")
    os.makedirs(f"outputs/weights_baseline/{model_name}", exist_ok=True)
    hf_hub_download(
        repo_id="AnonResearcher0029/CLEANsEMG-weights",
        filename=fname,
        local_dir=f"outputs/weights_baseline/{model_name}/"
    )
```

**2. Evaluate neural denoisers**

```bash
for model in FCN MSEMG SDEMG TrustEMGNet_RM; do
    python evaluation/evaluate_neural.py \
        --config   configs/config.yaml \
        --model    $model \
        --ckpt     outputs/weights_baseline/${model}/${model}_best.pth \
        --test-data data/cleansemg/test/test_combined.npz \
        --output   results/${model}/
done
```

**3. Evaluate classical methods**

```bash
for method in hp emd vmd ceemdan; do
    python evaluation/evaluate_classical.py \
        --config      configs/config.yaml \
        --trad-config configs/config_tradition.yaml \
        --params      data/cleansemg/metadata/tradition_params.json \
        --test-data   data/cleansemg/test/test_combined.npz \
        --output      results/${method}/ \
        --methods     $method
done
```

> **Note on HP/IIR:** The HP baseline implements the TrustEMG-Net contaminant-aware IIR filter chain (De Luca et al. [26] extended by Wang et al. [22]), not a single high-pass filter. The filter is noise-type-aware: PLI uses a notch filter, ECG/MOA use high-pass filters with source-specific cutoffs, and WGN/CLN use bandpass filtering.

---

## Full Reproduction

### Step 1: Configure paths

Edit `configs/config.yaml` and set `paths.root` and `paths.data_root`:

```yaml
paths:
  root: "."
  data_root: "/path/to/your/data"   # directory containing Ninapro and Cerqueira raw data
```

### Step 2: Data preparation

```bash
export CLEANSEMG_ROOT=/path/to/CleanSEMG
export DATA_ROOT=/path/to/your/data
bash scripts/run_pipeline.sh --config configs/config.yaml --stages all
```

| Stage | Script | Description |
|---|---|---|
| data | `data_pipeline/step1_qc_raw.py`, `step2_quality_filter.py` | QC filtering (SMR, SHR, OHM, Hampel) |
| data | `data_pipeline/step3_subject_split.py` | Cross-DB stratified split; DB2 held out as test |
| data | `data_pipeline/step4_preproc_and_segment.py` | Bandpass 20–500 Hz → 1 kHz → 2 s segments (unnormalized) |
| noise | `noise/noise_generator.py` | Generate PLI, ECG, MOA, WGN, Colored noise libraries |
| testdata | `evaluation/generate_test_data.py` | Fixed offline mixing: 7 SNR × 5 k = 35 conditions |

### Step 3: Train neural denoisers

```bash
for model in TrustEMGNet_RM FCN MSEMG SDEMG; do
    bash scripts/run_train.sh --config configs/config.yaml --model $model
done
```

### Step 4: Calibrate classical methods

```bash
python training/train_classical.py \
    --config        configs/config.yaml \
    --trad-config   configs/config_tradition.yaml \
    --segments-root outputs/segments/data_crossDB_seg2s \
    --noise-root    outputs/noise/sEMG_noise_train \
    --weights       outputs/weights_tradition \
    --methods       all
```

### Step 5: Evaluate

```bash
# Neural denoisers
bash scripts/run_evaluate.sh --config configs/config.yaml

# Classical methods
bash scripts/run_evaluate_classical.sh --config configs/config.yaml --methods all
```

---

## Downstream Tasks

### Hand Gesture Recognition (Ninapro DB2)

The gesture recognition backbone is **STCNet** (Yang et al., 2025).

```bash
# Preprocess DB2 and apply denoiser (neural)
python downstream/gesture/prepare_neural.py \
    --model-name      TrustEMGNet_RM \
    --model-path      outputs/weights_baseline/TrustEMGNet_RM/TrustEMGNet_RM_best.pth \
    --db2-root        $DATA_ROOT/DB2 \
    --test-npz        outputs/test_data/test_combined.npz \
    --qc-index        outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy    outputs/downstream/gesture/noisy/DB2 \
    --output-denoised outputs/downstream/gesture/trustemg/DB2

# Run all 8 denoisers
bash downstream/gesture/run_all.sh
```

### Fatigue Classification (Cerqueira Dataset)

Uses the Cerqueira et al. (2024) dataset with a Dilated CNN classifier.

```bash
# Preprocess Cerqueira dataset to 1 kHz cache (one-time)
python downstream/fatigue/preprocess_and_denoise.py \
    --cerqueira-data $DATA_ROOT/Cerqueira \
    --output-cache   outputs/downstream/fatigue/cache

# Run single denoiser
export CLEANSEMG_ROOT=/path/to/CleanSEMG
python downstream/fatigue/fatigue_cnn.py --denoiser trustemg

# Run all 8 denoisers
bash downstream/fatigue/run_all.sh --classifiers cnn
```

Supported `--denoiser` values: `hp`, `emd`, `vmd`, `ceemdan`, `fcn`, `msemg`, `sdemg`, `trustemg`

---

## Results

**Table 1.** Signal reconstruction and feature preservation on Ninapro DB2 (macro-average over all SNR × contaminant-type conditions).

| Method | SNRimp (dB) ↑ | RMSE (×10⁻⁵) ↓ | PRD (%) ↓ | LSD (dB) ↓ | RMSE-ZCR ↓ | RMSE-MNF ↓ | RMSE-MDF ↓ |
|---|---|---|---|---|---|---|---|
| HP/IIR | 1.18 | 5.86 | 130.82 | 0.72 | 91.17 | 34.94 | 43.53 |
| EMD | 0.46 | 5.32 | 116.78 | 0.57 | 106.06 | 31.01 | 44.06 |
| CEEMDAN | 0.58 | 5.17 | 113.88 | 0.56 | 96.88 | 31.90 | 46.12 |
| VMD | 0.44 | 5.30 | 117.32 | 0.60 | 99.17 | 35.64 | 64.74 |
| FCN | 6.60 | 2.59 | 57.96 | 0.24 | 50.34 | 18.79 | 26.29 |
| SDEMG | 5.55 | 3.05 | 68.38 | 0.29 | 49.10 | 17.18 | 26.80 |
| MSEMG | 6.96 | 2.55 | 56.94 | 0.24 | 48.33 | **17.05** | **25.95** |
| **TrustEMG-Net** | **7.36** | **2.45** | **54.73** | **0.23** | 49.91 | 17.97 | 26.01 |

**Table 2.** Downstream task performance.

| Method | Gesture Acc (%) ↑ | Gesture F1 (%) ↑ | Fatigue Acc (%) ↑ | Fatigue F1 (%) ↑ |
|---|---|---|---|---|
| HP/IIR | 65.82 | 61.60 | 42.78 | 39.49 |
| EMD | 62.24 | 57.60 | 40.93 | 34.93 |
| CEEMDAN | 61.12 | 57.80 | 40.89 | 39.75 |
| VMD | 62.73 | 59.24 | 38.94 | 34.93 |
| FCN | 67.24 | 63.81 | **61.60** | **61.70** |
| SDEMG | **68.46** | **66.11** | 57.55 | 57.68 |
| MSEMG | 68.04 | 65.31 | 58.84 | 58.94 |
| **TrustEMG-Net** | 68.52 | 65.59 | 60.05 | 60.22 |

Fatigue results use the Dilated CNN classifier evaluated via 3-fold cross-validation.

---

## Verification

Run the smoke test to verify all model imports and forward passes:

```bash
cd /path/to/CleanSEMG
python scripts/smoke_test.py
```

Expected: all tests PASS in ~60 seconds on CPU.

---

## Project Structure

```
CleanSEMG/
├── configs/
│   ├── config.yaml                  # Main config (set data_root here)
│   ├── baseline_train_config.yaml   # Neural model training config
│   └── config_tradition.yaml        # Classical method config
├── data_pipeline/
│   ├── step1_qc_raw.py
│   ├── step2_quality_filter.py
│   ├── step3_subject_split.py
│   └── step4_preproc_and_segment.py
├── denoising/
│   ├── FCN.py                       # Adapted from Wang et al., IEEE Sensors J., 2023
│   ├── MSEMG.py                     # Adapted from Liu et al., ICASSP 2025
│   ├── SDEMG.py                     # Adapted from Liu et al., ICASSP 2024 (self-contained)
│   ├── TrustEMGNet.py               # Adapted from Wang et al., IEEE JBHI, 2025
│   ├── classical_filters.py         # HP/IIR, EMD, VMD, CEEMDAN implementations
│   └── __init__.py                  # BASELINE_MODEL_REGISTRY
├── training/
│   ├── train_neural.py
│   └── train_classical.py
├── evaluation/
│   ├── generate_test_data.py        # Offline test set generation (v6.8.0)
│   ├── evaluate_neural.py           # Neural denoiser inference + metrics
│   └── evaluate_classical.py        # Classical method inference + metrics
├── noise/
│   └── noise_generator.py           # Noise library generation + online mixing
├── downstream/
│   ├── gesture/
│   │   ├── prepare_neural.py
│   │   ├── prepare_classical.py
│   │   ├── emg_preprocess.py
│   │   ├── run_all.sh
│   │   └── STCNet/                  # Adapted from Yang et al., Comput. Biol. Med., 2025
│   └── fatigue/
│       ├── preprocess_and_denoise.py
│       ├── fatigue_cnn.py           # Dilated CNN classifier
│       └── run_all.sh
└── scripts/
    ├── run_pipeline.sh              # Data preparation (steps 1–4, noise, test data)
    ├── run_train.sh                 # Neural model training
    ├── run_evaluate.sh              # Neural denoiser evaluation
    ├── run_evaluate_classical.sh    # Classical method evaluation
    └── smoke_test.py                # Installation verification
```

---

## Third-Party Code

This repository adapts code from the following open-source projects. All original licenses are retained in their respective subdirectories, and all original papers are cited.

| File / Directory | Original work | Paper | Code |
|---|---|---|---|
| `denoising/FCN.py` | Wang et al., IEEE Sensors J., 2023 | [doi:10.1109/JSEN.2023.3234567](https://doi.org/10.1109/JSEN.2023.3234567) | [GitHub](https://github.com/ASUS217/ECG-removal-from-sEMG) |
| `denoising/MSEMG.py` | Liu et al., ICASSP 2025 | [doi:10.1109/ICASSP49660.2025.10887547](https://doi.org/10.1109/ICASSP49660.2025.10887547) | — |
| `denoising/SDEMG.py` | Liu et al., ICASSP 2024 | [doi:10.1109/ICASSP48485.2024.10446154](https://doi.org/10.1109/ICASSP48485.2024.10446154) | [GitHub](https://github.com/tonyliu0910/SDEMG) |
| `denoising/TrustEMGNet.py` | Wang et al., IEEE JBHI, 2025 | [doi:10.1109/JBHI.2024.3504378](https://doi.org/10.1109/JBHI.2024.3504378) | [GitHub](https://github.com/eric-wang135/TrustEMG) |
| `downstream/gesture/STCNet/` | Yang et al., Comput. Biol. Med., 2025 | [doi:10.1016/j.compbiomed.2024.109525](https://doi.org/10.1016/j.compbiomed.2024.109525) | [GitHub](https://github.com/jaemoyang/STCNet) |

Each file contains an attribution header specifying the original source and modifications made for this benchmark.

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
<summary>Dataset and method citations</summary>

```bibtex
% Ninapro Database
@article{atzori2014ninapro,
  title   = {Electromyography data for non-invasive naturally-controlled
             robotic hand prostheses},
  author  = {Atzori, Manfredo and others},
  journal = {Scientific Data},
  volume  = {1}, pages = {140053}, year = {2014},
  doi     = {10.1038/sdata.2014.53}
}

% Cerqueira Fatigue Dataset
@article{cerqueira2024fatigue,
  title   = {Muscular Fatigue Dataset},
  author  = {Cerqueira, Sofia M. and others},
  journal = {Sensors},
  volume  = {24}, number = {24}, pages = {8081}, year = {2024},
  doi     = {10.3390/s24248081}
}

% MIT-BIH databases (PhysioNet)
@article{goldberger2000physiobank,
  title   = {{PhysioBank, PhysioToolkit, and PhysioNet}},
  author  = {Goldberger, Ary L and others},
  journal = {Circulation},
  volume  = {101}, number = {23}, pages = {e215--e220}, year = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}

% MIT-BIH NSTDB
@inproceedings{moody1984noise,
  title     = {A noise stress test for arrhythmia detectors},
  author    = {Moody, George B. and Muldrow, W. K. and Mark, Roger G.},
  booktitle = {Computers in Cardiology},
  volume    = {11}, pages = {381--384}, year = {1984}
}

% Machado MOA dataset
@article{machado2021motion,
  title   = {A dataset of surface electromyography signals obtained under
             controlled artifact-contamination conditions},
  author  = {Machado, Juliano and others},
  journal = {Biomedical Signal Processing and Control},
  volume  = {68}, pages = {102752}, year = {2021},
  doi     = {10.1016/j.bspc.2021.102752}
}

% FCN
@article{wang2023ecgsemg,
  title   = {Removing {ECG} Artifacts from Surface Electromyogram Signals
             Using Fully Convolutional Networks},
  author  = {Wang, Kuan-Chen and Liu, Kai-Chun and Tsao, Yu},
  journal = {{IEEE} Sensors Journal},
  year    = {2023},
  doi     = {10.1109/JSEN.2023.3234567}
}

% TrustEMG-Net
@article{wang2025trustemg,
  title   = {{TrustEMG-Net}: Using Representation-Masking Transformer
             with U-Net for Surface Electromyography Enhancement},
  author  = {Wang, Kuan-Chen and others},
  journal = {{IEEE} Journal of Biomedical and Health Informatics},
  volume  = {29}, number = {4}, year = {2025},
  doi     = {10.1109/JBHI.2024.3504378}
}

% SDEMG
@inproceedings{liu2024sdemg,
  title     = {{SDEMG}: Score-Based Diffusion Model for Surface
               Electromyographic Signal Denoising},
  author    = {Liu, Yu-Tung and others},
  booktitle = {{IEEE} ICASSP},
  pages     = {1736--1740}, year = {2024},
  doi       = {10.1109/ICASSP48485.2024.10446154}
}

% MSEMG
@inproceedings{liu2025msemg,
  title     = {{MSEMG}: Surface Electromyography Denoising with a
               Mamba-Based Efficient Network},
  author    = {Liu, Yu-Tung and others},
  booktitle = {{IEEE} ICASSP},
  pages     = {1--5}, year = {2025},
  doi       = {10.1109/ICASSP49660.2025.10887547}
}

% STCNet
@article{yang2025stcnet,
  title   = {{STCNet}: Spatio-Temporal Cross Network with subject-aware
             contrastive learning for hand gesture recognition in surface {EMG}},
  author  = {Yang, Jaemo and Cha, Doheun and Lee, Dong-Gyu and Ahn, Sangtae},
  journal = {Computers in Biology and Medicine},
  volume  = {185}, pages = {109525}, year = {2025},
  doi     = {10.1016/j.compbiomed.2024.109525}
}

% VMD
@article{dragomiretskiy2014vmd,
  title   = {Variational Mode Decomposition},
  author  = {Dragomiretskiy, Konstantin and Zosso, Dominique},
  journal = {{IEEE} Transactions on Signal Processing},
  volume  = {62}, number = {3}, pages = {531--544}, year = {2014},
  doi     = {10.1109/TSP.2013.2288675}
}

% CEEMDAN
@inproceedings{torres2011ceemdan,
  title     = {A complete ensemble empirical mode decomposition with adaptive noise},
  author    = {Torres, Mar{\'{i}}a E and others},
  booktitle = {{IEEE} ICASSP},
  pages     = {4144--4147}, year = {2011},
  doi       = {10.1109/ICASSP.2011.5947265}
}

% EMD
@article{huang1998emd,
  title   = {The empirical mode decomposition and the {Hilbert} spectrum for
             nonlinear and non-stationary time series analysis},
  author  = {Huang, Norden E. and others},
  journal = {Proceedings of the Royal Society A},
  volume  = {454}, number = {1971}, pages = {903--995}, year = {1998},
  doi     = {10.1098/rspa.1998.0193}
}

% HP/IIR filter reference
@article{deluca2010filtering,
  title   = {Filtering the surface {EMG} signal: Movement artifact and
             baseline noise contamination},
  author  = {De Luca, Carlo J. and others},
  journal = {Journal of Biomechanics},
  volume  = {43}, number = {8}, pages = {1573--1579}, year = {2010},
  doi     = {10.1016/j.jbiomech.2010.01.027}
}
```

</details>

---

## License

Code written for this benchmark is released under the **MIT License**. Third-party code in `denoising/` and `downstream/gesture/STCNet/` retains its original license — see individual files and subdirectory `LICENSE` files for details.

Datasets retain their original licenses: the Cerqueira fatigue dataset is CC BY 4.0; Ninapro datasets are academic non-commercial; MIT-BIH databases are ODC-BY; the Machado et al. motion artifact dataset is subject to the terms in the original publication.