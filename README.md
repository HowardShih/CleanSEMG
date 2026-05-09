# CLEANSEMG: A Benchmark for Single-Channel sEMG Denoising

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)](https://anonymous.4open.science/r/CleanSEMG)
[![Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/AnonResearcher0029/CLEANsEMG)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CLEANSEMG is the first standardized benchmark for evaluating single-channel surface EMG (sEMG) denoising methods. It provides a unified evaluation framework covering **signal-level reconstruction quality**, **feature-level preservation**, and **downstream task performance** across multiple datasets, noise types, and denoising approaches.

---

## Overview

| | Details |
|---|---|
| **sEMG datasets** | Ninapro DB2, DB3, DB4, DB7, DB8, DB10; Upper-Limb sEMG Fatigue Dataset |
| **Noise types** | PLI, ECG artifact, Motion artifact (MOA), WGN, Colored (pink/brown) |
| **Classical methods** | HP (high-pass filter), EMD, CEEMDAN, VMD |
| **Neural methods** | FCN, SDEMG, MSEMG (EMG-MAMBA), TrustEMG-Net |
| **Signal metrics** | SNRimp, RMSE, PRD, LSD |
| **Feature metrics** | ARV, ZCR, MNF, MDF, Kurtosis (RMSE vs. clean) |
| **Downstream tasks** | Hand gesture recognition (STCNet), Fatigue classification (RBF-SVM + Dilated CNN) |

---

## Installation

```bash
git clone https://github.com/HowardShih/CleanSEMG.git
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
| **Ninapro DB2** | [ninapro.hevs.ch](http://ninapro.hevs.ch) | Academic non-commercial | Test (gesture) |
| **Ninapro DB3, DB4, DB7, DB8, DB10** | [ninapro.hevs.ch](http://ninapro.hevs.ch) | Academic non-commercial | Train |
| **Upper-Limb sEMG Fatigue Dataset** | [Zenodo 13860256](https://zenodo.org/records/13860256) | CC BY 4.0 | Test (fatigue) |

> Ninapro raw .mat files cannot be redistributed. Download them directly from ninapro.hevs.ch.

### Contaminant sources

| Source | Type | License |
|---|---|---|
| Synthetic | PLI, WGN, Colored noise | — (generated) |
| MIT-BIH NSR Database ([PhysioNet](https://physionet.org/content/nsrdb/1.0.0/)) | ECG artifact | ODC-BY |
| MIT-BIH NSTDB ([PhysioNet](https://physionet.org/content/nstdb/1.0.0/)) | Motion artifact (electrode motion channel) | ODC-BY |
| Machado et al., BSPC 2021 ([doi](https://doi.org/10.1016/j.bspc.2021.102752)) | Motion artifact | See paper |

### HuggingFace artifacts

We provide a pre-built evaluation package on HuggingFace containing the fixed test set, noise pool, and pre-calibrated classical parameters — sufficient to reproduce Table 1 without downloading raw Ninapro data.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AnonResearcher0029/CLEANsEMG",
    repo_type="dataset",
    local_dir="data/sample"
)
```

| File | Description |
|---|---|
| `test_combined.npz` | Fixed test set: clean/noisy pairs at 7 SNR levels × 5 noise types |
| `noise_pool/{PLI,ECG,MOA,WGN,Color}/` | Noise libraries |
| `tradition_params.json` | Pre-calibrated HP/EMD/VMD/CEEMDAN parameters |

---

## Quick Start

### Evaluate with pre-built data and weights

Download the test set and pre-trained weights, then run the evaluation scripts.

**1. Download weights**

```python
from huggingface_hub import hf_hub_download
import os

for fname in [
    "FCN_best.pth", "MSEMG_best.pth",
    "SDEMG_best.pth", "TrustEMGNet_RM_best.pth",
]:
    name = fname.replace("_best.pth", "")
    os.makedirs(f"outputs/weights_baseline/{name}", exist_ok=True)
    hf_hub_download(
        repo_id="AnonResearcher0029/CLEANsEMG-weights",
        filename=fname,
        local_dir=f"outputs/weights_baseline/{name}/"
    )
```

**2. Evaluate**

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

# Classical methods
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

## Full Reproduction

### Step 1: Configure paths

Set `data_root` in `configs/config.yaml` — this is the only line that changes per machine:

```yaml
paths:
  root: "."
  data_root: "/path/to/your/data"
```

### Step 2: Data preparation

```bash
export CLEANSEMG_ROOT=/path/to/CleanSEMG
export DATA_ROOT=/path/to/your/data
bash scripts/run_pipeline.sh --config configs/config.yaml --stages all
```

| Step | Script | Description |
|---|---|---|
| 1–2 | `step1_qc_raw.py`, `step2_quality_filter.py` | QC metrics (SMR, SHR, OHM, Hampel); grade A/B/C |
| 3 | `step3_subject_split.py` | Cross-DB stratified split; DB2 held out as test |
| 4 | `step4_preproc_and_segment.py` | Bandpass 20–500 Hz → 1 kHz → 2 s windows |
| Noise | `noise/noise_generator.py` | PLI/ECG/MOA/WGN/Color noise libraries |
| Test set | `evaluation/generate_test_data.py` | Fixed offline mixing at 7 SNR levels |

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
bash scripts/run_evaluate.sh --test-data outputs/test_data/test_combined.npz
```

---

## Downstream Tasks

### Hand Gesture Recognition

The gesture recognition backbone is **STCNet** (Yang et al., 2025), a spatio-temporal cross-network for sEMG-based hand gesture classification.

- **Paper:** Yang et al., "STCNet: Spatio-Temporal Cross Network with subject-aware contrastive learning for hand gesture recognition in surface EMG," *Computers in Biology and Medicine*, vol. 185, p. 109525, 2025. [doi:10.1016/j.compbiomed.2024.109525](https://doi.org/10.1016/j.compbiomed.2024.109525)
- **Original code:** [https://github.com/jaemoyang/STCNet](https://github.com/jaemoyang/STCNet)

The STCNet code in `downstream/gesture/STCNet/` is adapted from the original repository with modifications for the CLEANSEMG evaluation pipeline. See `downstream/gesture/STCNet/LICENSE` for the original license.

```bash
# Generate denoised inputs — neural denoiser
python downstream/gesture/prepare_neural.py \
    --model-name      TrustEMGNet_RM \
    --model-path      outputs/weights_baseline/TrustEMGNet_RM/TrustEMGNet_RM_best.pth \
    --db2-root        $DATA_ROOT/DB2 \
    --test-npz        outputs/test_data/test_combined.npz \
    --qc-index        outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy    outputs/downstream/gesture/noisy/DB2 \
    --output-denoised outputs/downstream/gesture/trustemg/DB2

# Generate denoised inputs — classical denoiser
python downstream/gesture/prepare_classical.py \
    --method hp --db2-root $DATA_ROOT/DB2 \
    --test-npz        outputs/test_data/test_combined.npz \
    --qc-index        outputs/preprocessed/DB2/logs/qc_index.csv \
    --output-noisy    outputs/downstream/gesture/noisy/DB2 \
    --output-denoised outputs/downstream/gesture/hp/DB2

# Run all 8 denoisers
bash downstream/gesture/run_all.sh
```

### Fatigue Classification

Uses the Upper-Limb sEMG Fatigue Dataset (Cerqueira et al., Sensors, 2024) with two classifier families: RBF-SVM and Dilated CNN.

```bash
# Preprocess dataset to 1 kHz cache (one-time)
python downstream/fatigue/preprocess_and_denoise.py \
    --cerqueira-data $DATA_ROOT/Cerqueira \
    --output-cache   outputs/downstream/fatigue/cache

# Single experiment
export CLEANSEMG_ROOT=/path/to/CleanSEMG
python downstream/fatigue/fatigue_svm.py --denoiser trustemg
python downstream/fatigue/fatigue_cnn.py --denoiser trustemg

# All 8 denoisers × 2 classifiers = 16 experiments
bash downstream/fatigue/run_all.sh
```

Supported `--denoiser`: `hp`, `emd`, `vmd`, `ceemdan`, `fcn`, `msemg`, `sdemg`, `trustemg`

---

## Results

**Table 1.** Signal-level and feature-level metrics on Ninapro DB2 (macro-average over all SNR × noise-type).

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

Fatigue results use the Dilated CNN classifier. See the paper appendix for SVM results.

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
│   ├── FCN.py          # Adapted from Wang et al., IEEE Sensors J., 2023
│   ├── MSEMG.py        # Adapted from Liu et al., ICASSP 2025
│   ├── SDEMG.py        # Adapted from Liu et al., ICASSP 2024 (self-contained)
│   ├── TrustEMGNet.py  # Adapted from Wang et al., IEEE JBHI, 2025
│   ├── classical_filters.py
│   └── __init__.py     # BASELINE_MODEL_REGISTRY
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
│   │   └── STCNet/     # Adapted from Yang et al., Comput. Biol. Med., 2025
│   └── fatigue/
│       ├── preprocess_and_denoise.py
│       ├── fatigue_svm.py    # --denoiser {hp,...,trustemg}
│       ├── fatigue_cnn.py
│       └── run_all.sh
└── scripts/
    ├── run_pipeline.sh
    ├── run_train.sh
    └── run_evaluate.sh
```

---

## Third-Party Code

This repository adapts code from the following open-source projects. All original licenses are retained in their respective subdirectories, and all original papers are cited in the methods section.

| File / Directory | Original work | Paper | Code |
|---|---|---|---|
| `denoising/FCN.py` | Wang et al., IEEE Sensors J., 2023 | [doi:10.1109/JSEN.2023.3234567](https://doi.org/10.1109/JSEN.2023.3234567) | [GitHub](https://github.com/ASUS217/ECG-removal-from-sEMG) |
| `denoising/MSEMG.py` | Liu et al., ICASSP 2025 | [doi:10.1109/ICASSP49660.2025.10887547](https://doi.org/10.1109/ICASSP49660.2025.10887547) | — |
| `denoising/SDEMG.py` | Liu et al., ICASSP 2024 | [doi:10.1109/ICASSP48485.2024.10446154](https://doi.org/10.1109/ICASSP48485.2024.10446154) | [GitHub](https://github.com/tonyliu0910/SDEMG) |
| `denoising/TrustEMGNet.py` | Wang et al., IEEE JBHI, 2025 | [doi:10.1109/JBHI.2024.3504378](https://doi.org/10.1109/JBHI.2024.3504378) | [GitHub](https://github.com/eric-wang135/TrustEMG) |
| `downstream/gesture/STCNet/` | Yang et al., Comput. Biol. Med., 2025 | [doi:10.1016/j.compbiomed.2024.109525](https://doi.org/10.1016/j.compbiomed.2024.109525) | [GitHub](https://github.com/jaemoyang/STCNet) |

Each file contains an attribution header at the top specifying the original source and the modifications made for this benchmark.

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
  volume  = {1}, pages = {140053}, year = {2014},
  doi     = {10.1038/sdata.2014.53}
}

% Upper-Limb sEMG Fatigue Dataset
@article{cerqueira2024fatigue,
  title   = {Muscular Fatigue Dataset},
  author  = {Cerqueira, Sofia M. and others},
  journal = {Sensors},
  volume  = {24}, number = {24}, pages = {8081}, year = {2024},
  doi     = {10.3390/s24248081}
}

% MIT-BIH databases (ECG + MOA noise)
@article{goldberger2000physiobank,
  title   = {{PhysioBank, PhysioToolkit, and PhysioNet}},
  author  = {Goldberger, Ary L and others},
  journal = {Circulation},
  volume  = {101}, number = {23}, pages = {e215--e220}, year = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}

% MIT-BIH NSTDB (MOA noise)
@inproceedings{moody1984noise,
  title     = {A noise stress test for arrhythmia detectors},
  author    = {Moody, George B. and Muldrow, W. K. and Mark, Roger G.},
  booktitle = {Computers in Cardiology},
  volume    = {11}, pages = {381--384}, year = {1984}
}

% Machado MOA noise dataset
@article{machado2021motion,
  title   = {A dataset of surface electromyography signals obtained under
             controlled artifact-contamination conditions},
  author  = {Machado, Juliano and others},
  journal = {Biomedical Signal Processing and Control},
  volume  = {68}, pages = {102752}, year = {2021},
  doi     = {10.1016/j.bspc.2021.102752}
}

% FCN (ECG removal baseline, also used as FCN denoiser)
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
  title   = {{TrustEMG-Net}: Trustworthy {EMG} Denoising with Evidential
             Uncertainty Estimation Based on Dynamic Residual Masking},
  author  = {Wang, Kuan-Chen and others},
  journal = {{IEEE} Journal of Biomedical and Health Informatics},
  volume  = {29}, number = {4}, pages = {2506--2520}, year = {2025},
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

% MSEMG / EMG-MAMBA
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

% HP filter
@article{deluca2010filtering,
  title   = {Filtering the surface {EMG} signal: Movement artifact and
             baseline noise contamination},
  author  = {De Luca, Carlo J. and others},
  journal = {Journal of Biomechanics},
  volume  = {43}, number = {8}, pages = {1573--1579}, year = {2010},
  doi     = {10.1016/j.jbiomech.2010.01.027}
}

% EMG feature extraction (SVM fatigue)
@article{phinyomark2012feature,
  title   = {Feature Reduction and Selection for {EMG} Signal Classification},
  author  = {Phinyomark, Angkoon and Phukpattaranont, Pornchai and
             Limsakul, Chusak},
  journal = {Expert Systems with Applications},
  volume  = {39}, number = {8}, pages = {7420--7431}, year = {2012},
  doi     = {10.1016/j.eswa.2012.01.102}
}
```

</details>

---

## License

Code written for this benchmark is released under the **MIT License**. Third-party code in `denoising/` and `downstream/gesture/STCNet/` retains its original license — see individual files and subdirectory LICENSE files for details.

Datasets retain their original licenses: the Upper-Limb sEMG Fatigue Dataset is CC BY 4.0; Ninapro is academic non-commercial; MIT-BIH databases are ODC-BY.