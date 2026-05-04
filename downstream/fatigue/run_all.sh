#!/usr/bin/env bash
# ==============================================================================
# CleanSEMG — Fatigue Downstream: Run All 8 Denoisers
# downstream/fatigue/run_all.sh
#
# Runs all 8 denoisers × 2 classifiers (SVM + CNN) = 16 experiments.
#
# Prerequisites:
#   1. Download Cerqueira dataset from Zenodo:
#      https://zenodo.org/records/13860256
#   2. Run preprocessing to build cache:
#      python downstream/fatigue/preprocess_and_denoise.py \
#          --cerqueira-data /path/to/Cerqueira
#   3. Train neural denoisers (or download weights from HuggingFace):
#      bash scripts/run_train.sh
#   4. Calibrate classical methods:
#      python training/train_classical.py --config configs/config.yaml
#
# Usage:
#   export CLEANSEMG_ROOT=/path/to/CleanSEMG
#   export DATA_ROOT=/path/to/your/data
#   bash downstream/fatigue/run_all.sh
#
#   # Only SVM:
#   bash downstream/fatigue/run_all.sh --classifiers svm
#
#   # Only one denoiser:
#   bash downstream/fatigue/run_all.sh --denoisers trustemg
# ==============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CLEANSEMG_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"

CLASSIFIERS="svm cnn"           # space-separated subset or "all"
DENOISERS="hp emd vmd ceemdan fcn msemg sdemg trustemg"
GPU="0"
EXTRA_ARGS=""

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --classifiers) CLASSIFIERS="$2"; shift 2;;
        --denoisers)   DENOISERS="$2";   shift 2;;
        --gpu)         GPU="$2";         shift 2;;
        *) echo "[WARN] Unknown arg: $1"; shift;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export CLEANSEMG_ROOT="${ROOT}"
[[ -n "${DATA_ROOT}" ]] && export DATA_ROOT

# ── Derived paths ─────────────────────────────────────────────────────────────
CACHE_DIR="${ROOT}/outputs/downstream/fatigue/cache"
NOISE_ROOT="${ROOT}/outputs/noise/sEMG_noise_test"
WEIGHTS_DIR="${ROOT}/outputs/weights_baseline"
TRAD_PARAMS="${ROOT}/outputs/weights_tradition/tradition_params.json"
RESULTS_DIR="${ROOT}/outputs/downstream/fatigue/results"
LOG_DIR="${ROOT}/outputs/downstream/fatigue/logs"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -d "${CACHE_DIR}/baseline" ]]; then
    echo "[ERROR] Cache not found: ${CACHE_DIR}/baseline"
    echo "        Run preprocess_and_denoise.py first."
    exit 1
fi

if [[ ! -d "${NOISE_ROOT}" ]]; then
    echo "[ERROR] Noise pool not found: ${NOISE_ROOT}"
    echo "        Run: bash scripts/run_pipeline.sh --stages noise"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  CleanSEMG Fatigue Downstream — All Experiments"
echo "============================================================"
echo "  GPU           : ${GPU}"
echo "  Classifiers   : ${CLASSIFIERS}"
echo "  Denoisers     : ${DENOISERS}"
echo "  Cache dir     : ${CACHE_DIR}"
echo "  Noise root    : ${NOISE_ROOT}"
echo "  Results dir   : ${RESULTS_DIR}"
echo "============================================================"
echo ""

FAILED=()

run_one() {
    local clf="$1"
    local denoiser="$2"
    local script="${SCRIPT_DIR}/fatigue_${clf}.py"
    local log="${LOG_DIR}/${clf}_${denoiser}.log"

    echo "──── ${clf^^} + ${denoiser^^} ────"

    python3 "${script}" \
        --denoiser       "${denoiser}" \
        --cache-dir      "${CACHE_DIR}" \
        --noise-root     "${NOISE_ROOT}" \
        --weights-dir    "${WEIGHTS_DIR}" \
        --tradition-params "${TRAD_PARAMS}" \
        --output-dir     "${RESULTS_DIR}/${clf}_${denoiser}" \
        2>&1 | tee "${log}"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "  ✓ Done → ${RESULTS_DIR}/${clf}_${denoiser}"
    else
        echo "  ✗ FAILED (see ${log})"
        FAILED+=("${clf}+${denoiser}")
    fi
    echo ""
}

for denoiser in ${DENOISERS}; do
    for clf in ${CLASSIFIERS}; do
        run_one "${clf}" "${denoiser}"
    done
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Summary"
echo "============================================================"

python3 - <<PYEOF
import os, glob, pandas as pd
from pathlib import Path

results_dir = "${RESULTS_DIR}"
denoisers   = "${DENOISERS}".split()
classifiers = "${CLASSIFIERS}".split()

print(f"\n{'Denoiser':<14}", end="")
for clf in classifiers:
    print(f"{'SVM Acc':>12}" if clf == "svm" else f"{'CNN Acc':>12}", end="")
print()
print("─" * (14 + 12 * len(classifiers)))

for d in denoisers:
    print(f"{d.upper():<14}", end="")
    for clf in classifiers:
        csv_path = Path(results_dir) / f"{clf}_{d}" / "fold_results.csv"
        try:
            df  = pd.read_csv(csv_path)
            acc = df["acc_denoised"].mean()
            print(f"{acc:>11.2f}%", end="")
        except Exception:
            print(f"{'—':>12}", end="")
    print()

print()
PYEOF

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[WARN] Failed experiments: ${FAILED[*]}"
fi

echo "Done.  Results → ${RESULTS_DIR}"