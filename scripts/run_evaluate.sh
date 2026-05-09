#!/usr/bin/env bash
# Baseline Waveform Models — Inference Runner
# Metrics synced with inference.py
# SDEMG added: uses T=50 reverse diffusion steps (slower than L1 models)

set -euo pipefail

CONFIG="config.yaml"
GPU="1"
MODELS=""
TEST_DATA_FILE=""
METRICS="SNRimp,RMSE,PRD,LSD,RMSE_ARV,RMSE_ZCR,RMSE_MNF,RMSE_MDF,RMSE_Kurtosis"
SAMPLING_RATE="1000"
BATCH_SIZE="32"
FORCE=0
WEIGHTS_DIR=""
TEST_DATA_DIR=""
OUT_DIR=""

ALL_MODELS=(TrustEMGNet_RM MSEMG TrustEMGNet_UNetonly FCN CNN_waveform SDEMG)

usage() {
  echo "Usage: bash run_inference_baseline.sh [OPTIONS]"
  echo "  --config CONFIG       (default: config.yaml)"
  echo "  --gpu GPU_ID          (default: 1)"
  echo "  --models MODELS       comma-separated, default: all 6"
  echo "  --test-data-file FILE (default: test_combined.npz)"
  echo "  --metrics METRICS     (default: all 9)"
  echo "  --sr RATE             (default: 1000)"
  echo "  --batch SIZE          (default: 32)"
  echo "  --force               re-run even if output exists"
  echo "  --weights PATH        weights root override"
  echo "  --test-data-dir PATH  test data dir override"
  echo "  --out PATH            inference output root override"
  echo ""
  echo "SDEMG note: inference runs T=50 diffusion steps per sample — expect"
  echo "  longer runtime. Reduce --batch if GPU OOM."
  echo "  export SDEMG_REPO_PATH=/path/to/SDEMG  (default: ./SDEMG/)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         CONFIG="$2";         shift 2;;
    --gpu)            GPU="$2";            shift 2;;
    --models)         MODELS="$2";         shift 2;;
    --test-data-file) TEST_DATA_FILE="$2"; shift 2;;
    --metrics)        METRICS="$2";        shift 2;;
    --sr)             SAMPLING_RATE="$2";  shift 2;;
    --batch)          BATCH_SIZE="$2";     shift 2;;
    --force)          FORCE=1;             shift 1;;
    --weights)        WEIGHTS_DIR="$2";    shift 2;;
    --test-data-dir)  TEST_DATA_DIR="$2";  shift 2;;
    --out)            OUT_DIR="$2";        shift 2;;
    -h|--help)        usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "[CONFIG] Parsing: $CONFIG"
eval "$(python3 - <<PY
import os, yaml, sys
try:
    with open("$CONFIG", "r") as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"echo '[ERROR] {e}'; exit 1", file=sys.stderr); sys.exit(1)
def get_nested(d, keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d
root     = get_nested(cfg, ["paths", "root"], ".")
base     = get_nested(cfg, ["paths", "output", "base"], "outputs")
out_base = base if os.path.isabs(base) else os.path.join(root, base)
td_rel   = get_nested(cfg, ["paths", "output", "test_data"], "test_data")
print(f'OUT_BASE="{out_base}"')
print(f'TEST_DATA_REL="{td_rel}"')
PY
)"

OUT_BASE="${OUT_BASE:?}"
TD_DIR="${TEST_DATA_DIR:-${OUT_BASE}/${TEST_DATA_REL}}"
W_ROOT="${WEIGHTS_DIR:-${OUT_BASE}/weights_baseline}"
INFER_ROOT="${OUT_DIR:-${OUT_BASE}/inference_baseline}"

[[ -z "$TEST_DATA_FILE" ]] && TEST_DATA_FILE="test_combined.npz"
TEST_DATA_PATH="${TD_DIR}/${TEST_DATA_FILE}"

if [[ ! -f "$TEST_DATA_PATH" ]]; then
  echo "[ERROR] Test data not found: $TEST_DATA_PATH"
  exit 1
fi

if [[ -n "$MODELS" ]]; then
  IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
else
  MODEL_LIST=("${ALL_MODELS[@]}")
fi

echo "=============================================================="
echo "Baseline Inference Runner"
echo "  config:           $CONFIG"
echo "  gpu:              $GPU"
echo "  test_data:        $TEST_DATA_PATH"
echo "  weights:          $W_ROOT"
echo "  output:           $INFER_ROOT"
echo "  models:           ${MODEL_LIST[*]}"
echo "  metrics:          $METRICS"
echo "  SDEMG_REPO_PATH:  ${SDEMG_REPO_PATH:-./SDEMG/ (default)}"
echo "=============================================================="

pick_ckpt() {
  local mn="$1"; local wroot="$2"
  python3 - "$mn" "$wroot" <<'PY'
import os, glob, sys
mn   = sys.argv[1]
wdir = os.path.join(sys.argv[2], mn)
p1   = os.path.join(wdir, f"{mn}_best.pth")
if os.path.exists(p1): print(p1); raise SystemExit
cands = sorted(glob.glob(os.path.join(wdir, "*.pth")), key=os.path.getmtime, reverse=True)
print(cands[0] if cands else "")
PY
}

FAILED=()
for MODEL in "${MODEL_LIST[@]}"; do
  echo ""
  echo "╔══ ${MODEL} ══╗"
  CKPT="$(pick_ckpt "$MODEL" "$W_ROOT")"
  if [[ -z "$CKPT" ]]; then
    echo "[SKIP] No checkpoint for ${MODEL}"
    FAILED+=("$MODEL")
    continue
  fi
  echo "  checkpoint: $CKPT"
  OUT="${INFER_ROOT}/${MODEL}"
  [[ $FORCE -eq 1 ]] && rm -rf "${OUT:?}" || true
  mkdir -p "$OUT"

  python3 inference_baseline.py \
    --config    "$CONFIG" \
    --model     "$MODEL" \
    --ckpt      "$CKPT" \
    --test-data "$TEST_DATA_PATH" \
    --output    "$OUT" \
    --batch     "$BATCH_SIZE" \
    --metrics   "$METRICS" \
    --sr        "$SAMPLING_RATE" \
  && echo "✓ ${MODEL}" || { echo "✗ ${MODEL} failed"; FAILED+=("$MODEL"); }
done

echo ""
echo "=============================================================="
echo "✓ Baseline Inference Complete → $INFER_ROOT"
[[ ${#FAILED[@]} -gt 0 ]] && echo "⚠  Failed: ${FAILED[*]}"
echo "=============================================================="