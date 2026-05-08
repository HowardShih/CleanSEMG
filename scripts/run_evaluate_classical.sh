#!/usr/bin/env bash
# Traditional Baseline Inference v2  (HP + TS + EMD + VMD + CEEMDAN)
# Metrics synced with inference.py v6.7.0

set -euo pipefail

CONFIG="config.yaml"
TRAD_CONFIG="tradition_train_config.yaml"
PARAMS=""
TEST_DATA_FILE=""
METHODS="all"
METRICS="SNRimp,RMSE,PRD,LSD,RMSE_ARV,RMSE_ZCR,RMSE_MNF,RMSE_MDF,RMSE_Kurtosis"
SR="1000"
N_JOBS="1"
FORCE=0
WEIGHTS_DIR=""
OUT_DIR=""
TEST_DATA_DIR=""

usage() {
  echo "Usage: bash run_inference_tradition.sh [OPTIONS]"
  echo ""
  echo "  --config CONFIG        Main config.yaml (default: config.yaml)"
  echo "  --trad-config CONFIG   Traditional baseline config"
  echo "  --params PATH          tradition_params.json (auto-detected)"
  echo "  --test-data-file FILE  (default: test_combined.npz)"
  echo "  --methods METHODS      Comma-separated or 'all' (default: hp,ts)"
  echo "                         Choices: hp ts emd vmd ceemdan"
  echo "  --metrics METRICS      Comma-separated metric names"
  echo "  --sr RATE              Sampling rate in Hz (default: 1000)"
  echo "  --weights PATH         Weights dir override"
  echo "  --out PATH             Output dir override"
  echo "  --test-data-dir PATH   Test data dir override"
  echo "  --force                Clear output dir before running"
  echo ""
  echo "Examples:"
  echo "  bash run_inference_tradition.sh --methods hp,ts"
  echo "  bash run_inference_tradition.sh --methods all"
  echo "  bash run_inference_tradition.sh --methods emd,vmd,ceemdan"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         CONFIG="$2";         shift 2;;
    --trad-config)    TRAD_CONFIG="$2";    shift 2;;
    --params)         PARAMS="$2";         shift 2;;
    --test-data-file) TEST_DATA_FILE="$2"; shift 2;;
    --methods)        METHODS="$2";        shift 2;;
    --metrics)        METRICS="$2";        shift 2;;
    --sr)             SR="$2";             shift 2;;
    --force)          FORCE=1;             shift 1;;
    --weights)        WEIGHTS_DIR="$2";    shift 2;;
    --out)            OUT_DIR="$2";        shift 2;;
    --test-data-dir)  TEST_DATA_DIR="$2";  shift 2;;
    --n-jobs)         N_JOBS="$2";         shift 2;;
    -h|--help)        usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "[CONFIG] Parsing: $CONFIG"
eval "$(python3 - <<PY
import os, yaml, sys

def get_nested(d, keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d

try:
    with open("$CONFIG") as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"echo '[ERROR] {e}'; exit 1", file=sys.stderr); sys.exit(1)

root     = get_nested(cfg, ["paths", "root"], ".")
base     = get_nested(cfg, ["paths", "output", "base"], "outputs")
out_base = base if os.path.isabs(base) else os.path.join(root, base)
td_rel   = get_nested(cfg, ["paths", "output", "test_data"], "test_data")
exp_name = get_nested(cfg, ["exp", "name"], "exp")

print(f'OUT_BASE="{out_base}"')
print(f'TEST_DATA_REL="{td_rel}"')
print(f'EXP_NAME="{exp_name}"')
PY
)"

OUT_BASE="${OUT_BASE:?}"
TD_DIR="${TEST_DATA_DIR:-${OUT_BASE}/${TEST_DATA_REL}}"
W_DIR="${WEIGHTS_DIR:-${OUT_BASE}/weights_tradition}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

[[ -z "$TEST_DATA_FILE" ]] && TEST_DATA_FILE="test_combined.npz"
DATA_TAG="$(basename "$TEST_DATA_FILE" .npz)"
METHOD_TAG="$(echo "$METHODS" | tr ',' '_' | tr '/' '_')"

INFER_OUT="${OUT_DIR:-${OUT_BASE}/inference_tradition/${EXP_NAME}/${DATA_TAG}/${METHOD_TAG}_${RUN_ID}}"

[[ -z "$PARAMS" ]] && PARAMS="${W_DIR}/tradition_params.json"
if [[ ! -f "$PARAMS" ]]; then
  echo "[ERROR] Params not found: $PARAMS"
  echo "        Run first:  bash run_train_tradition.sh --methods ${METHODS}"
  exit 1
fi

TEST_DATA_PATH="${TD_DIR}/${TEST_DATA_FILE}"
if [[ ! -f "$TEST_DATA_PATH" ]]; then
  echo "[ERROR] Test data not found: $TEST_DATA_PATH"
  exit 1
fi

[[ $FORCE -eq 1 ]] && { echo "[FORCE] Removing ${INFER_OUT}"; rm -rf "${INFER_OUT:?}" || true; }
mkdir -p "$INFER_OUT"

echo "=============================================================="
echo "Traditional Baseline Inference v2"
echo "  config:      $CONFIG"
echo "  params:      $PARAMS"
echo "  test_data:   $TEST_DATA_PATH"
echo "  methods:     $METHODS"
echo "  output:      $INFER_OUT"
echo "  metrics:     $METRICS"
echo "=============================================================="

python3 inference_tradition.py \
  --config      "$CONFIG" \
  --trad-config "$TRAD_CONFIG" \
  --params      "$PARAMS" \
  --test-data   "$TEST_DATA_PATH" \
  --output      "$INFER_OUT" \
  --methods     "$METHODS" \
  --metrics     "$METRICS" \
  --sr          "$SR" \
  --n-jobs      "$N_JOBS"

if [[ $? -eq 0 ]]; then
  echo ""
  echo "=============================================================="
  echo "✓ Tradition Inference Complete → $INFER_OUT"
  # Show summary CSVs that were generated
  for m in hp ts emd vmd ceemdan; do
    F="${INFER_OUT}/${m}_overall_summary.csv"
    [[ -f "$F" ]] && echo "  ${m}: $F"
  done
  echo "=============================================================="
else
  echo "✗ Tradition Inference Failed"; exit 1
fi