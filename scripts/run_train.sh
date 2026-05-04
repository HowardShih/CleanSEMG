#!/usr/bin/env bash
# ==============================================================================
# Baseline Waveform Models — Training Runner
# Trains TrustEMGNet variants, FCN, CNN_waveform, SDEMG using the same data
# pipeline as the main MECGE model (same segments, same noise, same normalization).
#
# Outputs: outputs/weights_baseline/{ModelName}/
# Use: bash run_train_baseline.sh --gpu 7
#
# SDEMG note:
#   SDEMG uses diffusion MSE loss internally — val/train loss values are NOT
#   comparable to those of L1 waveform models. Set SDEMG_REPO_PATH if the
#   cloned SDEMG repo is not at ./SDEMG/ relative to this script.
#     export SDEMG_REPO_PATH=/path/to/SDEMG
# ==============================================================================

set -euo pipefail

CONFIG="config.yaml"
BASELINE_CONFIG="baseline_train_config.yaml"
GPU="0"
MODELS=""
TRAIN_DBS=""
FORCE=0

EPOCHS=""
BATCH_SIZE=""
LR=""
PATIENCE=""

SEGMENTS_ROOT=""
NOISE_ROOT=""
WEIGHTS_DIR=""

ALL_MODELS=(
    TrustEMGNet_RM
    MSEMG
    TrustEMGNet_UNetonly
    FCN
    CNN_waveform
    SDEMG
)

usage() {
  echo "Usage: bash run_train_baseline.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --config CONFIG        Config file (default: config.yaml)"
  echo "  --gpu GPU_ID           GPU to use (default: 0)"
  echo "  --models MODELS        Comma-separated model names (default: all)"
  echo "                         ${ALL_MODELS[*]}"
  echo "  --train-datasets DBs   Comma-separated DB names (default: from config)"
  echo "  --force                Remove existing best.pth before training"
  echo ""
  echo "Hyper-param overrides (optional):"
  echo "  --epochs N"
  echo "  --batch-size N"
  echo "  --lr FLOAT"
  echo "  --patience N"
  echo ""
  echo "Path overrides (optional):"
  echo "  --segments-root PATH"
  echo "  --noise-root PATH"
  echo "  --weights PATH"
  echo ""
  echo "SDEMG env var:"
  echo "  export SDEMG_REPO_PATH=/path/to/SDEMG   (default: ./SDEMG/)"
  echo ""
  echo "Examples:"
  echo "  # Train all models"
  echo "  bash run_train_baseline.sh --gpu 1"
  echo ""
  echo "  # Train only SDEMG"
  echo "  bash run_train_baseline.sh --models SDEMG --gpu 2"
  echo ""
  echo "  # Train only TrustEMGNet_RM and FCN"
  echo "  bash run_train_baseline.sh --models \"TrustEMGNet_RM,FCN\" --gpu 2"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         CONFIG="$2";       shift 2;;
    --baseline-config) BASELINE_CONFIG="$2"; shift 2;;
    --gpu)            GPU="$2";          shift 2;;
    --models)         MODELS="$2";       shift 2;;
    --train-datasets) TRAIN_DBS="$2";    shift 2;;
    --force)          FORCE=1;           shift 1;;
    --epochs)         EPOCHS="$2";       shift 2;;
    --batch-size)     BATCH_SIZE="$2";   shift 2;;
    --lr)             LR="$2";           shift 2;;
    --patience)       PATIENCE="$2";     shift 2;;
    --segments-root)  SEGMENTS_ROOT="$2"; shift 2;;
    --noise-root)     NOISE_ROOT="$2";   shift 2;;
    --weights)        WEIGHTS_DIR="$2";  shift 2;;
    -h|--help)        usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ── Parse config paths ──────────────────────────────────────────────────────
echo "[CONFIG] Parsing: $CONFIG"
eval "$(python3 - <<PY
import os, yaml, sys
try:
    with open("$CONFIG", "r") as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"echo '[ERROR] Failed to parse config: {e}'; exit 1", file=sys.stderr)
    sys.exit(1)

def get_nested(d, keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

root     = get_nested(cfg, ["paths", "root"], ".")
base     = get_nested(cfg, ["paths", "output", "base"], "outputs")
out_base = base if os.path.isabs(base) else os.path.join(root, base)
seg      = get_nested(cfg, ["paths", "output", "segments"], "segments")
noise_tr = get_nested(cfg, ["paths", "output", "noise_train"], "noise_train")

train_dbs = get_nested(cfg, ["datasets", "train_valid_dbs"], []) or []
if isinstance(train_dbs, str):
    train_dbs = [x.strip() for x in train_dbs.split(",") if x.strip()]
elif isinstance(train_dbs, dict):
    train_dbs = list(train_dbs.keys())
train_csv = ",".join(str(x) for x in train_dbs)

print(f'OUT_BASE="{out_base}"')
print(f'SEG_REL="{seg}"')
print(f'NOISE_TRAIN_REL="{noise_tr}"')
print(f'TRAIN_DBS_CSV="{train_csv}"')
PY
)"

OUT_BASE="${OUT_BASE:?}"
SEG_DIR="${SEGMENTS_ROOT:-${OUT_BASE}/${SEG_REL}}"
NOISE_TR_DIR="${NOISE_ROOT:-${OUT_BASE}/${NOISE_TRAIN_REL}}"
W_DIR="${WEIGHTS_DIR:-${OUT_BASE}/weights_baseline}"

TRAIN_DBS_USE="${TRAIN_DBS:-$TRAIN_DBS_CSV}"

# ── Resolve model list ───────────────────────────────────────────────────────
if [[ -n "$MODELS" ]]; then
  IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
else
  MODEL_LIST=("${ALL_MODELS[@]}")
fi

echo "=============================================================="
echo "Baseline Training Runner"
echo "=============================================================="
echo "config:          $CONFIG"
echo "baseline cfg:    $BASELINE_CONFIG"
echo "gpu:             $GPU"
echo "segments_root:   $SEG_DIR"
echo "noise_root:      $NOISE_TR_DIR"
echo "weights_dir:     $W_DIR"
echo "train_datasets:  $TRAIN_DBS_USE"
echo "models:          ${MODEL_LIST[*]}"
echo "force:           $FORCE"
echo "SDEMG_REPO_PATH: ${SDEMG_REPO_PATH:-./SDEMG/ (default)}"
echo "=============================================================="
echo ""
echo "📌 Using SAME segments and noise as MECGE (shared pipeline)"
echo "=============================================================="

EXTRA_ARGS=()
[[ -n "$EPOCHS"     ]] && EXTRA_ARGS+=(--epochs     "$EPOCHS")
[[ -n "$BATCH_SIZE" ]] && EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
[[ -n "$LR"         ]] && EXTRA_ARGS+=(--lr         "$LR")
[[ -n "$PATIENCE"   ]] && EXTRA_ARGS+=(--patience   "$PATIENCE")

# ── Train each model ─────────────────────────────────────────────────────────
for MODEL in "${MODEL_LIST[@]}"; do
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║  Training: $MODEL"
  echo "╚════════════════════════════════════════════════════════════╝"

  MODEL_W_DIR="${W_DIR}/${MODEL}"
  mkdir -p "$MODEL_W_DIR"
  BEST_PTH="${MODEL_W_DIR}/${MODEL}_best.pth"

  if [[ $FORCE -eq 1 ]]; then
    if [[ -f "$BEST_PTH" ]]; then
      echo "[FORCE] Removing: $BEST_PTH"
      rm -f "$BEST_PTH"
    fi
  elif [[ -f "$BEST_PTH" ]]; then
    echo "⏭  Skipping ${MODEL} — already trained: $BEST_PTH"
    echo "   (Use --force to retrain)"
    continue
  fi

  python3 train_baseline.py \
    --config         "$CONFIG" \
    --baseline-config "$BASELINE_CONFIG" \
    --model          "$MODEL" \
    --segments-root  "$SEG_DIR" \
    --noise-root     "$NOISE_TR_DIR" \
    --weights        "$W_DIR" \
    --train-datasets "$TRAIN_DBS_USE" \
    "${EXTRA_ARGS[@]}"

  echo "✓ ${MODEL} done → ${MODEL_W_DIR}/${MODEL}_best.pth"
done

echo ""
echo "=============================================================="
echo "✓ All baseline models trained"
echo "  Weights: $W_DIR"
echo ""
echo "Next: bash run_inference_baseline.sh --config $CONFIG"
echo "=============================================================="