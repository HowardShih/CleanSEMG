#!/usr/bin/env bash
# ==============================================================================
# Cross-DB sEMG Denoising Pipeline Runner (DATA-ONLY)
# Scope: Data preparation only (Step1-4, noise generation, test data)
# Training: Use run_train.sh separately
# Inference: Use run_inference.sh separately
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="config.yaml"
STAGES="all"          # all | data,noise,testdata
DB=""                 # optional: run data stage for a single DB
FORCE=0

usage() {
  echo "Usage: bash run_pipeline.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --config CONFIG        Config file (default: config.yaml)"
  echo "  --stages STAGES        Stages to run (default: all)"
  echo "                         Available: all | data,noise,testdata"
  echo "  --db DB                Process specific database for data stage"
  echo "  --force                Force regeneration (overwrite existing)"
  echo ""
  echo "Pipeline stages (DATA PREPARATION ONLY):"
  echo "  data      - Step1-4: QC, filtering, segmentation (saves RAW segments)"
  echo "  noise     - Generate noise pools (train + test)"
  echo "  testdata  - Generate offline test data with deterministic mixing"
  echo ""
  echo "Note: This script only handles DATA PREPARATION."
  echo "      For training, use: bash run_train.sh"
  echo "      For inference, use: bash run_inference.sh"
  echo ""
  echo "Examples:"
  echo "  bash run_pipeline.sh --stages all"
  echo "  bash run_pipeline.sh --stages data,noise,testdata"
  echo "  bash run_pipeline.sh --stages testdata --force"
  echo "  bash run_pipeline.sh --stages data --db DB2"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --db) DB="$2"; shift 2;;
    --force) FORCE=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

# ----------------------------
# Extract config
# ----------------------------
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

root = get_nested(cfg, ["paths", "root"], ".")
base = get_nested(cfg, ["paths", "output", "base"], "outputs")
out_base = base if os.path.isabs(base) else os.path.join(root, base)

seg = get_nested(cfg, ["paths", "output", "segments"], "segments")
noise_train = get_nested(cfg, ["paths", "output", "noise_train"], "noise_train")
noise_test = get_nested(cfg, ["paths", "output", "noise_test"], "noise_test")
test_data = get_nested(cfg, ["paths", "output", "test_data"], "test_data")
test_db = get_nested(cfg, ["datasets", "test_db"], "")

train_dbs = get_nested(cfg, ["datasets", "train_valid_dbs"], [])
if train_dbs is None:
    train_dbs = []
if isinstance(train_dbs, str):
    train_dbs = [x.strip() for x in train_dbs.split(",") if x.strip()]
elif isinstance(train_dbs, dict):
    train_dbs = list(train_dbs.keys())
elif not isinstance(train_dbs, list):
    train_dbs = [str(train_dbs)]
train_csv = ",".join([str(x) for x in train_dbs])

print(f'ROOT="{root}"')
print(f'OUT_BASE="{out_base}"')
print(f'SEG_REL="{seg}"')
print(f'NOISE_TRAIN_REL="{noise_train}"')
print(f'NOISE_TEST_REL="{noise_test}"')
print(f'TEST_DATA_REL="{test_data}"')
print(f'TEST_DB="{test_db}"')
print(f'TRAIN_DBS_CSV="{train_csv}"')
PY
)"

if [[ -z "${OUT_BASE:-}" ]]; then
  echo "[ERROR] Failed to extract config variables"
  exit 1
fi

SEG_DIR="${OUT_BASE}/${SEG_REL}"
NOISE_TRAIN_DIR="${OUT_BASE}/${NOISE_TRAIN_REL}"
NOISE_TEST_DIR="${OUT_BASE}/${NOISE_TEST_REL}"
TEST_DATA_DIR="${OUT_BASE}/${TEST_DATA_REL:-test_data}"

FORCE_FLAG=""
if [[ $FORCE -eq 1 ]]; then FORCE_FLAG="--force"; fi

# Parse stages (only data, noise, testdata allowed)
if [[ "$STAGES" == "all" ]]; then
  STAGE_LIST=(data noise testdata)
else
  IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
fi

# Validate stages
for stage in "${STAGE_LIST[@]}"; do
  if [[ ! "$stage" =~ ^(data|noise|testdata)$ ]]; then
    echo "[ERROR] Invalid stage: $stage"
    echo "        This script only supports: data, noise, testdata"
    echo "        For training, use: bash run_train.sh"
    echo "        For inference, use: bash run_inference.sh"
    exit 1
  fi
done

echo "=============================================================="
echo "Cross-DB sEMG Pipeline (DATA-ONLY)"
echo "=============================================================="
echo "Config:           $CONFIG"
echo "Stages:           ${STAGE_LIST[*]}"
echo "Root:             $ROOT"
echo "Out base:         $OUT_BASE"
echo "Segments dir:     $SEG_DIR"
echo "Noise train dir:  $NOISE_TRAIN_DIR"
echo "Noise test dir:   $NOISE_TEST_DIR"
echo "Test data dir:    $TEST_DATA_DIR"
echo "Test DB:          ${TEST_DB}"
echo "Train DBs:        ${TRAIN_DBS_CSV}"
echo "DB (data-only):   ${DB:-'(all from config)'}"
echo "Force:            $FORCE"
echo "=============================================================="
echo ""
echo "📌 Note: This pipeline handles DATA PREPARATION only"
echo "   After completion, run training with: bash run_train.sh"
echo "=============================================================="

# ----------------------------
# Stage: DATA (Step1-4)
# ----------------------------
run_data() {
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║               DATA STAGE: Step 1-4                         ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  
  echo "---- [DATA] Step1 QC raw ----"
  if [[ -n "$DB" ]]; then
    python3 step1_qc_raw.py --config "$CONFIG" --db "$DB" $FORCE_FLAG
  else
    python3 step1_qc_raw.py --config "$CONFIG" $FORCE_FLAG
  fi

  echo ""
  echo "---- [DATA] Step2 quality filter ----"
  if [[ -n "$DB" ]]; then
    python3 step2_quality_filter.py --config "$CONFIG" --db "$DB" $FORCE_FLAG
  else
    python3 step2_quality_filter.py --config "$CONFIG" $FORCE_FLAG
  fi

  echo ""
  echo "---- [DATA] Step3 subject split ----"
  python3 step3_subject_split.py --config "$CONFIG" $FORCE_FLAG

  echo ""
  echo "---- [DATA] Step4 preprocess + segment ----"
  echo "⚠️  Saving RAW segments + clean_scale_factor (reference only)"
  python3 step4_preproc_and_segment.py --config "$CONFIG" $FORCE_FLAG
  
  echo ""
  echo "✓ DATA stage complete"
  echo "  Output: $SEG_DIR"
}

# ----------------------------
# Stage: NOISE
# ----------------------------
run_noise() {
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║               NOISE STAGE: Generate Pools                  ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  
  if [[ $FORCE -eq 1 ]]; then
    echo "[NOISE] --force: removing old noise pools"
    rm -rf "$NOISE_TRAIN_DIR" "$NOISE_TEST_DIR" || true
  fi
  
  python3 noise.py --config "$CONFIG" --mode both
  
  echo ""
  echo "✓ NOISE stage complete"
  echo "  Train noise: $NOISE_TRAIN_DIR"
  echo "  Test noise:  $NOISE_TEST_DIR"
}

# ----------------------------
# Stage: TESTDATA
# ----------------------------
run_testdata() {
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║          TESTDATA STAGE: Offline Test Data Gen             ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  
  if [[ $FORCE -eq 1 ]]; then
    echo "[TESTDATA] --force: removing old test data"
    rm -rf "$TEST_DATA_DIR" || true
  fi
  
  echo "⚠️  Generating with Noisy-Scale Policy"
  echo "   Scale computed from noisy_raw to avoid clipping"
  
  python3 generate_test_data.py \
    --config "$CONFIG" \
    --segments-root "$SEG_DIR" \
    --noise-root "$NOISE_TEST_DIR" \
    --output-dir "$TEST_DATA_DIR" \
    --split test \
    $FORCE_FLAG
  
  echo ""
  echo "✓ TESTDATA stage complete"
  echo "  Output: $TEST_DATA_DIR"
  
  # Show generated files
  if [[ -d "$TEST_DATA_DIR" ]]; then
    echo ""
    echo "Generated test data files:"
    ls -lh "$TEST_DATA_DIR"/*.npz 2>/dev/null | head -5 | awk '{print "  " $9 " (" $5 ")"}'
    total_files=$(ls "$TEST_DATA_DIR"/*.npz 2>/dev/null | wc -l)
    if [[ $total_files -gt 5 ]]; then
      echo "  ... and $((total_files - 5)) more files"
    fi
  fi
}

# ----------------------------
# Run stages
# ----------------------------
for st in "${STAGE_LIST[@]}"; do
  case "$st" in
    data)     run_data ;;
    noise)    run_noise ;;
    testdata) run_testdata ;;
  esac
done

echo ""
echo "=============================================================="
echo "✓ Data Preparation Pipeline Complete"
echo "=============================================================="
echo ""
echo "📊 Summary:"
echo "  ✅ Segments:   $SEG_DIR"
echo "  ✅ Noise:      $NOISE_TRAIN_DIR (train), $NOISE_TEST_DIR (test)"
echo "  ✅ Test Data:  $TEST_DATA_DIR"
echo ""
echo "📌 Next Steps:"
echo "  1. Train model:     bash run_train.sh --config $CONFIG"
echo "  2. Run inference:   bash run_inference.sh --config $CONFIG"
echo ""
echo "=============================================================="