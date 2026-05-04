#!/usr/bin/env bash
# ${CLEANSEMG_ROOT}/downstream_tasks/stcnet/run_stcnet_all_denoisers.sh
################################################################################
# STCNet Downstream Task — All 8 Denoisers
#
# Denoisers:
#   Traditional : HP  EMD  VMD  CEEMDAN
#   Neural      : FCN  SDEMG  MSEMG  TrustEMGNet_RM
#
# Shared (run once):
#   Noisy MAT   → outputs/noisy_data_1k_baseline/DB2   (reuse if exists)
#   Noisy PKL   → STCNet/pkl_noisy_v11                 (reuse if exists)
#   Baseline test (pkl_baseline)                        (reuse if exists)
#
# Per denoiser:
#   Step 1: Generate denoised MAT
#   Step 2: emg_preprocess_fixed_v4.py  → denoised PKL
#   Step 3: test.py                     → denoised results
#
# Final: summary table across all 8 denoisers + baseline + noisy
################################################################################

set -euo pipefail

# ==============================================================================
# Denoisers
# ==============================================================================
TRAD_METHODS=("hp" "emd" "vmd" "ceemdan")
NEURAL_MODELS=("FCN" "SDEMG" "MSEMG" "TrustEMGNet_RM")

# ==============================================================================
# Paths
# ==============================================================================

BASE_DIR="${CLEANSEMG_ROOT}"
DOWNSTREAM_DIR="${BASE_DIR}/downstream_tasks/stcnet"
DB2_ROOT="${DATA_ROOT}/DB2"

TEST_NPZ="${BASE_DIR}/outputs/test_data/test_combined.npz"
QC_INDEX_PATH="${BASE_DIR}/outputs/preprocessed/DB2/logs/qc_index.csv"

# Calibrated tradition params (optional; built-in defaults used if absent)
TRAD_PARAMS_JSON="${BASE_DIR}/outputs/weights_tradition/tradition_params.json"

# Neural model weights
WEIGHTS_BASELINE_DIR="${BASE_DIR}/outputs/weights_baseline"

# Shared noisy MAT + PKL (reuse from baseline-model run if they exist)
NOISY_MAT_DIR="${DOWNSTREAM_DIR}/outputs/noisy_data_1k_baseline/DB2"
NOISY_PKL_DIR="${DOWNSTREAM_DIR}/STCNet/pkl_noisy_v11"

# Baseline PKL (already exists from v10a)
BASELINE_PKL_DIR="${DOWNSTREAM_DIR}/STCNet/pkl_baseline"

# STCNet CE model (v10a)
CE_MODEL="${DOWNSTREAM_DIR}/STCNet/save/CE/nina2_STCNet_models/lr_0.0001_decay_0.0001_bsz_64_tri_v10a_enc_lr_0.05_decay_0.0001_bsz_1024_temp_0.07_tri_v10a_gamma_0.3_cos_warm_cos_aug_0.5/best_model.pth"

TEST_RESULTS_DIR="${DOWNSTREAM_DIR}/test_results_all_denoisers"
LOG_DIR="${DOWNSTREAM_DIR}/logs/all_denoisers"

DATASET="nina2"
DEVICE="cuda"
GPU_ID="6"
BATCH_SIZE=16

# ==============================================================================
# Control flags
# ==============================================================================
FORCE_NOISY_MAT=false       # Regenerate noisy MAT even if it exists
FORCE_DENOISED_MAT=false    # Regenerate denoised MAT even if it exists
SKIP_BASELINE_TEST=true     # Skip baseline test if result CSV already exists
SKIP_DENOISER_IF_DONE=false # Skip entire denoiser if PKL + test result exist

# ==============================================================================
# Helpers
# ==============================================================================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] ${*:2}" | tee -a "${LOG_DIR}/run.log"; }
log_info()  { log "INFO"  "$@"; }
log_warn()  { log "WARN"  "$@"; }
log_error() { log "ERROR" "$@"; }

sep() { echo ""; echo "========================================"; echo "$*"; echo "========================================"; echo ""; }

mat_count() { [[ -d "$1" ]] && find "$1" -name "*.mat" -type f 2>/dev/null | wc -l || echo "0"; }

verify_pkl() {
    local dir="$1" label="$2"
    log_info "Verifying ${label} PKL…"
    python3 - <<PYEOF
import pandas as pd, sys
try:
    tr = pd.read_pickle('${dir}/train_${DATASET}.pkl')
    te = pd.read_pickle('${dir}/test_${DATASET}.pkl')
    print(f"  Train={len(tr)}  Test={len(te)}  Classes={tr['stimulus'].nunique()}")
    s = tr.iloc[0]['sampled_normalized']
    print(f"  Shape={s.shape}  range=[{s.min():.4f}, {s.max():.4f}]")
    if len(tr) == 0 or len(te) == 0:
        print("ERROR: empty!"); sys.exit(1)
    print("  ✓ PKL OK")
except Exception as e:
    print(f"ERROR: {e}"); sys.exit(1)
PYEOF
}

# ==============================================================================
# Init
# ==============================================================================

sep "STCNet — All 8 Denoisers"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
log_info "GPU=${GPU_ID}"
log_info "Traditional: ${TRAD_METHODS[*]}"
log_info "Neural:      ${NEURAL_MODELS[*]}"
mkdir -p "${LOG_DIR}" "${TEST_RESULTS_DIR}"

# Sanity checks
[[ -d  "${DB2_ROOT}" ]]                              || { log_error "DB2 root not found";    exit 1; }
[[ -f  "${QC_INDEX_PATH}" ]]                         || { log_error "qc_index not found";    exit 1; }
[[ -f  "${TEST_NPZ}" ]]                              || { log_error "test_combined.npz not found"; exit 1; }
[[ -f  "${CE_MODEL}" ]]                              || { log_error "CE model not found: ${CE_MODEL}"; exit 1; }
[[ -f  "${BASELINE_PKL_DIR}/train_${DATASET}.pkl" ]] || { log_error "Baseline PKL not found"; exit 1; }

cd "${DOWNSTREAM_DIR}"

# ==============================================================================
# Step 0: Noisy MAT (shared — generated once using the first neural model)
# ==============================================================================

sep "Step 0: Noisy MAT (shared)"

noisy_count=$(mat_count "${NOISY_MAT_DIR}")

if [[ "${noisy_count}" -gt 0 && "${FORCE_NOISY_MAT}" == "false" ]]; then
    log_info "Noisy MAT already exists (${noisy_count} files) — skipping"
else
    FIRST_NEURAL="${NEURAL_MODELS[0]}"
    FIRST_PTH="${WEIGHTS_BASELINE_DIR}/${FIRST_NEURAL}/${FIRST_NEURAL}_best.pth"
    [[ -f "${FIRST_PTH}" ]] || { log_error "Checkpoint not found: ${FIRST_PTH}"; exit 1; }

    FORCE_FLAG=""; [[ "${FORCE_NOISY_MAT}" == "true" ]] && FORCE_FLAG="--force"
    TMP_DEN="${DOWNSTREAM_DIR}/outputs/_tmp_noisy_gen/DB2"

    python3 prepare_denoised_mat_baseline_model.py \
        --model-name      "${FIRST_NEURAL}" \
        --model-path      "${FIRST_PTH}" \
        --db2-root        "${DB2_ROOT}" \
        --test-npz        "${TEST_NPZ}" \
        --qc-index        "${QC_INDEX_PATH}" \
        --output-noisy    "${NOISY_MAT_DIR}" \
        --output-denoised "${TMP_DEN}" \
        --device          "${DEVICE}" \
        --batch-size      "${BATCH_SIZE}" \
        ${FORCE_FLAG} \
        2>&1 | tee "${LOG_DIR}/step0_noisy_mat.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log_error "Noisy MAT generation failed"; exit 1; }
    rm -rf "${DOWNSTREAM_DIR}/outputs/_tmp_noisy_gen"
    log_info "✓ Noisy MAT done ($(mat_count "${NOISY_MAT_DIR}") files)"
fi

# ==============================================================================
# Step 0b: Noisy PKL (shared)
# ==============================================================================

sep "Step 0b: Noisy PKL (shared)"

if [[ -f "${NOISY_PKL_DIR}/train_${DATASET}.pkl" ]]; then
    log_info "Noisy PKL already exists"
    verify_pkl "${NOISY_PKL_DIR}" "Noisy" || exit 1
else
    python3 emg_preprocess_fixed_v4.py \
        --mode    denoised \
        --path    "${NOISY_MAT_DIR}" \
        --dataset "${DATASET}" \
        --output  "${NOISY_PKL_DIR}" \
        2>&1 | tee "${LOG_DIR}/step0b_noisy_pkl.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log_error "Noisy PKL failed"; exit 1; }
    verify_pkl "${NOISY_PKL_DIR}" "Noisy" || exit 1
    log_info "✓ Noisy PKL done"
fi

# ==============================================================================
# Step 1: Baseline test (shared)
# ==============================================================================

sep "Step 1: Baseline test (shared)"

BASELINE_RESULT_DIR="${TEST_RESULTS_DIR}/baseline"
mkdir -p "${BASELINE_RESULT_DIR}"
EXISTING_B=$(find "${BASELINE_RESULT_DIR}" -name "*_baseline_overall_*.csv" 2>/dev/null | head -1 || true)

if [[ -n "${EXISTING_B}" && "${SKIP_BASELINE_TEST}" == "true" ]]; then
    log_info "Baseline result exists — skipping"
else
    cd "${DOWNSTREAM_DIR}/STCNet"
    rm -f ./pkl; ln -sf pkl_baseline ./pkl
    python3 test.py \
        --model_path "${CE_MODEL}" --dataset "${DATASET}" \
        --model STCNet --batch_size 64 \
        --output_dir "${BASELINE_RESULT_DIR}" --mode baseline \
        2>&1 | tee "${LOG_DIR}/step1_baseline.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log_error "Baseline test failed"; exit 1; }
    log_info "✓ Baseline test done"
    cd "${DOWNSTREAM_DIR}"
fi

# ==============================================================================
# Helper: run one denoiser end-to-end
#   $1 = denoiser label (e.g. "hp", "FCN")
#   $2 = "trad" | "neural"
# ==============================================================================

run_one_denoiser() {
    local LABEL="$1"
    local KIND="$2"   # trad | neural

    sep "=== Denoiser: ${LABEL} ==="

    DENOISED_MAT_DIR="${DOWNSTREAM_DIR}/outputs/denoised_data_1k_${LABEL}/DB2"
    DENOISED_PKL_DIR="${DOWNSTREAM_DIR}/STCNet/pkl_denoised_${LABEL}"
    RESULT_DIR="${TEST_RESULTS_DIR}/${LABEL}"
    MLOG="${LOG_DIR}/${LABEL}"
    mkdir -p "${MLOG}" "${RESULT_DIR}"

    # Optional: skip if fully done
    if [[ "${SKIP_DENOISER_IF_DONE}" == "true" ]]; then
        DONE=$(find "${RESULT_DIR}" -name "*_denoised_overall_*.csv" 2>/dev/null | head -1 || true)
        if [[ -f "${DENOISED_PKL_DIR}/train_${DATASET}.pkl" && -n "${DONE}" ]]; then
            log_info "[${LABEL}] Already complete — skipping"
            return 0
        fi
    fi

    # ---- Step A: Denoised MAT -------------------------------------------
    log_info "[${LABEL}] Step A: Denoised MAT"
    den_count=$(mat_count "${DENOISED_MAT_DIR}")
    FORCE_FLAG=""; [[ "${FORCE_DENOISED_MAT}" == "true" ]] && FORCE_FLAG="--force"

    if [[ "${den_count}" -gt 0 && "${FORCE_DENOISED_MAT}" == "false" ]]; then
        log_info "[${LABEL}] Denoised MAT already exists (${den_count} files)"
    else
        if [[ "${KIND}" == "trad" ]]; then
            # Traditional method
            PARAMS_FLAG=""
            [[ -f "${TRAD_PARAMS_JSON}" ]] && PARAMS_FLAG="--params-json ${TRAD_PARAMS_JSON}"

            python3 prepare_denoised_mat_traditional.py \
                --method          "${LABEL}" \
                ${PARAMS_FLAG} \
                --db2-root        "${DB2_ROOT}" \
                --test-npz        "${TEST_NPZ}" \
                --qc-index        "${QC_INDEX_PATH}" \
                --output-noisy    "${NOISY_MAT_DIR}" \
                --output-denoised "${DENOISED_MAT_DIR}" \
                ${FORCE_FLAG} \
                2>&1 | tee "${MLOG}/step_a_mat.log"
        else
            # Neural model
            MODEL_PTH="${WEIGHTS_BASELINE_DIR}/${LABEL}/${LABEL}_best.pth"
            if [[ ! -f "${MODEL_PTH}" ]]; then
                log_warn "[${LABEL}] Checkpoint not found: ${MODEL_PTH} — skipping"
                return 0
            fi

            python3 prepare_denoised_mat_baseline_model.py \
                --model-name      "${LABEL}" \
                --model-path      "${MODEL_PTH}" \
                --db2-root        "${DB2_ROOT}" \
                --test-npz        "${TEST_NPZ}" \
                --qc-index        "${QC_INDEX_PATH}" \
                --output-noisy    "${NOISY_MAT_DIR}" \
                --output-denoised "${DENOISED_MAT_DIR}" \
                --device          "${DEVICE}" \
                --batch-size      "${BATCH_SIZE}" \
                ${FORCE_FLAG} \
                2>&1 | tee "${MLOG}/step_a_mat.log"
        fi

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "[${LABEL}] Denoised MAT failed — skipping"
            return 0
        fi
        log_info "[${LABEL}] ✓ Denoised MAT ($(mat_count "${DENOISED_MAT_DIR}") files)"
    fi

    # ---- Step B: Denoised PKL -------------------------------------------
    log_info "[${LABEL}] Step B: Denoised PKL"
    rm -rf "${DENOISED_PKL_DIR}"
    python3 emg_preprocess_fixed_v4.py \
        --mode    denoised \
        --path    "${DENOISED_MAT_DIR}" \
        --dataset "${DATASET}" \
        --output  "${DENOISED_PKL_DIR}" \
        2>&1 | tee "${MLOG}/step_b_pkl.log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "[${LABEL}] Denoised PKL failed — skipping"
        return 0
    fi
    verify_pkl "${DENOISED_PKL_DIR}" "${LABEL}" || { log_error "[${LABEL}] PKL verify failed"; return 0; }
    log_info "[${LABEL}] ✓ Denoised PKL"

    # ---- Step C: Test denoised ------------------------------------------
    log_info "[${LABEL}] Step C: Test denoised"
    DENOISED_RESULT_DIR="${RESULT_DIR}/denoised"
    mkdir -p "${DENOISED_RESULT_DIR}"

    cd "${DOWNSTREAM_DIR}/STCNet"
    rm -f ./pkl; ln -sf "pkl_denoised_${LABEL}" ./pkl
    python3 test.py \
        --model_path "${CE_MODEL}" --dataset "${DATASET}" \
        --model STCNet --batch_size 64 \
        --output_dir "${DENOISED_RESULT_DIR}" --mode denoised \
        2>&1 | tee "${MLOG}/step_c_test.log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "[${LABEL}] Test failed"
        cd "${DOWNSTREAM_DIR}"
        return 0
    fi
    cd "${DOWNSTREAM_DIR}"
    log_info "[${LABEL}] ✓ Complete"
}

# ==============================================================================
# Run all traditional denoisers
# ==============================================================================

sep "Traditional Methods"
for METHOD in "${TRAD_METHODS[@]}"; do
    run_one_denoiser "${METHOD}" "trad"
done

# ==============================================================================
# Run all neural denoisers
# ==============================================================================

sep "Neural Models"
for MODEL in "${NEURAL_MODELS[@]}"; do
    run_one_denoiser "${MODEL}" "neural"
done

# ==============================================================================
# Noisy test (shared, run once after all denoisers)
# ==============================================================================

sep "Noisy test (shared)"

NOISY_RESULT_DIR="${TEST_RESULTS_DIR}/noisy"
mkdir -p "${NOISY_RESULT_DIR}"

cd "${DOWNSTREAM_DIR}/STCNet"
rm -f ./pkl; ln -sf pkl_noisy_v11 ./pkl
python3 test.py \
    --model_path "${CE_MODEL}" --dataset "${DATASET}" \
    --model STCNet --batch_size 64 \
    --output_dir "${NOISY_RESULT_DIR}" --mode noisy \
    2>&1 | tee "${LOG_DIR}/noisy_test.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] && log_info "✓ Noisy test done" || log_warn "Noisy test failed"
cd "${DOWNSTREAM_DIR}"

# ==============================================================================
# Final Summary Table
# ==============================================================================

sep "Final Summary — All Denoisers"

ALL_DENOISERS=("${TRAD_METHODS[@]}" "${NEURAL_MODELS[@]}")

python3 - <<PYEOF
import pandas as pd, glob, os, sys

results_dir  = "${TEST_RESULTS_DIR}"
all_denoisers = "${ALL_DENOISERS[*]}".split()

def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    return pd.read_csv(files[-1]) if files else None

def get_val(df, col):
    if df is None or col not in df.columns:
        return None
    return float(df[col].iloc[0])

b_df = load_latest(f"{results_dir}/baseline/*_baseline_overall_*.csv")
n_df = load_latest(f"{results_dir}/noisy/*_noisy_overall_*.csv")

if b_df is None:
    print("[WARN] Baseline result not found.")
    sys.exit(0)

acc_b = get_val(b_df, "accuracy")
f1_b  = get_val(b_df, "f1_score")
acc_n = get_val(n_df, "accuracy")
f1_n  = get_val(n_df, "f1_score")

W = 90
print()
print("=" * W)
print(f"  STCNet DOWNSTREAM — ALL DENOISERS (CE model, Dataset=nina2)")
print("=" * W)

def fmt_acc(v):
    return f"{v:>8.2f}%" if v is not None else f"{'—':>9}"
def fmt_diff(a, b):
    if a is None or b is None: return f"{'—':>7}"
    return f"{a-b:>+6.2f}%"

header = f"  {'Denoiser':<20} {'Acc':>9} {'F1':>9}  {'Acc-B':>7}  {'F1-B':>7}  {'Acc-N':>7}"
print(header)
print("-" * W)

# Baseline row
print(f"  {'[Baseline]':<20} {fmt_acc(acc_b)} {fmt_acc(f1_b)}  {'—':>7}  {'—':>7}  {'—':>7}")
# Noisy row
print(f"  {'[Noisy]':<20} {fmt_acc(acc_n)} {fmt_acc(f1_n)}"
      f"  {fmt_diff(acc_n, acc_b)}  {fmt_diff(f1_n, f1_b)}  {'—':>7}")
print("-" * W)

rows = []
for label in all_denoisers:
    d_df = load_latest(f"{results_dir}/{label}/denoised/*_denoised_overall_*.csv")
    acc_d = get_val(d_df, "accuracy")
    f1_d  = get_val(d_df, "f1_score")
    rows.append((label, acc_d, f1_d))

    recovery_str = fmt_diff(acc_d, acc_n) if acc_n is not None else "—"
    print(f"  {label:<20} {fmt_acc(acc_d)} {fmt_acc(f1_d)}"
          f"  {fmt_diff(acc_d, acc_b)}  {fmt_diff(f1_d, f1_b)}  {recovery_str}")

print("=" * W)

# Best / worst summary
valid = [(l, a, f) for l, a, f in rows if a is not None]
if valid:
    best_acc = max(valid, key=lambda r: r[1])
    if acc_n is not None:
        best_rec = max(valid, key=lambda r: r[1] - acc_n)
        print(f"\n  ✅ Best denoised acc : {best_acc[0]} → {best_acc[1]:.2f}%")
        print(f"  ✅ Best recovery (D-N): {best_rec[0]} → {best_rec[1] - acc_n:+.2f}%")

# Save summary CSV
summary_rows = [{"denoiser": "baseline_clean",
                 "acc_denoised": acc_b, "f1_denoised": f1_b,
                 "acc_vs_baseline": 0.0, "acc_vs_noisy": None}]
if acc_n is not None:
    summary_rows.append({"denoiser": "noisy",
                         "acc_denoised": acc_n, "f1_denoised": f1_n,
                         "acc_vs_baseline": (acc_n - acc_b) if acc_b else None,
                         "acc_vs_noisy": 0.0})
for label, acc_d, f1_d in rows:
    summary_rows.append({
        "denoiser": label,
        "acc_denoised": acc_d,
        "f1_denoised": f1_d,
        "acc_vs_baseline": (acc_d - acc_b) if (acc_d is not None and acc_b is not None) else None,
        "acc_vs_noisy":    (acc_d - acc_n) if (acc_d is not None and acc_n is not None) else None,
    })

csv_path = f"{results_dir}/summary_all_denoisers.csv"
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")
print()
PYEOF

# ==============================================================================
# Done
# ==============================================================================

sep "Done"
log_info "Results: ${TEST_RESULTS_DIR}/summary_all_denoisers.csv"
log_info "Trad:    ${TRAD_METHODS[*]}"
log_info "Neural:  ${NEURAL_MODELS[*]}"

exit 0