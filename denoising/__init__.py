# Baseline waveform denoising models
# ${CLEANSEMG_ROOT}/baseline_models/__init__.py
from .TrustEMGNet import (
    TrustEMGNet_UNetonly,
    TrustEMGNet_DM,
    TrustEMGNet_RM,
    TrustEMGNet_skipall_DM,
    TrustEMGNet_skipall_RM,
    TrustEMGNet_LSTM_DM,
    TrustEMGNet_LSTM_RM,
)
from .FCN import FCN
from .CNN import CNN_waveform
from .MSEMG import MSEMG
from .SDEMG import SDEMG

BASELINE_MODEL_REGISTRY = {
    "TrustEMGNet_UNetonly":   TrustEMGNet_UNetonly,
    "TrustEMGNet_DM":         TrustEMGNet_DM,
    "TrustEMGNet_RM":         TrustEMGNet_RM,
    "TrustEMGNet_skipall_DM": TrustEMGNet_skipall_DM,
    "TrustEMGNet_skipall_RM": TrustEMGNet_skipall_RM,
    "TrustEMGNet_LSTM_DM":    TrustEMGNet_LSTM_DM,
    "TrustEMGNet_LSTM_RM":    TrustEMGNet_LSTM_RM,
    "FCN":                    FCN,
    "CNN_waveform":           CNN_waveform,
    "MSEMG":                  MSEMG,
    "SDEMG":                  SDEMG,
}

__all__ = list(BASELINE_MODEL_REGISTRY.keys()) + ["BASELINE_MODEL_REGISTRY"]