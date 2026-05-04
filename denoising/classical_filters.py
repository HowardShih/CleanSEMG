#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tradition_filters.py  v2  —  Traditional baseline filter implementations.

Methods
-------
HP       : Butterworth high-pass filter
TS       : Template subtraction (ECG artefact removal)
EMD      : EMD + noise-index-based thresholding per IMF
VMD      : VMD + interval soft-thresholding per VMF  (VMD-SIT / VMD-IIT)
CEEMDAN  : CEEMDAN + noise-index-based thresholding per IMF

Noise-type handling summary
---------------------------
             HP   TS    EMD               VMD               CEEMDAN
PLI (P)      ○    ×     notch each IMF    notch each VMF    notch each IMF
ECG (E)      ×    ○     corr+HP each IMF  (skip)            corr+HP each IMF
MOA (m)      ×    ×     freq<20→zero      freq<20→zero      freq<20→zero
BW  (B)      ×    ×     freq<10→zero      freq<10→zero      freq<10→zero
WGN (WG)     ×    ×     noise-idx thr     VMD-IIT           noise-idx thr
Color (Co)   △    ×     HP-pre + generic  HP-pre + generic  HP-pre + generic

△ HP is partially effective for brown noise (most energy below ~20 Hz)
  but cannot remove pink noise whose energy overlaps the sEMG band.

Color noise (pink / brown) references
--------------------------------------
  Flandrin et al., "EMD as a filter bank," IEEE SP Letters 11(2):112–114, 2004.
    → Shows EMD decomposes 1/f^β noise in a predictable filter-bank manner;
      IMF noise-index approach generalises from WGN to coloured noise.
  Wu & Huang, "A study of WGN using the EMD method,"
    Proc. R. Soc. London A 460(2046):1597–1611, 2004.
    → Derives the WGN_NOISE_IDXS used in TrustEMG-Net's baseline.
  Ma et al., "EMG signal filtering based on VMD and sub-band thresholding,"
    IEEE JBHI 25(1):47–58, 2021. doi: 10.1109/JBHI.2020.3001440
    → VMD-WST / VMD-SIT for broadband noise in sEMG; alpha=2000, K=10.

Other references
----------------
  Wang et al. (TrustEMG-Net), IEEE JBHI 29(4):2506–2520, 2025.
  Huang et al., Proc. R. Soc. London A 454:903–995, 1998.   (EMD)
  Torres et al., ICASSP 2011.                               (CEEMDAN)
  Dragomiretskiy & Zosso, IEEE TSP 62(3):531–544, 2014.    (VMD)
  GitHub: https://github.com/eric-wang135/TrustEMG          (TrustEMG code)
"""

import math
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.optimize import minimize


# ============================================================================
# ── WGN noise-index arrays  (Wu & Huang 2004; used by TrustEMG-Net) ─────────
#   Normalised std of each IMF when input is WGN.  IMF-0 = highest freq.
#   Pre-computed empirically for max_imf=8 over 2000-sample segments at 1000 Hz.
#   Source: TrustEMG-Net GitHub, baseline/tradition.py
# ============================================================================
WGN_NOISE_IDXS_EMD = np.array(
    [1.0, 0.4751015, 0.29994139, 0.21576524,
     0.13714082, 0.09261460, 0.05600730, 0.05497522]
)
WGN_NOISE_IDXS_CEEMDAN = np.array(
    [1.0, 0.45871, 0.309190, 0.19652,
     0.14093, 0.084863, 0.06222, 0.05597]
)


# ============================================================================
# ── Noise-type keyword helpers ────────────────────────────────────────────
# ============================================================================

# Mapping: TrustEMG-style single-letter tags → our noise type names
# (noise.py stores types as "PLI", "ECG", "MOA", "WGN", "Color")
_TRUSTEMG_TAG_MAP = {
    "P":  "PLI",
    "E":  "ECG",
    "WG": "WGN",
    "m":  "MOA",   # TrustEMG uses lowercase 'm' for motion artifact
    "Q":  "MOA",   # TrustEMG uses 'Q' for electrode motion (same as MOA here)
    "B":  "BW",    # baseline wander (not in our noise set, kept for completeness)
}


def _has(noise_type: str, tag: str) -> bool:
    """
    Check if a noise_types string contains the given noise type.
    Accepts both our system names ("PLI", "ECG", "MOA", "WGN", "Color")
    and TrustEMG-style single-char tags ("P", "E", "WG", "m", "Q").

    Our test data stores noise_types as e.g. "PLI+ECG", "Color+WGN", "MOA"
    (from generate_test_data.py: "+".join(info["noise_types"])).
    """
    nt  = (noise_type or "").upper()
    our = _TRUSTEMG_TAG_MAP.get(tag, tag).upper()
    return our in nt


def _is_color(noise_type: str) -> bool:
    """True if noise contains any broadband coloured noise (Pink or Brown)."""
    nt = (noise_type or "").lower()
    return "color" in nt or "colour" in nt or "pink" in nt or "brown" in nt


# ============================================================================
# ── Soft / interval thresholding helpers ─────────────────────────────────
# ============================================================================

def _mad_sigma(x: np.ndarray) -> float:
    """Noise-level estimate via MAD (Donoho & Johnstone 1994)."""
    return float(np.median(np.abs(x)) / 0.6745)


def _universal_threshold(sigma: float, n: int) -> float:
    """VisuShrink universal threshold: σ √(2 ln N)."""
    return sigma * math.sqrt(2.0 * math.log(max(n, 2)))


def _soft_threshold(x: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def _interval_soft_threshold(x: np.ndarray, thr: float) -> np.ndarray:
    """
    Interval soft thresholding (SIT): threshold applied per zero-crossing
    interval rather than sample-by-sample.

    Ref: Ma et al., IEEE JBHI 25(1):47-58, 2021  (VMD-SIT);
         Wang et al. TrustEMG-Net GitHub – interval_thresholding().
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    # Find zero-crossing boundaries
    zc = [0]
    for i in range(1, len(x)):
        if (x[i - 1] >= 0 and x[i] < 0) or (x[i - 1] < 0 and x[i] >= 0):
            zc.append(i)
    intervals = []
    for i in range(len(zc) - 1):
        seg = x[zc[i]:zc[i + 1]]
        ext = np.max(np.abs(seg))
        if ext == 0:
            intervals.append(seg)
        else:
            intervals.append(seg * max(0.0, (ext - thr) / ext))
    # Last interval
    seg = x[zc[-1]:]
    if len(seg):
        ext = np.max(np.abs(seg))
        intervals.append(seg if ext == 0 else seg * max(0.0, (ext - thr) / ext))
    return np.concatenate(intervals) if intervals else x.copy()


def _threshold_component(component: np.ndarray,
                          sigma_ref: float,
                          mode: str = "soft") -> np.ndarray:
    """Apply soft or interval-soft threshold to one IMF/VMF."""
    n = len(component)
    sigma = _mad_sigma(component) if sigma_ref <= 0.0 else sigma_ref
    thr = _universal_threshold(sigma, n) / 4.0   # /4 matches TrustEMG convention
    if mode == "interval":
        return _interval_soft_threshold(component, thr)
    return _soft_threshold(component, thr)


# ============================================================================
# ── Spectral helpers ──────────────────────────────────────────────────────
# ============================================================================

def _dominant_freq_hz(x: np.ndarray, fs: int) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd   = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    return float(freqs[np.argmax(psd)])


# ============================================================================
# ── HP filter ─────────────────────────────────────────────────────────────
# ============================================================================

def apply_hp_filter(x: np.ndarray, fs: int, cutoff_hz: float,
                    order: int = 4) -> np.ndarray:
    """
    Butterworth high-pass filter (zero-phase).
    Ref: Wang et al. TrustEMG-Net, IEEE JBHI 2025.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    nyq = fs / 2.0
    Wn  = min(max(float(cutoff_hz) / nyq, 1e-4), 0.9999)
    b, a = butter(order, Wn, btype="high")
    if x.size < max(27, 3 * order + 1):
        return x.copy()
    return filtfilt(b, a, x)


# ============================================================================
# ── Template subtraction ──────────────────────────────────────────────────
# ============================================================================

def _ts_scalar(waveform: np.ndarray, template: np.ndarray) -> float:
    def _mse(s):
        return float(np.sum((waveform - s[0] * template) ** 2))
    try:
        res = minimize(_mse, [1.0])
        return float(res.x[0]) if res.success else 1.0
    except Exception:
        return 1.0


def apply_template_subtraction(
    n_emg: np.ndarray, fs: int = 1000,
    peak_detect_bp_low: float = 2.5,
    peak_detect_bp_high: float = 50.0,
    peak_detect_order: int = 4,
    avg_window: int = 11,
    min_peaks: int = 2,
    min_beat_gap: int = 140,
    tile_factor: int = 8,
) -> Tuple[np.ndarray, int]:
    """
    Averaged-template subtraction for ECG artefact removal.
    Uses averaged template + per-peak scalar fitting (robust for 2 s segments).

    Ref: Wang et al. TrustEMG-Net, IEEE JBHI 2025;
         Xu et al., Sensors 20(17):4890, 2020.
    Returns (enhanced, error_code): error_code==0 → success.
    """
    n_emg = np.asarray(n_emg, dtype=np.float64).reshape(-1)
    L   = n_emg.size
    nyq = fs / 2.0
    lo  = max(1e-4, peak_detect_bp_low  / nyq)
    hi  = min(0.9999, peak_detect_bp_high / nyq)
    if hi <= lo:
        return n_emg.copy(), 2
    b, a  = butter(peak_detect_order, [lo, hi], btype="bandpass")
    if L < max(27, 3 * peak_detect_order + 1):
        return n_emg.copy(), 1
    bp = filtfilt(b, a, n_emg)

    rec = np.abs(bp)
    ma1 = np.convolve(rec, np.ones(fs)             / fs,             "same")
    ma2 = np.convolve(rec, np.ones(max(1, fs // 10)) / max(1, fs // 10), "same")
    r_peaks, mark = [], 0
    for i in range(L):
        if i < mark: continue
        if ma1[i] < ma2[i]:
            for j in range(i, L):
                if ma1[j] >= ma2[j]:
                    mark = j
                    if j - i >= min_beat_gap:
                        r_peaks.append(i + int(np.argmax(bp[i:j])))
                    break

    if len(r_peaks) < min_peaks:
        return n_emg.copy(), 1

    trr   = min(r_peaks[k] - r_peaks[k - 1] for k in range(1, len(r_peaks)))
    left  = math.floor(0.25 * trr)
    right = math.floor(0.45 * trr)

    waveforms, valid_peaks = [], []
    for pk in r_peaks:
        if pk - left < 0 or pk + right + 1 > L: continue
        waveforms.append(bp[pk - left: pk + right + 1])
        valid_peaks.append(pk)
    if not waveforms:
        return n_emg.copy(), 1

    template = np.stack(waveforms).mean(axis=0)
    enh = n_emg.copy()
    for pk, wav in zip(valid_peaks, waveforms):
        s = _ts_scalar(wav, template)
        enh[pk - left: pk + right + 1] -= s * template
    return enh, 0


# ============================================================================
# ── EMD  ─────────────────────────────────────────────────────────────────
# ============================================================================

def apply_emd_filter(
    x: np.ndarray,
    fs: int = 1000,
    f_min: float = 20.0,
    f_max: float = 500.0,
    max_imfs: int = 8,
    noise_type: str = "",
) -> Tuple[np.ndarray, bool]:
    """
    EMD-based sEMG denoiser.

    IMF processing strategy (matches TrustEMG-Net baseline approach):

    Noise type   Action
    ----------   ------
    PLI (P)      iirnotch at dominant freq if 50-70 Hz
    MOA (m)      zero IMFs with dominant freq < 20 Hz; HP filter the rest
    BW  (B)      zero IMFs with dominant freq < 10 Hz
    WGN (WG)     noise-index threshold (Wu & Huang 2004); wavelet thr on survivors
    ECG (E)      correlate IMFs with LP-filtered ECG template; HP filter top-6
    Color (Co)   HP pre-filter + noise-index threshold (generalised WGN approach)
                 Justification: Flandrin et al. (2004) show that EMD acts as a
                 dyadic filter bank for 1/f^β noise, so the WGN noise-index
                 thresholding approach extends to coloured noise.
    Unknown      frequency-based selection + noise-index threshold

    References
    ----------
    Huang et al., Proc. R. Soc. London A, 1998.
    Wu & Huang, Proc. R. Soc. London A, 2004.          (noise indices)
    Flandrin et al., IEEE SP Letters 11(2):112, 2004.  (colour noise)
    TrustEMG-Net GitHub: baseline/tradition.py
    """
    try:
        from PyEMD import EMD
    except ImportError:
        warnings.warn("PyEMD not installed.  Run: pip install EMD-signal")
        return x.copy(), False

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    N = len(x)
    if N < 64:
        return x.copy(), False

    # For Color noise: HP pre-filter to remove sub-sEMG-band energy before EMD
    x_in = x.copy()
    if _is_color(noise_type):
        nyq = fs / 2.0
        Wn  = min(max(f_min / nyq, 1e-4), 0.9999)
        b, a = butter(4, Wn, btype="high")
        if N >= max(27, 13):
            x_in = filtfilt(b, a, x_in)

    try:
        emd = EMD()
        emd.MAX_ITERATION = 1000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imfs = emd(x_in, max_imf=max_imfs)   # shape [N, K] → we get [K, N]
    except Exception:
        return x.copy(), False

    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]

    K     = imfs.shape[0]
    freq  = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / fs))
    u_out = imfs.copy()

    # Noise std estimate from highest-frequency IMF (IMF-0)
    imf_std = np.std(imfs, axis=1)
    noise_idxs = WGN_NOISE_IDXS_EMD[:K] if K <= len(WGN_NOISE_IDXS_EMD) \
                 else np.concatenate([WGN_NOISE_IDXS_EMD,
                                      np.full(K - len(WGN_NOISE_IDXS_EMD),
                                              WGN_NOISE_IDXS_EMD[-1])])
    est_noise_std = imf_std[0] * noise_idxs   # expected std if pure WGN

    # Dominant frequency of each IMF
    u_spectrum = np.abs(np.fft.fft(imfs, axis=1))
    P_max = np.array([
        freq[N // 2 + np.argmax(u_spectrum[i, :N // 2])]
        for i in range(K)
    ])

    for i in range(K):
        pm = P_max[i]

        # PLI: notch if IMF has dominant freq in 50-70 Hz
        if _has(noise_type, "P") and 50 < pm < 70:
            b_n, a_n = iirnotch(pm, 20, fs=fs)
            u_out[i] = filtfilt(b_n, a_n, u_out[i])

        # MOA: zero if below 20 Hz, else high-pass
        if _has(noise_type, "m"):
            if pm < f_min:
                u_out[i] = 0.0
            else:
                nyq = fs / 2.0
                b_h, a_h = butter(4, min(max(f_min / nyq, 1e-4), 0.9999), btype="high")
                if N >= max(27, 13): u_out[i] = filtfilt(b_h, a_h, u_out[i])
        elif _has(noise_type, "Q"):
            if pm < f_min:
                u_out[i] = 0.0
            else:
                nyq = fs / 2.0
                b_l, a_l = butter(4, min(max(f_min / nyq, 1e-4), 0.9999), btype="low")
                if N >= max(27, 13): u_out[i] = u_out[i] - filtfilt(b_l, a_l, u_out[i])
        elif _has(noise_type, "B"):
            if pm < 10:
                u_out[i] = 0.0
            else:
                nyq = fs / 2.0
                b_l, a_l = butter(4, min(max(10.0 / nyq, 1e-4), 0.9999), btype="low")
                if N >= max(27, 13): u_out[i] = u_out[i] - filtfilt(b_l, a_l, u_out[i])

    # WGN or Color noise: noise-index threshold
    use_noise_idx = _has(noise_type, "WG") or _is_color(noise_type) or not noise_type.strip()
    if use_noise_idx:
        u_out[0] = 0.0   # First IMF is typically noise-dominant (highest freq)
        for i in range(1, K):
            if est_noise_std[i] > imf_std[i] or imf_std[0] == 0:
                u_out[i] = 0.0   # Noise-only IMF → discard
            else:
                # Noisy sEMG IMF → wavelet-style soft threshold
                sigma = _mad_sigma(imfs[i])
                thr   = _universal_threshold(sigma, N) / 4.0
                u_out[i] = _soft_threshold(u_out[i], thr)

    # ECG: correlate with LP template, HP filter top-6 correlated IMFs
    if _has(noise_type, "E"):
        try:
            from scipy.stats import pearsonr
            nyq = fs / 2.0
            b_lp, a_lp = butter(4, min(40.0 / nyq, 0.9999), btype="low")
            b_hp, a_hp = butter(4, min(max(30.0 / nyq, 1e-4), 0.9999), btype="high")
            if N >= max(27, 13):
                ecg_tmpl = filtfilt(b_lp, a_lp, x_in)
                active = ~np.all(u_out == 0, axis=1)
                u_active = u_out[active]
                corr = [abs(pearsonr(ecg_tmpl, u_active[j])[0])
                        for j in range(u_active.shape[0])]
                top6 = np.argsort(corr)[-6:]
                for idx in top6:
                    u_active[idx] = filtfilt(b_hp, a_hp, u_active[idx])
                u_out[active] = u_active
        except Exception:
            pass

    return u_out.sum(axis=0), True


# ============================================================================
# ── VMD  ─────────────────────────────────────────────────────────────────
# ============================================================================

def apply_vmd_filter(
    x: np.ndarray,
    fs: int = 1000,
    K: int = 10,
    alpha: float = 2000.0,
    tau: float = 0.0,
    f_min: float = 20.0,
    f_max: float = 500.0,
    tol: float = 1e-7,
    noise_type: str = "",
) -> Tuple[np.ndarray, bool]:
    """
    VMD-IIT sEMG denoiser (Iterative Interval Thresholding on each VMF).

    Parameter notes
    ---------------
    alpha : 2000 (Ma et al. JBHI 2021, for 2-second sEMG segments).
            TrustEMG-Net code uses 1000 for longer segments. We default to
            2000 following the primary reference (Ma et al. 2021).
            Calibration will tune K; alpha is kept fixed per literature.
    K     : Swept during calibration (default 10 per Ma et al. 2021 / TrustEMG).

    Color noise
    -----------
    Coloured noise is broadband; VMD decomposes it into modes whose centre
    frequencies span the spectrum.  Modes with centre freq outside the sEMG
    band [f_min, f_max] are zeroed; modes inside the band are soft-thresholded
    via interval thresholding (SIT).  This is conservative and avoids
    distortion of the sEMG.
    Ref: Ma et al. IEEE JBHI 25(1):47-58, 2021.

    References
    ----------
    Dragomiretskiy & Zosso, IEEE TSP 62(3):531-544, 2014.
    Ma et al., IEEE JBHI 25(1):47-58, 2021.
    vmdpy library: https://github.com/vrcarva/vmdpy
    TrustEMG-Net GitHub: baseline/VMD.py
    """
    try:
        from vmdpy import VMD
    except ImportError:
        warnings.warn("vmdpy not installed.  Run: pip install vmdpy")
        return x.copy(), False

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    N = len(x)
    if N < 64:
        return x.copy(), False

    # For Color noise: HP pre-filter to remove sub-band energy
    x_in = x.copy()
    if _is_color(noise_type):
        nyq = fs / 2.0
        Wn  = min(max(f_min / nyq, 1e-4), 0.9999)
        b, a = butter(4, Wn, btype="high")
        if N >= max(27, 13):
            x_in = filtfilt(b, a, x_in)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u, u_hat, omega = VMD(x_in, alpha, tau, int(K),
                                  DC=0, init=1, tol=tol)
        # u shape: [K, N];  omega[-1] shape: [K] — centre freqs in [0, 0.5]
    except Exception:
        return x.copy(), False

    omega_hz = omega[-1] * fs      # convert normalised → Hz
    seg_len  = N

    u_out = u.copy()
    for k in range(int(K)):
        pm = omega_hz[k]

        # PLI: notch if VMF centre in 50-70 Hz
        if _has(noise_type, "P") and 50 < pm < 70:
            b_n, a_n = iirnotch(pm, 20, fs=fs)
            if seg_len >= max(27, 13):
                u_out[k] = filtfilt(b_n, a_n, u_out[k])

        if _has(noise_type, "E"):
            # Skip ECG processing for VMD (handled by TS separately if needed)
            pass
        elif _has(noise_type, "m"):
            if pm < f_min:
                u_out[k] = 0.0
            else:
                nyq = fs / 2.0
                b_h, a_h = butter(4, min(max(f_min / nyq, 1e-4), 0.9999), btype="high")
                if seg_len >= max(27, 13): u_out[k] = filtfilt(b_h, a_h, u_out[k])
        elif _has(noise_type, "Q"):
            if pm < f_min:
                u_out[k] = 0.0
            else:
                nyq = fs / 2.0
                b_l, a_l = butter(4, min(max(f_min / nyq, 1e-4), 0.9999), btype="low")
                if seg_len >= max(27, 13): u_out[k] = u_out[k] - filtfilt(b_l, a_l, u_out[k])
        elif _has(noise_type, "B"):
            if pm < 10:
                u_out[k] = 0.0
            else:
                nyq = fs / 2.0
                b_l, a_l = butter(4, min(max(10.0 / nyq, 1e-4), 0.9999), btype="low")
                if seg_len >= max(27, 13): u_out[k] = u_out[k] - filtfilt(b_l, a_l, u_out[k])

        # WGN / Color / unknown: interval soft-threshold
        # FIX: use per-VMF sigma to avoid over-thresholding signal-band VMFs.
        # Using the global highest-freq VMF sigma causes all in-band VMFs to be
        # zeroed (the highest-freq VMF is noise-dominated → huge sigma → huge thr).
        use_thr = (_has(noise_type, "WG") or _is_color(noise_type)
                   or not noise_type.strip())
        if use_thr:
            if not (f_min <= pm <= f_max):
                u_out[k] = 0.0       # Outside sEMG band → discard
            else:
                # Per-VMF sigma: threshold calibrated to each mode's own noise level
                sigma_k = _mad_sigma(u_out[k])
                thr_k   = _universal_threshold(sigma_k, N) / 4.0
                u_out[k] = _interval_soft_threshold(u_out[k], thr_k)

    return u_out.sum(axis=0), True


# ============================================================================
# ── CEEMDAN ───────────────────────────────────────────────────────────────
# ============================================================================

def apply_ceemdan_filter(
    x: np.ndarray,
    fs: int = 1000,
    trials: int = 20,
    epsilon: float = None,    # None → use TrustEMG default (range_thr / power_thr)
    f_min: float = 20.0,
    f_max: float = 500.0,
    max_imfs: int = 8,
    noise_type: str = "",
    # trials reduced to 20 (TrustEMG default); further speed improvement
    # comes from parallel inference (see run_one_method n_jobs param)
) -> Tuple[np.ndarray, bool]:
    """
    CEEMDAN-based sEMG denoiser.

    Uses the same noise-index thresholding strategy as the EMD filter,
    with CEEMDAN-specific noise indices (WGN_NOISE_IDXS_CEEMDAN).
    trials=20 matches TrustEMG-Net's baseline code (not 100).

    Color noise handling: identical to EMD — HP pre-filter + noise-index
    threshold, justified by Flandrin et al. (2004) filter-bank property.

    References
    ----------
    Torres et al., ICASSP 2011.
    TrustEMG-Net GitHub: baseline/tradition.py  (CEEMDAN_method)
    Flandrin et al., IEEE SP Letters 11(2):112, 2004.
    """
    try:
        from PyEMD import CEEMDAN
    except ImportError:
        warnings.warn("PyEMD not installed.  Run: pip install EMD-signal")
        return x.copy(), False

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    N = len(x)
    if N < 64:
        return x.copy(), False

    # Color noise: HP pre-filter
    x_in = x.copy()
    if _is_color(noise_type):
        nyq = fs / 2.0
        Wn  = min(max(f_min / nyq, 1e-4), 0.9999)
        b, a = butter(4, Wn, btype="high")
        if N >= max(27, 13):
            x_in = filtfilt(b, a, x_in)

    try:
        # range_thr / total_power_thr stopping criteria from TrustEMG code
        ceemdan = CEEMDAN(trials=int(trials),
                          range_thr=0.001, total_power_thr=0.01)
        ceemdan.noise_seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imfs = ceemdan(x_in, max_imf=max_imfs)
    except Exception:
        return x.copy(), False

    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]
    # TrustEMG code transposes: u = ceemdan(...).T → shape [K, N]
    # PyEMD already returns [K, N] after ceemdan(), no transpose needed.

    K    = imfs.shape[0]
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / fs))

    imf_std = np.std(imfs, axis=1)
    noise_idxs = WGN_NOISE_IDXS_CEEMDAN[:K] if K <= len(WGN_NOISE_IDXS_CEEMDAN) \
                 else np.concatenate([WGN_NOISE_IDXS_CEEMDAN,
                                      np.full(K - len(WGN_NOISE_IDXS_CEEMDAN),
                                              WGN_NOISE_IDXS_CEEMDAN[-1])])
    est_noise_std = imf_std[0] * noise_idxs

    u_spectrum = np.abs(np.fft.fft(imfs, axis=1))
    P_max = np.array([
        freq[N // 2 + np.argmax(u_spectrum[i, :N // 2])]
        for i in range(K)
    ])

    u_out = imfs.copy()
    for i in range(K):
        pm = P_max[i]

        if _has(noise_type, "P") and 50 < pm < 70:
            b_n, a_n = iirnotch(pm, 20, fs=fs)
            u_out[i] = filtfilt(b_n, a_n, u_out[i])

        if _has(noise_type, "m"):
            if pm < f_min: u_out[i] = 0.0
            else:
                nyq = fs / 2.0
                b_h, a_h = butter(4, min(max(f_min / nyq, 1e-4), 0.9999), btype="high")
                if N >= max(27, 13): u_out[i] = filtfilt(b_h, a_h, u_out[i])
        elif _has(noise_type, "Q"):
            if pm < f_min: u_out[i] = 0.0
            else:
                nyq = fs / 2.0
                b_l, a_l = butter(4, min(max(f_min / nyq, 1e-4), 0.9999), btype="low")
                if N >= max(27, 13): u_out[i] = u_out[i] - filtfilt(b_l, a_l, u_out[i])
        elif _has(noise_type, "B"):
            if pm < 10: u_out[i] = 0.0
            else:
                nyq = fs / 2.0
                b_l, a_l = butter(4, min(max(10.0 / nyq, 1e-4), 0.9999), btype="low")
                if N >= max(27, 13): u_out[i] = u_out[i] - filtfilt(b_l, a_l, u_out[i])

    # WGN / Color / unknown: noise-index threshold
    if _has(noise_type, "WG") or _is_color(noise_type) or not noise_type.strip():
        u_out[0] = 0.0
        for i in range(1, K - 1):   # TrustEMG skips last IMF (residual)
            if est_noise_std[i] > imf_std[i] or imf_std[0] == 0:
                u_out[i] = 0.0
            else:
                sigma = _mad_sigma(imfs[i])
                thr   = _universal_threshold(sigma, N) / 4.0
                u_out[i] = _soft_threshold(u_out[i], thr)

    # ECG
    if _has(noise_type, "E"):
        try:
            from scipy.stats import pearsonr
            nyq = fs / 2.0
            b_lp, a_lp = butter(4, min(40.0 / nyq, 0.9999), btype="low")
            b_hp, a_hp = butter(4, min(max(30.0 / nyq, 1e-4), 0.9999), btype="high")
            if N >= max(27, 13):
                ecg_tmpl = filtfilt(b_lp, a_lp, x_in)
                active = ~np.all(u_out == 0, axis=1)
                u_active = u_out[active]
                corr = [abs(pearsonr(ecg_tmpl, u_active[j])[0])
                        for j in range(u_active.shape[0])]
                top6 = np.argsort(corr)[-6:]
                for idx in top6:
                    u_active[idx] = filtfilt(b_hp, a_hp, u_active[idx])
                u_out[active] = u_active
        except Exception:
            pass

    return u_out.sum(axis=0), True


# ============================================================================
# ── Dispatch helper ───────────────────────────────────────────────────────
# ============================================================================

ALL_METHODS = ["hp", "ts", "emd", "vmd", "ceemdan"]


def apply_method(
    method: str,
    noisy_raw: np.ndarray,
    params: dict,
    fs: int = 1000,
    noise_type: str = "",
) -> Tuple[np.ndarray, bool]:
    """
    Unified dispatch.
    Returns (enhanced_signal, success).
    `params` must contain sub-dicts keyed by method name.
    `noise_type` is the noise_types field from the test sample (e.g. "WGN+ECG").
    """
    method = method.lower()

    def _hp_fallback(p_hp):
        return apply_hp_filter(noisy_raw, fs=fs,
                                cutoff_hz=p_hp["best_cutoff_hz"],
                                order=p_hp["order"])

    if method == "hp":
        p = params["hp"]
        return apply_hp_filter(noisy_raw, fs=fs,
                                cutoff_hz=p["best_cutoff_hz"],
                                order=p["order"]), True

    elif method == "ts":
        p = params["ts"]
        enh, err = apply_template_subtraction(
            noisy_raw, fs=fs,
            peak_detect_bp_low=p["peak_detect_bp_low_hz"],
            peak_detect_bp_high=p["peak_detect_bp_high_hz"],
            peak_detect_order=p["peak_detect_order"],
            avg_window=p["avg_window"],
            min_peaks=p["min_peaks"],
            min_beat_gap=p["min_beat_gap_samples"],
            tile_factor=p.get("tile_factor", 8),
        )
        if err != 0:
            if p.get("fallback", "noisy") == "hp":
                return _hp_fallback(params["hp"]), False
            return noisy_raw.copy(), False
        return enh, True

    elif method == "emd":
        p = params.get("emd", {})
        enh, ok = apply_emd_filter(
            noisy_raw, fs=fs,
            f_min=p.get("best_f_min_hz", p.get("f_min_hz", 20.0)),
            f_max=p.get("f_max_hz", 500.0),
            max_imfs=p.get("max_imfs", 8),
            noise_type=noise_type,
        )
        if not ok:
            if p.get("fallback", "noisy") == "hp":
                return _hp_fallback(params["hp"]), False
            return noisy_raw.copy(), False
        return enh, True

    elif method == "vmd":
        p = params.get("vmd", {})
        enh, ok = apply_vmd_filter(
            noisy_raw, fs=fs,
            K=p.get("best_K", p.get("K", 10)),
            alpha=p.get("alpha", 2000.0),
            tau=p.get("tau", 0.0),
            f_min=p.get("f_min_hz", 20.0),
            f_max=p.get("f_max_hz", 500.0),
            tol=p.get("tol", 1e-7),
            noise_type=noise_type,
        )
        if not ok:
            if p.get("fallback", "noisy") == "hp":
                return _hp_fallback(params["hp"]), False
            return noisy_raw.copy(), False
        return enh, True

    elif method == "ceemdan":
        p = params.get("ceemdan", {})
        enh, ok = apply_ceemdan_filter(
            noisy_raw, fs=fs,
            trials=p.get("trials", 20),
            f_min=p.get("best_f_min_hz", p.get("f_min_hz", 20.0)),
            f_max=p.get("f_max_hz", 500.0),
            max_imfs=p.get("max_imfs", 8),
            noise_type=noise_type,
        )
        if not ok:
            if p.get("fallback", "noisy") == "hp":
                return _hp_fallback(params["hp"]), False
            return noisy_raw.copy(), False
        return enh, True

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from: {ALL_METHODS}")