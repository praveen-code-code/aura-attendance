"""ECG feature extraction (Section 10): R-R interval / heart-rate features,
amplitude statistics and frequency-domain characteristics.
"""
import numpy as np
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq

FEATURE_NAMES = [
    "mean", "std", "variance", "skewness", "kurtosis",
    "mean_rr", "std_rr", "rmssd", "mean_hr",
    "dominant_freq", "spectral_energy", "spectral_entropy",
    "signal_range", "zero_crossing_rate",
]


def _rr_features(r_peaks: np.ndarray, fs: int) -> dict:
    if len(r_peaks) < 2:
        return {"mean_rr": 0.0, "std_rr": 0.0, "rmssd": 0.0, "mean_hr": 0.0}

    rr_intervals = np.diff(r_peaks) / fs  # seconds
    mean_rr = float(np.mean(rr_intervals))
    std_rr = float(np.std(rr_intervals))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))) if len(rr_intervals) > 1 else 0.0
    mean_hr = float(60.0 / mean_rr) if mean_rr > 0 else 0.0
    return {"mean_rr": mean_rr, "std_rr": std_rr, "rmssd": rmssd, "mean_hr": mean_hr}


def _frequency_features(sig: np.ndarray, fs: int) -> dict:
    n = len(sig)
    freqs = rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(rfft(sig))
    if spectrum.sum() < 1e-8:
        return {"dominant_freq": 0.0, "spectral_energy": 0.0, "spectral_entropy": 0.0}

    dominant_freq = float(freqs[np.argmax(spectrum)])
    spectral_energy = float(np.sum(spectrum ** 2))
    psd_norm = spectrum / np.sum(spectrum)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    return {
        "dominant_freq": dominant_freq,
        "spectral_energy": spectral_energy,
        "spectral_entropy": spectral_entropy,
    }


def extract_ecg_features(preprocessed: dict) -> np.ndarray:
    """Takes the dict returned by preprocess_ecg() and returns a fixed-length
    numerical feature vector suitable for classical ML models."""
    sig = preprocessed["signal"]
    r_peaks = preprocessed["r_peaks"]
    fs = preprocessed["fs"]

    zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
    stats = {
        "mean": float(np.mean(sig)),
        "std": float(np.std(sig)),
        "variance": float(np.var(sig)),
        "skewness": float(sp_stats.skew(sig)),
        "kurtosis": float(sp_stats.kurtosis(sig)),
        "signal_range": float(np.max(sig) - np.min(sig)),
        "zero_crossing_rate": float(zero_crossings / len(sig)),
    }
    stats.update(_rr_features(r_peaks, fs))
    stats.update(_frequency_features(sig, fs))

    return np.array([stats[name] for name in FEATURE_NAMES], dtype=float)
