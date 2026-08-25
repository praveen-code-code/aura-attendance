"""EEG feature extraction (Section 10): band power (delta/theta/alpha/beta/
gamma), statistical and power-spectral-density features.
"""
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

FEATURE_NAMES = [
    "mean", "std", "variance", "skewness", "kurtosis",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "total_power", "mean_psd", "spectral_entropy",
]


def _band_powers(sig: np.ndarray, fs: int) -> dict:
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), fs * 2))
    powers = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[f"{band}_power"] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0

    total_power = float(np.sum(psd))
    mean_psd = float(np.mean(psd))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    powers.update({
        "total_power": total_power,
        "mean_psd": mean_psd,
        "spectral_entropy": spectral_entropy,
    })
    return powers


def extract_eeg_features(preprocessed: dict) -> np.ndarray:
    """Takes the dict returned by preprocess_eeg() and returns a fixed-length
    numerical feature vector."""
    sig = preprocessed["signal"]
    fs = preprocessed["fs"]

    stats = {
        "mean": float(np.mean(sig)),
        "std": float(np.std(sig)),
        "variance": float(np.var(sig)),
        "skewness": float(sp_stats.skew(sig)),
        "kurtosis": float(sp_stats.kurtosis(sig)),
    }
    stats.update(_band_powers(sig, fs))

    return np.array([stats[name] for name in FEATURE_NAMES], dtype=float)
