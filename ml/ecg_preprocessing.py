"""ECG preprocessing pipeline (Section 9): filtering, baseline correction,
normalization and R-peak detection.
"""
import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(ecg: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Removes baseline wander (<0.5 Hz) and high-frequency noise (>40 Hz)."""
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, ecg)


def remove_baseline_wander(ecg: np.ndarray, fs: int) -> np.ndarray:
    """High-pass filter to strip slow baseline drift."""
    nyq = 0.5 * fs
    b, a = sp_signal.butter(2, 0.5 / nyq, btype="high")
    return sp_signal.filtfilt(b, a, ecg)


def normalize(ecg: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    std = np.std(ecg)
    if std < 1e-8:
        return ecg - np.mean(ecg)
    return (ecg - np.mean(ecg)) / std


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Simple R-peak detector: bandpass -> squared derivative -> peak picking.
    Adequate for a prototype; swap for Pan-Tompkins if higher accuracy is needed.
    """
    filtered = bandpass_filter(ecg, fs, low=5.0, high=15.0)
    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2
    window = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    min_distance = int(0.3 * fs)  # refractory period ~300ms (max ~200bpm)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=min_distance)
    return peaks


def preprocess_ecg(raw_ecg: np.ndarray, fs: int) -> dict:
    """Full ECG preprocessing pipeline. Returns the cleaned signal plus
    detected R-peaks for downstream feature extraction."""
    ecg = np.asarray(raw_ecg, dtype=float)
    ecg = remove_baseline_wander(ecg, fs)
    ecg = bandpass_filter(ecg, fs)
    ecg = normalize(ecg)
    r_peaks = detect_r_peaks(ecg, fs)
    return {"signal": ecg, "r_peaks": r_peaks, "fs": fs}
