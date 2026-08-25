"""EEG preprocessing pipeline (Section 9): filtering, artifact removal,
normalization and segmentation.
"""
import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(eeg: np.ndarray, fs: int, low: float = 1.0, high: float = 45.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, eeg)


def notch_filter(eeg: np.ndarray, fs: int, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Removes mains-hum artifact (50/60 Hz)."""
    nyq = 0.5 * fs
    if freq >= nyq:
        return eeg
    b, a = sp_signal.iirnotch(freq / nyq, quality)
    return sp_signal.filtfilt(b, a, eeg)


def remove_artifacts(eeg: np.ndarray, threshold_std: float = 5.0) -> np.ndarray:
    """Clips extreme amplitude spikes (eye blinks / muscle artifacts) rather
    than dropping samples, so segment length stays constant."""
    mean, std = np.mean(eeg), np.std(eeg)
    if std < 1e-8:
        return eeg
    limit = mean + threshold_std * std
    return np.clip(eeg, mean - threshold_std * std, limit)


def normalize(eeg: np.ndarray) -> np.ndarray:
    std = np.std(eeg)
    if std < 1e-8:
        return eeg - np.mean(eeg)
    return (eeg - np.mean(eeg)) / std


def preprocess_eeg(raw_eeg: np.ndarray, fs: int) -> dict:
    """Full EEG preprocessing pipeline for a single-channel segment."""
    eeg = np.asarray(raw_eeg, dtype=float)
    eeg = notch_filter(eeg, fs)
    eeg = bandpass_filter(eeg, fs)
    eeg = remove_artifacts(eeg)
    eeg = normalize(eeg)
    return {"signal": eeg, "fs": fs}
