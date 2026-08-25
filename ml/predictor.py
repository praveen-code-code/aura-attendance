"""Loads the trained ECG/EEG models once and exposes a simple predict()
call used by the attendance route (Section 13, 14 — ECG/EEG Identification).
"""
import os
import joblib
import numpy as np

from ml.ecg_preprocessing import preprocess_ecg
from ml.eeg_preprocessing import preprocess_eeg
from ml.ecg_features import extract_ecg_features
from ml.eeg_features import extract_eeg_features
from ml.fusion import ModalityPrediction


class ModelNotTrainedError(RuntimeError):
    pass


class BiometricPredictor:
    """Singleton-style holder for the trained models + label encoder.
    Call `load(config)` once at app startup; models are re-loaded lazily
    if the underlying files change (e.g. after a retrain)."""

    def __init__(self):
        self.ecg_model = None
        self.eeg_model = None
        self.label_encoder = None
        self._mtimes = {}

    def load(self, cfg) -> bool:
        paths = {
            "ecg": cfg["ECG_MODEL_PATH"],
            "eeg": cfg["EEG_MODEL_PATH"],
            "encoder": cfg["LABEL_ENCODER_PATH"],
        }
        if not all(os.path.exists(p) for p in paths.values()):
            return False

        self.ecg_model = joblib.load(paths["ecg"])
        self.eeg_model = joblib.load(paths["eeg"])
        self.label_encoder = joblib.load(paths["encoder"])
        self._mtimes = {k: os.path.getmtime(v) for k, v in paths.items()}
        return True

    def is_ready(self) -> bool:
        return self.ecg_model is not None and self.eeg_model is not None

    def _predict_one(self, model, feature_vector: np.ndarray) -> tuple[str, float]:
        proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
        best_idx = int(np.argmax(proba))
        predicted_label = self.label_encoder.inverse_transform([best_idx])[0]
        confidence = float(proba[best_idx])
        return predicted_label, confidence

    def predict_ecg(self, raw_signal: np.ndarray, fs: int) -> ModalityPrediction:
        if not self.is_ready():
            raise ModelNotTrainedError("ECG model is not trained yet. Run `python -m ml.train`.")
        pre = preprocess_ecg(raw_signal, fs)
        feats = extract_ecg_features(pre)
        predicted_id, confidence = self._predict_one(self.ecg_model, feats)
        return ModalityPrediction(predicted_id=predicted_id, confidence=confidence)

    def predict_eeg(self, raw_signal: np.ndarray, fs: int) -> ModalityPrediction:
        if not self.is_ready():
            raise ModelNotTrainedError("EEG model is not trained yet. Run `python -m ml.train`.")
        pre = preprocess_eeg(raw_signal, fs)
        feats = extract_eeg_features(pre)
        predicted_id, confidence = self._predict_one(self.eeg_model, feats)
        return ModalityPrediction(predicted_id=predicted_id, confidence=confidence)


predictor = BiometricPredictor()
