"""Model training workflow (Section 27): load dataset -> preprocess ->
extract features -> split -> train -> evaluate -> save best model.

Trains separate ECG and EEG identification models, as specified in the
proposed system (Section 6).

Usage:
    python -m ml.train
"""
import os
import glob
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import Config
from ml.ecg_preprocessing import preprocess_ecg
from ml.eeg_preprocessing import preprocess_eeg
from ml.ecg_features import extract_ecg_features
from ml.eeg_features import extract_eeg_features

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "svm": SVC(probability=True, kernel="rbf", random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
}


def _load_modality_dataset(dataset_dir: str, modality: str, cfg: Config):
    """Walks dataset_dir/<subject_id>/sample_*.csv, preprocesses each signal,
    and returns (feature_matrix, labels)."""
    X, y = [], []
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject folders found in {dataset_dir}. "
            f"Run `python -m ml.generate_demo_dataset` first, or place a real "
            f"dataset there organized as <subject_id>/sample_*.csv"
        )

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        sample_files = sorted(glob.glob(os.path.join(subject_dir, "sample_*.csv")))
        for sample_file in sample_files:
            raw = np.loadtxt(sample_file, delimiter=",")
            if modality == "ecg":
                pre = preprocess_ecg(raw, cfg.ECG_SAMPLING_RATE)
                feats = extract_ecg_features(pre)
            else:
                pre = preprocess_eeg(raw, cfg.EEG_SAMPLING_RATE)
                feats = extract_eeg_features(pre)
            X.append(feats)
            y.append(subject_id)

    return np.array(X), np.array(y)


def _train_and_select_best(X_train, X_test, y_train, y_test, modality_name: str):
    """Trains each candidate algorithm (Section 12) and picks the one with
    the highest F1-score on the held-out test split (Section 28)."""
    best_name, best_pipeline, best_f1, best_report = None, None, -1.0, None

    for name, estimator in CANDIDATE_MODELS.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clone(estimator)),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  [{modality_name}] {name}: acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

        if f1 > best_f1:
            best_name, best_pipeline, best_f1 = name, pipeline, f1
            best_report = {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

    print(f"  -> Best {modality_name} model: {best_name} (f1={best_f1:.3f})")
    return best_pipeline, best_report


def train_all(test_size: float = 0.2, random_state: int = 42):
    cfg = Config()
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    label_encoder = LabelEncoder()

    print("Loading + preprocessing ECG dataset...")
    X_ecg, y_ecg_raw = _load_modality_dataset(cfg.DATASET_DIR_ECG, "ecg", cfg)
    print("Loading + preprocessing EEG dataset...")
    X_eeg, y_eeg_raw = _load_modality_dataset(cfg.DATASET_DIR_EEG, "eeg", cfg)

    # Fit the shared label encoder across the union of subjects seen in
    # either modality so ECG/EEG predictions map to the same ID space.
    all_subjects = sorted(set(y_ecg_raw) | set(y_eeg_raw))
    label_encoder.fit(all_subjects)

    y_ecg = label_encoder.transform(y_ecg_raw)
    y_eeg = label_encoder.transform(y_eeg_raw)

    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        X_ecg, y_ecg, test_size=test_size, random_state=random_state, stratify=y_ecg
    )
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_eeg, y_eeg, test_size=test_size, random_state=random_state, stratify=y_eeg
    )

    print("\nTraining ECG models...")
    ecg_model, ecg_report = _train_and_select_best(Xe_train, Xe_test, ye_train, ye_test, "ECG")

    print("\nTraining EEG models...")
    eeg_model, eeg_report = _train_and_select_best(Xg_train, Xg_test, yg_train, yg_test, "EEG")

    joblib.dump(ecg_model, cfg.ECG_MODEL_PATH)
    joblib.dump(eeg_model, cfg.EEG_MODEL_PATH)
    joblib.dump(label_encoder, cfg.LABEL_ENCODER_PATH)

    print(f"\nSaved models to:\n  {cfg.ECG_MODEL_PATH}\n  {cfg.EEG_MODEL_PATH}\n  {cfg.LABEL_ENCODER_PATH}")
    return {"ecg": ecg_report, "eeg": eeg_report}


if __name__ == "__main__":
    train_all()
