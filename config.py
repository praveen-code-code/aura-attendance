import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Central configuration for the Aura Attendance System.

    All values can be overridden with environment variables so the same
    codebase works locally (SQLite fallback) and in a real deployment
    (PostgreSQL via DATABASE_URL).
    """

    SECRET_KEY = os.environ.get("AURA_SECRET_KEY", "dev-secret-change-me")

    # Prefer PostgreSQL (as specified in the project doc). Falls back to a
    # local SQLite file so the app runs immediately without a DB server.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'aura.db')}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER_ECG = os.path.join(BASE_DIR, "uploads", "ecg")
    UPLOAD_FOLDER_EEG = os.path.join(BASE_DIR, "uploads", "eeg")
    ALLOWED_EXTENSIONS = {"csv", "npy", "txt"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB per upload

    DATASET_DIR_ECG = os.path.join(BASE_DIR, "dataset", "ecg")
    DATASET_DIR_EEG = os.path.join(BASE_DIR, "dataset", "eeg")
    MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
    ECG_MODEL_PATH = os.path.join(MODEL_DIR, "ecg_model.pkl")
    EEG_MODEL_PATH = os.path.join(MODEL_DIR, "eeg_model.pkl")
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

    # Multimodal fusion weights (Section 15 of the doc) and the minimum
    # combined confidence required to mark attendance (Section 16 / 29).
    ECG_WEIGHT = float(os.environ.get("AURA_ECG_WEIGHT", 0.5))
    EEG_WEIGHT = float(os.environ.get("AURA_EEG_WEIGHT", 0.5))
    AUTH_THRESHOLD = float(os.environ.get("AURA_AUTH_THRESHOLD", 0.60))

    # Signal characteristics used by both the synthetic dataset generator
    # and the preprocessing/feature-extraction modules.
    ECG_SAMPLING_RATE = 250  # Hz
    EEG_SAMPLING_RATE = 128  # Hz
    ECG_SEGMENT_SECONDS = 10
    EEG_SEGMENT_SECONDS = 10

    PERMANENT_SESSION_LIFETIME = timedelta(hours=4)
