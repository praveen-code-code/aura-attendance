
import os
import numpy as np
from datetime import date
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename

from models import db, Student, Attendance
from ml.predictor import predictor, ModelNotTrainedError
from ml.fusion import fuse_predictions

attendance_bp = Blueprint("attendance", __name__)


def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def _load_signal(file_storage) -> np.ndarray:
    """Reads an uploaded CSV/TXT/NPY signal file into a 1D numpy array."""
    filename = file_storage.filename
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "npy":
        data = np.load(file_storage.stream, allow_pickle=False)
    else:
        data = np.loadtxt(file_storage.stream, delimiter=",")
    return np.asarray(data, dtype=float).flatten()


@attendance_bp.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


@attendance_bp.route("/attendance/mark", methods=["POST"])
def mark_attendance():
    student_id = request.form.get("student_id", "").strip()
    ecg_file = request.files.get("ecg_file")
    eeg_file = request.files.get("eeg_file")

    # --- Validation (Section 22 #4, Section 29) ---
    if not student_id:
        return jsonify({"error": "Student ID is required."}), 400

    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"error": f"No registered student with ID {student_id}."}), 404

    if not ecg_file or not eeg_file:
        return jsonify({"error": "Both an ECG and an EEG file are required."}), 400

    for f in (ecg_file, eeg_file):
        if not _allowed_file(f.filename):
            return jsonify({"error": f"Unsupported file type: {f.filename}"}), 400

    if not predictor.is_ready():
        cfg = current_app.config
        if not predictor.load(current_app.config):
            return jsonify({
                "error": "Models are not trained yet. Run `python -m ml.train` first."
            }), 503

    # --- Duplicate attendance check (Section 16) ---
    existing = Attendance.query.filter_by(student_id=student.id, date=date.today()).first()
    if existing:
        return jsonify({
            "message": f"{student.name} already marked present today at {existing.time.strftime('%H:%M:%S')}.",
            "duplicate": True,
        }), 200

    # --- Save uploads (Section 18) ---
    ecg_filename = secure_filename(f"{student_id}_{ecg_file.filename}")
    eeg_filename = secure_filename(f"{student_id}_{eeg_file.filename}")
    ecg_path = os.path.join(current_app.config["UPLOAD_FOLDER_ECG"], ecg_filename)
    eeg_path = os.path.join(current_app.config["UPLOAD_FOLDER_EEG"], eeg_filename)

    try:
        ecg_signal = _load_signal(ecg_file)
        eeg_signal = _load_signal(eeg_file)
        ecg_file.stream.seek(0)
        eeg_file.stream.seek(0)
        ecg_file.save(ecg_path)
        eeg_file.save(eeg_path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a validation error
        return jsonify({"error": f"Could not read uploaded signal files: {exc}"}), 400

    # --- Preprocessing + prediction (Section 13, 14) ---
    try:
        ecg_pred = predictor.predict_ecg(ecg_signal, current_app.config["ECG_SAMPLING_RATE"])
        eeg_pred = predictor.predict_eeg(eeg_signal, current_app.config["EEG_SAMPLING_RATE"])
    except ModelNotTrainedError as exc:
        return jsonify({"error": str(exc)}), 503

    # --- Fusion (Section 15) ---
    result = fuse_predictions(
        ecg_pred, eeg_pred, claimed_id=student_id,
        ecg_weight=current_app.config["ECG_WEIGHT"],
        eeg_weight=current_app.config["EEG_WEIGHT"],
        threshold=current_app.config["AUTH_THRESHOLD"],
    )

    record = Attendance(
        student_id=student.id,
        status="PRESENT" if result.authenticated else "REJECTED",
        ecg_predicted_id=ecg_pred.predicted_id,
        eeg_predicted_id=eeg_pred.predicted_id,
        ecg_score=ecg_pred.confidence,
        eeg_score=eeg_pred.confidence,
        combined_score=result.combined_score,
        authentication_result="VERIFIED" if result.authenticated else "REJECTED",
    )

    if result.authenticated:
        db.session.add(record)
        db.session.commit()

    return jsonify({
        "student_name": student.name,
        "student_id": student.student_id,
        "ecg_prediction": ecg_pred.predicted_id,
        "ecg_confidence": round(ecg_pred.confidence * 100, 2),
        "eeg_prediction": eeg_pred.predicted_id,
        "eeg_confidence": round(eeg_pred.confidence * 100, 2),
        "combined_score": round(result.combined_score * 100, 2),
        "authenticated": result.authenticated,
        "attendance": "PRESENT" if result.authenticated else "NOT MARKED",
        "date": date.today().isoformat(),
    }), 200 if result.authenticated else 401
