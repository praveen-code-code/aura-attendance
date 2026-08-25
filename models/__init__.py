from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date

db = SQLAlchemy()


class Admin(db.Model, UserMixin):
    """Administrator account (Section 17 — Users/Admin Table)."""

    __tablename__ = "admins"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    def get_id(self):
        # Prefix so Flask-Login can distinguish Admin sessions from Student
        # sessions if both ever share a login flow.
        return f"admin:{self.id}"


class Student(db.Model):
    """Registered student / subject (Section 17 — Students Table)."""

    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    department = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    biometric_features = db.relationship(
        "BiometricFeature", backref="student", cascade="all, delete-orphan"
    )
    attendance_records = db.relationship(
        "Attendance", backref="student", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Student {self.student_id} {self.name}>"


class BiometricFeature(db.Model):
    """Stored ECG/EEG feature vectors used to (re)train models
    (Section 17 — Biometric Features Table)."""

    __tablename__ = "biometric_features"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    modality = db.Column(db.String(10), nullable=False)  # 'ecg' or 'eeg'
    feature_data = db.Column(db.Text, nullable=False)  # JSON-encoded feature vector
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Attendance(db.Model):
    """One attendance event, produced after multimodal authentication
    (Section 17 — Attendance Table)."""

    __tablename__ = "attendance"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    date = db.Column(db.Date, default=date.today, nullable=False)
    time = db.Column(db.Time, default=lambda: datetime.now().time(), nullable=False)
    status = db.Column(db.String(20), default="PRESENT", nullable=False)
    ecg_predicted_id = db.Column(db.String(50), nullable=True)
    eeg_predicted_id = db.Column(db.String(50), nullable=True)
    ecg_score = db.Column(db.Float, nullable=True)
    eeg_score = db.Column(db.Float, nullable=True)
    combined_score = db.Column(db.Float, nullable=True)
    authentication_result = db.Column(db.String(20), nullable=False)  # VERIFIED / REJECTED
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("student_id", "date", name="uq_student_date"),
    )
