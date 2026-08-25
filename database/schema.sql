-- Aura Attendance System — PostgreSQL schema
-- Matches the SQLAlchemy models in models/__init__.py.
-- db.create_all() will create these automatically on first run;
-- this file is provided for reference / manual provisioning.

CREATE TABLE IF NOT EXISTS admins (
    id SERIAL PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(120) UNIQUE,
    department VARCHAR(120),
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_students_student_id ON students (student_id);

CREATE TABLE IF NOT EXISTS biometric_features (
    id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    modality VARCHAR(10) NOT NULL CHECK (modality IN ('ecg', 'eeg')),
    feature_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    time TIME NOT NULL DEFAULT CURRENT_TIME,
    status VARCHAR(20) NOT NULL DEFAULT 'PRESENT',
    ecg_predicted_id VARCHAR(50),
    eeg_predicted_id VARCHAR(50),
    ecg_score FLOAT,
    eeg_score FLOAT,
    combined_score FLOAT,
    authentication_result VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (student_id, date)
);
