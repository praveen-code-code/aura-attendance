# Aura — AI-Based Multimodal Biometric Attendance System (ECG + EEG)

A working prototype implementing the pipeline from the project documentation:
ECG/EEG preprocessing → feature extraction → per-modality ML classifiers →
weighted multimodal fusion → identity verification → attendance recorded
in PostgreSQL, with a Flask web app for students (attendance capture) and
admins (student management, monitoring, reports).

## 1. Setup

```bash
cd aura-attendance
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit values, especially DATABASE_URL
```

By default the app falls back to a local SQLite file (`aura.db`) if
`DATABASE_URL` isn't set, so you can run everything without PostgreSQL
installed. Point `DATABASE_URL` at Postgres when you're ready — the schema
is identical either way (see `database/schema.sql`).

## 2. Generate a demo dataset and train the models

No physical ECG/EEG hardware is used (per the project's scope). To get the
full pipeline running immediately, generate a synthetic dataset with a
distinct per-subject signal "fingerprint":

```bash
python -m ml.generate_demo_dataset --subjects 10 --samples 15
python -m ml.train
```

This writes `dataset/ecg/<subject>/sample_*.csv` and
`dataset/eeg/<subject>/sample_*.csv`, trains Random Forest / SVM / KNN
candidates for each modality, picks the best by F1-score, and saves them to
`trained_models/`.

**Swap in a real dataset later** by replacing the contents of `dataset/ecg/`
and `dataset/eeg/` with the same `<subject_id>/sample_NN.csv` layout, using
a public ECG biometrics corpus (e.g. PhysioNet ECG-ID / MIT-BIH) and a
public EEG biometrics corpus — ideally recorded from the *same* subjects,
since cross-dataset fusion across different subject pools isn't scientifically
valid (see the project doc, Section 26/31). Then re-run `ml.train`.

## 3. Run the app

```bash
python app.py
```

Visit `http://localhost:5000`:

- `/` and `/attendance` — student-facing attendance capture (student ID +
  ECG file + EEG file upload).
- `/setup` — one-time admin account creation (only works until the first
  admin exists).
- `/login` → `/dashboard`, `/students`, `/reports` — admin area.

### Trying attendance with the demo dataset

Any `dataset/ecg/STU10X/sample_NN.csv` / `dataset/eeg/STU10X/sample_NN.csv`
pair generated in step 2 can be uploaded directly as a test "attendance"
submission for `STU10X` (after registering that student in `/students`).

## 4. Project layout

```
aura-attendance/
├── app.py                  Flask application factory
├── config.py                Central configuration (env-driven)
├── models/__init__.py       SQLAlchemy models: Admin, Student, BiometricFeature, Attendance
├── ml/
│   ├── ecg_preprocessing.py Filtering, baseline correction, R-peak detection
│   ├── eeg_preprocessing.py Filtering, notch, artifact clipping
│   ├── ecg_features.py      R-R interval / HR / frequency-domain features
│   ├── eeg_features.py      Band power (delta..gamma) + spectral features
│   ├── fusion.py            Weighted multimodal fusion + identity match check
│   ├── train.py             Load → preprocess → extract → train → evaluate → save
│   ├── predictor.py         Loads trained models for use in the attendance route
│   └── generate_demo_dataset.py  Synthetic per-subject ECG/EEG dataset generator
├── routes/
│   ├── auth.py               Admin login/logout/setup
│   ├── students.py           Student registration & management
│   ├── attendance.py         Upload → authenticate → mark attendance
│   └── admin.py               Dashboard, reports, analytics API
├── templates/                Jinja2 templates (dark clinical-monitor UI)
├── static/css/style.css
├── database/schema.sql       Reference PostgreSQL schema
└── dataset/, trained_models/, uploads/   (created/populated at runtime)
```


