"""Generates a small synthetic ECG + EEG dataset organized by subject
(Section 8, Step 2), so the full pipeline can be trained and demoed without
first sourcing a real public dataset.

IMPORTANT: This is for development/demo purposes only. Swap in a real
public ECG/EEG biometric dataset (e.g. PhysioNet ECG-ID / MIT-BIH for ECG,
a public EEG biometrics corpus for EEG) before treating results as
meaningful — see Section 26 of the project doc for dataset requirements,
and note the caveat that ECG and EEG should ideally come from the SAME
subjects for multimodal fusion to be scientifically valid.

Usage:
    python -m ml.generate_demo_dataset --subjects 10 --samples 15
"""
import argparse
import os
import numpy as np

from config import Config


def _synthetic_ecg(duration_s: int, fs: int, subject_seed: int, rng: np.random.Generator) -> np.ndarray:
    """Generates a plausible-looking ECG-like signal with a subject-specific
    heart rate and waveform morphology, plus noise, so different subjects are
    separable and different samples of the same subject are not identical."""
    t = np.arange(0, duration_s, 1 / fs)
    base_hr = 60 + (subject_seed % 40)  # 60-99 bpm, subject-specific
    hr = base_hr + rng.normal(0, 2)  # sample-to-sample variability
    beat_interval = 60.0 / hr

    signal = np.zeros_like(t)
    beat_times = np.arange(0, duration_s, beat_interval)
    qrs_amplitude = 1.0 + 0.1 * (subject_seed % 5) + rng.normal(0, 0.05)
    qrs_width = 0.03 + 0.005 * (subject_seed % 3)

    for bt in beat_times:
        signal += qrs_amplitude * np.exp(-((t - bt) ** 2) / (2 * qrs_width ** 2))
        # small P/T waves for realism
        signal += 0.15 * np.exp(-((t - (bt - 0.15)) ** 2) / (2 * (qrs_width * 2) ** 2))
        signal += 0.25 * np.exp(-((t - (bt + 0.25)) ** 2) / (2 * (qrs_width * 3) ** 2))

    noise = rng.normal(0, 0.03, size=t.shape)
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.2 * t + subject_seed)
    return signal + noise + baseline_wander


def _synthetic_eeg(duration_s: int, fs: int, subject_seed: int, rng: np.random.Generator) -> np.ndarray:
    """Generates an EEG-like signal as a subject-specific mixture of band
    oscillations (delta/theta/alpha/beta/gamma) plus noise."""
    t = np.arange(0, duration_s, 1 / fs)
    rng_local = rng

    # Band center frequency is fixed per subject (a stable per-subject seed),
    # only jittered slightly per-sample, so PSD-based features stay separable
    # across subjects while still varying sample to sample.
    band_rng = np.random.default_rng(subject_seed * 97 + 7)
    band_centers = {
        "delta": band_rng.uniform(1, 4),
        "theta": band_rng.uniform(4, 8),
        "alpha": band_rng.uniform(8, 13),
        "beta": band_rng.uniform(13, 30),
        "gamma": band_rng.uniform(30, 45),
    }
    band_freqs = {b: f + rng_local.normal(0, 0.15) for b, f in band_centers.items()}

    # Subject-specific relative band weights give each subject a distinct
    # spectral "fingerprint" — spread out more widely than before, and
    # scaled by subject_seed so each subject has a clearly different overall
    # power profile rather than a small offset easily lost in noise.
    base_weights = {
        "delta": 0.4 + 0.35 * ((subject_seed * 3) % 7) / 6,
        "theta": 0.3 + 0.35 * ((subject_seed * 5 + 1) % 7) / 6,
        "alpha": 0.5 + 0.40 * ((subject_seed * 7 + 2) % 7) / 6,
        "beta": 0.2 + 0.30 * ((subject_seed * 11 + 3) % 7) / 6,
        "gamma": 0.1 + 0.20 * ((subject_seed * 13 + 4) % 7) / 6,
    }

    signal = np.zeros_like(t)
    for band, freq in band_freqs.items():
        amp = base_weights[band] + rng_local.normal(0, 0.02)
        phase = rng_local.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    noise = rng_local.normal(0, 0.08, size=t.shape)
    return signal + noise


def generate_dataset(n_subjects: int, n_samples: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    cfg = Config()

    for subject_idx in range(1, n_subjects + 1):
        student_id = f"STU{100 + subject_idx}"
        ecg_dir = os.path.join(cfg.DATASET_DIR_ECG, student_id)
        eeg_dir = os.path.join(cfg.DATASET_DIR_EEG, student_id)
        os.makedirs(ecg_dir, exist_ok=True)
        os.makedirs(eeg_dir, exist_ok=True)

        for sample_idx in range(1, n_samples + 1):
            ecg = _synthetic_ecg(cfg.ECG_SEGMENT_SECONDS, cfg.ECG_SAMPLING_RATE, subject_idx, rng)
            eeg = _synthetic_eeg(cfg.EEG_SEGMENT_SECONDS, cfg.EEG_SAMPLING_RATE, subject_idx, rng)

            np.savetxt(os.path.join(ecg_dir, f"sample_{sample_idx:02d}.csv"), ecg, delimiter=",")
            np.savetxt(os.path.join(eeg_dir, f"sample_{sample_idx:02d}.csv"), eeg, delimiter=",")

        print(f"Generated {n_samples} ECG + EEG samples for {student_id}")

    print(f"\nDone. Dataset written to:\n  {cfg.DATASET_DIR_ECG}\n  {cfg.DATASET_DIR_EEG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic ECG/EEG demo dataset.")
    parser.add_argument("--subjects", type=int, default=10, help="Number of subjects to generate")
    parser.add_argument("--samples", type=int, default=15, help="Samples per subject")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.subjects, args.samples, args.seed)
