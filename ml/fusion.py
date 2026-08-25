"""Multimodal fusion of ECG and EEG identity predictions (Section 15).

Combines two independent classifier outputs into a single authentication
decision using weighted confidence scoring, with an explicit identity
mismatch check.
"""
from dataclasses import dataclass


@dataclass
class ModalityPrediction:
    predicted_id: str
    confidence: float  # 0.0 - 1.0


@dataclass
class FusionResult:
    ecg_prediction: ModalityPrediction
    eeg_prediction: ModalityPrediction
    identity_match: bool
    combined_score: float
    authenticated: bool
    final_identity: str | None


def fuse_predictions(
    ecg: ModalityPrediction,
    eeg: ModalityPrediction,
    claimed_id: str,
    ecg_weight: float = 0.5,
    eeg_weight: float = 0.5,
    threshold: float = 0.60,
) -> FusionResult:
    """Implements the weighted-fusion formula from Section 15:

        combined_score = (ecg_score * ecg_weight) + (eeg_score * eeg_weight)

    Authentication requires BOTH modalities to agree with the claimed
    student ID and the combined score to clear `threshold`.
    """
    identity_match = (
        ecg.predicted_id == claimed_id and eeg.predicted_id == claimed_id
    )
    combined_score = (ecg.confidence * ecg_weight) + (eeg.confidence * eeg_weight)
    authenticated = identity_match and combined_score >= threshold

    return FusionResult(
        ecg_prediction=ecg,
        eeg_prediction=eeg,
        identity_match=identity_match,
        combined_score=combined_score,
        authenticated=authenticated,
        final_identity=claimed_id if authenticated else None,
    )
