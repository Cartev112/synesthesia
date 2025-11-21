"""
Machine learning components for BCI system.

Includes:
- Artifact classification
- Mental state classification
"""

from backend.ml.artifact_classifier import ArtifactCNN, ArtifactClassifier
from backend.ml.state_classifier import (
    MentalStateClassifier,
    MentalStateTracker
)
from backend.ml.calibration import (
    UserCalibration,
    CalibrationSession,
    CalibrationProtocol
)

__all__ = [
    'ArtifactCNN',
    'ArtifactClassifier',
    'MentalStateClassifier',
    'MentalStateTracker',
    'UserCalibration',
    'CalibrationSession',
    'CalibrationProtocol'
]
