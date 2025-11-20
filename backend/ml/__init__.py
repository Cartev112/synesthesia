"""
Machine learning components for BCI system.

Includes:
- Artifact classification
- Mental state classification
- State tracking with temporal smoothing
"""

from backend.ml.artifact_classifier import ArtifactClassifier, ArtifactCNN
from backend.ml.state_classifier import MentalStateClassifier, MentalStateTracker

__all__ = [
    'ArtifactClassifier',
    'ArtifactCNN',
    'MentalStateClassifier',
    'MentalStateTracker',
]
