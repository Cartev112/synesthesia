"""
Signal processing pipeline for EEG data.
"""

from backend.signal_processing.buffer import CircularBuffer
from backend.signal_processing.feature_extraction import FeatureExtractor
from backend.signal_processing.preprocessing import SignalPreprocessor

__all__ = [
    "CircularBuffer",
    "SignalPreprocessor",
    "FeatureExtractor",
]





