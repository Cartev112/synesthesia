"""
EEG device interfaces and simulator.
"""

from backend.eeg.device_interface import EEGDeviceInterface
from backend.eeg.simulator import EEGSimulator, StreamingEEGSimulator

__all__ = [
    "EEGDeviceInterface",
    "EEGSimulator",
    "StreamingEEGSimulator",
]



