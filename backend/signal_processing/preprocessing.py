"""
Signal preprocessing: filtering, re-referencing, and cleaning.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class SignalPreprocessor:
    """
    Preprocessing pipeline for raw EEG signals.
    
    Applies:
    - Bandpass filtering (0.5-50 Hz)
    - Notch filtering (60 Hz power line noise)
    - Common average re-referencing
    """
    
    def __init__(
        self,
        sampling_rate: int,
        bandpass_low: float = 0.5,
        bandpass_high: float = 50.0,
        notch_freq: float = 60.0,
        notch_quality: float = 30.0
    ) -> None:
        """
        Initialize signal preprocessor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            bandpass_low: Low cutoff frequency for bandpass filter
            bandpass_high: High cutoff frequency for bandpass filter
            notch_freq: Notch filter frequency (power line noise)
            notch_quality: Quality factor for notch filter
        """
        self.sampling_rate = sampling_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        
        # Design bandpass filter (Butterworth)
        nyquist = sampling_rate / 2.0
        low = bandpass_low / nyquist
        high = bandpass_high / nyquist
        
        self.bp_sos = signal.butter(
            4,  # Filter order
            [low, high],
            btype='band',
            output='sos'
        )
        
        # Design notch filter
        self.notch_sos = signal.iirnotch(
            notch_freq,
            notch_quality,
            sampling_rate
        )
        # Convert to second-order sections for stability
        self.notch_sos = signal.tf2sos(*self.notch_sos)
        
        # Filter states for continuous filtering
        self.bp_zi: NDArray[np.float64] | None = None
        self.notch_zi: NDArray[np.float64] | None = None
        
        logger.info(
            "preprocessor_initialized",
            sampling_rate=sampling_rate,
            bandpass=f"{bandpass_low}-{bandpass_high} Hz",
            notch=f"{notch_freq} Hz"
        )
    
    def process_chunk(
        self,
        data: NDArray[np.float64],
        reset_state: bool = False
    ) -> NDArray[np.float64]:
        """
        Process a chunk of EEG data.
        
        Args:
            data: Input data of shape (n_samples, n_channels)
            reset_state: If True, reset filter states
            
        Returns:
            Filtered data of shape (n_samples, n_channels)
        """
        if data.size == 0:
            return data
        
        if reset_state:
            self.reset_filter_state()
        
        # Apply filters to each channel
        filtered = np.zeros_like(data)
        n_channels = data.shape[1]
        
        # Initialize filter states if needed
        if self.bp_zi is None:
            self.bp_zi = np.zeros((self.bp_sos.shape[0], 2, n_channels))
        if self.notch_zi is None:
            self.notch_zi = np.zeros((self.notch_sos.shape[0], 2, n_channels))
        
        for ch in range(n_channels):
            # Bandpass filter
            filtered[:, ch], self.bp_zi[:, :, ch] = signal.sosfilt(
                self.bp_sos,
                data[:, ch],
                zi=self.bp_zi[:, :, ch]
            )
            
            # Notch filter
            filtered[:, ch], self.notch_zi[:, :, ch] = signal.sosfilt(
                self.notch_sos,
                filtered[:, ch],
                zi=self.notch_zi[:, :, ch]
            )
        
        # Common average re-referencing
        filtered = self.rereference(filtered)
        
        return filtered
    
    def rereference(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply common average re-referencing.
        
        Args:
            data: Input data of shape (n_samples, n_channels)
            
        Returns:
            Re-referenced data
        """
        avg = np.mean(data, axis=1, keepdims=True)
        return data - avg
    
    def reset_filter_state(self) -> None:
        """Reset filter states (for new recording sessions)."""
        self.bp_zi = None
        self.notch_zi = None
        logger.debug("filter_state_reset")





