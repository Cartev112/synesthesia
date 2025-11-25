"""
Abstract interface for EEG devices.

Provides a common interface for all EEG data sources (simulator, OpenBCI, Muse, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from backend.core.logging import get_logger

logger = get_logger(__name__)


class EEGDeviceInterface(ABC):
    """
    Abstract base class for EEG devices.
    
    All EEG data sources (simulator, real hardware) must implement this interface.
    """
    
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize the EEG device."""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the EEG device.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the EEG device."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if device is connected.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def start_stream(self) -> None:
        """Start streaming data from the device."""
        pass
    
    @abstractmethod
    def stop_stream(self) -> None:
        """Stop streaming data from the device."""
        pass
    
    @abstractmethod
    def read_samples(self, n_samples: int = 1) -> Optional[NDArray[np.float64]]:
        """
        Read samples from the device.
        
        Args:
            n_samples: Number of samples to read
            
        Returns:
            Array of shape (n_samples, n_channels) or None if no data available
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """
        Get device information.
        
        Returns:
            Dictionary with device info (name, channels, sampling rate, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Get device sampling rate in Hz."""
        pass
    
    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Get number of channels."""
        pass
    
    @property
    @abstractmethod
    def channel_names(self) -> list[str]:
        """Get list of channel names."""
        pass





