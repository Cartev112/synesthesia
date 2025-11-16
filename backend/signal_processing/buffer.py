"""
Circular buffer for efficient real-time EEG data streaming.
"""

import numpy as np
from numpy.typing import NDArray

from backend.core.logging import get_logger

logger = get_logger(__name__)


class CircularBuffer:
    """
    Efficient circular buffer for streaming EEG data.
    
    Uses a fixed-size numpy array with write pointer tracking.
    Enables efficient append and retrieval operations without reallocation.
    """
    
    def __init__(
        self,
        n_channels: int,
        buffer_duration: float,
        sampling_rate: int
    ) -> None:
        """
        Initialize circular buffer.
        
        Args:
            n_channels: Number of EEG channels
            buffer_duration: Buffer size in seconds
            sampling_rate: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.buffer_duration = buffer_duration
        self.n_samples = int(buffer_duration * sampling_rate)
        
        # Pre-allocate buffer
        self.buffer = np.zeros((self.n_samples, n_channels), dtype=np.float64)
        self.write_idx = 0
        self.is_full = False
        
        logger.debug(
            "circular_buffer_initialized",
            n_channels=n_channels,
            buffer_duration=buffer_duration,
            buffer_samples=self.n_samples
        )
    
    def append(self, new_data: NDArray[np.float64]) -> None:
        """
        Append new samples to buffer.
        
        Args:
            new_data: Array of shape (n_samples, n_channels)
        """
        if new_data.shape[1] != self.n_channels:
            raise ValueError(
                f"Data has {new_data.shape[1]} channels, "
                f"expected {self.n_channels}"
            )
        
        n_new = new_data.shape[0]
        
        # Handle wraparound
        if self.write_idx + n_new <= self.n_samples:
            # No wraparound needed
            self.buffer[self.write_idx:self.write_idx + n_new] = new_data
        else:
            # Need to wrap around
            part1_size = self.n_samples - self.write_idx
            part2_size = n_new - part1_size
            
            self.buffer[self.write_idx:] = new_data[:part1_size]
            self.buffer[:part2_size] = new_data[part1_size:]
            
            self.is_full = True
        
        self.write_idx = (self.write_idx + n_new) % self.n_samples
    
    def get_latest(
        self,
        duration: float
    ) -> NDArray[np.float64]:
        """
        Get most recent N seconds of data.
        
        Args:
            duration: Duration in seconds to retrieve
            
        Returns:
            Array of shape (n_samples, n_channels)
        """
        n_samples = int(duration * self.sampling_rate)
        
        if n_samples > self.n_samples:
            raise ValueError(
                f"Requested {duration}s ({n_samples} samples) "
                f"exceeds buffer size ({self.n_samples} samples)"
            )
        
        if not self.is_full and n_samples > self.write_idx:
            # Not enough data yet
            return self.buffer[:self.write_idx]
        
        # Calculate start index
        start_idx = (self.write_idx - n_samples) % self.n_samples
        
        if start_idx < self.write_idx:
            # No wraparound
            return self.buffer[start_idx:self.write_idx].copy()
        else:
            # Wraparound case
            part1 = self.buffer[start_idx:]
            part2 = self.buffer[:self.write_idx]
            return np.concatenate([part1, part2], axis=0)
    
    def get_all(self) -> NDArray[np.float64]:
        """
        Get all valid data in buffer.
        
        Returns:
            Array of shape (n_valid_samples, n_channels)
        """
        if not self.is_full:
            return self.buffer[:self.write_idx].copy()
        
        # Return in chronological order
        return np.concatenate([
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx]
        ], axis=0)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_idx = 0
        self.is_full = False
        logger.debug("circular_buffer_cleared")
    
    @property
    def current_samples(self) -> int:
        """Get number of valid samples currently in buffer."""
        return self.n_samples if self.is_full else self.write_idx
    
    @property
    def current_duration(self) -> float:
        """Get duration of valid data currently in buffer."""
        return self.current_samples / self.sampling_rate



