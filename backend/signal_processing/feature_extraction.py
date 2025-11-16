"""
Feature extraction from preprocessed EEG signals.

Extracts:
- Band powers (delta, theta, alpha, beta, gamma)
- Derived metrics (focus, relax)
- Hemispheric asymmetry
"""

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from backend.core.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Extract features from EEG data for brain state classification.
    
    Features:
    - Band powers (delta, theta, alpha, beta, gamma)
    - Focus metric (beta / (alpha + theta))
    - Relax metric (alpha / (beta + gamma))
    - Hemispheric asymmetry
    """
    
    def __init__(
        self,
        sampling_rate: int,
        window_size: float = 1.0,
        overlap: float = 0.5
    ) -> None:
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            window_size: Window size in seconds for PSD computation
            overlap: Overlap fraction (0-1)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
        self.window_samples = int(window_size * sampling_rate)
        self.hop_samples = int((1 - overlap) * self.window_samples)
        
        # Frequency bands (Hz)
        self.bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 50.0)
        }
        
        logger.info(
            "feature_extractor_initialized",
            sampling_rate=sampling_rate,
            window_size=window_size,
            overlap=overlap
        )
    
    def extract_features(self, data: NDArray[np.float64]) -> Dict[str, float]:
        """
        Extract features from EEG window.
        
        Args:
            data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            Dictionary of features
        """
        if data.shape[0] < self.window_samples:
            logger.warning(
                "insufficient_data_for_features",
                provided_samples=data.shape[0],
                required_samples=self.window_samples
            )
            return self._get_default_features()
        
        features = {}
        
        # Compute power spectral density
        freqs, psd = self._compute_psd(data)
        
        # Extract band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            power = self._band_power(psd, freqs, low_freq, high_freq)
            band_powers[band_name] = power
            features[f'{band_name}_power'] = float(power)
        
        # Compute derived metrics
        features['focus_metric'] = float(self._compute_focus(band_powers))
        features['relax_metric'] = float(self._compute_relax(band_powers))
        
        # Compute hemispheric asymmetry
        features['hemispheric_asymmetry'] = float(self._compute_asymmetry(data))
        
        return features
    
    def _compute_psd(
        self,
        data: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute power spectral density using Welch's method.
        
        Args:
            data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            freqs: Frequency bins
            psd: Power spectral density averaged across channels
        """
        # Average PSD across all channels
        psd_channels = []
        
        for ch in range(data.shape[1]):
            freqs, psd_ch = signal.welch(
                data[:, ch],
                fs=self.sampling_rate,
                nperseg=min(self.window_samples, data.shape[0]),
                noverlap=self.window_samples // 2
            )
            psd_channels.append(psd_ch)
        
        # Average across channels
        psd = np.mean(psd_channels, axis=0)
        
        return freqs, psd
    
    def _band_power(
        self,
        psd: NDArray[np.float64],
        freqs: NDArray[np.float64],
        low_freq: float,
        high_freq: float
    ) -> float:
        """
        Calculate power in a frequency band.
        
        Args:
            psd: Power spectral density
            freqs: Frequency bins
            low_freq: Lower frequency bound
            high_freq: Upper frequency bound
            
        Returns:
            Band power
        """
        # Find indices for frequency range
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        
        # Integrate power (trapezoidal rule)
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        
        return band_power
    
    def _compute_focus(self, band_powers: Dict[str, float]) -> float:
        """
        Compute focus metric: beta / (alpha + theta)
        
        Args:
            band_powers: Dictionary of band powers
            
        Returns:
            Focus metric (0-infinity, typically 0-2)
        """
        numerator = band_powers['beta']
        denominator = band_powers['alpha'] + band_powers['theta']
        
        if denominator < 1e-6:
            return 0.0
        
        return numerator / denominator
    
    def _compute_relax(self, band_powers: Dict[str, float]) -> float:
        """
        Compute relax metric: alpha / (beta + gamma)
        
        Args:
            band_powers: Dictionary of band powers
            
        Returns:
            Relax metric (0-infinity, typically 0-3)
        """
        numerator = band_powers['alpha']
        denominator = band_powers['beta'] + band_powers['gamma']
        
        if denominator < 1e-6:
            return 0.0
        
        return numerator / denominator
    
    def _compute_asymmetry(self, data: NDArray[np.float64]) -> float:
        """
        Compute hemispheric asymmetry from alpha power.
        
        Assumes channels are: Fp1, Fp2, C3, C4, P7, P8, O1, O2
        Left: 0, 2, 4, 6 (Fp1, C3, P7, O1)
        Right: 1, 3, 5, 7 (Fp2, C4, P8, O2)
        
        Args:
            data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            Asymmetry index (-1 to 1, negative = left dominant, positive = right dominant)
        """
        if data.shape[1] < 8:
            return 0.0
        
        left_indices = [0, 2, 4, 6]
        right_indices = [1, 3, 5, 7]
        
        # Get alpha band power for left and right hemispheres
        left_data = data[:, left_indices]
        right_data = data[:, right_indices]
        
        left_freqs, left_psd = self._compute_psd(left_data)
        right_freqs, right_psd = self._compute_psd(right_data)
        
        # Alpha power
        left_alpha = self._band_power(left_psd, left_freqs, 8.0, 13.0)
        right_alpha = self._band_power(right_psd, right_freqs, 8.0, 13.0)
        
        # Asymmetry index
        total = left_alpha + right_alpha
        if total < 1e-6:
            return 0.0
        
        asymmetry = (right_alpha - left_alpha) / total
        
        return asymmetry
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when extraction fails."""
        return {
            'delta_power': 0.0,
            'theta_power': 0.0,
            'alpha_power': 0.0,
            'beta_power': 0.0,
            'gamma_power': 0.0,
            'focus_metric': 0.0,
            'relax_metric': 0.0,
            'hemispheric_asymmetry': 0.0
        }



