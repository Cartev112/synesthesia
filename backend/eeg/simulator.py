"""
EEG Simulator for testing and development without physical hardware.

Generates realistic multi-channel EEG data with:
- Controllable brain states (focus, relax, neutral)
- Realistic frequency band characteristics
- Spatial correlation between electrodes
- Artifact injection (blinks, muscle, movement)
"""

import time
from typing import Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

BrainState = Literal["neutral", "focus", "relax"]


class EEGSimulator:
    """
    Simulates realistic EEG data with controllable parameters.
    
    Generates signals based on:
    - Frequency band characteristics (delta, theta, alpha, beta, gamma)
    - Mental state modulation
    - Realistic spatial correlations
    - Artifact injection
    """
    
    def __init__(
        self,
        n_channels: int = 8,
        sampling_rate: int = 256,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize EEG simulator.
        
        Args:
            n_channels: Number of EEG channels (default: 8)
            sampling_rate: Sampling rate in Hz (default: 256)
            seed: Random seed for reproducibility
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'][:n_channels]
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Current state
        self.mental_state: BrainState = "neutral"
        self.state_intensity: float = 1.0
        self.hemispheric_asymmetry: float = 0.0  # -1 (left) to 1 (right)
        
        # Transition parameters
        self.target_state: BrainState = "neutral"
        self.transition_start_time: Optional[float] = None
        self.transition_duration: float = 2.0  # seconds
        
        # Auto-variation for demo mode
        self.auto_vary_states: bool = True
        self.last_state_change: float = time.time()
        self.state_change_interval: float = 10.0  # Change state every 10 seconds
        
        # Phase accumulator for oscillators
        self.phase: NDArray[np.float64] = np.zeros(n_channels)
        self.time_step = 1.0 / sampling_rate
        
        # Band power parameters (baseline)
        self.band_params = {
            'delta': {'freq_range': (0.5, 4), 'power': 1.0},
            'theta': {'freq_range': (4, 8), 'power': 0.8},
            'alpha': {'freq_range': (8, 13), 'power': 1.2},
            'beta': {'freq_range': (13, 30), 'power': 0.6},
            'gamma': {'freq_range': (30, 50), 'power': 0.3}
        }
        
        logger.info(
            "eeg_simulator_initialized",
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            channels=self.channels
        )
    
    def set_mental_state(
        self,
        state: BrainState,
        intensity: float = 1.0,
        transition_time: float = 2.0
    ) -> None:
        """
        Set target mental state with smooth transition.
        
        Args:
            state: Target mental state ('neutral', 'focus', 'relax')
            intensity: State intensity (0.0 to 1.0)
            transition_time: Transition duration in seconds
        """
        self.target_state = state
        self.state_intensity = np.clip(intensity, 0.0, 1.0)
        self.transition_start_time = time.time()
        self.transition_duration = transition_time
        
        logger.debug(
            "mental_state_transition_started",
            current_state=self.mental_state,
            target_state=state,
            intensity=intensity,
            transition_time=transition_time
        )
    
    def set_hemispheric_asymmetry(self, asymmetry: float) -> None:
        """
        Set hemispheric asymmetry.
        
        Args:
            asymmetry: -1.0 (left dominant) to 1.0 (right dominant)
        """
        self.hemispheric_asymmetry = np.clip(asymmetry, -1.0, 1.0)
    
    def generate_sample(self) -> NDArray[np.float64]:
        """
        Generate one sample across all channels.
        
        Returns:
            Array of shape (n_channels,) in microvolts
        """
        # Update state transition
        self._update_state_transition()
        
        # Get current band powers based on mental state
        band_powers = self._get_current_band_powers()
        
        # Generate signal for each frequency band
        sample = np.zeros(self.n_channels)
        
        for band_name, params in self.band_params.items():
            freq_low, freq_high = params['freq_range']
            power = band_powers[band_name]
            
            # Generate oscillation for this band
            freq = self.rng.uniform(freq_low, freq_high)
            amplitude = np.sqrt(power) * 10.0  # Scale to microvolts
            
            # Different amplitude per channel (spatial variation)
            channel_amplitudes = self._get_channel_amplitudes(band_name)
            
            # Generate sinusoidal component
            oscillation = amplitude * channel_amplitudes * np.sin(2 * np.pi * freq * self.phase)
            sample += oscillation
        
        # Add pink noise (1/f characteristic)
        noise = self._generate_pink_noise()
        sample += noise * 5.0  # Scale noise
        
        # Apply hemispheric asymmetry
        sample = self._apply_hemispheric_asymmetry(sample)
        
        # Update phase
        self.phase += self.time_step
        
        return sample
    
    def generate_chunk(self, duration: float) -> NDArray[np.float64]:
        """
        Generate a chunk of EEG data.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Array of shape (n_samples, n_channels)
        """
        n_samples = int(duration * self.sampling_rate)
        data = np.zeros((n_samples, self.n_channels))
        
        for i in range(n_samples):
            data[i, :] = self.generate_sample()
        
        return data
    
    def add_artifact(
        self,
        artifact_type: Literal["blink", "muscle", "movement"],
        intensity: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Generate an artifact and return it.
        
        Args:
            artifact_type: Type of artifact
            intensity: Artifact intensity (0.0 to 1.0)
            
        Returns:
            Artifact signal of shape (n_channels,)
        """
        artifact = np.zeros(self.n_channels)
        
        if artifact_type == "blink":
            # Eye blinks: high amplitude in frontal channels
            artifact[:2] = self.rng.normal(0, 100 * intensity, 2)  # Fp1, Fp2
            
        elif artifact_type == "muscle":
            # Muscle artifacts: high frequency, temporal regions
            artifact[4:6] = self.rng.normal(0, 50 * intensity, 2)  # P7, P8
            
        elif artifact_type == "movement":
            # Movement: affects all channels
            artifact = self.rng.normal(0, 30 * intensity, self.n_channels)
        
        return artifact
    
    def _update_state_transition(self) -> None:
        """Update current state based on transition progress."""
        # Auto-vary states for demo mode
        if self.auto_vary_states:
            current_time = time.time()
            if current_time - self.last_state_change >= self.state_change_interval:
                # Randomly pick a new state
                states: list[BrainState] = ["neutral", "focus", "relax"]
                new_state = self.rng.choice(states)
                intensity = self.rng.uniform(0.6, 1.0)
                self.set_mental_state(new_state, intensity, transition_time=3.0)
                self.last_state_change = current_time
                logger.info(
                    "auto_state_change",
                    new_state=new_state,
                    intensity=intensity
                )
        
        if self.transition_start_time is None:
            return
        
        elapsed = time.time() - self.transition_start_time
        
        # Handle instant transition
        if self.transition_duration <= 0:
            self.mental_state = self.target_state
            self.transition_start_time = None
            return
        
        progress = min(elapsed / self.transition_duration, 1.0)
        
        # Smooth transition (ease-out cubic)
        eased_progress = 1 - (1 - progress) ** 3
        
        # Interpolate state
        if progress >= 1.0:
            self.mental_state = self.target_state
            self.transition_start_time = None
        else:
            # For now, snap to target state at 50% progress
            if progress >= 0.5 and self.mental_state != self.target_state:
                self.mental_state = self.target_state
    
    def _get_current_band_powers(self) -> Dict[str, float]:
        """
        Get current band powers based on mental state.
        
        Returns:
            Dictionary of band powers
        """
        powers = {
            'delta': 1.0,
            'theta': 0.8,
            'alpha': 1.2,
            'beta': 0.6,
            'gamma': 0.3
        }
        
        intensity = self.state_intensity
        
        if self.mental_state == "focus":
            # Focus: increased beta and gamma, decreased alpha
            powers['beta'] = 0.6 + (0.8 * intensity)
            powers['gamma'] = 0.3 + (0.5 * intensity)
            powers['alpha'] = 1.2 - (0.6 * intensity)
            powers['theta'] = 0.8 - (0.3 * intensity)
            
        elif self.mental_state == "relax":
            # Relax: increased alpha and theta, decreased beta
            powers['alpha'] = 1.2 + (0.8 * intensity)
            powers['theta'] = 0.8 + (0.6 * intensity)
            powers['beta'] = 0.6 - (0.3 * intensity)
            powers['gamma'] = 0.3 - (0.2 * intensity)
        
        return powers
    
    def _get_channel_amplitudes(self, band_name: str) -> NDArray[np.float64]:
        """
        Get amplitude scaling for each channel based on band.
        
        Different frequency bands have different spatial distributions.
        """
        amplitudes = np.ones(self.n_channels)
        
        if band_name == 'alpha':
            # Alpha strongest in occipital (back of head)
            amplitudes[6:8] *= 1.5  # O1, O2
            
        elif band_name == 'beta':
            # Beta strongest in frontal/central
            amplitudes[0:4] *= 1.3  # Fp1, Fp2, C3, C4
            
        elif band_name == 'theta':
            # Theta more frontal
            amplitudes[0:2] *= 1.2  # Fp1, Fp2
        
        # Add small random variation
        amplitudes *= self.rng.uniform(0.9, 1.1, self.n_channels)
        
        return amplitudes
    
    def _generate_pink_noise(self) -> NDArray[np.float64]:
        """
        Generate pink noise (1/f characteristic).
        
        Returns:
            Pink noise array of shape (n_channels,)
        """
        # Simplified pink noise using multiple white noise sources
        white = self.rng.randn(self.n_channels)
        pink = white * 0.5  # Simplified approximation
        return pink
    
    def _apply_hemispheric_asymmetry(
        self,
        sample: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Apply hemispheric asymmetry to sample.
        
        Args:
            sample: EEG sample of shape (n_channels,)
            
        Returns:
            Modified sample with hemispheric bias
        """
        if abs(self.hemispheric_asymmetry) < 0.01:
            return sample
        
        # Left hemisphere channels: odd indices (Fp1, C3, P7, O1)
        # Right hemisphere channels: even indices (Fp2, C4, P8, O2)
        
        modified = sample.copy()
        
        # Apply asymmetry (positive = right dominant)
        left_indices = [0, 2, 4, 6]
        right_indices = [1, 3, 5, 7]
        
        if self.hemispheric_asymmetry > 0:  # Right dominant
            modified[right_indices[:len(right_indices)]] *= (1 + self.hemispheric_asymmetry * 0.3)
            modified[left_indices[:len(left_indices)]] *= (1 - self.hemispheric_asymmetry * 0.2)
        else:  # Left dominant
            modified[left_indices[:len(left_indices)]] *= (1 + abs(self.hemispheric_asymmetry) * 0.3)
            modified[right_indices[:len(right_indices)]] *= (1 - abs(self.hemispheric_asymmetry) * 0.2)
        
        return modified
    
    def get_info(self) -> Dict:
        """
        Get simulator information.
        
        Returns:
            Dictionary with simulator info
        """
        return {
            'device_type': 'simulator',
            'n_channels': self.n_channels,
            'sampling_rate': self.sampling_rate,
            'channels': self.channels,
            'current_state': self.mental_state,
            'state_intensity': self.state_intensity,
            'hemispheric_asymmetry': self.hemispheric_asymmetry
        }


class StreamingEEGSimulator(EEGSimulator):
    """
    EEG Simulator with async streaming capabilities.
    
    Simulates real-time data streaming similar to actual EEG hardware.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.start_time: Optional[float] = None
        self.sample_count = 0
    
    def start_stream(self) -> None:
        """Start streaming EEG data."""
        self.is_streaming = True
        self.start_time = time.time()
        self.sample_count = 0
        logger.info("eeg_streaming_started")
    
    def stop_stream(self) -> None:
        """Stop streaming EEG data."""
        self.is_streaming = False
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(
            "eeg_streaming_stopped",
            duration_seconds=duration,
            total_samples=self.sample_count
        )
    
    def read_samples(self, n_samples: int = 1) -> Optional[NDArray[np.float64]]:
        """
        Read samples from stream.
        
        Args:
            n_samples: Number of samples to read
            
        Returns:
            Array of shape (n_samples, n_channels) or None if not streaming
        """
        if not self.is_streaming:
            return None
        
        data = np.zeros((n_samples, self.n_channels))
        for i in range(n_samples):
            data[i, :] = self.generate_sample()
        
        self.sample_count += n_samples
        return data



