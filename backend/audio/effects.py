"""
Audio effects for post-processing.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy import signal

from backend.core.logging import get_logger

logger = get_logger(__name__)


class EffectBase(ABC):
    """Base class for audio effects."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize effect.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.enabled = True
    
    @abstractmethod
    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process audio samples.
        
        Args:
            samples: Input audio samples
            
        Returns:
            Processed audio samples
        """
        pass
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the effect."""
        self.enabled = enabled


class ReverbEffect(EffectBase):
    """
    Simple reverb effect using comb filters.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3
    ):
        """
        Initialize reverb effect.
        
        Args:
            sample_rate: Audio sample rate
            room_size: Room size (0-1)
            damping: High frequency damping (0-1)
            wet_level: Wet/dry mix (0-1)
        """
        super().__init__(sample_rate)
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
        
        # Comb filter delays (in samples)
        self.delays = [
            int(0.0297 * sample_rate * (1 + room_size)),
            int(0.0371 * sample_rate * (1 + room_size)),
            int(0.0411 * sample_rate * (1 + room_size)),
            int(0.0437 * sample_rate * (1 + room_size))
        ]
        
        # Feedback coefficients
        self.feedback = 0.5 + 0.3 * room_size
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply reverb effect."""
        if not self.enabled:
            return samples
        
        # Simple reverb using delayed copies
        reverb = np.zeros_like(samples)
        
        for delay in self.delays:
            if delay < len(samples):
                # Create delayed version
                delayed = np.zeros_like(samples)
                delayed[delay:] = samples[:-delay] * self.feedback
                reverb += delayed
        
        # Normalize
        if len(self.delays) > 0:
            reverb /= len(self.delays)
        
        # Mix wet and dry
        output = (1 - self.wet_level) * samples + self.wet_level * reverb
        
        return output


class DelayEffect(EffectBase):
    """
    Delay/echo effect.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        delay_time: float = 0.25,
        feedback: float = 0.4,
        wet_level: float = 0.3
    ):
        """
        Initialize delay effect.
        
        Args:
            sample_rate: Audio sample rate
            delay_time: Delay time in seconds
            feedback: Feedback amount (0-1)
            wet_level: Wet/dry mix (0-1)
        """
        super().__init__(sample_rate)
        self.delay_time = delay_time
        self.feedback = feedback
        self.wet_level = wet_level
        self.delay_samples = int(delay_time * sample_rate)
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply delay effect."""
        if not self.enabled:
            return samples
        
        # Create delayed signal
        delayed = np.zeros_like(samples)
        
        if self.delay_samples < len(samples):
            delayed[self.delay_samples:] = samples[:-self.delay_samples]
            
            # Add feedback
            for i in range(self.delay_samples, len(samples)):
                delayed[i] += delayed[i - self.delay_samples] * self.feedback
        
        # Mix wet and dry
        output = (1 - self.wet_level) * samples + self.wet_level * delayed
        
        return output


class FilterEffect(EffectBase):
    """
    Frequency filter effect (low-pass, high-pass, band-pass).
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        filter_type: str = 'lowpass',
        cutoff_freq: float = 1000.0,
        resonance: float = 0.7
    ):
        """
        Initialize filter effect.
        
        Args:
            sample_rate: Audio sample rate
            filter_type: 'lowpass', 'highpass', or 'bandpass'
            cutoff_freq: Cutoff frequency in Hz
            resonance: Filter resonance/Q (0-1)
        """
        super().__init__(sample_rate)
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
        self.resonance = resonance
        
        # Design filter
        self._design_filter()
    
    def _design_filter(self):
        """Design the filter coefficients."""
        # Normalize cutoff frequency
        nyquist = self.sample_rate / 2
        normalized_cutoff = self.cutoff_freq / nyquist
        
        # Clamp to valid range
        normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
        
        # Q factor from resonance
        Q = 0.5 + self.resonance * 9.5  # Map 0-1 to 0.5-10
        
        # Design filter
        if self.filter_type == 'lowpass':
            self.b, self.a = signal.butter(2, normalized_cutoff, btype='low')
        elif self.filter_type == 'highpass':
            self.b, self.a = signal.butter(2, normalized_cutoff, btype='high')
        elif self.filter_type == 'bandpass':
            # Bandpass around cutoff
            low = normalized_cutoff * 0.8
            high = normalized_cutoff * 1.2
            self.b, self.a = signal.butter(2, [low, high], btype='band')
        else:
            # Default to lowpass
            self.b, self.a = signal.butter(2, normalized_cutoff, btype='low')
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply filter effect."""
        if not self.enabled:
            return samples
        
        # Apply filter
        filtered = signal.filtfilt(self.b, self.a, samples)
        
        return filtered
    
    def set_cutoff(self, cutoff_freq: float):
        """Update cutoff frequency."""
        self.cutoff_freq = cutoff_freq
        self._design_filter()


class CompressorEffect(EffectBase):
    """
    Dynamic range compressor.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        threshold: float = 0.5,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1
    ):
        """
        Initialize compressor effect.
        
        Args:
            sample_rate: Audio sample rate
            threshold: Compression threshold (0-1)
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds
        """
        super().__init__(sample_rate)
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        
        # Calculate attack/release coefficients
        self.attack_coeff = np.exp(-1.0 / (attack * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release * sample_rate))
        
        self.envelope = 0.0
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply compression."""
        if not self.enabled:
            return samples
        
        output = np.zeros_like(samples)
        
        for i, sample in enumerate(samples):
            # Calculate envelope
            abs_sample = abs(sample)
            
            if abs_sample > self.envelope:
                # Attack
                self.envelope = self.attack_coeff * self.envelope + \
                               (1 - self.attack_coeff) * abs_sample
            else:
                # Release
                self.envelope = self.release_coeff * self.envelope + \
                               (1 - self.release_coeff) * abs_sample
            
            # Calculate gain reduction
            if self.envelope > self.threshold:
                # Compress
                excess = self.envelope - self.threshold
                compressed = self.threshold + excess / self.ratio
                gain = compressed / (self.envelope + 1e-6)
            else:
                gain = 1.0
            
            output[i] = sample * gain
        
        return output


# Effect registry
EFFECT_REGISTRY = {
    'reverb': ReverbEffect,
    'delay': DelayEffect,
    'filter': FilterEffect,
    'compressor': CompressorEffect
}


def get_effect(effect_type: str, sample_rate: int = 44100, **kwargs) -> EffectBase:
    """
    Get an effect instance by type.
    
    Args:
        effect_type: Effect type name
        sample_rate: Audio sample rate
        **kwargs: Additional effect parameters
        
    Returns:
        Effect instance
    """
    if effect_type not in EFFECT_REGISTRY:
        logger.warning(
            "unknown_effect_type",
            effect_type=effect_type,
            available=list(EFFECT_REGISTRY.keys())
        )
        return None
    
    return EFFECT_REGISTRY[effect_type](sample_rate=sample_rate, **kwargs)
