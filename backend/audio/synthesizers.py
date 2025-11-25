"""
Audio synthesizers for different sound types.

Provides multiple synthesis options per musical layer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ADSREnvelope:
    """ADSR envelope parameters."""
    attack: float = 0.005   # seconds - fast but not instant to avoid clicks
    decay: float = 0.05     # seconds
    sustain: float = 0.8    # level (0-1)
    release: float = 0.05   # seconds - short release for buffer continuity


class SynthesizerBase(ABC):
    """
    Base class for all synthesizers.
    
    Each synthesizer generates audio samples for a given note.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize synthesizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.envelope = ADSREnvelope()
        
    @abstractmethod
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """
        Generate audio samples for a note.
        
        Args:
            frequency: Note frequency in Hz
            duration: Note duration in seconds
            velocity: Note velocity (0-1)
            
        Returns:
            Audio samples as numpy array
        """
        pass
    
    def apply_envelope(
        self,
        samples: np.ndarray,
        duration: float
    ) -> np.ndarray:
        """
        Apply ADSR envelope to samples.
        
        Args:
            samples: Audio samples
            duration: Total duration in seconds
            
        Returns:
            Samples with envelope applied
        """
        n_samples = len(samples)
        envelope = np.ones(n_samples)
        
        # Calculate sample counts for each phase
        attack_samples = int(self.envelope.attack * self.sample_rate)
        decay_samples = int(self.envelope.decay * self.sample_rate)
        release_samples = int(self.envelope.release * self.sample_rate)
        
        # Ensure we don't exceed array bounds
        attack_samples = min(attack_samples, n_samples)
        decay_samples = min(decay_samples, n_samples - attack_samples)
        release_samples = min(release_samples, n_samples)
        
        sustain_samples = n_samples - attack_samples - decay_samples - release_samples
        sustain_samples = max(0, sustain_samples)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            start_idx = attack_samples
            end_idx = start_idx + decay_samples
            envelope[start_idx:end_idx] = np.linspace(
                1, self.envelope.sustain, decay_samples
            )
        
        # Sustain phase
        if sustain_samples > 0:
            start_idx = attack_samples + decay_samples
            end_idx = start_idx + sustain_samples
            envelope[start_idx:end_idx] = self.envelope.sustain
        
        # Release phase
        if release_samples > 0:
            start_idx = n_samples - release_samples
            envelope[start_idx:] = np.linspace(
                self.envelope.sustain, 0, release_samples
            )
        
        return samples * envelope
    
    def midi_to_frequency(self, midi_note: int) -> float:
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


class SineWaveSynth(SynthesizerBase):
    """
    Pure sine wave synthesizer.
    
    Clean, simple tone - good for melody and harmony.
    """
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """Generate sine wave."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate sine wave
        samples = np.sin(2 * np.pi * frequency * t) * velocity
        
        # Apply envelope
        samples = self.apply_envelope(samples, duration)
        
        return samples


class SquareWaveSynth(SynthesizerBase):
    """
    Square wave synthesizer.
    
    Bright, hollow tone - good for bass and leads.
    """
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """Generate square wave."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate square wave
        samples = np.sign(np.sin(2 * np.pi * frequency * t)) * velocity
        
        # Apply envelope
        samples = self.apply_envelope(samples, duration)
        
        return samples


class SawtoothWaveSynth(SynthesizerBase):
    """
    Sawtooth wave synthesizer.
    
    Bright, buzzy tone - good for bass and pads.
    """
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """Generate sawtooth wave."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate sawtooth wave
        samples = 2 * (t * frequency - np.floor(t * frequency + 0.5)) * velocity
        
        # Apply envelope
        samples = self.apply_envelope(samples, duration)
        
        return samples


class TriangleWaveSynth(SynthesizerBase):
    """
    Triangle wave synthesizer.
    
    Mellow tone - good for pads and texture.
    """
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """Generate triangle wave."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate triangle wave
        samples = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
        samples *= velocity
        
        # Apply envelope
        samples = self.apply_envelope(samples, duration)
        
        return samples


class FMSynth(SynthesizerBase):
    """
    Frequency Modulation synthesizer.
    
    Complex, evolving tones - good for bells, pads, and texture.
    """
    
    def __init__(self, sample_rate: int = 44100, mod_ratio: float = 2.0, mod_index: float = 5.0):
        """
        Initialize FM synthesizer.
        
        Args:
            sample_rate: Audio sample rate
            mod_ratio: Modulator frequency ratio
            mod_index: Modulation index (depth)
        """
        super().__init__(sample_rate)
        self.mod_ratio = mod_ratio
        self.mod_index = mod_index
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """Generate FM synthesized tone."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Modulator frequency
        mod_freq = frequency * self.mod_ratio
        
        # Generate modulator
        modulator = np.sin(2 * np.pi * mod_freq * t)
        
        # Generate carrier with FM
        samples = np.sin(2 * np.pi * frequency * t + self.mod_index * modulator)
        samples *= velocity
        
        # Apply envelope
        samples = self.apply_envelope(samples, duration)
        
        return samples


class SubtractiveSynth(SynthesizerBase):
    """
    Subtractive synthesis with filter.
    
    Rich, analog-style tones - good for bass and leads.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        cutoff_ratio: float = 2.0,
        resonance: float = 0.7
    ):
        """
        Initialize subtractive synthesizer.
        
        Args:
            sample_rate: Audio sample rate
            cutoff_ratio: Filter cutoff as ratio of fundamental
            resonance: Filter resonance (0-1)
        """
        super().__init__(sample_rate)
        self.cutoff_ratio = cutoff_ratio
        self.resonance = resonance
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """Generate subtractive synthesized tone."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate sawtooth as source
        samples = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        
        # Simple low-pass filter (moving average)
        cutoff_freq = frequency * self.cutoff_ratio
        window_size = max(1, int(self.sample_rate / cutoff_freq))
        
        if window_size > 1:
            # Apply moving average filter
            kernel = np.ones(window_size) / window_size
            samples = np.convolve(samples, kernel, mode='same')
        
        samples *= velocity
        
        # Apply envelope
        samples = self.apply_envelope(samples, duration)
        
        return samples


# Synthesizer registry for easy selection
SYNTHESIZER_REGISTRY = {
    'sine': SineWaveSynth,
    'square': SquareWaveSynth,
    'sawtooth': SawtoothWaveSynth,
    'triangle': TriangleWaveSynth,
    'fm': FMSynth,
    'subtractive': SubtractiveSynth
}


def get_synthesizer(synth_type: str, sample_rate: int = 44100) -> SynthesizerBase:
    """
    Get a synthesizer instance by type.
    
    Args:
        synth_type: Synthesizer type name
        sample_rate: Audio sample rate
        
    Returns:
        Synthesizer instance
    """
    if synth_type not in SYNTHESIZER_REGISTRY:
        logger.warning(
            "unknown_synthesizer_type",
            synth_type=synth_type,
            available=list(SYNTHESIZER_REGISTRY.keys())
        )
        synth_type = 'sine'  # Default fallback
    
    return SYNTHESIZER_REGISTRY[synth_type](sample_rate=sample_rate)
