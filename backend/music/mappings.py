"""
Brain state to musical parameter mappings.

Defines how brain states map to musical parameters across
different layers and instruments.
"""

from typing import Dict, Callable
import numpy as np

from backend.core.logging import get_logger

logger = get_logger(__name__)


class BrainMusicMapper:
    """
    Maps brain states to musical parameters.
    
    Provides configurable mappings for different musical layers
    and instruments.
    """
    
    def __init__(self):
        """Initialize brain-music mapper."""
        # Default mapping functions
        self.mappings: Dict[str, Callable] = {
            'tempo': self._map_tempo,
            'density': self._map_density,
            'pitch_center': self._map_pitch_center,
            'scale': self._map_scale,
            'reverb': self._map_reverb,
            'filter_cutoff': self._map_filter,
            'harmonic_complexity': self._map_harmony,
        }
        
        logger.info("brain_music_mapper_initialized")
    
    def map_all(self, brain_state: Dict[str, float]) -> Dict[str, any]:
        """
        Map brain state to all musical parameters.
        
        Args:
            brain_state: Dictionary with brain state values
            
        Returns:
            Dictionary of musical parameters
        """
        params = {}
        
        for param_name, mapping_func in self.mappings.items():
            try:
                params[param_name] = mapping_func(brain_state)
            except Exception as e:
                logger.error(
                    "mapping_error",
                    parameter=param_name,
                    error=str(e)
                )
                # Use safe default
                params[param_name] = self._get_default(param_name)
        
        return params
    
    def _map_tempo(self, brain_state: Dict[str, float]) -> float:
        """
        Map brain state to tempo (BPM).
        
        Focus increases tempo, relax decreases it.
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Tempo in BPM (60-180)
        """
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        
        base_tempo = 100.0
        tempo = base_tempo + (focus * 60) - (relax * 30)
        
        return float(np.clip(tempo, 60, 180))
    
    def _map_density(self, brain_state: Dict[str, float]) -> float:
        """
        Map brain state to note density.
        
        Focus increases density, relax decreases it.
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Density (0-1)
        """
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        
        density = 0.3 + (focus * 0.4) - (relax * 0.2)
        
        return float(np.clip(density, 0.1, 0.8))
    
    def _map_pitch_center(self, brain_state: Dict[str, float]) -> int:
        """
        Map brain state to pitch center (MIDI note).
        
        Focus raises pitch, relax lowers it.
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            MIDI note number (48-84, C3-C6)
        """
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        
        base_pitch = 60  # Middle C
        offset = int((focus - relax) * 12)  # Up to 1 octave
        
        pitch = base_pitch + offset
        
        return int(np.clip(pitch, 48, 84))
    
    def _map_scale(self, brain_state: Dict[str, float]) -> str:
        """
        Map brain state to musical scale.
        
        Focus → more complex scales
        Relax → simpler scales
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Scale name
        """
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        
        if focus > 0.7:
            # High focus: complex modes
            return np.random.choice(['dorian', 'lydian', 'mixolydian'])
        elif relax > 0.7:
            # High relax: simple pentatonic
            return 'major_pentatonic'
        elif relax > 0.5:
            # Moderate relax: minor
            return 'minor'
        else:
            # Default: major
            return 'major'
    
    def _map_reverb(self, brain_state: Dict[str, float]) -> float:
        """
        Map brain state to reverb amount.
        
        Relax increases reverb (more spacious).
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Reverb wet/dry mix (0-1)
        """
        relax = brain_state.get('relax', 0.5)
        theta_power = brain_state.get('theta_power', 0.5)
        
        # Relax and theta both increase reverb
        reverb = (relax * 0.5) + (theta_power * 0.3)
        
        return float(np.clip(reverb, 0.1, 0.8))
    
    def _map_filter(self, brain_state: Dict[str, float]) -> float:
        """
        Map brain state to filter cutoff frequency.
        
        Relax lowers cutoff (darker sound).
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Filter cutoff in Hz (200-5000)
        """
        relax = brain_state.get('relax', 0.5)
        focus = brain_state.get('focus', 0.5)
        
        # Focus increases brightness, relax decreases
        base_cutoff = 1000.0
        cutoff = base_cutoff + (focus * 2000) - (relax * 600)
        
        return float(np.clip(cutoff, 200, 5000))
    
    def _map_harmony(self, brain_state: Dict[str, float]) -> float:
        """
        Map brain state to harmonic complexity.
        
        Focus increases complexity.
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Complexity (0-1)
        """
        focus = brain_state.get('focus', 0.5)
        gamma_power = brain_state.get('gamma_power', 0.5)
        
        # Focus and gamma increase complexity
        complexity = (focus * 0.6) + (gamma_power * 0.4)
        
        return float(np.clip(complexity, 0.2, 1.0))
    
    def _get_default(self, param_name: str) -> any:
        """Get safe default value for parameter."""
        defaults = {
            'tempo': 100.0,
            'density': 0.3,
            'pitch_center': 60,
            'scale': 'major',
            'reverb': 0.3,
            'filter_cutoff': 1000.0,
            'harmonic_complexity': 0.5,
        }
        return defaults.get(param_name, 0.5)


class LayeredMusicMapper:
    """
    Maps brain states to parameters for multiple musical layers.
    
    Supports different mapping strategies for:
    - Bass/rhythm layer
    - Harmonic pad layer
    - Melodic lead layer
    - Textural elements layer
    """
    
    def __init__(self):
        """Initialize layered music mapper."""
        self.base_mapper = BrainMusicMapper()
        logger.info("layered_music_mapper_initialized")
    
    def map_bass_layer(self, brain_state: Dict[str, float]) -> Dict[str, any]:
        """
        Map brain state to bass/rhythm layer parameters.
        
        Bass responds to:
        - Focus → faster, more syncopated
        - Relax → slower, simpler (drone-like)
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Bass layer parameters
        """
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        
        return {
            'tempo': self.base_mapper._map_tempo(brain_state),
            'density': 0.1 + (focus * 0.2),  # Sparser than melody
            'pitch_range': (40, 55),  # Bass range (MIDI)
            'syncopation': focus * 0.8,  # More syncopation with focus
            'note_duration': 0.5 + (relax * 1.5),  # Longer notes when relaxed
        }
    
    def map_harmony_layer(self, brain_state: Dict[str, float]) -> Dict[str, any]:
        """
        Map brain state to harmonic pad layer.
        
        Harmony responds to:
        - Hemispheric asymmetry → consonance vs dissonance
        - Theta power → reverb/space
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Harmony layer parameters
        """
        asymmetry = brain_state.get('hemispheric_asymmetry', 0.0)
        theta = brain_state.get('theta_power', 0.5)
        
        return {
            'pitch_range': (55, 72),  # Mid range
            'consonance': 1.0 - abs(asymmetry),  # Symmetric = consonant
            'reverb': theta * 0.7,
            'note_duration': 2.0 + (theta * 2.0),  # Long sustained notes
            'chord_complexity': abs(asymmetry) * 0.5,
        }
    
    def map_melody_layer(self, brain_state: Dict[str, float]) -> Dict[str, any]:
        """
        Map brain state to melodic lead layer.
        
        Main CA-generated melody responds to:
        - Focus → note density, pitch height
        - Stability → repetition vs variation
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Melody layer parameters
        """
        focus = brain_state.get('focus', 0.5)
        stability = brain_state.get('stability', 0.5)
        
        return {
            'tempo': self.base_mapper._map_tempo(brain_state),
            'density': self.base_mapper._map_density(brain_state),
            'pitch_center': self.base_mapper._map_pitch_center(brain_state),
            'pitch_range': (60, 84),  # Melody range
            'scale': self.base_mapper._map_scale(brain_state),
            'repetition': stability,  # More stable = more repetitive
            'variation': 1.0 - stability,
        }
    
    def map_texture_layer(self, brain_state: Dict[str, float]) -> Dict[str, any]:
        """
        Map brain state to textural elements (sparkle notes).
        
        Texture responds to:
        - Gamma power → density of high notes
        - Focus trend → rising or falling gestures
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Texture layer parameters
        """
        gamma = brain_state.get('gamma_power', 0.5)
        focus_trend = brain_state.get('focus_trend', 0.0)
        
        return {
            'density': gamma * 0.4,  # Sparse, high notes
            'pitch_range': (75, 96),  # High range
            'gesture_direction': focus_trend,  # Rising/falling
            'note_duration': 0.1 + (gamma * 0.2),  # Short notes
            'velocity_range': (40, 80),  # Softer than melody
        }
    
    def map_all_layers(self, brain_state: Dict[str, float]) -> Dict[str, Dict]:
        """
        Map brain state to all layer parameters.
        
        Args:
            brain_state: Brain state dict
            
        Returns:
            Dictionary of layer parameters
        """
        return {
            'bass': self.map_bass_layer(brain_state),
            'harmony': self.map_harmony_layer(brain_state),
            'melody': self.map_melody_layer(brain_state),
            'texture': self.map_texture_layer(brain_state),
        }
