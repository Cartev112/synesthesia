"""
Main music generator coordinating multiple layers.

Combines cellular automaton, mappings, and layering
to create responsive music from brain states.
"""

from typing import Dict, List, Optional
import time
import numpy as np

from backend.music.cellular_automaton import MusicalCellularAutomaton, MidiEvent
from backend.music.mappings import LayeredMusicMapper
from backend.core.logging import get_logger

logger = get_logger(__name__)


class MusicGenerator:
    """
    Main music generation system.
    
    Coordinates multiple musical layers and responds to brain state.
    """
    
    def __init__(
        self,
        enable_bass: bool = True,
        enable_harmony: bool = True,
        enable_melody: bool = True,
        enable_texture: bool = False
    ):
        """
        Initialize music generator.
        
        Args:
            enable_bass: Enable bass/rhythm layer
            enable_harmony: Enable harmonic pad layer
            enable_melody: Enable melodic lead layer
            enable_texture: Enable textural elements layer
        """
        self.mapper = LayeredMusicMapper()
        
        # Create cellular automaton for melody layer
        self.melody_ca = MusicalCellularAutomaton(
            width=16,
            height=8,
            scale_name='major',
            root_note=60,
            base_tempo=120.0
        )
        
        # Layer enable flags
        self.layers_enabled = {
            'bass': enable_bass,
            'harmony': enable_harmony,
            'melody': enable_melody,
            'texture': enable_texture,
        }
        
        # Current brain state
        self.current_brain_state: Dict[str, float] = {
            'focus': 0.5,
            'relax': 0.5,
            'neutral': 0.5,
        }
        
        # Timing
        self.last_step_time = time.time()
        self.step_count = 0
        
        logger.info(
            "music_generator_initialized",
            layers=self.layers_enabled
        )
    
    def update_brain_state(self, brain_state: Dict[str, float]):
        """
        Update current brain state.
        
        Args:
            brain_state: New brain state values
        """
        self.current_brain_state = brain_state
        
        # Update melody CA with brain state
        self.melody_ca.update_from_brain_state(brain_state)
        
        logger.debug(
            "brain_state_updated",
            focus=brain_state.get('focus', 0.5),
            relax=brain_state.get('relax', 0.5)
        )
    
    def generate_step(self) -> Dict[str, List[MidiEvent]]:
        """
        Generate one step of music across all layers.
        
        Returns:
            Dictionary mapping layer names to MIDI events
        """
        events_by_layer = {}
        
        # Get layer parameters from brain state
        layer_params = self.mapper.map_all_layers(self.current_brain_state)
        
        # Generate melody layer (CA-based)
        if self.layers_enabled['melody']:
            melody_events = self.melody_ca.step()
            events_by_layer['melody'] = melody_events
        
        # Generate bass layer
        if self.layers_enabled['bass']:
            bass_events = self._generate_bass(layer_params['bass'])
            events_by_layer['bass'] = bass_events
        
        # Generate harmony layer
        if self.layers_enabled['harmony']:
            harmony_events = self._generate_harmony(layer_params['harmony'])
            events_by_layer['harmony'] = harmony_events
        
        # Generate texture layer
        if self.layers_enabled['texture']:
            texture_events = self._generate_texture(layer_params['texture'])
            events_by_layer['texture'] = texture_events
        
        self.step_count += 1
        self.last_step_time = time.time()
        
        return events_by_layer
    
    def _generate_bass(self, params: Dict) -> List[MidiEvent]:
        """
        Generate bass/rhythm layer events.
        
        Args:
            params: Bass layer parameters
            
        Returns:
            List of MIDI events
        """
        events = []
        
        # Simple bass pattern: play on certain beats
        density = params['density']
        
        if np.random.random() < density:
            # Choose bass note from range
            pitch_min, pitch_max = params['pitch_range']
            note = np.random.randint(pitch_min, pitch_max + 1)
            
            # Syncopation: occasionally offset timing
            syncopation = params['syncopation']
            time_offset = 0.0
            if np.random.random() < syncopation:
                time_offset = np.random.uniform(-0.05, 0.05)
            
            events.append(MidiEvent(
                note=note,
                velocity=np.random.randint(80, 110),
                duration=params['note_duration'],
                time=time_offset
            ))
        
        return events
    
    def _generate_harmony(self, params: Dict) -> List[MidiEvent]:
        """
        Generate harmonic pad layer events.
        
        Args:
            params: Harmony layer parameters
            
        Returns:
            List of MIDI events
        """
        events = []
        
        # Harmony plays sustained chords
        # Only update occasionally (every 4-8 steps)
        if self.step_count % np.random.randint(4, 9) == 0:
            pitch_min, pitch_max = params['pitch_range']
            consonance = params['consonance']
            
            # Generate chord based on consonance
            if consonance > 0.7:
                # Consonant: major/minor triad
                root = np.random.randint(pitch_min, pitch_max - 7)
                intervals = [0, 4, 7] if np.random.random() > 0.5 else [0, 3, 7]
            elif consonance > 0.4:
                # Moderate: add 7th
                root = np.random.randint(pitch_min, pitch_max - 10)
                intervals = [0, 4, 7, 10]
            else:
                # Dissonant: clusters
                root = np.random.randint(pitch_min, pitch_max - 4)
                intervals = [0, 1, 2, 6]
            
            # Create chord events
            for interval in intervals:
                events.append(MidiEvent(
                    note=root + interval,
                    velocity=np.random.randint(40, 60),
                    duration=params['note_duration']
                ))
        
        return events
    
    def _generate_texture(self, params: Dict) -> List[MidiEvent]:
        """
        Generate textural elements (sparkle notes).
        
        Args:
            params: Texture layer parameters
            
        Returns:
            List of MIDI events
        """
        events = []
        
        density = params['density']
        
        if np.random.random() < density:
            pitch_min, pitch_max = params['pitch_range']
            gesture_direction = params['gesture_direction']
            
            # Bias pitch selection based on gesture direction
            if gesture_direction > 0.3:
                # Rising gesture: favor higher notes
                note = np.random.randint(
                    int(pitch_min + (pitch_max - pitch_min) * 0.5),
                    pitch_max + 1
                )
            elif gesture_direction < -0.3:
                # Falling gesture: favor lower notes
                note = np.random.randint(
                    pitch_min,
                    int(pitch_min + (pitch_max - pitch_min) * 0.5) + 1
                )
            else:
                # Neutral: random
                note = np.random.randint(pitch_min, pitch_max + 1)
            
            vel_min, vel_max = params['velocity_range']
            
            events.append(MidiEvent(
                note=note,
                velocity=np.random.randint(vel_min, vel_max),
                duration=params['note_duration']
            ))
        
        return events
    
    def get_step_duration(self) -> float:
        """
        Get duration of one generation step in seconds.
        
        Returns:
            Step duration based on current tempo
        """
        return self.melody_ca.get_step_duration()
    
    def set_layer_enabled(self, layer: str, enabled: bool):
        """
        Enable or disable a musical layer.
        
        Args:
            layer: Layer name ('bass', 'harmony', 'melody', 'texture')
            enabled: Whether to enable the layer
        """
        if layer in self.layers_enabled:
            self.layers_enabled[layer] = enabled
            logger.info("layer_toggled", layer=layer, enabled=enabled)
    
    def reset(self):
        """Reset the music generator to initial state."""
        self.melody_ca.reset()
        self.step_count = 0
        self.current_brain_state = {
            'focus': 0.5,
            'relax': 0.5,
            'neutral': 0.5,
        }
        logger.info("music_generator_reset")
    
    def get_current_parameters(self) -> Dict:
        """
        Get current musical parameters for all layers.
        
        Returns:
            Dictionary of current parameters
        """
        layer_params = self.mapper.map_all_layers(self.current_brain_state)
        
        return {
            'tempo': self.melody_ca.tempo,
            'density': self.melody_ca.density,
            'pitch_center': self.melody_ca.pitch_center,
            'scale': self.melody_ca.scale.name,
            'layers': layer_params,
            'step_count': self.step_count,
        }
