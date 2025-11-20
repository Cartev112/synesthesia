"""
Music generation system for BCI.

Includes:
- Cellular automaton music generation
- Brain-to-music parameter mappings
- Multi-layer music coordination
- Musical scales and theory utilities
"""

from backend.music.cellular_automaton import MusicalCellularAutomaton, MidiEvent
from backend.music.mappings import BrainMusicMapper, LayeredMusicMapper
from backend.music.music_generator import MusicGenerator
from backend.music.scales import Scale, get_scale, SCALES

__all__ = [
    'MusicalCellularAutomaton',
    'MidiEvent',
    'BrainMusicMapper',
    'LayeredMusicMapper',
    'MusicGenerator',
    'Scale',
    'get_scale',
    'SCALES',
]
