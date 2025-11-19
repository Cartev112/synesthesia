"""
Musical scales and theory utilities.

Provides scale definitions and MIDI note mappings.
"""

from typing import List, Dict
import numpy as np

# MIDI note number for middle C
MIDDLE_C = 60


class Scale:
    """Represents a musical scale."""
    
    def __init__(self, name: str, intervals: List[int], root: int = MIDDLE_C):
        """
        Initialize a scale.
        
        Args:
            name: Scale name
            intervals: Semitone intervals from root
            root: Root note MIDI number
        """
        self.name = name
        self.intervals = intervals
        self.root = root
    
    def get_note(self, degree: int) -> int:
        """
        Get MIDI note for a scale degree.
        
        Args:
            degree: Scale degree (0-indexed)
            
        Returns:
            MIDI note number
        """
        octave = degree // len(self.intervals)
        scale_degree = degree % len(self.intervals)
        
        return self.root + (octave * 12) + self.intervals[scale_degree]
    
    def get_notes(self, num_octaves: int = 2) -> List[int]:
        """
        Get all notes in the scale for given octaves.
        
        Args:
            num_octaves: Number of octaves to generate
            
        Returns:
            List of MIDI note numbers
        """
        notes = []
        for octave in range(num_octaves):
            for interval in self.intervals:
                notes.append(self.root + (octave * 12) + interval)
        return notes
    
    def transpose(self, semitones: int) -> 'Scale':
        """
        Transpose the scale.
        
        Args:
            semitones: Number of semitones to transpose
            
        Returns:
            New transposed scale
        """
        return Scale(self.name, self.intervals, self.root + semitones)


# Common scale definitions
SCALES: Dict[str, List[int]] = {
    # Major scales
    'major': [0, 2, 4, 5, 7, 9, 11],
    'ionian': [0, 2, 4, 5, 7, 9, 11],
    
    # Minor scales
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    
    # Modes
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    
    # Pentatonic
    'major_pentatonic': [0, 2, 4, 7, 9],
    'minor_pentatonic': [0, 3, 5, 7, 10],
    
    # Other scales
    'blues': [0, 3, 5, 6, 7, 10],
    'whole_tone': [0, 2, 4, 6, 8, 10],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    
    # Eastern scales
    'hirajoshi': [0, 2, 3, 7, 8],
    'pelog': [0, 1, 3, 7, 8],
}


def get_scale(name: str, root: int = MIDDLE_C) -> Scale:
    """
    Get a scale by name.
    
    Args:
        name: Scale name
        root: Root note MIDI number
        
    Returns:
        Scale object
        
    Raises:
        ValueError: If scale name not found
    """
    if name not in SCALES:
        raise ValueError(f"Unknown scale: {name}. Available: {list(SCALES.keys())}")
    
    return Scale(name, SCALES[name], root)


def midi_to_note_name(midi_note: int) -> str:
    """
    Convert MIDI note number to note name.
    
    Args:
        midi_note: MIDI note number (0-127)
        
    Returns:
        Note name (e.g., 'C4', 'A#3')
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"


def note_name_to_midi(note_name: str) -> int:
    """
    Convert note name to MIDI number.
    
    Args:
        note_name: Note name (e.g., 'C4', 'A#3')
        
    Returns:
        MIDI note number
    """
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    # Parse note name
    note = note_name[0].upper()
    accidental = 0
    octave_start = 1
    
    if len(note_name) > 1 and note_name[1] in ['#', 'b']:
        accidental = 1 if note_name[1] == '#' else -1
        octave_start = 2
    
    octave = int(note_name[octave_start:])
    
    return (octave + 1) * 12 + note_map[note] + accidental


def frequency_to_midi(frequency: float) -> int:
    """
    Convert frequency to nearest MIDI note.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        MIDI note number
    """
    return int(round(69 + 12 * np.log2(frequency / 440.0)))


def midi_to_frequency(midi_note: int) -> float:
    """
    Convert MIDI note to frequency.
    
    Args:
        midi_note: MIDI note number
        
    Returns:
        Frequency in Hz
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
