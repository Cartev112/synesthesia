"""
Audio mixer for combining multiple tracks.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field

from backend.audio.synthesizers import SynthesizerBase, get_synthesizer
from backend.audio.effects import EffectBase, get_effect
from backend.music.scales import midi_to_frequency
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioTrack:
    """
    Represents a single audio track with synthesizer and effects.
    """
    name: str
    synthesizer_type: str = 'sine'
    volume: float = 0.8
    pan: float = 0.5  # 0=left, 0.5=center, 1=right
    mute: bool = False
    solo: bool = False
    effects: List[str] = field(default_factory=list)
    
    # Internal state
    _synthesizer: Optional[SynthesizerBase] = field(default=None, init=False, repr=False)
    _effect_chain: List[EffectBase] = field(default_factory=list, init=False, repr=False)
    
    def initialize(self, sample_rate: int = 44100):
        """Initialize synthesizer and effects."""
        self._synthesizer = get_synthesizer(self.synthesizer_type, sample_rate)
        
        # Initialize effects
        self._effect_chain = []
        for effect_type in self.effects:
            effect = get_effect(effect_type, sample_rate)
            if effect:
                self._effect_chain.append(effect)
    
    def render_note(
        self,
        midi_note: int,
        duration: float,
        velocity: float = 1.0
    ) -> np.ndarray:
        """
        Render a single note.
        
        Args:
            midi_note: MIDI note number
            duration: Duration in seconds
            velocity: Note velocity (0-1)
            
        Returns:
            Audio samples
        """
        if self._synthesizer is None:
            raise ValueError("Track not initialized")
        
        # Convert MIDI to frequency
        frequency = midi_to_frequency(midi_note)
        
        # Generate note
        samples = self._synthesizer.generate_note(frequency, duration, velocity)
        
        # Apply effects
        for effect in self._effect_chain:
            samples = effect.process(samples)
        
        # Apply volume
        samples *= self.volume
        
        return samples
    
    def set_synthesizer(self, synth_type: str, sample_rate: int = 44100):
        """Change the synthesizer type."""
        self.synthesizer_type = synth_type
        self._synthesizer = get_synthesizer(synth_type, sample_rate)
        logger.info("track_synthesizer_changed", track=self.name, synth=synth_type)
    
    def add_effect(self, effect_type: str, sample_rate: int = 44100, **kwargs):
        """Add an effect to the chain."""
        effect = get_effect(effect_type, sample_rate, **kwargs)
        if effect:
            self._effect_chain.append(effect)
            self.effects.append(effect_type)
            logger.info("effect_added", track=self.name, effect=effect_type)
    
    def remove_effect(self, effect_index: int):
        """Remove an effect from the chain."""
        if 0 <= effect_index < len(self._effect_chain):
            removed = self.effects.pop(effect_index)
            self._effect_chain.pop(effect_index)
            logger.info("effect_removed", track=self.name, effect=removed)
    
    def clear_effects(self):
        """Remove all effects."""
        self.effects.clear()
        self._effect_chain.clear()


class AudioMixer:
    """
    Multi-track audio mixer.
    
    Combines multiple tracks with individual synthesizers and effects.
    """
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        """
        Initialize audio mixer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Audio buffer size in samples
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.tracks: Dict[str, AudioTrack] = {}
        self.master_volume = 0.8
        
        logger.info("audio_mixer_initialized", sample_rate=sample_rate)
    
    def add_track(
        self,
        name: str,
        synthesizer_type: str = 'sine',
        volume: float = 0.8,
        effects: Optional[List[str]] = None
    ) -> AudioTrack:
        """
        Add a new track to the mixer.
        
        Args:
            name: Track name
            synthesizer_type: Synthesizer type
            volume: Track volume (0-1)
            effects: List of effect types
            
        Returns:
            Created AudioTrack
        """
        if effects is None:
            effects = []
        
        track = AudioTrack(
            name=name,
            synthesizer_type=synthesizer_type,
            volume=volume,
            effects=effects
        )
        
        track.initialize(self.sample_rate)
        self.tracks[name] = track
        
        logger.info(
            "track_added",
            track=name,
            synthesizer=synthesizer_type,
            effects=effects
        )
        
        return track
    
    def remove_track(self, name: str):
        """Remove a track from the mixer."""
        if name in self.tracks:
            del self.tracks[name]
            logger.info("track_removed", track=name)
    
    def get_track(self, name: str) -> Optional[AudioTrack]:
        """Get a track by name."""
        return self.tracks.get(name)
    
    def render_events(
        self,
        events: Dict[str, List],
        duration: float
    ) -> np.ndarray:
        """
        Render musical events from all tracks.
        
        Args:
            events: Dictionary mapping track names to event lists
            duration: Total duration in seconds
            
        Returns:
            Mixed audio samples
        """
        n_samples = int(duration * self.sample_rate)
        output = np.zeros(n_samples)
        
        # Check if any track is soloed
        has_solo = any(track.solo for track in self.tracks.values())
        
        # Render each track
        for track_name, track_events in events.items():
            track = self.tracks.get(track_name)
            
            if track is None:
                logger.warning("track_not_found", track=track_name)
                continue
            
            # Skip if muted or if another track is soloed
            if track.mute or (has_solo and not track.solo):
                continue
            
            # Render track
            track_audio = self._render_track(track, track_events, duration)
            
            # Apply panning (simple stereo for now, we'll mix to mono)
            # For now, just add to mono output
            output += track_audio
        
        # Apply master volume
        output *= self.master_volume
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 1.0:
            output /= max_amplitude
        
        return output
    
    def _render_track(
        self,
        track: AudioTrack,
        events: List,
        duration: float
    ) -> np.ndarray:
        """
        Render a single track's events.
        
        Args:
            track: AudioTrack to render
            events: List of MidiEvent objects
            duration: Total duration in seconds
            
        Returns:
            Track audio samples
        """
        n_samples = int(duration * self.sample_rate)
        output = np.zeros(n_samples)
        
        for event in events:
            # Render note
            note_samples = track.render_note(
                event.note,
                event.duration,
                event.velocity
            )
            
            # Calculate start position
            start_sample = int(event.time * self.sample_rate)
            end_sample = start_sample + len(note_samples)
            
            # Ensure we don't exceed buffer
            if start_sample >= n_samples:
                continue
            
            end_sample = min(end_sample, n_samples)
            note_length = end_sample - start_sample
            
            # Mix into output
            output[start_sample:end_sample] += note_samples[:note_length]
        
        return output
    
    def set_track_volume(self, track_name: str, volume: float):
        """Set track volume."""
        track = self.tracks.get(track_name)
        if track:
            track.volume = np.clip(volume, 0.0, 1.0)
            logger.debug("track_volume_changed", track=track_name, volume=volume)
    
    def set_track_mute(self, track_name: str, mute: bool):
        """Mute or unmute a track."""
        track = self.tracks.get(track_name)
        if track:
            track.mute = mute
            logger.debug("track_mute_changed", track=track_name, mute=mute)
    
    def set_track_solo(self, track_name: str, solo: bool):
        """Solo or unsolo a track."""
        track = self.tracks.get(track_name)
        if track:
            track.solo = solo
            logger.debug("track_solo_changed", track=track_name, solo=solo)
    
    def set_master_volume(self, volume: float):
        """Set master volume."""
        self.master_volume = np.clip(volume, 0.0, 1.0)
        logger.debug("master_volume_changed", volume=volume)
    
    def get_track_list(self) -> List[Dict]:
        """Get list of all tracks with their settings."""
        return [
            {
                'name': track.name,
                'synthesizer': track.synthesizer_type,
                'volume': track.volume,
                'pan': track.pan,
                'mute': track.mute,
                'solo': track.solo,
                'effects': track.effects
            }
            for track in self.tracks.values()
        ]
    
    def reset(self):
        """Reset mixer to default state."""
        self.tracks.clear()
        self.master_volume = 0.8
        logger.info("mixer_reset")
