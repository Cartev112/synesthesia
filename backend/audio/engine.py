"""
Main audio engine for synesthesia BCI system.

Integrates synthesizers, effects, and mixer with the music generator.
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import queue
import threading
import time

from backend.audio.mixer import AudioMixer, AudioTrack
from backend.music import MusicGenerator
from backend.core.logging import get_logger

logger = get_logger(__name__)


class AudioEngine:
    """
    Main audio engine.
    
    Coordinates music generation, audio synthesis, and real-time playback.
    """
    
    # Default synthesizer configurations for each layer
    DEFAULT_SYNTH_CONFIG = {
        'bass': {
            'synthesizer': 'subtractive',
            'volume': 0.7,
            'effects': ['compressor']
        },
        'harmony': {
            'synthesizer': 'sine',
            'volume': 0.5,
            'effects': ['reverb']
        },
        'melody': {
            'synthesizer': 'fm',
            'volume': 0.6,
            'effects': ['reverb', 'delay']
        },
        'texture': {
            'synthesizer': 'triangle',
            'volume': 0.4,
            'effects': ['reverb', 'filter']
        }
    }
    
    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 512,
        music_generator: Optional[MusicGenerator] = None
    ):
        """
        Initialize audio engine.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Audio buffer size in samples
            music_generator: MusicGenerator instance (optional)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Create mixer
        self.mixer = AudioMixer(sample_rate=sample_rate, buffer_size=buffer_size)
        
        # Music generator
        self.music_generator = music_generator
        
        # Audio buffer queue
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Playback state
        self.is_playing = False
        self.playback_thread = None
        
        # Timing
        self.step_duration = 0.125  # 125ms per step (8th notes at 120 BPM)
        self.current_time = 0.0
        
        # Callbacks
        self.on_audio_buffer: Optional[Callable] = None
        
        logger.info(
            "audio_engine_initialized",
            sample_rate=sample_rate,
            buffer_size=buffer_size
        )
    
    def setup_default_tracks(self):
        """Setup default tracks for each musical layer."""
        for layer_name, config in self.DEFAULT_SYNTH_CONFIG.items():
            self.mixer.add_track(
                name=layer_name,
                synthesizer_type=config['synthesizer'],
                volume=config['volume'],
                effects=config['effects']
            )
        
        logger.info("default_tracks_setup", tracks=list(self.DEFAULT_SYNTH_CONFIG.keys()))
    
    def set_music_generator(self, music_generator: MusicGenerator):
        """Set the music generator."""
        self.music_generator = music_generator
        logger.info("music_generator_set")
    
    def set_track_synthesizer(self, track_name: str, synth_type: str):
        """
        Change synthesizer for a track.
        
        Args:
            track_name: Track name
            synth_type: Synthesizer type ('sine', 'square', 'sawtooth', 'triangle', 'fm', 'subtractive')
        """
        track = self.mixer.get_track(track_name)
        if track:
            track.set_synthesizer(synth_type, self.sample_rate)
            logger.info(
                "track_synthesizer_changed",
                track=track_name,
                synthesizer=synth_type
            )
        else:
            logger.warning("track_not_found", track=track_name)
    
    def add_track_effect(self, track_name: str, effect_type: str, **kwargs):
        """
        Add an effect to a track.
        
        Args:
            track_name: Track name
            effect_type: Effect type ('reverb', 'delay', 'filter', 'compressor')
            **kwargs: Effect parameters
        """
        track = self.mixer.get_track(track_name)
        if track:
            track.add_effect(effect_type, self.sample_rate, **kwargs)
            logger.info("effect_added", track=track_name, effect=effect_type)
        else:
            logger.warning("track_not_found", track=track_name)
    
    def generate_audio_buffer(self) -> np.ndarray:
        """
        Generate one audio buffer from music generator.
        
        Returns:
            Audio samples for one buffer
        """
        if self.music_generator is None:
            # Return silence if no generator
            return np.zeros(self.buffer_size)
        
        # Generate musical events
        events = self.music_generator.generate_step()
        
        # Calculate buffer duration
        buffer_duration = self.buffer_size / self.sample_rate
        
        # Render events to audio
        audio = self.mixer.render_events(events, buffer_duration)
        
        # Ensure correct length
        if len(audio) < self.buffer_size:
            # Pad with zeros
            audio = np.pad(audio, (0, self.buffer_size - len(audio)))
        elif len(audio) > self.buffer_size:
            # Truncate
            audio = audio[:self.buffer_size]
        
        return audio
    
    def start_playback(self):
        """Start real-time audio playback."""
        if self.is_playing:
            logger.warning("playback_already_started")
            return
        
        self.is_playing = True
        self.current_time = 0.0
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        
        logger.info("playback_started")
    
    def stop_playback(self):
        """Stop real-time audio playback."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        
        # Wait for thread to finish
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("playback_stopped")
    
    def _playback_loop(self):
        """Playback loop running in separate thread."""
        logger.info("playback_loop_started")
        
        while self.is_playing:
            try:
                # Generate audio buffer
                audio_buffer = self.generate_audio_buffer()
                
                # Add to queue (blocks if queue is full)
                self.audio_queue.put(audio_buffer, timeout=1.0)
                
                # Call callback if set
                if self.on_audio_buffer:
                    self.on_audio_buffer(audio_buffer)
                
                # Update time
                buffer_duration = self.buffer_size / self.sample_rate
                self.current_time += buffer_duration
                
                # Sleep to maintain timing
                time.sleep(buffer_duration * 0.8)  # Sleep slightly less to stay ahead
                
            except queue.Full:
                logger.warning("audio_queue_full")
            except Exception as e:
                logger.exception("playback_loop_error", error=str(e))
        
        logger.info("playback_loop_stopped")
    
    def get_audio_buffer(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio buffer from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio buffer or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def render_to_file(
        self,
        duration: float,
        filename: str,
        format: str = 'wav'
    ):
        """
        Render audio to file.
        
        Args:
            duration: Duration in seconds
            filename: Output filename
            format: Audio format ('wav', 'mp3', etc.)
        """
        import soundfile as sf
        
        # Calculate total samples
        total_samples = int(duration * self.sample_rate)
        audio = np.zeros(total_samples)
        
        # Generate in chunks
        samples_generated = 0
        
        while samples_generated < total_samples:
            # Generate buffer
            buffer = self.generate_audio_buffer()
            
            # Copy to output
            end_idx = min(samples_generated + len(buffer), total_samples)
            length = end_idx - samples_generated
            audio[samples_generated:end_idx] = buffer[:length]
            
            samples_generated += length
        
        # Write to file
        sf.write(filename, audio, self.sample_rate)
        
        logger.info(
            "audio_rendered_to_file",
            filename=filename,
            duration=duration,
            samples=total_samples
        )
    
    def get_configuration(self) -> Dict:
        """
        Get current audio engine configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'is_playing': self.is_playing,
            'tracks': self.mixer.get_track_list(),
            'master_volume': self.mixer.master_volume
        }
    
    def update_brain_state(self, brain_state: Dict):
        """
        Update brain state (passed to music generator).
        
        Args:
            brain_state: Brain state dictionary
        """
        if self.music_generator:
            self.music_generator.update_brain_state(brain_state)
    
    def set_tempo(self, bpm: float):
        """
        Set tempo.
        
        Args:
            bpm: Beats per minute
        """
        if self.music_generator:
            # Update step duration based on BPM
            # 8th notes at given BPM
            self.step_duration = 60.0 / (bpm * 2)
            logger.info("tempo_changed", bpm=bpm, step_duration=self.step_duration)
