"""
Tests for audio engine.
"""

import pytest
import numpy as np

from backend.audio.synthesizers import (
    SineWaveSynth,
    SquareWaveSynth,
    SawtoothWaveSynth,
    TriangleWaveSynth,
    FMSynth,
    SubtractiveSynth,
    get_synthesizer
)
from backend.audio.effects import (
    ReverbEffect,
    DelayEffect,
    FilterEffect,
    CompressorEffect
)
from backend.audio.mixer import AudioMixer, AudioTrack
from backend.audio.engine import AudioEngine
from backend.music import MusicGenerator, MidiEvent


class TestSynthesizers:
    """Test synthesizer implementations."""
    
    @pytest.fixture
    def sample_rate(self):
        return 44100
    
    def test_sine_wave_synth(self, sample_rate):
        """Test sine wave synthesizer."""
        synth = SineWaveSynth(sample_rate=sample_rate)
        
        # Generate A440
        audio = synth.generate_note(frequency=440.0, duration=0.5, velocity=0.8)
        
        assert len(audio) == int(0.5 * sample_rate)
        assert audio.dtype == np.float64
        assert np.max(np.abs(audio)) <= 1.0  # Check for clipping
    
    def test_square_wave_synth(self, sample_rate):
        """Test square wave synthesizer."""
        synth = SquareWaveSynth(sample_rate=sample_rate)
        audio = synth.generate_note(frequency=220.0, duration=0.3, velocity=1.0)
        
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0
    
    def test_sawtooth_wave_synth(self, sample_rate):
        """Test sawtooth wave synthesizer."""
        synth = SawtoothWaveSynth(sample_rate=sample_rate)
        audio = synth.generate_note(frequency=330.0, duration=0.4, velocity=0.9)
        
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0
    
    def test_triangle_wave_synth(self, sample_rate):
        """Test triangle wave synthesizer."""
        synth = TriangleWaveSynth(sample_rate=sample_rate)
        audio = synth.generate_note(frequency=550.0, duration=0.2, velocity=0.7)
        
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0
    
    def test_fm_synth(self, sample_rate):
        """Test FM synthesizer."""
        synth = FMSynth(sample_rate=sample_rate, mod_ratio=2.0, mod_index=5.0)
        audio = synth.generate_note(frequency=440.0, duration=0.5, velocity=0.8)
        
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0
    
    def test_subtractive_synth(self, sample_rate):
        """Test subtractive synthesizer."""
        synth = SubtractiveSynth(sample_rate=sample_rate, cutoff_ratio=2.0)
        audio = synth.generate_note(frequency=110.0, duration=0.6, velocity=1.0)
        
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0
    
    def test_envelope_applied(self, sample_rate):
        """Test that ADSR envelope is applied."""
        synth = SineWaveSynth(sample_rate=sample_rate)
        audio = synth.generate_note(frequency=440.0, duration=0.5, velocity=1.0)
        
        # Check that audio starts and ends near zero (envelope)
        assert abs(audio[0]) < 0.1  # Attack phase
        assert abs(audio[-1]) < 0.1  # Release phase
    
    def test_midi_to_frequency(self, sample_rate):
        """Test MIDI to frequency conversion."""
        synth = SineWaveSynth(sample_rate=sample_rate)
        
        # A4 = MIDI 69 = 440 Hz
        assert abs(synth.midi_to_frequency(69) - 440.0) < 0.01
        
        # C4 = MIDI 60 = 261.63 Hz
        assert abs(synth.midi_to_frequency(60) - 261.63) < 0.01
    
    def test_get_synthesizer(self, sample_rate):
        """Test synthesizer factory function."""
        synth = get_synthesizer('sine', sample_rate)
        assert isinstance(synth, SineWaveSynth)
        
        synth = get_synthesizer('fm', sample_rate)
        assert isinstance(synth, FMSynth)
        
        # Unknown type should default to sine
        synth = get_synthesizer('unknown', sample_rate)
        assert isinstance(synth, SineWaveSynth)


class TestEffects:
    """Test audio effects."""
    
    @pytest.fixture
    def sample_rate(self):
        return 44100
    
    @pytest.fixture
    def test_audio(self, sample_rate):
        """Generate test audio signal."""
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        return np.sin(2 * np.pi * 440 * t)
    
    def test_reverb_effect(self, sample_rate, test_audio):
        """Test reverb effect."""
        effect = ReverbEffect(sample_rate=sample_rate, room_size=0.5, wet_level=0.3)
        
        processed = effect.process(test_audio)
        
        assert len(processed) == len(test_audio)
        assert not np.array_equal(processed, test_audio)  # Should be different
    
    def test_delay_effect(self, sample_rate, test_audio):
        """Test delay effect."""
        effect = DelayEffect(sample_rate=sample_rate, delay_time=0.25, feedback=0.4)
        
        processed = effect.process(test_audio)
        
        assert len(processed) == len(test_audio)
        assert not np.array_equal(processed, test_audio)
    
    def test_filter_effect(self, sample_rate, test_audio):
        """Test filter effect."""
        effect = FilterEffect(
            sample_rate=sample_rate,
            filter_type='lowpass',
            cutoff_freq=1000.0
        )
        
        processed = effect.process(test_audio)
        
        assert len(processed) == len(test_audio)
    
    def test_compressor_effect(self, sample_rate, test_audio):
        """Test compressor effect."""
        effect = CompressorEffect(
            sample_rate=sample_rate,
            threshold=0.5,
            ratio=4.0
        )
        
        processed = effect.process(test_audio)
        
        assert len(processed) == len(test_audio)
    
    def test_effect_enable_disable(self, sample_rate, test_audio):
        """Test enabling/disabling effects."""
        effect = ReverbEffect(sample_rate=sample_rate)
        
        # Enabled
        effect.set_enabled(True)
        processed_enabled = effect.process(test_audio)
        
        # Disabled
        effect.set_enabled(False)
        processed_disabled = effect.process(test_audio)
        
        # When disabled, should return original
        assert np.array_equal(processed_disabled, test_audio)
        assert not np.array_equal(processed_enabled, test_audio)


class TestAudioMixer:
    """Test audio mixer."""
    
    @pytest.fixture
    def mixer(self):
        return AudioMixer(sample_rate=44100, buffer_size=512)
    
    def test_mixer_initialization(self, mixer):
        """Test mixer initialization."""
        assert mixer.sample_rate == 44100
        assert mixer.buffer_size == 512
        assert len(mixer.tracks) == 0
    
    def test_add_track(self, mixer):
        """Test adding tracks."""
        track = mixer.add_track(
            name='test_track',
            synthesizer_type='sine',
            volume=0.8
        )
        
        assert track.name == 'test_track'
        assert track.synthesizer_type == 'sine'
        assert track.volume == 0.8
        assert 'test_track' in mixer.tracks
    
    def test_remove_track(self, mixer):
        """Test removing tracks."""
        mixer.add_track('track1', 'sine')
        mixer.add_track('track2', 'square')
        
        assert len(mixer.tracks) == 2
        
        mixer.remove_track('track1')
        
        assert len(mixer.tracks) == 1
        assert 'track1' not in mixer.tracks
        assert 'track2' in mixer.tracks
    
    def test_render_events(self, mixer):
        """Test rendering musical events."""
        # Add track
        mixer.add_track('melody', 'sine', volume=0.8)
        
        # Create test events
        events = {
            'melody': [
                MidiEvent(note=60, velocity=0.8, duration=0.25, time=0.0),
                MidiEvent(note=64, velocity=0.7, duration=0.25, time=0.25)
            ]
        }
        
        # Render
        audio = mixer.render_events(events, duration=0.5)
        
        assert len(audio) == int(0.5 * mixer.sample_rate)
        assert audio.dtype == np.float64
    
    def test_track_mute(self, mixer):
        """Test track muting."""
        mixer.add_track('track1', 'sine')
        
        events = {
            'track1': [MidiEvent(note=60, velocity=0.8, duration=0.5, time=0.0)]
        }
        
        # Render unmuted
        audio_unmuted = mixer.render_events(events, duration=0.5)
        
        # Mute and render
        mixer.set_track_mute('track1', True)
        audio_muted = mixer.render_events(events, duration=0.5)
        
        # Muted should be silent
        assert np.max(np.abs(audio_muted)) < 0.01
        assert np.max(np.abs(audio_unmuted)) > 0.01
    
    def test_track_solo(self, mixer):
        """Test track solo."""
        mixer.add_track('track1', 'sine')
        mixer.add_track('track2', 'square')
        
        events = {
            'track1': [MidiEvent(note=60, velocity=0.8, duration=0.5, time=0.0)],
            'track2': [MidiEvent(note=64, velocity=0.8, duration=0.5, time=0.0)]
        }
        
        # Solo track1
        mixer.set_track_solo('track1', True)
        audio = mixer.render_events(events, duration=0.5)
        
        # Should only hear track1
        assert len(audio) > 0
    
    def test_master_volume(self, mixer):
        """Test master volume control."""
        mixer.add_track('track1', 'sine')
        
        events = {
            'track1': [MidiEvent(note=60, velocity=1.0, duration=0.5, time=0.0)]
        }
        
        # Render at full volume
        mixer.set_master_volume(1.0)
        audio_full = mixer.render_events(events, duration=0.5)
        
        # Render at half volume
        mixer.set_master_volume(0.5)
        audio_half = mixer.render_events(events, duration=0.5)
        
        # Half volume should be quieter
        assert np.max(np.abs(audio_half)) < np.max(np.abs(audio_full))


class TestAudioEngine:
    """Test audio engine."""
    
    @pytest.fixture
    def engine(self):
        return AudioEngine(sample_rate=44100, buffer_size=512)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.sample_rate == 44100
        assert engine.buffer_size == 512
        assert engine.mixer is not None
    
    def test_setup_default_tracks(self, engine):
        """Test setting up default tracks."""
        engine.setup_default_tracks()
        
        assert 'bass' in engine.mixer.tracks
        assert 'harmony' in engine.mixer.tracks
        assert 'melody' in engine.mixer.tracks
        assert 'texture' in engine.mixer.tracks
    
    def test_set_track_synthesizer(self, engine):
        """Test changing track synthesizer."""
        engine.setup_default_tracks()
        
        engine.set_track_synthesizer('melody', 'square')
        
        track = engine.mixer.get_track('melody')
        assert track.synthesizer_type == 'square'
    
    def test_add_track_effect(self, engine):
        """Test adding effects to tracks."""
        engine.setup_default_tracks()
        
        engine.add_track_effect('melody', 'delay', delay_time=0.3)
        
        track = engine.mixer.get_track('melody')
        assert 'delay' in track.effects
    
    def test_generate_audio_buffer(self, engine):
        """Test generating audio buffer."""
        engine.setup_default_tracks()
        
        # Create music generator
        music_gen = MusicGenerator()
        engine.set_music_generator(music_gen)
        
        # Generate buffer
        buffer = engine.generate_audio_buffer()
        
        assert len(buffer) == engine.buffer_size
        assert buffer.dtype == np.float64
    
    def test_get_configuration(self, engine):
        """Test getting engine configuration."""
        engine.setup_default_tracks()
        
        config = engine.get_configuration()
        
        assert 'sample_rate' in config
        assert 'buffer_size' in config
        assert 'tracks' in config
        assert len(config['tracks']) == 4


class TestIntegration:
    """Integration tests."""
    
    def test_full_audio_pipeline(self):
        """Test complete audio pipeline."""
        # Create music generator
        music_gen = MusicGenerator(
            enable_bass=True,
            enable_harmony=True,
            enable_melody=True
        )
        
        # Update brain state
        brain_state = {
            'focus': 0.7,
            'relax': 0.3,
            'neutral': 0.0,
            'stability': 0.8
        }
        music_gen.update_brain_state(brain_state)
        
        # Create audio engine
        engine = AudioEngine(sample_rate=44100, buffer_size=512)
        engine.setup_default_tracks()
        engine.set_music_generator(music_gen)
        
        # Generate audio
        buffer = engine.generate_audio_buffer()
        
        assert len(buffer) == 512
        assert buffer.dtype == np.float64
    
    def test_different_synthesizers_per_track(self):
        """Test using different synthesizers for each track."""
        engine = AudioEngine()
        engine.setup_default_tracks()
        
        # Set different synthesizers
        engine.set_track_synthesizer('bass', 'subtractive')
        engine.set_track_synthesizer('harmony', 'sine')
        engine.set_track_synthesizer('melody', 'fm')
        engine.set_track_synthesizer('texture', 'triangle')
        
        # Verify
        assert engine.mixer.get_track('bass').synthesizer_type == 'subtractive'
        assert engine.mixer.get_track('harmony').synthesizer_type == 'sine'
        assert engine.mixer.get_track('melody').synthesizer_type == 'fm'
        assert engine.mixer.get_track('texture').synthesizer_type == 'triangle'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
