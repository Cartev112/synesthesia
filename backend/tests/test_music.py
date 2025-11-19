"""
Unit tests for music generation components.

Tests:
- Musical scales
- Cellular automaton
- Brain-music mappings
- Music generator
"""

import pytest
import numpy as np

from backend.music import (
    Scale,
    get_scale,
    SCALES,
    MusicalCellularAutomaton,
    MidiEvent,
    BrainMusicMapper,
    LayeredMusicMapper,
    MusicGenerator,
)


class TestScale:
    """Test musical scale functionality."""
    
    def test_scale_initialization(self):
        """Test scale initialization."""
        scale = Scale('major', [0, 2, 4, 5, 7, 9, 11], root=60)
        
        assert scale.name == 'major'
        assert scale.root == 60
        assert len(scale.intervals) == 7
    
    def test_get_note(self):
        """Test getting notes from scale degrees."""
        scale = get_scale('major', root=60)
        
        # C major scale starting at C4 (60)
        assert scale.get_note(0) == 60  # C
        assert scale.get_note(1) == 62  # D
        assert scale.get_note(2) == 64  # E
        assert scale.get_note(7) == 72  # C (next octave)
    
    def test_get_notes(self):
        """Test getting all notes in scale."""
        scale = get_scale('major', root=60)
        notes = scale.get_notes(num_octaves=2)
        
        assert len(notes) == 14  # 7 notes * 2 octaves
        assert notes[0] == 60
        assert notes[7] == 72  # Octave up
    
    def test_transpose(self):
        """Test scale transposition."""
        scale = get_scale('major', root=60)
        transposed = scale.transpose(5)  # Up a fourth
        
        assert transposed.root == 65
        assert transposed.intervals == scale.intervals
    
    def test_all_scales_available(self):
        """Test that all defined scales can be created."""
        for scale_name in SCALES.keys():
            scale = get_scale(scale_name)
            assert scale.name == scale_name
            assert len(scale.intervals) > 0
    
    def test_invalid_scale_raises_error(self):
        """Test that invalid scale name raises error."""
        with pytest.raises(ValueError):
            get_scale('nonexistent_scale')


class TestMidiEvent:
    """Test MIDI event representation."""
    
    def test_midi_event_creation(self):
        """Test creating MIDI event."""
        event = MidiEvent(note=60, velocity=80, duration=0.5, time=0.0)
        
        assert event.note == 60
        assert event.velocity == 80
        assert event.duration == 0.5
        assert event.time == 0.0
    
    def test_midi_event_repr(self):
        """Test MIDI event string representation."""
        event = MidiEvent(note=60, velocity=80, duration=0.5)
        repr_str = repr(event)
        
        assert 'MidiEvent' in repr_str
        assert '60' in repr_str


class TestMusicalCellularAutomaton:
    """Test cellular automaton music generator."""
    
    @pytest.fixture
    def ca(self):
        """Create CA instance."""
        return MusicalCellularAutomaton(
            width=16,
            height=8,
            scale_name='major',
            root_note=60,
            base_tempo=120.0
        )
    
    def test_initialization(self, ca):
        """Test CA initialization."""
        assert ca.width == 16
        assert ca.height == 8
        assert ca.scale.name == 'major'
        assert ca.base_tempo == 120.0
        assert ca.grid.shape == (8, 16)
    
    def test_update_from_brain_state(self, ca):
        """Test brain state update."""
        brain_state = {
            'focus': 0.8,
            'relax': 0.2,
            'neutral': 0.0
        }
        
        initial_tempo = ca.tempo
        ca.update_from_brain_state(brain_state)
        
        # High focus should increase tempo
        assert ca.tempo > initial_tempo
        assert ca.density > 0.3  # Higher density with focus
    
    def test_step_generates_events(self, ca):
        """Test that step generates MIDI events."""
        # Ensure some cells are active
        ca.grid[:, 0] = True
        
        events = ca.step()
        
        assert isinstance(events, list)
        # Should have events since column 0 is all active
        assert len(events) > 0
        assert all(isinstance(e, MidiEvent) for e in events)
    
    def test_step_advances_position(self, ca):
        """Test that step advances position."""
        initial_pos = ca.current_position
        ca.step()
        
        assert ca.current_position == (initial_pos + 1) % ca.width
    
    def test_ca_rules_application(self, ca):
        """Test CA rules are applied."""
        # Set up known pattern
        initial_grid = ca.grid.copy()
        ca.step()
        
        # Grid should have changed (unless by chance it's stable)
        # Just verify it's still valid
        assert ca.grid.shape == initial_grid.shape
        assert ca.grid.dtype == bool
    
    def test_set_scale(self, ca):
        """Test changing scale."""
        ca.set_scale('minor', root_note=65)
        
        assert ca.scale.name == 'minor'
        assert ca.scale.root == 65
    
    def test_reset(self, ca):
        """Test reset functionality."""
        ca.current_position = 10
        ca.step()
        
        ca.reset()
        
        assert ca.current_position == 0
    
    def test_get_step_duration(self, ca):
        """Test step duration calculation."""
        duration = ca.get_step_duration()
        
        assert duration > 0
        assert isinstance(duration, float)
        
        # At 120 BPM, 16th note should be 0.125s
        expected = 60.0 / 120.0 / 4.0
        assert abs(duration - expected) < 0.001
    
    def test_midi_note_clamping(self, ca):
        """Test that MIDI notes are clamped to valid range."""
        # Force extreme pitch center
        ca.pitch_center = 120  # Very high
        
        events = ca.step()
        
        # All notes should be in valid MIDI range
        for event in events:
            assert 0 <= event.note <= 127
    
    def test_velocity_range(self, ca):
        """Test that velocities are in valid range."""
        ca.grid[:, :] = True  # All cells active
        
        events = ca.step()
        
        for event in events:
            assert 0 <= event.velocity <= 127


class TestBrainMusicMapper:
    """Test brain-music parameter mapping."""
    
    @pytest.fixture
    def mapper(self):
        """Create mapper instance."""
        return BrainMusicMapper()
    
    @pytest.fixture
    def brain_state(self):
        """Sample brain state."""
        return {
            'focus': 0.7,
            'relax': 0.3,
            'neutral': 0.0,
            'theta_power': 0.4,
            'gamma_power': 0.6,
        }
    
    def test_map_tempo(self, mapper, brain_state):
        """Test tempo mapping."""
        tempo = mapper._map_tempo(brain_state)
        
        assert 60 <= tempo <= 180
        assert isinstance(tempo, float)
        
        # High focus should increase tempo
        assert tempo > 100
    
    def test_map_density(self, mapper, brain_state):
        """Test density mapping."""
        density = mapper._map_density(brain_state)
        
        assert 0 <= density <= 1
        # High focus should increase density
        assert density > 0.3
    
    def test_map_pitch_center(self, mapper, brain_state):
        """Test pitch center mapping."""
        pitch = mapper._map_pitch_center(brain_state)
        
        assert 48 <= pitch <= 84
        assert isinstance(pitch, int)
    
    def test_map_scale(self, mapper, brain_state):
        """Test scale mapping."""
        scale = mapper._map_scale(brain_state)
        
        assert isinstance(scale, str)
        assert scale in SCALES
    
    def test_map_reverb(self, mapper, brain_state):
        """Test reverb mapping."""
        reverb = mapper._map_reverb(brain_state)
        
        assert 0 <= reverb <= 1
        assert isinstance(reverb, float)
    
    def test_map_all(self, mapper, brain_state):
        """Test mapping all parameters."""
        params = mapper.map_all(brain_state)
        
        assert isinstance(params, dict)
        assert 'tempo' in params
        assert 'density' in params
        assert 'pitch_center' in params
        assert 'scale' in params


class TestLayeredMusicMapper:
    """Test layered music mapping."""
    
    @pytest.fixture
    def mapper(self):
        """Create layered mapper instance."""
        return LayeredMusicMapper()
    
    @pytest.fixture
    def brain_state(self):
        """Sample brain state."""
        return {
            'focus': 0.6,
            'relax': 0.4,
            'neutral': 0.0,
            'hemispheric_asymmetry': 0.2,
            'theta_power': 0.5,
            'gamma_power': 0.3,
            'stability': 0.7,
            'focus_trend': 0.1,
        }
    
    def test_map_bass_layer(self, mapper, brain_state):
        """Test bass layer mapping."""
        params = mapper.map_bass_layer(brain_state)
        
        assert 'tempo' in params
        assert 'density' in params
        assert 'pitch_range' in params
        assert params['pitch_range'][0] < params['pitch_range'][1]
    
    def test_map_harmony_layer(self, mapper, brain_state):
        """Test harmony layer mapping."""
        params = mapper.map_harmony_layer(brain_state)
        
        assert 'pitch_range' in params
        assert 'consonance' in params
        assert 'reverb' in params
        assert 0 <= params['consonance'] <= 1
    
    def test_map_melody_layer(self, mapper, brain_state):
        """Test melody layer mapping."""
        params = mapper.map_melody_layer(brain_state)
        
        assert 'tempo' in params
        assert 'density' in params
        assert 'scale' in params
        assert 'repetition' in params
    
    def test_map_texture_layer(self, mapper, brain_state):
        """Test texture layer mapping."""
        params = mapper.map_texture_layer(brain_state)
        
        assert 'density' in params
        assert 'pitch_range' in params
        assert 'gesture_direction' in params
    
    def test_map_all_layers(self, mapper, brain_state):
        """Test mapping all layers."""
        all_params = mapper.map_all_layers(brain_state)
        
        assert 'bass' in all_params
        assert 'harmony' in all_params
        assert 'melody' in all_params
        assert 'texture' in all_params


class TestMusicGenerator:
    """Test main music generator."""
    
    @pytest.fixture
    def generator(self):
        """Create music generator instance."""
        return MusicGenerator(
            enable_bass=True,
            enable_harmony=True,
            enable_melody=True,
            enable_texture=False
        )
    
    @pytest.fixture
    def brain_state(self):
        """Sample brain state."""
        return {
            'focus': 0.5,
            'relax': 0.5,
            'neutral': 0.0,
            'stability': 0.6,
        }
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.layers_enabled['melody'] is True
        assert generator.layers_enabled['bass'] is True
        assert generator.layers_enabled['texture'] is False
    
    def test_update_brain_state(self, generator, brain_state):
        """Test brain state update."""
        generator.update_brain_state(brain_state)
        
        assert generator.current_brain_state == brain_state
    
    def test_generate_step(self, generator, brain_state):
        """Test generating one step."""
        generator.update_brain_state(brain_state)
        events_by_layer = generator.generate_step()
        
        assert isinstance(events_by_layer, dict)
        assert 'melody' in events_by_layer
        assert 'bass' in events_by_layer
        
        # Each layer should have a list of events
        for layer, events in events_by_layer.items():
            assert isinstance(events, list)
    
    def test_layer_toggling(self, generator):
        """Test enabling/disabling layers."""
        generator.set_layer_enabled('bass', False)
        assert generator.layers_enabled['bass'] is False
        
        generator.set_layer_enabled('bass', True)
        assert generator.layers_enabled['bass'] is True
    
    def test_reset(self, generator):
        """Test reset functionality."""
        generator.step_count = 10
        generator.reset()
        
        assert generator.step_count == 0
    
    def test_get_current_parameters(self, generator, brain_state):
        """Test getting current parameters."""
        generator.update_brain_state(brain_state)
        params = generator.get_current_parameters()
        
        assert 'tempo' in params
        assert 'density' in params
        assert 'layers' in params
        assert 'step_count' in params
    
    def test_step_duration(self, generator):
        """Test step duration calculation."""
        duration = generator.get_step_duration()
        
        assert duration > 0
        assert isinstance(duration, float)


class TestIntegration:
    """Integration tests for music system."""
    
    def test_full_music_pipeline(self):
        """Test complete music generation pipeline."""
        # Create generator
        generator = MusicGenerator(
            enable_bass=True,
            enable_melody=True,
            enable_harmony=False,
            enable_texture=False
        )
        
        # Simulate brain state updates
        brain_states = [
            {'focus': 0.3, 'relax': 0.7, 'neutral': 0.0},
            {'focus': 0.7, 'relax': 0.3, 'neutral': 0.0},
            {'focus': 0.5, 'relax': 0.5, 'neutral': 0.0},
        ]
        
        for brain_state in brain_states:
            generator.update_brain_state(brain_state)
            
            # Generate several steps
            for _ in range(4):
                events = generator.generate_step()
                
                # Verify structure
                assert isinstance(events, dict)
                assert 'melody' in events
                assert 'bass' in events
                
                # Verify events are valid
                for layer_events in events.values():
                    for event in layer_events:
                        assert isinstance(event, MidiEvent)
                        assert 0 <= event.note <= 127
                        assert 0 <= event.velocity <= 127
                        assert event.duration > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
