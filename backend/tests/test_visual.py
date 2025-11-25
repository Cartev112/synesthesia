"""
Tests for visual parameter generation.
"""

import pytest
import numpy as np

from backend.visual.parameter_generator import VisualParameterGenerator
from backend.visual.algorithms import (
    LissajousGenerator,
    HarmonographGenerator,
    FourierEpicycleGenerator,
    VisualAlgorithmFactory
)


class TestVisualParameterGenerator:
    """Test visual parameter generator."""
    
    @pytest.fixture
    def generator(self):
        return VisualParameterGenerator()
    
    @pytest.fixture
    def brain_state(self):
        return {
            'focus': 0.7,
            'relax': 0.3,
            'neutral': 0.0,
            'hemispheric_asymmetry': 0.2,
            'theta_power': 0.6,
            'alpha_power': 0.5,
            'beta_power': 0.8,
            'stability': 0.7
        }
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.last_params is None
        assert generator.smoothing_factor == 0.3
    
    def test_generate_params(self, generator, brain_state):
        """Test parameter generation."""
        params = generator.generate_params(brain_state)
        
        # Check all required keys present
        assert 'timestamp' in params
        assert 'frequency_ratio_x' in params
        assert 'frequency_ratio_y' in params
        assert 'phase_offset' in params
        assert 'hue_base' in params
        assert 'saturation' in params
        assert 'brightness' in params
        assert 'num_harmonics' in params
        assert 'point_density' in params
        assert 'brain_state' in params
    
    def test_focus_affects_complexity(self, generator):
        """Test that focus increases complexity."""
        # Low focus
        low_focus = generator.generate_params({'focus': 0.1, 'relax': 0.5})
        
        # High focus
        high_focus = generator.generate_params({'focus': 0.9, 'relax': 0.5})
        
        # High focus should have more harmonics
        assert high_focus['num_harmonics'] > low_focus['num_harmonics']
        
        # High focus should have higher frequency ratios
        assert high_focus['frequency_ratio_x'] > low_focus['frequency_ratio_x']
        
        # High focus should have more points
        assert high_focus['point_density'] > low_focus['point_density']
    
    def test_relax_affects_trails(self, generator):
        """Test that relax affects trail length."""
        # Low relax
        low_relax = generator.generate_params({'focus': 0.5, 'relax': 0.1})
        
        # High relax
        high_relax = generator.generate_params({'focus': 0.5, 'relax': 0.9})
        
        # High relax should have longer trails
        assert high_relax['trail_length'] > low_relax['trail_length']
        
        # High relax should have less distortion
        assert high_relax['distortion_amount'] < low_relax['distortion_amount']
    
    def test_asymmetry_affects_rotation(self):
        """Test that hemispheric asymmetry affects rotation."""
        # Create fresh generator (no smoothing history)
        gen = VisualParameterGenerator()
        
        # Negative asymmetry
        neg_asym = gen.generate_params({
            'focus': 0.5,
            'relax': 0.5,
            'hemispheric_asymmetry': -0.5
        })
        
        # Create another fresh generator
        gen2 = VisualParameterGenerator()
        
        # Positive asymmetry
        pos_asym = gen2.generate_params({
            'focus': 0.5,
            'relax': 0.5,
            'hemispheric_asymmetry': 0.5
        })
        
        # Should have opposite rotation speeds
        assert neg_asym['rotation_speed'] < 0
        assert pos_asym['rotation_speed'] > 0
    
    def test_theta_affects_hue(self, generator):
        """Test that theta power affects hue."""
        # Different theta values
        low_theta = generator.generate_params({
            'focus': 0.5,
            'relax': 0.5,
            'theta_power': 0.1
        })
        
        high_theta = generator.generate_params({
            'focus': 0.5,
            'relax': 0.5,
            'theta_power': 0.9
        })
        
        # Hues should be different
        assert low_theta['hue_base'] != high_theta['hue_base']
    
    def test_smoothing(self, generator, brain_state):
        """Test parameter smoothing."""
        # Generate first params
        params1 = generator.generate_params(brain_state)
        
        # Change brain state significantly
        brain_state2 = {
            'focus': 0.1,
            'relax': 0.9,
            'neutral': 0.0,
            'hemispheric_asymmetry': -0.5,
            'stability': 0.3
        }
        
        # Generate second params (should be smoothed)
        params2 = generator.generate_params(brain_state2)
        
        # Params should be different but not drastically
        # (smoothing prevents sudden jumps)
        freq_diff = abs(params2['frequency_ratio_x'] - params1['frequency_ratio_x'])
        assert freq_diff < 2.0  # Should not jump by more than 2
    
    def test_set_smoothing(self, generator):
        """Test setting smoothing factor."""
        generator.set_smoothing(0.5)
        assert generator.smoothing_factor == 0.5
        
        generator.set_smoothing(1.5)  # Should clamp to 1.0
        assert generator.smoothing_factor == 1.0
        
        generator.set_smoothing(-0.5)  # Should clamp to 0.0
        assert generator.smoothing_factor == 0.0
    
    def test_reset(self, generator, brain_state):
        """Test generator reset."""
        # Generate params
        generator.generate_params(brain_state)
        assert generator.last_params is not None
        
        # Reset
        generator.reset()
        assert generator.last_params is None
    
    def test_presets(self, generator):
        """Test preset parameters."""
        # Test each preset
        for preset in ['calm', 'energetic', 'meditative']:
            params = generator.get_preset_params(preset)
            
            assert 'timestamp' in params
            assert 'frequency_ratio_x' in params
            assert 'hue_base' in params
            assert 'brain_state' in params
    
    def test_unknown_preset(self, generator):
        """Test unknown preset defaults to calm."""
        params = generator.get_preset_params('unknown_preset')
        
        # Should get calm preset
        assert params is not None
        assert 'hue_base' in params


class TestLissajousGenerator:
    """Test Lissajous curve generator."""
    
    @pytest.fixture
    def generator(self):
        return LissajousGenerator()
    
    @pytest.fixture
    def params(self):
        return {
            'frequency_ratio_x': 3.0,
            'frequency_ratio_y': 2.0,
            'phase_offset': 0.0,
            'amplitude_x': 1.0,
            'amplitude_y': 1.0
        }
    
    def test_generate_points(self, generator, params):
        """Test point generation."""
        points = generator.generate_points(params, num_points=100, duration=1.0)
        
        assert len(points) == 100
        assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
        
        # Check points are in valid range
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        assert all(-1.5 <= x <= 1.5 for x in x_vals)
        assert all(-1.5 <= y <= 1.5 for y in y_vals)
    
    def test_generate_formula(self, generator, params):
        """Test formula generation."""
        formula = generator.generate_formula(params)
        
        assert 'x' in formula
        assert 'y' in formula
        assert 'type' in formula
        assert formula['type'] == 'lissajous'


class TestHarmonographGenerator:
    """Test harmonograph generator."""
    
    @pytest.fixture
    def generator(self):
        return HarmonographGenerator()
    
    @pytest.fixture
    def params(self):
        return {
            'num_harmonics': 4,
            'damping_x': 0.02,
            'damping_y': 0.02,
            'phase_offset': 0.0
        }
    
    def test_generate_points(self, generator, params):
        """Test point generation."""
        points = generator.generate_points(params, num_points=200, duration=5.0)
        
        assert len(points) == 200
        assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
        
        # Check normalization
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        assert max(abs(x) for x in x_vals) <= 1.1  # Allow small margin
        assert max(abs(y) for y in y_vals) <= 1.1
    
    def test_damping_effect(self, generator):
        """Test that damping reduces amplitude over time."""
        params = {
            'num_harmonics': 2,
            'damping_x': 0.1,
            'damping_y': 0.1,
            'phase_offset': 0.0
        }
        
        points = generator.generate_points(params, num_points=1000, duration=10.0)
        
        # Early points should have larger amplitude than late points
        early_amp = max(abs(p[0]) for p in points[:100])
        late_amp = max(abs(p[0]) for p in points[-100:])
        
        assert early_amp > late_amp
    
    def test_generate_formula(self, generator, params):
        """Test formula generation."""
        formula = generator.generate_formula(params)
        
        assert 'x' in formula
        assert 'y' in formula
        assert 'type' in formula
        assert formula['type'] == 'harmonograph'
        assert 'num_harmonics' in formula


class TestFourierEpicycleGenerator:
    """Test Fourier epicycle generator."""
    
    @pytest.fixture
    def generator(self):
        return FourierEpicycleGenerator()
    
    @pytest.fixture
    def params(self):
        return {
            'num_epicycles': 5,
            'epicycle_decay': 0.7,
            'rotation_speed': 0.5
        }
    
    def test_generate_points(self, generator, params):
        """Test point generation."""
        points = generator.generate_points(params, num_points=150, duration=3.0)
        
        assert len(points) == 150
        assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
        
        # Check normalization
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        assert max(abs(x) for x in x_vals) <= 1.1
        assert max(abs(y) for y in y_vals) <= 1.1
    
    def test_generate_formula(self, generator, params):
        """Test formula generation."""
        formula = generator.generate_formula(params)
        
        assert 'z' in formula
        assert 'type' in formula
        assert formula['type'] == 'fourier_epicycle'
        assert 'num_epicycles' in formula


class TestVisualAlgorithmFactory:
    """Test visual algorithm factory."""
    
    def test_create_lissajous(self):
        """Test creating Lissajous generator."""
        gen = VisualAlgorithmFactory.create('lissajous')
        assert isinstance(gen, LissajousGenerator)
    
    def test_create_harmonograph(self):
        """Test creating harmonograph generator."""
        gen = VisualAlgorithmFactory.create('harmonograph')
        assert isinstance(gen, HarmonographGenerator)
    
    def test_create_epicycle(self):
        """Test creating epicycle generator."""
        gen = VisualAlgorithmFactory.create('epicycle')
        assert isinstance(gen, FourierEpicycleGenerator)
        
        # Test alias
        gen2 = VisualAlgorithmFactory.create('fourier')
        assert isinstance(gen2, FourierEpicycleGenerator)
    
    def test_create_unknown(self):
        """Test creating unknown algorithm defaults to Lissajous."""
        gen = VisualAlgorithmFactory.create('unknown')
        assert isinstance(gen, LissajousGenerator)
    
    def test_get_available_algorithms(self):
        """Test getting available algorithms."""
        algorithms = VisualAlgorithmFactory.get_available_algorithms()
        
        assert len(algorithms) == 3
        assert all('id' in alg for alg in algorithms)
        assert all('name' in alg for alg in algorithms)
        assert all('description' in alg for alg in algorithms)


class TestIntegration:
    """Integration tests."""
    
    def test_full_visual_pipeline(self):
        """Test complete visual generation pipeline."""
        # Create generator
        param_gen = VisualParameterGenerator()
        
        # Generate parameters from brain state
        brain_state = {
            'focus': 0.8,
            'relax': 0.2,
            'neutral': 0.0,
            'hemispheric_asymmetry': 0.3,
            'theta_power': 0.6,
            'stability': 0.7
        }
        
        params = param_gen.generate_params(brain_state)
        
        # Generate visual points
        liss_gen = LissajousGenerator()
        points = liss_gen.generate_points(params, num_points=500)
        
        assert len(points) == 500
        assert all(isinstance(p, tuple) for p in points)
    
    def test_different_algorithms_same_params(self):
        """Test different algorithms with same parameters."""
        params = {
            'frequency_ratio_x': 3.0,
            'frequency_ratio_y': 2.0,
            'phase_offset': 0.5,
            'num_harmonics': 4,
            'damping_x': 0.02,
            'damping_y': 0.02,
            'num_epicycles': 5,
            'epicycle_decay': 0.7,
            'rotation_speed': 0.5
        }
        
        # Generate with each algorithm
        liss = LissajousGenerator().generate_points(params, num_points=100)
        harm = HarmonographGenerator().generate_points(params, num_points=100)
        epic = FourierEpicycleGenerator().generate_points(params, num_points=100)
        
        # All should generate valid points
        assert len(liss) == 100
        assert len(harm) == 100
        assert len(epic) == 100
        
        # But patterns should be different
        assert liss != harm
        assert harm != epic


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
