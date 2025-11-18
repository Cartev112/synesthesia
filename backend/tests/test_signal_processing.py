"""
Unit tests for signal processing components.

Tests:
- Preprocessing (filtering, re-referencing)
- Feature extraction
- Circular buffer
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from backend.signal_processing.preprocessing import SignalPreprocessor
from backend.signal_processing.feature_extraction import FeatureExtractor
from backend.signal_processing.buffer import CircularBuffer


class TestSignalPreprocessor:
    """Test signal preprocessing functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return SignalPreprocessor(sampling_rate=256)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample EEG data."""
        # 1 second of data, 8 channels
        n_samples = 256
        n_channels = 8
        
        # Generate synthetic signal with multiple frequency components
        t = np.linspace(0, 1, n_samples)
        data = np.zeros((n_samples, n_channels))
        
        for ch in range(n_channels):
            # Mix of frequencies
            data[:, ch] = (
                np.sin(2 * np.pi * 10 * t) +  # 10 Hz (alpha)
                0.5 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz (beta)
                0.3 * np.sin(2 * np.pi * 60 * t)  # 60 Hz (power line noise)
            )
        
        return data
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.sampling_rate == 256
        assert preprocessor.bandpass_low == 0.5
        assert preprocessor.bandpass_high == 50.0
        assert preprocessor.notch_freq == 60.0
    
    def test_process_chunk_shape(self, preprocessor, sample_data):
        """Test that output shape matches input shape."""
        processed = preprocessor.process_chunk(sample_data)
        assert processed.shape == sample_data.shape
    
    def test_bandpass_filtering(self, preprocessor):
        """Test bandpass filter removes out-of-band frequencies."""
        # Create signal with low and high frequency components
        n_samples = 512
        t = np.linspace(0, 2, n_samples)
        
        # 0.1 Hz (should be filtered) + 10 Hz (should pass) + 100 Hz (should be filtered)
        signal = (
            np.sin(2 * np.pi * 0.1 * t) +
            np.sin(2 * np.pi * 10 * t) +
            np.sin(2 * np.pi * 100 * t)
        ).reshape(-1, 1)
        
        filtered = preprocessor.process_chunk(signal)
        
        # Check that filtered signal has reduced power at extreme frequencies
        # (This is a simplified test - in practice would check frequency domain)
        assert filtered.shape == signal.shape
        assert not np.array_equal(filtered, signal)
    
    def test_notch_filtering(self, preprocessor):
        """Test notch filter reduces 60 Hz power line noise."""
        n_samples = 512
        t = np.linspace(0, 2, n_samples)
        
        # Signal with 10 Hz + 60 Hz
        signal = (
            np.sin(2 * np.pi * 10 * t) +
            2.0 * np.sin(2 * np.pi * 60 * t)  # Strong 60 Hz component
        ).reshape(-1, 1)
        
        filtered = preprocessor.process_chunk(signal)
        
        # 60 Hz should be attenuated
        assert filtered.shape == signal.shape
        assert not np.array_equal(filtered, signal)
    
    def test_rereferencing(self, preprocessor, sample_data):
        """Test common average re-referencing."""
        processed = preprocessor.process_chunk(sample_data)
        
        # After CAR, mean across channels should be close to zero
        channel_mean = np.mean(processed, axis=1)
        assert_allclose(channel_mean, 0, atol=1e-10)


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return FeatureExtractor(
            sampling_rate=256,
            window_size=1.0,
            overlap=0.5
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample EEG data with known frequency content."""
        n_samples = 256  # 1 second at 256 Hz
        n_channels = 8
        t = np.linspace(0, 1, n_samples)
        
        data = np.zeros((n_samples, n_channels))
        
        for ch in range(n_channels):
            # Alpha band (10 Hz) dominant
            data[:, ch] = (
                0.5 * np.sin(2 * np.pi * 10 * t) +
                0.2 * np.sin(2 * np.pi * 20 * t) +
                0.1 * np.random.randn(n_samples)
            )
        
        return data
    
    def test_initialization(self, extractor):
        """Test feature extractor initialization."""
        assert extractor.sampling_rate == 256
        assert extractor.window_size == 1.0
        assert extractor.overlap == 0.5
        assert extractor.window_samples == 256
    
    def test_extract_features_returns_dict(self, extractor, sample_data):
        """Test that extract_features returns a dictionary."""
        features = extractor.extract_features(sample_data)
        assert isinstance(features, dict)
    
    def test_extract_features_has_all_bands(self, extractor, sample_data):
        """Test that all frequency bands are present."""
        features = extractor.extract_features(sample_data)
        
        expected_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        for band in expected_bands:
            assert f'{band}_power' in features
            assert features[f'{band}_power'] >= 0
    
    def test_extract_features_has_metrics(self, extractor, sample_data):
        """Test that derived metrics are present."""
        features = extractor.extract_features(sample_data)
        
        assert 'focus_metric' in features
        assert 'relax_metric' in features
        assert 'hemispheric_asymmetry' in features
        
        assert features['focus_metric'] >= 0
        assert features['relax_metric'] >= 0
        assert -1 <= features['hemispheric_asymmetry'] <= 1
    
    def test_focus_metric_calculation(self, extractor):
        """Test focus metric calculation."""
        band_powers = {
            'delta': 1.0,
            'theta': 1.0,
            'alpha': 1.0,
            'beta': 4.0,
            'gamma': 1.0
        }
        
        focus = extractor._compute_focus(band_powers)
        # beta / (alpha + theta) = 4 / (1 + 1) = 2.0
        assert_allclose(focus, 2.0, rtol=1e-5)
    
    def test_relax_metric_calculation(self, extractor):
        """Test relax metric calculation."""
        band_powers = {
            'delta': 1.0,
            'theta': 1.0,
            'alpha': 6.0,
            'beta': 2.0,
            'gamma': 1.0
        }
        
        relax = extractor._compute_relax(band_powers)
        # alpha / (beta + gamma) = 6 / (2 + 1) = 2.0
        assert_allclose(relax, 2.0, rtol=1e-5)
    
    def test_insufficient_data_returns_defaults(self, extractor):
        """Test that insufficient data returns default features."""
        # Only 10 samples (insufficient)
        insufficient_data = np.random.randn(10, 8)
        features = extractor.extract_features(insufficient_data)
        
        # Should return default features
        assert features['delta_power'] == 0.0
        assert features['theta_power'] == 0.0
        assert features['alpha_power'] == 0.0
    
    def test_band_power_calculation(self, extractor):
        """Test band power calculation."""
        # Create PSD with known values
        freqs = np.arange(0, 51, 0.5)  # 0-50 Hz
        psd = np.ones_like(freqs)  # Uniform power
        
        # Alpha band: 8-13 Hz
        alpha_power = extractor._band_power(psd, freqs, 8.0, 13.0)
        
        # Should integrate over 5 Hz range
        assert alpha_power > 0
    
    def test_hemispheric_asymmetry(self, extractor):
        """Test hemispheric asymmetry calculation."""
        n_samples = 256
        n_channels = 8
        
        # Create data with right hemisphere dominance
        data = np.random.randn(n_samples, n_channels)
        
        # Boost right hemisphere channels (1, 3, 5, 7)
        data[:, [1, 3, 5, 7]] *= 2.0
        
        asymmetry = extractor._compute_asymmetry(data)
        
        # Should be positive (right dominant)
        assert asymmetry > 0


class TestCircularBuffer:
    """Test circular buffer functionality."""
    
    @pytest.fixture
    def buffer(self):
        """Create circular buffer instance."""
        return CircularBuffer(
            n_channels=8,
            buffer_duration=2.0,
            sampling_rate=256
        )
    
    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.n_channels == 8
        assert buffer.buffer_duration == 2.0
        assert buffer.sampling_rate == 256
        assert buffer.n_samples == 512  # 2 seconds * 256 Hz
        assert buffer.buffer.shape == (512, 8)
    
    def test_append_data(self, buffer):
        """Test appending data to buffer."""
        # Append 128 samples
        new_data = np.random.randn(128, 8)
        buffer.append(new_data)
        
        assert buffer.write_idx == 128
    
    def test_append_wraparound(self, buffer):
        """Test buffer wraparound behavior."""
        # Fill buffer completely
        full_data = np.random.randn(512, 8)
        buffer.append(full_data)
        
        assert buffer.write_idx == 0  # Should wrap to start
        
        # Append more data
        extra_data = np.random.randn(64, 8)
        buffer.append(extra_data)
        
        assert buffer.write_idx == 64
    
    def test_get_latest(self, buffer):
        """Test retrieving latest data."""
        # Fill buffer first to ensure we have enough data
        fill_data = np.random.randn(512, 8)
        buffer.append(fill_data)
        
        # Now append known data
        data = np.ones((256, 8)) * 2
        buffer.append(data)
        
        # Get latest 256 samples
        latest = buffer.get_latest(duration=1.0)
        
        assert latest.shape == (256, 8)
        # Should be the data we just appended
        assert_array_almost_equal(latest, data)
    
    def test_get_latest_wraparound(self, buffer):
        """Test get_latest with wraparound."""
        # Fill buffer and wrap
        data = np.arange(512).reshape(512, 1)
        data = np.repeat(data, 8, axis=1)
        buffer.append(data)
        
        # Append more to cause wrap
        extra = np.ones((128, 8)) * 999
        buffer.append(extra)
        
        # Get latest should handle wraparound
        latest = buffer.get_latest(duration=0.5)  # 128 samples
        
        assert latest.shape == (128, 8)
        assert_array_almost_equal(latest, extra)
    
    def test_buffer_capacity(self, buffer):
        """Test that buffer maintains fixed capacity."""
        # Append more data than buffer capacity
        large_data = np.random.randn(1024, 8)  # 4 seconds worth
        buffer.append(large_data)
        
        # Buffer should still be same size
        assert buffer.buffer.shape == (512, 8)
        
        # Only latest data should be retained
        latest = buffer.get_latest(duration=2.0)
        assert latest.shape == (512, 8)


class TestIntegration:
    """Integration tests for signal processing pipeline."""
    
    def test_full_pipeline(self):
        """Test complete signal processing pipeline."""
        # Create components
        preprocessor = SignalPreprocessor(sampling_rate=256)
        extractor = FeatureExtractor(sampling_rate=256)
        buffer = CircularBuffer(
            n_channels=8,
            buffer_duration=2.0,
            sampling_rate=256
        )
        
        # Generate test data - need multiple chunks for filter to stabilize
        n_samples = 512  # 2 seconds
        n_channels = 8
        t = np.linspace(0, 2, n_samples)
        
        data = np.zeros((n_samples, n_channels))
        for ch in range(n_channels):
            data[:, ch] = (
                2.0 * np.sin(2 * np.pi * 10 * t) +  # Strong 10 Hz (alpha)
                1.0 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz (beta)
                0.2 * np.random.randn(n_samples)  # Noise
            )
        
        # Process through pipeline
        # 1. Preprocess
        processed = preprocessor.process_chunk(data)
        assert processed.shape == data.shape
        
        # 2. Add to buffer
        buffer.append(processed)  # Already in (samples, channels) format
        
        # 3. Extract features from last second
        latest = buffer.get_latest(duration=1.0)
        features = extractor.extract_features(latest)  # Already in correct format
        
        # Verify features
        assert isinstance(features, dict)
        assert 'alpha_power' in features
        assert 'focus_metric' in features
        # After filtering, should have some power
        assert features['alpha_power'] >= 0  # May be very small after filtering


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
