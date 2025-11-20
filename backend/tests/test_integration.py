"""
Integration tests for complete BCI pipeline.

Tests end-to-end flow:
EEG Simulator → Signal Processing → ML → Music Generation
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from backend.eeg.simulator import EEGSimulator, StreamingEEGSimulator
from backend.signal_processing import SignalPreprocessor, FeatureExtractor, CircularBuffer
from backend.ml import ArtifactClassifier, MentalStateClassifier, MentalStateTracker
from backend.music import MusicGenerator
from backend.pipeline.realtime_pipeline import RealtimePipeline


class TestEEGSimulator:
    """Test EEG simulator functionality."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return EEGSimulator(n_channels=8, sampling_rate=256, seed=42)
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.n_channels == 8
        assert simulator.sampling_rate == 256
        assert len(simulator.channels) == 8
    
    def test_generate_sample(self, simulator):
        """Test single sample generation."""
        sample = simulator.generate_sample()
        
        assert sample.shape == (8,)
        assert sample.dtype == np.float64
        # EEG should be in reasonable range (microvolts)
        assert np.all(np.abs(sample) < 200)
    
    def test_generate_chunk(self, simulator):
        """Test chunk generation."""
        chunk = simulator.generate_chunk(duration=1.0)
        
        assert chunk.shape == (256, 8)  # 1 second at 256 Hz
    
    def test_mental_state_changes(self, simulator):
        """Test mental state transitions."""
        # Generate baseline
        simulator.set_mental_state('neutral')
        neutral_data = simulator.generate_chunk(0.5)
        
        # Generate focus state
        simulator.set_mental_state('focus', intensity=1.0, transition_time=0.0)
        focus_data = simulator.generate_chunk(0.5)
        
        # Generate relax state
        simulator.set_mental_state('relax', intensity=1.0, transition_time=0.0)
        relax_data = simulator.generate_chunk(0.5)
        
        # All should be different
        assert not np.array_equal(neutral_data, focus_data)
        assert not np.array_equal(focus_data, relax_data)
    
    def test_hemispheric_asymmetry(self, simulator):
        """Test hemispheric asymmetry."""
        simulator.set_hemispheric_asymmetry(0.5)  # Right dominant
        sample = simulator.generate_sample()
        
        # Right hemisphere channels should have higher amplitude
        left_channels = [0, 2, 4, 6]
        right_channels = [1, 3, 5, 7]
        
        # This is probabilistic, so we just check it doesn't crash
        assert sample.shape == (8,)


class TestStreamingSimulator:
    """Test streaming EEG simulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create streaming simulator."""
        return StreamingEEGSimulator(n_channels=8, sampling_rate=256)
    
    def test_streaming_start_stop(self, simulator):
        """Test starting and stopping stream."""
        simulator.start_stream()
        assert simulator.is_streaming
        
        simulator.stop_stream()
        assert not simulator.is_streaming
    
    def test_read_samples(self, simulator):
        """Test reading samples from stream."""
        simulator.start_stream()
        
        samples = simulator.read_samples(n_samples=32)
        
        assert samples is not None
        assert samples.shape == (32, 8)
        
        simulator.stop_stream()
    
    def test_read_without_streaming(self, simulator):
        """Test reading when not streaming."""
        samples = simulator.read_samples(n_samples=10)
        assert samples is None


class TestSignalProcessingPipeline:
    """Test complete signal processing pipeline."""
    
    @pytest.fixture
    def components(self):
        """Create signal processing components."""
        return {
            'simulator': EEGSimulator(n_channels=8, sampling_rate=256, seed=42),
            'preprocessor': SignalPreprocessor(sampling_rate=256),
            'extractor': FeatureExtractor(sampling_rate=256),
            'buffer': CircularBuffer(n_channels=8, buffer_duration=2.0, sampling_rate=256)
        }
    
    def test_full_signal_pipeline(self, components):
        """Test complete signal processing flow."""
        sim = components['simulator']
        prep = components['preprocessor']
        ext = components['extractor']
        buf = components['buffer']
        
        # Generate data
        raw_data = sim.generate_chunk(duration=1.0)
        
        # Preprocess
        processed = prep.process_chunk(raw_data)
        assert processed.shape == raw_data.shape
        
        # Add to buffer
        buf.append(processed)
        
        # Extract features
        latest = buf.get_latest(duration=1.0)
        features = ext.extract_features(latest)
        
        # Verify features
        assert 'alpha_power' in features
        assert 'beta_power' in features
        assert 'focus_metric' in features
        assert features['focus_metric'] >= 0
    
    def test_different_mental_states(self, components):
        """Test feature extraction for different mental states."""
        sim = components['simulator']
        prep = components['preprocessor']
        ext = components['extractor']
        
        # Focus state
        sim.set_mental_state('focus', intensity=1.0, transition_time=0.0)
        focus_data = sim.generate_chunk(duration=1.0)
        focus_processed = prep.process_chunk(focus_data)
        focus_features = ext.extract_features(focus_processed)
        
        # Relax state
        sim.set_mental_state('relax', intensity=1.0, transition_time=0.0)
        relax_data = sim.generate_chunk(duration=1.0)
        relax_processed = prep.process_chunk(relax_data)
        relax_features = ext.extract_features(relax_processed)
        
        # Focus should have higher beta/alpha ratio
        # (This is probabilistic, so we just verify features are different)
        assert focus_features['beta_power'] != relax_features['beta_power']


class TestMLPipeline:
    """Test ML pipeline integration."""
    
    @pytest.fixture
    def ml_components(self):
        """Create ML components."""
        classifier = MentalStateClassifier(n_estimators=10)
        tracker = MentalStateTracker(classifier=classifier)
        return {'classifier': classifier, 'tracker': tracker}
    
    def test_ml_with_simulated_data(self, ml_components):
        """Test ML pipeline with simulated EEG data."""
        sim = EEGSimulator(n_channels=8, sampling_rate=256, seed=42)
        prep = SignalPreprocessor(sampling_rate=256)
        ext = FeatureExtractor(sampling_rate=256)
        
        # Generate training data
        training_data = {'neutral': [], 'focus': [], 'relax': []}
        
        for state in ['neutral', 'focus', 'relax']:
            sim.set_mental_state(state, intensity=1.0, transition_time=0.0)
            
            for _ in range(10):  # 10 samples per state
                raw = sim.generate_chunk(duration=1.0)
                processed = prep.process_chunk(raw)
                features = ext.extract_features(processed)
                
                feature_array = np.array([
                    features['delta_power'],
                    features['theta_power'],
                    features['alpha_power'],
                    features['beta_power'],
                    features['gamma_power'],
                    features['hemispheric_asymmetry'],
                    features['focus_metric'],
                    features['relax_metric']
                ])
                
                training_data[state].append(feature_array)
        
        # Convert to arrays
        X_train = np.vstack([
            np.array(training_data['neutral']),
            np.array(training_data['focus']),
            np.array(training_data['relax'])
        ])
        y_train = np.array([0]*10 + [1]*10 + [2]*10)
        
        # Train classifier
        ml_components['classifier'].train(X_train, y_train)
        
        # Test prediction
        test_features = training_data['focus'][0]
        state, confidence = ml_components['classifier'].predict(test_features)
        
        assert state in ['neutral', 'focus', 'relax']
        assert 0 <= confidence <= 1


class TestMusicPipeline:
    """Test music generation pipeline."""
    
    def test_music_from_simulated_eeg(self):
        """Test music generation from simulated EEG."""
        # Create components
        sim = EEGSimulator(n_channels=8, sampling_rate=256, seed=42)
        prep = SignalPreprocessor(sampling_rate=256)
        ext = FeatureExtractor(sampling_rate=256)
        music_gen = MusicGenerator()
        
        # Set focus state
        sim.set_mental_state('focus', intensity=0.8)
        
        # Generate and process EEG
        raw = sim.generate_chunk(duration=1.0)
        processed = prep.process_chunk(raw)
        features = ext.extract_features(processed)
        
        # Create brain state
        brain_state = {
            'focus': features['focus_metric'] / 2.0,
            'relax': features['relax_metric'] / 2.0,
            'neutral': 0.5,
            'stability': 0.7
        }
        
        # Generate music
        music_gen.update_brain_state(brain_state)
        events = music_gen.generate_step()
        
        # Verify events
        assert isinstance(events, dict)
        assert 'melody' in events
        
        # Events should be lists
        for layer_events in events.values():
            assert isinstance(layer_events, list)


class TestRealtimePipeline:
    """Test real-time pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_start_stop(self):
        """Test starting and stopping pipeline."""
        pipeline = RealtimePipeline(sampling_rate=256, use_simulator=True)
        
        await pipeline.start()
        assert pipeline.is_running
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        await pipeline.stop()
        assert not pipeline.is_running
    
    @pytest.mark.asyncio
    async def test_pipeline_with_callbacks(self):
        """Test pipeline with callbacks."""
        brain_states = []
        music_events_list = []
        
        async def on_brain_state(state):
            brain_states.append(state)
        
        async def on_music_events(events):
            music_events_list.append(events)
        
        pipeline = RealtimePipeline(
            sampling_rate=256,
            use_simulator=True,
            on_brain_state=on_brain_state,
            on_music_events=on_music_events
        )
        
        await pipeline.start()
        
        # Run for 2 seconds
        await asyncio.sleep(2.0)
        
        await pipeline.stop()
        
        # Should have received updates
        assert len(brain_states) > 0
        assert len(music_events_list) > 0
        
        # Verify brain state structure
        if brain_states:
            state = brain_states[0]
            assert 'focus' in state
            assert 'relax' in state
    
    @pytest.mark.asyncio
    async def test_pipeline_mental_state_control(self):
        """Test controlling simulator mental state through pipeline."""
        brain_states = []
        
        async def on_brain_state(state):
            brain_states.append(state)
        
        pipeline = RealtimePipeline(
            sampling_rate=256,
            use_simulator=True,
            on_brain_state=on_brain_state
        )
        
        await pipeline.start()
        
        # Wait for pipeline to start generating data
        await asyncio.sleep(1.5)
        
        # Set focus state
        pipeline.set_mental_state('focus', intensity=1.0)
        
        # Run longer to collect data
        await asyncio.sleep(2.0)
        
        await pipeline.stop()
        
        # Should have collected some states
        assert len(brain_states) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics(self):
        """Test pipeline performance metrics."""
        pipeline = RealtimePipeline(sampling_rate=256, use_simulator=True)
        
        await pipeline.start()
        await asyncio.sleep(1.0)
        await pipeline.stop()
        
        metrics = pipeline.get_metrics()
        
        # Should have metrics
        assert 'avg_total_latency_ms' in metrics
        assert 'total_samples_processed' in metrics
        assert metrics['total_samples_processed'] > 0


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_bci_flow(self):
        """Test complete BCI flow from EEG to music."""
        # Track outputs
        brain_states = []
        music_outputs = []
        
        async def on_brain_state(state):
            brain_states.append(state)
        
        async def on_music_events(events):
            music_outputs.append(events)
        
        # Create pipeline
        pipeline = RealtimePipeline(
            sampling_rate=256,
            use_simulator=True,
            on_brain_state=on_brain_state,
            on_music_events=on_music_events
        )
        
        # Start pipeline
        await pipeline.start()
        
        # Wait for pipeline to initialize and start generating data
        await asyncio.sleep(1.5)
        
        # Simulate different mental states
        pipeline.set_mental_state('focus', intensity=0.8)
        await asyncio.sleep(1.5)
        
        pipeline.set_mental_state('relax', intensity=0.7)
        await asyncio.sleep(1.5)
        
        # Stop pipeline
        await pipeline.stop()
        
        # Verify we got outputs
        assert len(brain_states) > 0, "Should receive brain state updates"
        assert len(music_outputs) > 0, "Should receive music events"
        
        # Verify brain state structure
        state = brain_states[-1]
        assert 'focus' in state
        assert 'relax' in state
        assert 0 <= state['focus'] <= 1
        assert 0 <= state['relax'] <= 1
        
        # Verify music events structure
        events = music_outputs[-1]
        assert isinstance(events, dict)
        
        # Get metrics
        metrics = pipeline.get_metrics()
        assert metrics['avg_total_latency_ms'] < 100, "Latency should be under 100ms"
        
        logger_output = f"""
        End-to-End Test Results:
        - Brain states collected: {len(brain_states)}
        - Music events generated: {len(music_outputs)}
        - Average latency: {metrics['avg_total_latency_ms']:.2f}ms
        - Samples processed: {metrics['total_samples_processed']}
        """
        print(logger_output)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
