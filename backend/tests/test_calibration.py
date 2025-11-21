"""
Tests for user calibration system.
"""

import pytest
import numpy as np
from datetime import datetime

from backend.ml.calibration import (
    UserCalibration,
    CalibrationSession,
    CalibrationProtocol
)
from backend.eeg.simulator import EEGSimulator
from backend.signal_processing import SignalPreprocessor, FeatureExtractor


class TestUserCalibration:
    """Test UserCalibration class."""
    
    @pytest.fixture
    def calibration(self):
        """Create calibration instance."""
        return UserCalibration(user_id="test_user_123")
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample features."""
        return np.random.randn(8)
    
    def test_initialization(self, calibration):
        """Test calibration initialization."""
        assert calibration.user_id == "test_user_123"
        assert calibration.baseline_stats is None
        assert calibration.feature_scaler is None
        assert calibration.state_model is None
        assert len(calibration.calibration_data) == 3
    
    def test_add_calibration_sample(self, calibration, sample_features):
        """Test adding calibration samples."""
        calibration.add_calibration_sample('neutral', sample_features)
        
        assert len(calibration.calibration_data['neutral']) == 1
        assert np.array_equal(
            calibration.calibration_data['neutral'][0],
            sample_features
        )
    
    def test_add_sample_invalid_state(self, calibration, sample_features):
        """Test adding sample with invalid state."""
        with pytest.raises(ValueError, match="Invalid state"):
            calibration.add_calibration_sample('invalid', sample_features)
    
    def test_add_sample_wrong_shape(self, calibration):
        """Test adding sample with wrong shape."""
        wrong_features = np.random.randn(10)  # Should be 8
        
        with pytest.raises(ValueError, match="Expected 8 features"):
            calibration.add_calibration_sample('neutral', wrong_features)
    
    def test_compute_baseline_stats(self, calibration):
        """Test baseline statistics computation."""
        # Add neutral samples
        for _ in range(10):
            features = np.random.randn(8)
            calibration.add_calibration_sample('neutral', features)
        
        calibration.compute_baseline_stats()
        
        assert calibration.baseline_stats is not None
        assert 'mean' in calibration.baseline_stats
        assert 'std' in calibration.baseline_stats
        assert calibration.baseline_stats['mean'].shape == (8,)
        assert calibration.baseline_stats['std'].shape == (8,)
    
    def test_compute_baseline_no_data(self, calibration):
        """Test baseline computation without data."""
        with pytest.raises(ValueError, match="No neutral state data"):
            calibration.compute_baseline_stats()
    
    def test_train_success(self, calibration):
        """Test successful model training."""
        # Add samples for all states
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):  # 20 samples per state
                features = np.random.randn(8)
                calibration.add_calibration_sample(state, features)
        
        accuracy, sample_counts = calibration.train()
        
        assert 0 <= accuracy <= 1
        assert sample_counts['neutral'] == 20
        assert sample_counts['focus'] == 20
        assert sample_counts['relax'] == 20
        assert calibration.state_model is not None
        assert calibration.feature_scaler is not None
    
    def test_train_missing_state(self, calibration):
        """Test training with missing state data."""
        # Only add neutral samples
        for _ in range(10):
            calibration.add_calibration_sample('neutral', np.random.randn(8))
        
        with pytest.raises(ValueError, match="No samples for state"):
            calibration.train()
    
    def test_normalize(self, calibration):
        """Test feature normalization."""
        # Add samples and compute baseline
        for _ in range(10):
            calibration.add_calibration_sample('neutral', np.random.randn(8))
        
        calibration.compute_baseline_stats()
        
        # Normalize a feature vector
        features = np.random.randn(8)
        normalized = calibration.normalize(features)
        
        assert normalized.shape == (8,)
        # Normalized features should have roughly zero mean relative to baseline
        assert np.abs(np.mean(normalized)) < 2  # Within 2 std devs
    
    def test_predict_state(self, calibration):
        """Test state prediction."""
        # Train model
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):
                calibration.add_calibration_sample(state, np.random.randn(8))
        
        calibration.train()
        
        # Predict
        features = np.random.randn(8)
        state, confidence = calibration.predict_state(features)
        
        assert state in ['neutral', 'focus', 'relax']
        assert 0 <= confidence <= 1
    
    def test_predict_without_training(self, calibration):
        """Test prediction without training."""
        features = np.random.randn(8)
        
        with pytest.raises(ValueError, match="Model not trained"):
            calibration.predict_state(features)
    
    def test_get_state_probabilities(self, calibration):
        """Test getting state probabilities."""
        # Train model
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):
                calibration.add_calibration_sample(state, np.random.randn(8))
        
        calibration.train()
        
        # Get probabilities
        features = np.random.randn(8)
        probs = calibration.get_state_probabilities(features)
        
        assert 'neutral' in probs
        assert 'focus' in probs
        assert 'relax' in probs
        
        # Probabilities should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 0.01
    
    def test_save_and_load_session(self, calibration):
        """Test saving and loading calibration session."""
        # Train model
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):
                calibration.add_calibration_sample(state, np.random.randn(8))
        
        calibration.train()
        
        # Save session
        session = calibration.save_session()
        
        assert isinstance(session, CalibrationSession)
        assert session.user_id == "test_user_123"
        assert session.validation_accuracy > 0
        
        # Create new calibration and load
        new_calibration = UserCalibration(user_id="test_user_123")
        new_calibration.load_session(session)
        
        assert new_calibration.is_trained()
        
        # Should be able to predict
        features = np.random.randn(8)
        state, confidence = new_calibration.predict_state(features)
        assert state in ['neutral', 'focus', 'relax']
    
    def test_load_wrong_user(self, calibration):
        """Test loading session for wrong user."""
        # Train and save
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):
                calibration.add_calibration_sample(state, np.random.randn(8))
        
        calibration.train()
        session = calibration.save_session()
        
        # Try to load into different user
        wrong_user = UserCalibration(user_id="different_user")
        
        with pytest.raises(ValueError, match="user_id mismatch"):
            wrong_user.load_session(session)
    
    def test_is_trained(self, calibration):
        """Test is_trained method."""
        assert not calibration.is_trained()
        
        # Train model
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):
                calibration.add_calibration_sample(state, np.random.randn(8))
        
        calibration.train()
        
        assert calibration.is_trained()
    
    def test_get_feature_importance(self, calibration):
        """Test feature importance extraction."""
        # Train model
        for state in ['neutral', 'focus', 'relax']:
            for _ in range(20):
                calibration.add_calibration_sample(state, np.random.randn(8))
        
        calibration.train()
        
        importance = calibration.get_feature_importance()
        
        assert len(importance) == 8
        assert 'alpha_power' in importance
        assert 'beta_power' in importance
        assert 'focus_metric' in importance
        
        # All importances should be non-negative
        for value in importance.values():
            assert value >= 0


class TestCalibrationSession:
    """Test CalibrationSession dataclass."""
    
    def test_session_serialization(self):
        """Test session to_dict and from_dict."""
        # Create a mock session
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, 8))
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(30, 8)
        y = np.array([0]*10 + [1]*10 + [2]*10)
        model.fit(X, y)
        
        session = CalibrationSession(
            user_id="test_user",
            timestamp=datetime.now(),
            baseline_stats={
                'mean': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                'std': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            },
            scaler=scaler,
            model=model,
            validation_accuracy=0.85,
            training_samples={'neutral': 10, 'focus': 10, 'relax': 10}
        )
        
        # Serialize
        data = session.to_dict()
        
        assert data['user_id'] == "test_user"
        assert data['validation_accuracy'] == 0.85
        assert 'scaler' in data
        assert 'model' in data
        
        # Deserialize
        restored = CalibrationSession.from_dict(data)
        
        assert restored.user_id == session.user_id
        assert restored.validation_accuracy == session.validation_accuracy
        assert np.array_equal(
            restored.baseline_stats['mean'],
            session.baseline_stats['mean']
        )


class TestCalibrationProtocol:
    """Test CalibrationProtocol class."""
    
    @pytest.fixture
    def protocol(self):
        """Create protocol instance."""
        return CalibrationProtocol(user_id="test_user")
    
    def test_initialization(self, protocol):
        """Test protocol initialization."""
        assert protocol.user_id == "test_user"
        assert protocol.current_stage is None
        assert protocol.stage_start_time is None
    
    def test_start_stage(self, protocol):
        """Test starting a calibration stage."""
        info = protocol.start_stage('baseline')
        
        assert info['stage'] == 'baseline'
        assert info['state_label'] == 'neutral'
        assert info['duration'] == 60
        assert 'instructions' in info
        assert protocol.current_stage == 'baseline'
        assert protocol.stage_start_time is not None
    
    def test_start_invalid_stage(self, protocol):
        """Test starting invalid stage."""
        with pytest.raises(ValueError, match="Invalid stage"):
            protocol.start_stage('invalid')
    
    def test_add_sample_during_stage(self, protocol):
        """Test adding samples during a stage."""
        protocol.start_stage('baseline')
        
        features = np.random.randn(8)
        protocol.add_sample(features)
        
        # Check sample was added to calibration
        assert len(protocol.calibration.calibration_data['neutral']) == 1
    
    def test_add_sample_no_stage(self, protocol):
        """Test adding sample without active stage."""
        features = np.random.randn(8)
        
        with pytest.raises(ValueError, match="No active stage"):
            protocol.add_sample(features)
    
    def test_add_sample_during_validation(self, protocol):
        """Test that samples aren't collected during validation."""
        protocol.start_stage('validation')
        
        features = np.random.randn(8)
        protocol.add_sample(features)
        
        # No data should be collected
        assert len(protocol.calibration.calibration_data['neutral']) == 0
    
    def test_get_stage_progress(self, protocol):
        """Test getting stage progress."""
        import time
        
        # No active stage
        progress = protocol.get_stage_progress()
        assert not progress['active']
        
        # Start stage
        protocol.start_stage('baseline')
        time.sleep(0.1)
        
        progress = protocol.get_stage_progress()
        assert progress['active']
        assert progress['stage'] == 'baseline'
        assert progress['elapsed'] > 0
        assert progress['remaining'] < 60
        assert 0 < progress['progress'] < 1
    
    def test_train_model(self, protocol):
        """Test training model through protocol."""
        # Add samples for all stages
        for stage, state in [('baseline', 'neutral'), ('focus', 'focus'), ('relax', 'relax')]:
            protocol.start_stage(stage)
            
            for _ in range(20):
                features = np.random.randn(8)
                protocol.add_sample(features)
        
        # Train
        results = protocol.train_model()
        
        assert results['success']
        assert 'validation_accuracy' in results
        assert 'sample_counts' in results
        assert 'training_time' in results
        assert 'feature_importance' in results
        assert protocol.calibration.is_trained()
    
    def test_save_and_load(self, protocol):
        """Test saving and loading through protocol."""
        # Train model
        for stage, state in [('baseline', 'neutral'), ('focus', 'focus'), ('relax', 'relax')]:
            protocol.start_stage(stage)
            for _ in range(20):
                protocol.add_sample(np.random.randn(8))
        
        protocol.train_model()
        
        # Save
        session = protocol.save()
        assert isinstance(session, CalibrationSession)
        
        # Load into new protocol
        new_protocol = CalibrationProtocol(user_id="test_user")
        new_protocol.load(session)
        
        assert new_protocol.calibration.is_trained()


class TestCalibrationWithSimulator:
    """Test calibration with EEG simulator."""
    
    def test_calibration_with_simulated_data(self):
        """Test complete calibration workflow with simulator."""
        # Create components
        simulator = EEGSimulator(n_channels=8, sampling_rate=256, seed=42)
        preprocessor = SignalPreprocessor(sampling_rate=256)
        extractor = FeatureExtractor(sampling_rate=256)
        protocol = CalibrationProtocol(user_id="sim_user")
        
        # Simulate calibration stages
        for stage, sim_state in [
            ('baseline', 'neutral'),
            ('focus', 'focus'),
            ('relax', 'relax')
        ]:
            protocol.start_stage(stage)
            simulator.set_mental_state(sim_state, intensity=1.0, transition_time=0.0)
            
            # Collect 20 samples
            for _ in range(20):
                # Generate EEG
                raw = simulator.generate_chunk(duration=1.0)
                
                # Process
                processed = preprocessor.process_chunk(raw)
                features_dict = extractor.extract_features(processed)
                
                # Convert to array
                features = np.array([
                    features_dict['delta_power'],
                    features_dict['theta_power'],
                    features_dict['alpha_power'],
                    features_dict['beta_power'],
                    features_dict['gamma_power'],
                    features_dict['hemispheric_asymmetry'],
                    features_dict['focus_metric'],
                    features_dict['relax_metric']
                ])
                
                protocol.add_sample(features)
        
        # Train model
        results = protocol.train_model()
        
        assert results['success']
        assert results['validation_accuracy'] > 0
        
        # Test prediction
        simulator.set_mental_state('focus', intensity=1.0, transition_time=0.0)
        raw = simulator.generate_chunk(duration=1.0)
        processed = preprocessor.process_chunk(raw)
        features_dict = extractor.extract_features(processed)
        
        features = np.array([
            features_dict['delta_power'],
            features_dict['theta_power'],
            features_dict['alpha_power'],
            features_dict['beta_power'],
            features_dict['gamma_power'],
            features_dict['hemispheric_asymmetry'],
            features_dict['focus_metric'],
            features_dict['relax_metric']
        ])
        
        state, confidence = protocol.calibration.predict_state(features)
        
        assert state in ['neutral', 'focus', 'relax']
        assert 0 <= confidence <= 1
        
        # Get feature importance
        importance = protocol.calibration.get_feature_importance()
        assert len(importance) == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
