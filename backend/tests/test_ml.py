"""
Unit tests for machine learning components.

Tests:
- Artifact classifier
- Mental state classifier
- State tracker
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from backend.ml.artifact_classifier import ArtifactClassifier, ArtifactCNN
from backend.ml.state_classifier import MentalStateClassifier, MentalStateTracker


class TestArtifactCNN:
    """Test artifact detection CNN architecture."""
    
    @pytest.fixture
    def model(self):
        """Create CNN model instance."""
        return ArtifactCNN(n_channels=8, n_samples=128)
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_channels == 8
        assert model.n_samples == 128
    
    def test_forward_pass_shape(self, model):
        """Test forward pass output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 8, 128)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
    
    def test_forward_pass_single_sample(self, model):
        """Test forward pass with single sample."""
        x = torch.randn(1, 8, 128)
        output = model(x)
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1


class TestArtifactClassifier:
    """Test artifact classifier functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create artifact classifier instance."""
        return ArtifactClassifier(
            n_channels=8,
            window_samples=128,
            threshold=0.5
        )
    
    @pytest.fixture
    def clean_data(self):
        """Generate clean EEG-like data."""
        n_channels = 8
        n_samples = 128
        t = np.linspace(0, 0.5, n_samples)
        
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Normal EEG: mix of alpha and beta
            data[ch, :] = (
                np.sin(2 * np.pi * 10 * t) +
                0.5 * np.sin(2 * np.pi * 20 * t) +
                0.1 * np.random.randn(n_samples)
            )
        
        return data
    
    @pytest.fixture
    def artifact_data(self):
        """Generate data with artifact (simulated blink)."""
        n_channels = 8
        n_samples = 128
        
        data = np.random.randn(n_channels, n_samples) * 0.1
        
        # Add large amplitude spike in frontal channels (blink)
        data[0:2, 40:60] += 10.0  # Large amplitude in Fp1, Fp2
        
        return data
    
    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.n_channels == 8
        assert classifier.window_samples == 128
        assert classifier.threshold == 0.5
    
    def test_detect_artifact_returns_tuple(self, classifier, clean_data):
        """Test that detect_artifact returns (bool, float)."""
        is_artifact, confidence = classifier.detect_artifact(clean_data)
        
        assert isinstance(is_artifact, bool)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_detect_artifact_shape_validation(self, classifier):
        """Test handling of incorrect input shapes."""
        # Wrong shape
        wrong_data = np.random.randn(4, 64)  # Wrong channels and samples
        
        is_artifact, confidence = classifier.detect_artifact(wrong_data)
        
        # Should handle gracefully
        assert isinstance(is_artifact, bool)
        assert isinstance(confidence, float)
    
    def test_resize_window_padding(self, classifier):
        """Test window resizing with padding."""
        # Too few samples
        small_window = np.random.randn(8, 64)
        resized = classifier._resize_window(small_window)
        
        assert resized.shape == (8, 128)
    
    def test_resize_window_truncation(self, classifier):
        """Test window resizing with truncation."""
        # Too many samples
        large_window = np.random.randn(8, 256)
        resized = classifier._resize_window(large_window)
        
        assert resized.shape == (8, 128)
    
    def test_save_and_load_model(self, classifier):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.pt'
            
            # Save model
            classifier.save_model(model_path)
            assert model_path.exists()
            
            # Create new classifier and load
            new_classifier = ArtifactClassifier(
                n_channels=8,
                window_samples=128,
                model_path=model_path
            )
            
            # Models should produce same output
            test_data = np.random.randn(8, 128)
            
            _, conf1 = classifier.detect_artifact(test_data)
            _, conf2 = new_classifier.detect_artifact(test_data)
            
            assert abs(conf1 - conf2) < 1e-5
    
    def test_training_basic(self, classifier):
        """Test basic training functionality."""
        # Generate synthetic training data
        n_samples = 100
        train_data = np.random.randn(n_samples, 8, 128)
        train_labels = np.random.randint(0, 2, n_samples)
        
        # Train for just 2 epochs (quick test)
        classifier.train_model(
            train_data=train_data,
            train_labels=train_labels,
            epochs=2,
            batch_size=16
        )
        
        # Model should still work after training
        test_data = np.random.randn(8, 128)
        is_artifact, confidence = classifier.detect_artifact(test_data)
        
        assert isinstance(is_artifact, bool)
        assert 0 <= confidence <= 1


class TestMentalStateClassifier:
    """Test mental state classifier functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create mental state classifier instance."""
        return MentalStateClassifier(
            n_estimators=10,  # Small for testing
            max_depth=5
        )
    
    @pytest.fixture
    def training_data(self):
        """Generate synthetic training data."""
        n_samples_per_class = 50
        n_features = 8
        
        # Neutral: balanced features
        neutral = np.random.randn(n_samples_per_class, n_features)
        
        # Focus: high beta, low alpha
        focus = np.random.randn(n_samples_per_class, n_features)
        focus[:, 3] += 2.0  # Boost beta power
        focus[:, 2] -= 1.0  # Reduce alpha power
        
        # Relax: high alpha, low beta
        relax = np.random.randn(n_samples_per_class, n_features)
        relax[:, 2] += 2.0  # Boost alpha power
        relax[:, 3] -= 1.0  # Reduce beta power
        
        # Combine
        X = np.vstack([neutral, focus, relax])
        y = np.array([0] * n_samples_per_class + 
                     [1] * n_samples_per_class + 
                     [2] * n_samples_per_class)
        
        return X, y
    
    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert not classifier.is_trained
        assert len(classifier.state_map) == 3
        assert classifier.state_map[0] == 'neutral'
        assert classifier.state_map[1] == 'focus'
        assert classifier.state_map[2] == 'relax'
    
    def test_training(self, classifier, training_data):
        """Test classifier training."""
        X, y = training_data
        
        classifier.train(X, y)
        
        assert classifier.is_trained
    
    def test_predict_untrained(self, classifier):
        """Test prediction on untrained model."""
        features = np.random.randn(8)
        
        state, confidence = classifier.predict(features)
        
        assert state == 'neutral'
        assert confidence == 0.0
    
    def test_predict_trained(self, classifier, training_data):
        """Test prediction on trained model."""
        X, y = training_data
        classifier.train(X, y)
        
        # Predict on a sample
        features = X[0]
        state, confidence = classifier.predict(features)
        
        assert state in ['neutral', 'focus', 'relax']
        assert 0 <= confidence <= 1
    
    def test_predict_proba(self, classifier, training_data):
        """Test probability prediction."""
        X, y = training_data
        classifier.train(X, y)
        
        features = X[0]
        probs = classifier.predict_proba(features)
        
        assert isinstance(probs, dict)
        assert set(probs.keys()) == {'neutral', 'focus', 'relax'}
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-5  # Should sum to 1
    
    def test_save_and_load(self, classifier, training_data):
        """Test model saving and loading."""
        X, y = training_data
        classifier.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_classifier.pkl'
            
            # Save
            classifier.save(model_path)
            assert model_path.exists()
            
            # Load into new classifier
            new_classifier = MentalStateClassifier()
            new_classifier.load(model_path)
            
            assert new_classifier.is_trained
            
            # Should produce same predictions
            features = X[0]
            state1, conf1 = classifier.predict(features)
            state2, conf2 = new_classifier.predict(features)
            
            assert state1 == state2
            assert abs(conf1 - conf2) < 1e-5


class TestMentalStateTracker:
    """Test mental state tracker functionality."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Create and train a classifier."""
        classifier = MentalStateClassifier(n_estimators=10)
        
        # Quick training data
        n_samples = 30
        n_features = 8
        X = np.random.randn(n_samples * 3, n_features)
        y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)
        
        classifier.train(X, y)
        return classifier
    
    @pytest.fixture
    def tracker(self, trained_classifier):
        """Create state tracker instance."""
        return MentalStateTracker(
            classifier=trained_classifier,
            smoothing_factor=0.3,
            history_size=50
        )
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.smoothing_factor == 0.3
        assert len(tracker.state_history) == 0
        assert tracker.smoothed_probs['neutral'] == 1.0
        assert tracker.smoothed_probs['focus'] == 0.0
        assert tracker.smoothed_probs['relax'] == 0.0
    
    def test_update(self, tracker):
        """Test state update."""
        features = np.random.randn(8)
        
        result = tracker.update(features)
        
        assert isinstance(result, dict)
        assert 'state' in result
        assert 'focus' in result
        assert 'relax' in result
        assert 'neutral' in result
        assert 'focus_trend' in result
        assert 'relax_trend' in result
        assert 'stability' in result
    
    def test_smoothing(self, tracker):
        """Test temporal smoothing."""
        # Update multiple times with same features
        features = np.random.randn(8)
        
        results = []
        for _ in range(10):
            result = tracker.update(features)
            results.append(result['focus'])
        
        # Values should be smoothed (not identical)
        assert len(set(results)) > 1  # Should have variation
    
    def test_history_tracking(self, tracker):
        """Test state history tracking."""
        # Update several times
        for _ in range(5):
            features = np.random.randn(8)
            tracker.update(features)
        
        history = tracker.get_history()
        
        assert len(history) == 5
        assert all('timestamp' in entry for entry in history)
        assert all('state' in entry for entry in history)
        assert all('probabilities' in entry for entry in history)
    
    def test_trend_calculation(self, tracker):
        """Test trend calculation."""
        # Create increasing focus pattern
        for i in range(10):
            features = np.random.randn(8)
            features[3] = i * 0.5  # Increasing beta-like feature
            tracker.update(features)
        
        # Get latest result
        features = np.random.randn(8)
        result = tracker.update(features)
        
        # Trend should be a float between -1 and 1
        assert isinstance(result['focus_trend'], float)
        assert -1 <= result['focus_trend'] <= 1
    
    def test_stability_calculation(self, tracker):
        """Test stability calculation."""
        # Update with consistent state
        features = np.random.randn(8)
        for _ in range(10):
            tracker.update(features)
        
        result = tracker.update(features)
        
        # Stability should be high (close to 1)
        assert 0 <= result['stability'] <= 1
    
    def test_reset(self, tracker):
        """Test tracker reset."""
        # Add some history
        for _ in range(5):
            features = np.random.randn(8)
            tracker.update(features)
        
        assert len(tracker.state_history) > 0
        
        # Reset
        tracker.reset()
        
        assert len(tracker.state_history) == 0
        assert tracker.smoothed_probs['neutral'] == 1.0
        assert tracker.smoothed_probs['focus'] == 0.0
    
    def test_history_size_limit(self, tracker):
        """Test that history respects max size."""
        # Update more than history_size times
        for _ in range(100):
            features = np.random.randn(8)
            tracker.update(features)
        
        history = tracker.get_history()
        
        # Should not exceed max size (50)
        assert len(history) <= 50


class TestIntegration:
    """Integration tests for ML components."""
    
    def test_artifact_and_state_pipeline(self):
        """Test using artifact classifier and state classifier together."""
        # Create components
        artifact_classifier = ArtifactClassifier(
            n_channels=8,
            window_samples=128
        )
        
        state_classifier = MentalStateClassifier(n_estimators=10)
        
        # Train state classifier
        n_samples = 30
        n_features = 8
        X = np.random.randn(n_samples * 3, n_features)
        y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)
        state_classifier.train(X, y)
        
        tracker = MentalStateTracker(classifier=state_classifier)
        
        # Simulate processing pipeline
        eeg_window = np.random.randn(8, 128)
        
        # 1. Check for artifacts
        is_artifact, artifact_conf = artifact_classifier.detect_artifact(eeg_window)
        
        # 2. If clean, extract features and classify state
        if not is_artifact:
            # Simulate feature extraction
            features = np.random.randn(8)
            
            # 3. Update state tracker
            state_result = tracker.update(features)
            
            assert 'state' in state_result
            assert state_result['state'] in ['neutral', 'focus', 'relax']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
