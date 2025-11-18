"""
Mental state classifier for brain state detection.

Classifies mental states:
- Neutral (baseline)
- Focus (active concentration)
- Relax (calm, meditative)
"""

from typing import Dict, Tuple, Optional, List
from collections import deque
import time
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from backend.core.logging import get_logger

logger = get_logger(__name__)


class MentalStateClassifier:
    """
    Classifies mental states from EEG features.
    
    Uses Random Forest for robust classification with:
    - Feature normalization
    - Temporal smoothing
    - Confidence estimation
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5
    ):
        """
        Initialize mental state classifier.
        
        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # State mapping
        self.state_map = {
            0: 'neutral',
            1: 'focus',
            2: 'relax'
        }
        self.reverse_state_map = {v: k for k, v in self.state_map.items()}
        
        logger.info(
            "state_classifier_initialized",
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    
    def train(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64]
    ):
        """
        Train the classifier on labeled data.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: State labels (0=neutral, 1=focus, 2=relax)
        """
        logger.info(
            "training_started",
            n_samples=len(features),
            n_features=features.shape[1]
        )
        
        # Fit scaler and transform features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, labels)
        self.is_trained = True
        
        # Log training accuracy
        train_accuracy = self.model.score(features_scaled, labels)
        logger.info(
            "training_complete",
            accuracy=train_accuracy
        )
    
    def predict(
        self,
        features: NDArray[np.float64]
    ) -> Tuple[str, float]:
        """
        Predict mental state from features.
        
        Args:
            features: Feature vector of shape (n_features,)
            
        Returns:
            state: Predicted state ('neutral', 'focus', 'relax')
            confidence: Prediction confidence (0-1)
        """
        if not self.is_trained:
            logger.warning("predict_called_on_untrained_model")
            return 'neutral', 0.0
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        probabilities = self.model.predict_proba(features_scaled)[0]
        state_idx = np.argmax(probabilities)
        confidence = probabilities[state_idx]
        state = self.state_map[state_idx]
        
        logger.debug(
            "state_predicted",
            state=state,
            confidence=confidence,
            probabilities={
                self.state_map[i]: float(p)
                for i, p in enumerate(probabilities)
            }
        )
        
        return state, confidence
    
    def predict_proba(
        self,
        features: NDArray[np.float64]
    ) -> Dict[str, float]:
        """
        Get probability distribution over states.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary mapping states to probabilities
        """
        if not self.is_trained:
            return {
                'neutral': 1.0,
                'focus': 0.0,
                'relax': 0.0
            }
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            self.state_map[i]: float(p)
            for i, p in enumerate(probabilities)
        }
    
    def save(self, model_path: Path):
        """
        Save trained model and scaler.
        
        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            logger.warning("attempting_to_save_untrained_model")
            return
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'state_map': self.state_map
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("model_saved", path=str(model_path))
    
    def load(self, model_path: Path):
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to saved model
        """
        if not model_path.exists():
            logger.error("model_file_not_found", path=str(model_path))
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.state_map = model_data['state_map']
        self.reverse_state_map = {v: k for k, v in self.state_map.items()}
        self.is_trained = True
        
        logger.info("model_loaded", path=str(model_path))


class MentalStateTracker:
    """
    Tracks mental state over time with temporal smoothing.
    
    Uses exponential moving average to smooth predictions and
    reduce jitter in state transitions.
    """
    
    def __init__(
        self,
        classifier: Optional[MentalStateClassifier] = None,
        smoothing_factor: float = 0.3,
        history_size: int = 100
    ):
        """
        Initialize mental state tracker.
        
        Args:
            classifier: Mental state classifier instance
            smoothing_factor: EMA smoothing factor (0-1, higher = more smoothing)
            history_size: Number of historical states to keep
        """
        self.classifier = classifier or MentalStateClassifier()
        self.smoothing_factor = smoothing_factor
        
        # State history
        self.state_history: deque = deque(maxlen=history_size)
        
        # Smoothed probabilities
        self.smoothed_probs = {
            'neutral': 1.0,
            'focus': 0.0,
            'relax': 0.0
        }
        
        logger.info(
            "state_tracker_initialized",
            smoothing_factor=smoothing_factor,
            history_size=history_size
        )
    
    def update(
        self,
        features: NDArray[np.float64]
    ) -> Dict[str, float]:
        """
        Update state estimate with new features.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with smoothed state probabilities and metrics
        """
        # Get raw predictions
        raw_probs = self.classifier.predict_proba(features)
        
        # Apply exponential moving average
        for state in ['neutral', 'focus', 'relax']:
            self.smoothed_probs[state] = (
                self.smoothing_factor * self.smoothed_probs[state] +
                (1 - self.smoothing_factor) * raw_probs[state]
            )
        
        # Determine current state
        current_state = max(self.smoothed_probs, key=self.smoothed_probs.get)
        
        # Add to history
        self.state_history.append({
            'timestamp': time.time(),
            'state': current_state,
            'probabilities': self.smoothed_probs.copy(),
            'raw_probabilities': raw_probs
        })
        
        # Compute additional metrics
        result = {
            'state': current_state,
            'focus': self.smoothed_probs['focus'],
            'relax': self.smoothed_probs['relax'],
            'neutral': self.smoothed_probs['neutral'],
            'focus_trend': self._compute_trend('focus'),
            'relax_trend': self._compute_trend('relax'),
            'stability': self._compute_stability()
        }
        
        logger.debug(
            "state_updated",
            state=current_state,
            focus=result['focus'],
            relax=result['relax']
        )
        
        return result
    
    def _compute_trend(self, state: str, window: int = 10) -> float:
        """
        Compute trend for a specific state (rising or falling).
        
        Args:
            state: State to compute trend for
            window: Number of recent samples to use
            
        Returns:
            Trend value (-1 to 1, negative=falling, positive=rising)
        """
        if len(self.state_history) < 2:
            return 0.0
        
        # Get recent probabilities
        recent = list(self.state_history)[-window:]
        probs = [entry['probabilities'][state] for entry in recent]
        
        if len(probs) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(probs))
        trend = np.polyfit(x, probs, 1)[0]  # Slope
        
        # Normalize to [-1, 1]
        trend = np.clip(trend * 10, -1, 1)
        
        return float(trend)
    
    def _compute_stability(self, window: int = 20) -> float:
        """
        Compute stability of current state.
        
        Args:
            window: Number of recent samples to use
            
        Returns:
            Stability metric (0-1, higher = more stable)
        """
        if len(self.state_history) < 2:
            return 1.0
        
        # Get recent states
        recent = list(self.state_history)[-window:]
        states = [entry['state'] for entry in recent]
        
        # Count state changes
        changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        
        # Stability is inverse of change rate
        stability = 1.0 - (changes / len(states))
        
        return float(stability)
    
    def get_history(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get state history.
        
        Args:
            n: Number of recent entries to return (None = all)
            
        Returns:
            List of historical state entries
        """
        if n is None:
            return list(self.state_history)
        else:
            return list(self.state_history)[-n:]
    
    def reset(self):
        """Reset tracker to initial state."""
        self.state_history.clear()
        self.smoothed_probs = {
            'neutral': 1.0,
            'focus': 0.0,
            'relax': 0.0
        }
        logger.info("state_tracker_reset")


