"""
User calibration system for personalized mental state classification.

Implements the calibration protocol from the implementation plan:
1. Baseline recording (60s)
2. Focus task (60s) 
3. Relax task (60s)
4. Model training
5. Validation
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationSession:
    """Represents a calibration session."""
    user_id: str
    timestamp: datetime
    baseline_stats: Dict[str, np.ndarray]
    scaler: StandardScaler
    model: RandomForestClassifier
    validation_accuracy: float
    training_samples: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'baseline_stats': {
                'mean': self.baseline_stats['mean'].tolist(),
                'std': self.baseline_stats['std'].tolist()
            },
            'scaler': pickle.dumps(self.scaler),
            'model': pickle.dumps(self.model),
            'validation_accuracy': self.validation_accuracy,
            'training_samples': self.training_samples
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationSession':
        """Deserialize from storage."""
        return cls(
            user_id=data['user_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            baseline_stats={
                'mean': np.array(data['baseline_stats']['mean']),
                'std': np.array(data['baseline_stats']['std'])
            },
            scaler=pickle.loads(data['scaler']),
            model=pickle.loads(data['model']),
            validation_accuracy=data['validation_accuracy'],
            training_samples=data['training_samples']
        )


class UserCalibration:
    """
    Manages user-specific calibration for mental state classification.
    
    Follows the calibration protocol:
    - Baseline: 60s neutral state
    - Focus: 60s focused task (counting backwards)
    - Relax: 60s relaxation (breathing focus)
    """
    
    def __init__(self, user_id: str):
        """
        Initialize calibration for a user.
        
        Args:
            user_id: Unique user identifier
        """
        self.user_id = user_id
        self.baseline_stats: Optional[Dict[str, np.ndarray]] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.state_model: Optional[RandomForestClassifier] = None
        
        # Calibration data collection
        self.calibration_data: Dict[str, List[np.ndarray]] = {
            'neutral': [],
            'focus': [],
            'relax': []
        }
        
        logger.info("user_calibration_initialized", user_id=user_id)
    
    def add_calibration_sample(self, state: str, features: np.ndarray) -> None:
        """
        Add a feature sample for calibration.
        
        Args:
            state: Mental state label ('neutral', 'focus', 'relax')
            features: Feature vector (8 features)
        """
        if state not in self.calibration_data:
            raise ValueError(f"Invalid state: {state}")
        
        if features.shape != (8,):
            raise ValueError(f"Expected 8 features, got {features.shape}")
        
        self.calibration_data[state].append(features)
        
        logger.debug(
            "calibration_sample_added",
            user_id=self.user_id,
            state=state,
            total_samples=len(self.calibration_data[state])
        )
    
    def compute_baseline_stats(self) -> None:
        """
        Compute baseline statistics from neutral state data.
        
        Used for user-specific normalization.
        """
        if not self.calibration_data['neutral']:
            raise ValueError("No neutral state data for baseline")
        
        neutral_data = np.array(self.calibration_data['neutral'])
        
        self.baseline_stats = {
            'mean': np.mean(neutral_data, axis=0),
            'std': np.std(neutral_data, axis=0)
        }
        
        logger.info(
            "baseline_stats_computed",
            user_id=self.user_id,
            n_samples=len(neutral_data)
        )
    
    def train(self) -> Tuple[float, Dict[str, int]]:
        """
        Train the user-specific mental state classifier.
        
        Returns:
            validation_accuracy: Cross-validation accuracy
            sample_counts: Number of samples per state
        """
        # Check we have data for all states
        sample_counts = {
            state: len(samples)
            for state, samples in self.calibration_data.items()
        }
        
        for state, count in sample_counts.items():
            if count == 0:
                raise ValueError(f"No samples for state: {state}")
        
        # Compute baseline if not done
        if self.baseline_stats is None:
            self.compute_baseline_stats()
        
        # Prepare training data
        X_list = []
        y_list = []
        
        state_labels = {'neutral': 0, 'focus': 1, 'relax': 2}
        
        for state, label in state_labels.items():
            samples = np.array(self.calibration_data[state])
            X_list.append(samples)
            y_list.extend([label] * len(samples))
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        # Apply temporal smoothing (5-sample rolling average)
        X_smoothed = self._apply_temporal_smoothing(X, window=5)
        
        # Feature scaling
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_smoothed)
        
        # Train Random Forest classifier
        self.state_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        
        self.state_model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.state_model,
            X_scaled,
            y,
            cv=5,
            scoring='accuracy'
        )
        validation_accuracy = float(np.mean(cv_scores))
        
        logger.info(
            "calibration_model_trained",
            user_id=self.user_id,
            validation_accuracy=validation_accuracy,
            sample_counts=sample_counts
        )
        
        return validation_accuracy, sample_counts
    
    def _apply_temporal_smoothing(
        self,
        X: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Apply rolling average for temporal smoothing.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            window: Window size for rolling average
            
        Returns:
            Smoothed feature matrix
        """
        if len(X) < window:
            return X
        
        # Pad at the beginning to maintain shape
        padded = np.vstack([
            np.tile(X[0], (window - 1, 1)),
            X
        ])
        
        # Apply rolling average
        smoothed = np.array([
            np.mean(padded[i:i+window], axis=0)
            for i in range(len(X))
        ])
        
        return smoothed
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply user-specific normalization.
        
        Args:
            features: Raw feature vector
            
        Returns:
            Normalized features (z-score relative to baseline)
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline stats not computed")
        
        # Z-score normalization
        normalized = (features - self.baseline_stats['mean']) / \
                    (self.baseline_stats['std'] + 1e-6)
        
        return normalized
    
    def predict_state(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict mental state from features.
        
        Args:
            features: Feature vector (8 features)
            
        Returns:
            state: Predicted state ('neutral', 'focus', 'relax')
            confidence: Prediction confidence (0-1)
        """
        if self.state_model is None:
            raise ValueError("Model not trained")
        
        if self.feature_scaler is None:
            raise ValueError("Feature scaler not initialized")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        
        # Predict
        probabilities = self.state_model.predict_proba(X_scaled)[0]
        state_idx = np.argmax(probabilities)
        confidence = probabilities[state_idx]
        
        state_names = ['neutral', 'focus', 'relax']
        state = state_names[state_idx]
        
        return state, float(confidence)
    
    def get_state_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        """
        Get probabilities for all states.
        
        Args:
            features: Feature vector (8 features)
            
        Returns:
            Dictionary with probabilities for each state
        """
        if self.state_model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        probabilities = self.state_model.predict_proba(X_scaled)[0]
        
        return {
            'neutral': float(probabilities[0]),
            'focus': float(probabilities[1]),
            'relax': float(probabilities[2])
        }
    
    def save_session(self) -> CalibrationSession:
        """
        Save calibration session.
        
        Returns:
            CalibrationSession object for persistence
        """
        if self.state_model is None:
            raise ValueError("Model not trained")
        
        # Compute validation accuracy
        X_list = []
        y_list = []
        state_labels = {'neutral': 0, 'focus': 1, 'relax': 2}
        
        for state, label in state_labels.items():
            samples = np.array(self.calibration_data[state])
            X_list.append(samples)
            y_list.extend([label] * len(samples))
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        X_scaled = self.feature_scaler.transform(X)
        
        validation_accuracy = float(self.state_model.score(X_scaled, y))
        
        sample_counts = {
            state: len(samples)
            for state, samples in self.calibration_data.items()
        }
        
        session = CalibrationSession(
            user_id=self.user_id,
            timestamp=datetime.now(),
            baseline_stats=self.baseline_stats,
            scaler=self.feature_scaler,
            model=self.state_model,
            validation_accuracy=validation_accuracy,
            training_samples=sample_counts
        )
        
        logger.info(
            "calibration_session_saved",
            user_id=self.user_id,
            accuracy=validation_accuracy
        )
        
        return session
    
    def load_session(self, session: CalibrationSession) -> None:
        """
        Load a saved calibration session.
        
        Args:
            session: CalibrationSession to load
        """
        if session.user_id != self.user_id:
            raise ValueError("Session user_id mismatch")
        
        self.baseline_stats = session.baseline_stats
        self.feature_scaler = session.scaler
        self.state_model = session.model
        
        logger.info(
            "calibration_session_loaded",
            user_id=self.user_id,
            accuracy=session.validation_accuracy,
            timestamp=session.timestamp
        )
    
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.state_model is not None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.state_model is None:
            raise ValueError("Model not trained")
        
        feature_names = [
            'delta_power',
            'theta_power',
            'alpha_power',
            'beta_power',
            'gamma_power',
            'hemispheric_asymmetry',
            'focus_metric',
            'relax_metric'
        ]
        
        importances = self.state_model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }


class CalibrationProtocol:
    """
    Manages the calibration protocol workflow.
    
    Protocol stages:
    1. Baseline (60s) - neutral state
    2. Focus (60s) - counting task
    3. Relax (60s) - breathing focus
    4. Training (2-3s)
    5. Validation (30s)
    """
    
    # Stage durations in seconds
    BASELINE_DURATION = 60
    FOCUS_DURATION = 60
    RELAX_DURATION = 60
    VALIDATION_DURATION = 30
    
    def __init__(self, user_id: str):
        """
        Initialize calibration protocol.
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        self.calibration = UserCalibration(user_id)
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        
        logger.info("calibration_protocol_initialized", user_id=user_id)
    
    def start_stage(self, stage: str) -> Dict[str, any]:
        """
        Start a calibration stage.
        
        Args:
            stage: Stage name ('baseline', 'focus', 'relax', 'validation')
            
        Returns:
            Stage information including duration and instructions
        """
        valid_stages = ['baseline', 'focus', 'relax', 'validation']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}")
        
        self.current_stage = stage
        self.stage_start_time = time.time()
        
        # Map stage to state label
        stage_to_state = {
            'baseline': 'neutral',
            'focus': 'focus',
            'relax': 'relax',
            'validation': None  # No data collection during validation
        }
        
        # Get duration
        durations = {
            'baseline': self.BASELINE_DURATION,
            'focus': self.FOCUS_DURATION,
            'relax': self.RELAX_DURATION,
            'validation': self.VALIDATION_DURATION
        }
        
        # Get instructions
        instructions = {
            'baseline': "Sit comfortably and relax. Try to keep your mind calm and neutral.",
            'focus': "Count backwards from 100 by 7s (100, 93, 86...). Stay focused on the task.",
            'relax': "Close your eyes and focus on your breathing. Let your mind rest.",
            'validation': "Try to control your mental state. Move between focus and relaxation."
        }
        
        logger.info(
            "calibration_stage_started",
            user_id=self.user_id,
            stage=stage
        )
        
        return {
            'stage': stage,
            'state_label': stage_to_state[stage],
            'duration': durations[stage],
            'instructions': instructions[stage],
            'start_time': self.stage_start_time
        }
    
    def add_sample(self, features: np.ndarray) -> None:
        """
        Add a sample during calibration.
        
        Args:
            features: Feature vector
        """
        if self.current_stage is None:
            raise ValueError("No active stage")
        
        if self.current_stage == 'validation':
            # Don't collect data during validation
            return
        
        # Map stage to state
        stage_to_state = {
            'baseline': 'neutral',
            'focus': 'focus',
            'relax': 'relax'
        }
        
        state = stage_to_state[self.current_stage]
        self.calibration.add_calibration_sample(state, features)
    
    def get_stage_progress(self) -> Dict[str, any]:
        """
        Get current stage progress.
        
        Returns:
            Progress information
        """
        if self.current_stage is None or self.stage_start_time is None:
            return {'active': False}
        
        elapsed = time.time() - self.stage_start_time
        
        durations = {
            'baseline': self.BASELINE_DURATION,
            'focus': self.FOCUS_DURATION,
            'relax': self.RELAX_DURATION,
            'validation': self.VALIDATION_DURATION
        }
        
        duration = durations[self.current_stage]
        progress = min(elapsed / duration, 1.0)
        remaining = max(duration - elapsed, 0)
        
        return {
            'active': True,
            'stage': self.current_stage,
            'elapsed': elapsed,
            'remaining': remaining,
            'progress': progress,
            'complete': progress >= 1.0
        }
    
    def train_model(self) -> Dict[str, any]:
        """
        Train the calibration model.
        
        Returns:
            Training results
        """
        start_time = time.time()
        
        validation_accuracy, sample_counts = self.calibration.train()
        
        training_time = time.time() - start_time
        
        logger.info(
            "calibration_model_trained",
            user_id=self.user_id,
            accuracy=validation_accuracy,
            training_time=training_time
        )
        
        return {
            'success': True,
            'validation_accuracy': validation_accuracy,
            'sample_counts': sample_counts,
            'training_time': training_time,
            'feature_importance': self.calibration.get_feature_importance()
        }
    
    def save(self) -> CalibrationSession:
        """Save calibration session."""
        return self.calibration.save_session()
    
    def load(self, session: CalibrationSession) -> None:
        """Load calibration session."""
        self.calibration.load_session(session)
    
    def get_calibration(self) -> UserCalibration:
        """Get the underlying calibration object."""
        return self.calibration
