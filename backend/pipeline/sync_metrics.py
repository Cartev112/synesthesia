"""
Synchrony metrics for multi-user BCI.

Computes meaningful inter-brain synchrony metrics:
- Phase Locking Value (PLV) on alpha/theta bands
- Bandpower correlation
- Baseline-relative synchrony scoring
- Composite sync score with smoothing
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureFrame:
    """
    Timestamped feature frame from a single user's pipeline.
    
    Attributes:
        timestamp: Unix timestamp when features were extracted
        user_id: Identifier for the user/pipeline
        features: Feature dictionary from FeatureExtractor
        feature_array: Numpy array of features for ML
        is_artifact: Whether this window contains artifacts
        raw_bandpowers: Raw bandpower values for PLV computation
    """
    timestamp: float
    user_id: str
    features: Dict[str, float]
    feature_array: NDArray[np.float64]
    is_artifact: bool = False
    raw_bandpowers: Optional[Dict[str, NDArray[np.float64]]] = None


@dataclass
class SyncState:
    """
    Current synchronization state between two users.
    
    Attributes:
        sync_score: Composite synchrony score (0-1, 1=perfectly synced)
        dissonance_level: Inverse mapping for music (0=consonant, 1=dissonant)
        alpha_plv: Phase locking value for alpha band
        theta_plv: Phase locking value for theta band
        bandpower_correlation: Pearson correlation of bandpowers
        asymmetry_correlation: Correlation of hemispheric asymmetry
        baseline_delta: Deviation from baseline synchrony
        quality: Data quality indicator (0-1)
        timestamp: When this state was computed
    """
    sync_score: float = 0.5
    dissonance_level: float = 0.5
    alpha_plv: float = 0.0
    theta_plv: float = 0.0
    bandpower_correlation: float = 0.0
    asymmetry_correlation: float = 0.0
    baseline_delta: float = 0.0
    quality: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'sync_score': float(self.sync_score),
            'dissonance_level': float(self.dissonance_level),
            'alpha_plv': float(self.alpha_plv),
            'theta_plv': float(self.theta_plv),
            'bandpower_correlation': float(self.bandpower_correlation),
            'asymmetry_correlation': float(self.asymmetry_correlation),
            'baseline_delta': float(self.baseline_delta),
            'quality': float(self.quality),
            'timestamp': self.timestamp
        }


class SyncMetricsCalculator:
    """
    Computes inter-brain synchrony metrics from paired feature frames.
    
    Uses multiple metrics to capture different aspects of synchrony:
    - PLV captures phase relationships in oscillations
    - Bandpower correlation captures similar activation patterns
    - Baseline-relative scoring normalizes for individual differences
    """
    
    def __init__(
        self,
        window_size: int = 10,
        ema_alpha: float = 0.3,
        hysteresis_threshold: float = 0.1,
        sampling_rate: int = 256
    ):
        """
        Initialize sync metrics calculator.
        
        Args:
            window_size: Number of frames to use for sliding window metrics
            ema_alpha: Exponential moving average smoothing factor (0-1)
            hysteresis_threshold: Threshold for state change hysteresis
            sampling_rate: EEG sampling rate for PLV computation
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.hysteresis_threshold = hysteresis_threshold
        self.sampling_rate = sampling_rate
        
        # Rolling windows for each user's features
        self.user_a_frames: deque[FeatureFrame] = deque(maxlen=window_size)
        self.user_b_frames: deque[FeatureFrame] = deque(maxlen=window_size)
        
        # Baseline statistics (computed during baseline phase)
        self.baseline_sync: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.is_baseline_phase: bool = False
        self.baseline_scores: List[float] = []
        
        # Smoothed state
        self.current_state: SyncState = SyncState()
        self.previous_sync_score: float = 0.5
        
        # Metric weights for composite score
        self.weights = {
            'alpha_plv': 0.35,      # Alpha PLV most important for social sync
            'theta_plv': 0.25,      # Theta for emotional/memory alignment
            'bandpower_corr': 0.25, # Overall activation similarity
            'asymmetry_corr': 0.15  # Hemispheric pattern matching
        }
        
        logger.info(
            "sync_metrics_initialized",
            window_size=window_size,
            ema_alpha=ema_alpha
        )
    
    def add_frame(self, frame: FeatureFrame) -> None:
        """
        Add a feature frame from a user.
        
        Args:
            frame: FeatureFrame from one user's pipeline
        """
        if frame.user_id == 'user_a':
            self.user_a_frames.append(frame)
        elif frame.user_id == 'user_b':
            self.user_b_frames.append(frame)
        else:
            logger.warning("unknown_user_id", user_id=frame.user_id)
    
    def compute_sync(self) -> Optional[SyncState]:
        """
        Compute synchrony state from current frame windows.
        
        Returns:
            SyncState if enough data, None otherwise
        """
        # Need at least 3 frames from each user for meaningful computation
        if len(self.user_a_frames) < 3 or len(self.user_b_frames) < 3:
            return None
        
        # Align frames by timestamp (find overlapping windows)
        aligned_a, aligned_b = self._align_frames()
        
        if len(aligned_a) < 2:
            return None
        
        # Filter out artifact frames
        clean_pairs = [
            (a, b) for a, b in zip(aligned_a, aligned_b)
            if not a.is_artifact and not b.is_artifact
        ]
        
        if len(clean_pairs) < 2:
            # Low quality - too many artifacts
            return SyncState(quality=0.0, timestamp=time.time())
        
        quality = len(clean_pairs) / len(aligned_a)
        
        # Extract feature arrays
        a_features = np.array([p[0].feature_array for p in clean_pairs])
        b_features = np.array([p[1].feature_array for p in clean_pairs])
        
        # Compute individual metrics
        alpha_plv = self._compute_bandpower_plv(a_features, b_features, band='alpha')
        theta_plv = self._compute_bandpower_plv(a_features, b_features, band='theta')
        bandpower_corr = self._compute_bandpower_correlation(a_features, b_features)
        asymmetry_corr = self._compute_asymmetry_correlation(a_features, b_features)
        
        # Compute composite score (weighted average)
        raw_score = (
            self.weights['alpha_plv'] * alpha_plv +
            self.weights['theta_plv'] * theta_plv +
            self.weights['bandpower_corr'] * bandpower_corr +
            self.weights['asymmetry_corr'] * asymmetry_corr
        )
        
        # Compute baseline-relative delta if baseline is set
        baseline_delta = 0.0
        if self.baseline_sync is not None and self.baseline_std is not None:
            if self.baseline_std > 0.01:
                baseline_delta = (raw_score - self.baseline_sync) / self.baseline_std
            else:
                baseline_delta = raw_score - self.baseline_sync
        
        # Apply EMA smoothing with hysteresis
        smoothed_score = self._apply_smoothing(raw_score)
        
        # Map to dissonance (inverse, with some nonlinearity for perceptibility)
        dissonance = self._map_to_dissonance(smoothed_score)
        
        # If in baseline phase, collect data
        if self.is_baseline_phase:
            self.baseline_scores.append(raw_score)
        
        # Create new state
        new_state = SyncState(
            sync_score=smoothed_score,
            dissonance_level=dissonance,
            alpha_plv=alpha_plv,
            theta_plv=theta_plv,
            bandpower_correlation=bandpower_corr,
            asymmetry_correlation=asymmetry_corr,
            baseline_delta=baseline_delta,
            quality=quality,
            timestamp=time.time()
        )
        
        self.current_state = new_state
        return new_state
    
    def start_baseline(self) -> None:
        """Start baseline collection phase."""
        self.is_baseline_phase = True
        self.baseline_scores = []
        logger.info("sync_baseline_started")
    
    def finish_baseline(self) -> Dict[str, float]:
        """
        Finish baseline collection and compute baseline statistics.
        
        Returns:
            Dictionary with baseline statistics
        """
        self.is_baseline_phase = False
        
        if len(self.baseline_scores) >= 5:
            self.baseline_sync = float(np.mean(self.baseline_scores))
            self.baseline_std = float(np.std(self.baseline_scores))
        else:
            # Not enough data, use defaults
            self.baseline_sync = 0.5
            self.baseline_std = 0.15
        
        logger.info(
            "sync_baseline_finished",
            mean=self.baseline_sync,
            std=self.baseline_std,
            n_samples=len(self.baseline_scores)
        )
        
        return {
            'baseline_sync': self.baseline_sync,
            'baseline_std': self.baseline_std,
            'n_samples': len(self.baseline_scores)
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self.user_a_frames.clear()
        self.user_b_frames.clear()
        self.baseline_sync = None
        self.baseline_std = None
        self.is_baseline_phase = False
        self.baseline_scores = []
        self.current_state = SyncState()
        self.previous_sync_score = 0.5
    
    def _align_frames(self) -> Tuple[List[FeatureFrame], List[FeatureFrame]]:
        """
        Align frames from both users by timestamp.
        
        Uses nearest-neighbor matching within a tolerance window.
        """
        tolerance_ms = 500  # 500ms alignment tolerance
        tolerance_s = tolerance_ms / 1000.0
        
        aligned_a = []
        aligned_b = []
        
        # Use user_a as reference, find matching frames in user_b
        b_list = list(self.user_b_frames)
        
        for frame_a in self.user_a_frames:
            best_match = None
            best_delta = float('inf')
            
            for frame_b in b_list:
                delta = abs(frame_a.timestamp - frame_b.timestamp)
                if delta < tolerance_s and delta < best_delta:
                    best_delta = delta
                    best_match = frame_b
            
            if best_match is not None:
                aligned_a.append(frame_a)
                aligned_b.append(best_match)
        
        return aligned_a, aligned_b
    
    def _compute_bandpower_plv(
        self,
        a_features: NDArray[np.float64],
        b_features: NDArray[np.float64],
        band: str = 'alpha'
    ) -> float:
        """
        Compute Phase Locking Value approximation from bandpower time series.
        
        Uses the bandpower values as a proxy for oscillation amplitude envelope.
        True PLV would require raw EEG data with Hilbert transform.
        
        This approximation measures consistency of relative bandpower changes.
        """
        # Feature indices: delta=0, theta=1, alpha=2, beta=3, gamma=4
        band_idx = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}
        idx = band_idx.get(band, 2)
        
        a_band = a_features[:, idx]
        b_band = b_features[:, idx]
        
        if len(a_band) < 2:
            return 0.5
        
        # Normalize to zero mean
        a_norm = a_band - np.mean(a_band)
        b_norm = b_band - np.mean(b_band)
        
        # Compute instantaneous phase using Hilbert transform
        a_analytic = signal.hilbert(a_norm)
        b_analytic = signal.hilbert(b_norm)
        
        a_phase = np.angle(a_analytic)
        b_phase = np.angle(b_analytic)
        
        # Phase difference
        phase_diff = a_phase - b_phase
        
        # PLV = |mean(e^(i*phase_diff))|
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return float(np.clip(plv, 0.0, 1.0))
    
    def _compute_bandpower_correlation(
        self,
        a_features: NDArray[np.float64],
        b_features: NDArray[np.float64]
    ) -> float:
        """
        Compute correlation of bandpower patterns between users.
        """
        # Use delta through gamma (indices 0-4)
        a_bp = a_features[:, :5].flatten()
        b_bp = b_features[:, :5].flatten()
        
        if len(a_bp) < 2 or np.std(a_bp) < 1e-6 or np.std(b_bp) < 1e-6:
            return 0.5
        
        corr = np.corrcoef(a_bp, b_bp)[0, 1]
        
        # Map from [-1, 1] to [0, 1]
        return float(np.clip((corr + 1) / 2, 0.0, 1.0))
    
    def _compute_asymmetry_correlation(
        self,
        a_features: NDArray[np.float64],
        b_features: NDArray[np.float64]
    ) -> float:
        """
        Compute correlation of hemispheric asymmetry between users.
        """
        # Asymmetry is at index 5
        a_asym = a_features[:, 5]
        b_asym = b_features[:, 5]
        
        if len(a_asym) < 2 or np.std(a_asym) < 1e-6 or np.std(b_asym) < 1e-6:
            return 0.5
        
        corr = np.corrcoef(a_asym, b_asym)[0, 1]
        
        # Map from [-1, 1] to [0, 1]
        return float(np.clip((corr + 1) / 2, 0.0, 1.0))
    
    def _apply_smoothing(self, raw_score: float) -> float:
        """
        Apply EMA smoothing with hysteresis.
        
        Prevents jittery score changes while allowing significant shifts.
        """
        # EMA smoothing
        smoothed = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self.previous_sync_score
        
        # Hysteresis: only update if change exceeds threshold
        delta = abs(smoothed - self.previous_sync_score)
        if delta < self.hysteresis_threshold:
            # Small change - use even stronger smoothing
            smoothed = 0.1 * raw_score + 0.9 * self.previous_sync_score
        
        self.previous_sync_score = smoothed
        return float(np.clip(smoothed, 0.0, 1.0))
    
    def _map_to_dissonance(self, sync_score: float) -> float:
        """
        Map sync score to dissonance level for music generation.
        
        Uses a nonlinear mapping to make changes more perceptible:
        - High sync (0.7-1.0) -> Low dissonance (0.0-0.2) - consonant
        - Medium sync (0.4-0.7) -> Medium dissonance (0.2-0.6)
        - Low sync (0.0-0.4) -> High dissonance (0.6-1.0) - dissonant
        """
        # Inverse with slight S-curve for perceptibility
        # Using sigmoid-like mapping
        x = (sync_score - 0.5) * 4  # Scale to roughly [-2, 2]
        sigmoid = 1 / (1 + np.exp(x))
        
        return float(np.clip(sigmoid, 0.0, 1.0))

