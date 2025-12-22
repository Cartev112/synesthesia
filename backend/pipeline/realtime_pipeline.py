"""
Real-time BCI processing pipeline.

Coordinates EEG acquisition, signal processing, ML inference,
and music generation in real-time.
"""

import asyncio
from typing import Dict, Optional, Callable
import time
import numpy as np

from backend.eeg.simulator import StreamingEEGSimulator
from backend.eeg.device_interface import EEGDeviceInterface
from backend.eeg.devices.muse_s_athena import MuseSAthenaDevice
from backend.signal_processing import SignalPreprocessor, FeatureExtractor, CircularBuffer
from backend.ml import ArtifactClassifier, MentalStateTracker, MentalStateClassifier
from backend.ml.calibration import UserCalibration, CalibrationSession
from backend.visual import VisualParameterGenerator
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RealtimePipeline:
    """
    Real-time BCI processing pipeline.
    
    Orchestrates the complete flow:
    EEG → Preprocessing → Feature Extraction → ML → Music Generation
    """
    
    def __init__(
        self,
        sampling_rate: int = 256,
        device_type: str = "simulator",
        device_address: Optional[str] = None,
        device_preset: str = "full_research",
        on_brain_state: Optional[Callable] = None,
        on_music_events: Optional[Callable] = None,
        on_visual_params: Optional[Callable] = None,
        on_audio_buffer: Optional[Callable] = None,
        on_features: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Initialize real-time pipeline.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            device_type: Device type - "simulator" or "muse_s_athena"
            device_address: BLE address for Muse device (None for auto-discovery)
            device_preset: OpenMuse preset for Muse device
            on_brain_state: Callback for brain state updates
            on_music_events: Callback for music events
            on_visual_params: Callback for visual parameters
            on_features: Callback for feature vectors (used during calibration)
            on_error: Callback for errors
        """
        self.sampling_rate = sampling_rate
        self.device_type = device_type
        self.device_address = device_address
        self.device_preset = device_preset
        
        # Callbacks
        self.on_brain_state = on_brain_state
        self.on_music_events = on_music_events
        self.on_visual_params = on_visual_params
        self.on_audio_buffer = on_audio_buffer
        self.on_features = on_features
        self.on_error = on_error
        
        # Components
        self.eeg_source = None
        self.preprocessor = SignalPreprocessor(sampling_rate=sampling_rate)
        self.extractor = FeatureExtractor(sampling_rate=sampling_rate)
        self.buffer = CircularBuffer(
            n_channels=8,
            buffer_duration=2.0,
            sampling_rate=sampling_rate
        )
        
        # ML components
        self.artifact_classifier = ArtifactClassifier(
            n_channels=8,
            window_samples=128  # 0.5s at 256Hz
        )
        self.state_classifier = MentalStateClassifier()
        self.state_tracker = MentalStateTracker(classifier=self.state_classifier)
        
        # User calibration (personalized model)
        self.user_calibration: Optional[UserCalibration] = None
        
        # Visual parameter generator
        self.visual_generator = VisualParameterGenerator()
        
        # State
        self.is_running = False
        self.is_calibrated = False
        self.calibration_mode = False  # When True, sends features via on_features callback
        self.session_start_time = None
        self.sample_count = 0
        self.update_count = 0  # Track number of updates sent
        
        # Performance metrics
        self.metrics = {
            'total_latency': [],
            'preprocessing_time': [],
            'feature_extraction_time': [],
            'ml_inference_time': [],
            'updates_sent': 0
        }
        
        logger.info(
            "realtime_pipeline_initialized",
            sampling_rate=sampling_rate,
            device_type=device_type
        )
    
    async def start(self):
        """Start the real-time pipeline."""
        if self.is_running:
            logger.warning("pipeline_already_running")
            return
        
        # Initialize EEG source based on device type
        if self.device_type == "simulator":
            self.eeg_source = StreamingEEGSimulator(
                n_channels=8,
                sampling_rate=self.sampling_rate
            )
            self.eeg_source.start_stream()
            logger.info("using_eeg_simulator")
        elif self.device_type == "muse_s_athena":
            self.eeg_source = MuseSAthenaDevice(
                address=self.device_address,
                preset=self.device_preset
            )
            if not self.eeg_source.connect():
                raise RuntimeError("Failed to connect to Muse S Athena device")
            self.eeg_source.start_stream()
            logger.info("using_muse_s_athena", address=self.eeg_source._address)
        else:
            raise ValueError(f"Unknown device type: {self.device_type}")
        
        self.is_running = True
        self.session_start_time = time.time()
        self.sample_count = 0
        
        # Pre-fill buffer so data flows immediately on first iteration
        # Need 256 samples (1 second) for feature extraction
        warmup_samples = 256
        warmup_data = self.eeg_source.read_samples(warmup_samples)
        if warmup_data is not None:
            processed = self.preprocessor.process_chunk(warmup_data)
            self.buffer.append(processed)
            self.sample_count += warmup_samples
            logger.info("buffer_prefilled", samples=warmup_samples)
        
        logger.info("pipeline_started")
        
        # Start processing loop
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        # Critical: yield control multiple times to ensure the task starts
        # A single sleep(0) may not be enough if the event loop is busy
        for _ in range(3):
            await asyncio.sleep(0.01)
        
        logger.info("processing_loop_scheduled")
    
    async def stop(self):
        """Stop the real-time pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.eeg_source and hasattr(self.eeg_source, 'stop_stream'):
            self.eeg_source.stop_stream()
        
        # Log performance metrics
        if self.metrics['total_latency']:
            avg_latency = np.mean(self.metrics['total_latency'])
            logger.info(
                "pipeline_stopped",
                avg_latency_ms=avg_latency * 1000,
                total_samples=self.sample_count,
                total_updates_sent=self.update_count
            )
        else:
            # Log even if no latency metrics (debugging)
            logger.warning(
                "pipeline_stopped_no_updates",
                total_samples=self.sample_count,
                total_updates_sent=self.update_count,
                buffer_samples=self.buffer.current_samples if self.buffer else 0
            )
    
    async def _processing_loop(self):
        """Main processing loop."""
        chunk_size = 32  # Process 32 samples at a time (~125ms at 256Hz)
        update_interval = chunk_size / self.sampling_rate  # Target time per iteration
        
        # Track last brain state for continuous updates
        last_brain_state = None
        last_visual_params = None
        first_iteration = True
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # 1. Acquire EEG data
                raw_data = self.eeg_source.read_samples(chunk_size)
                
                if raw_data is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # 2. Preprocess
                t0 = time.time()
                processed = self.preprocessor.process_chunk(raw_data)
                self.metrics['preprocessing_time'].append(time.time() - t0)
                
                # 3. Add to buffer
                self.buffer.append(processed)
                self.sample_count += chunk_size
                
                # 4. Check for artifacts (on 0.5s window)
                is_artifact = False
                if self.buffer.current_samples >= 128:
                    latest_window = self.buffer.get_latest(duration=0.5)
                    
                    is_artifact, artifact_conf = self.artifact_classifier.detect_artifact(
                        latest_window.T  # Transpose to (channels, samples)
                    )
                    
                    if is_artifact:
                        logger.debug("artifact_detected", confidence=artifact_conf)
                
                # 5. Extract features and update state (on 1s window)
                should_update = False
                
                # Debug logging for first iteration
                if first_iteration:
                    logger.info(
                        "first_iteration_state",
                        buffer_samples=self.buffer.current_samples,
                        is_artifact=is_artifact,
                        will_extract_features=self.buffer.current_samples >= 256 and not is_artifact
                    )
                    first_iteration = False
                
                if self.buffer.current_samples >= 256 and not is_artifact:
                    t0 = time.time()
                    feature_window = self.buffer.get_latest(duration=1.0)
                    features = self.extractor.extract_features(feature_window)
                    self.metrics['feature_extraction_time'].append(time.time() - t0)
                    
                    # Convert to feature array for ML/calibration
                    feature_array = self._features_to_array(features)
                    
                    # 5b. Send features if in calibration mode
                    if self.calibration_mode and self.on_features:
                        await self.on_features(feature_array)
                    
                    # 6. ML inference
                    if self.is_calibrated and self.user_calibration is not None:
                        # Use personalized calibration model
                        t0 = time.time()
                        state, confidence = self.user_calibration.predict_state(feature_array)
                        probs = self.user_calibration.get_state_probabilities(feature_array)
                        self.metrics['ml_inference_time'].append(time.time() - t0)
                        
                        brain_state = {
                            'focus': probs['focus'],
                            'relax': probs['relax'],
                            'neutral': probs['neutral'],
                            'predicted_state': state,
                            'confidence': confidence,
                            'stability': confidence,
                            **features
                        }
                    elif self.is_calibrated:
                        # Use generic trained classifier
                        t0 = time.time()
                        brain_state = self.state_tracker.update(feature_array)
                        self.metrics['ml_inference_time'].append(time.time() - t0)
                    else:
                        # Use raw features without classification
                        # Normalize metrics using empirically measured ranges from simulator
                        focus_raw = features['focus_metric']
                        relax_raw = features['relax_metric']
                        
                        # Empirical baselines from test_state_switching.py:
                        # focus_metric = beta / (alpha + theta)
                        #   - Neutral: ~0.30, Focus: ~0.53, Relax: ~0.17
                        # relax_metric = alpha / (beta + gamma)
                        #   - Neutral: ~2.5, Relax: ~4.0, Focus: ~1.4
                        
                        # Use neutral as baseline, measure deviation
                        focus_baseline = 0.30
                        relax_baseline = 2.5
                        
                        # Expected ranges for each state
                        focus_range = 0.23   # 0.53 - 0.30 for focus
                        relax_range = 1.5    # 4.0 - 2.5 for relax
                        
                        # Deviation from baseline, normalized to expected range
                        focus_deviation = (focus_raw - focus_baseline) / focus_range
                        relax_deviation = (relax_raw - relax_baseline) / relax_range
                        
                        # Convert deviations to 0-1 scores using sigmoid
                        # Steeper sigmoid (3.0) for more responsive state detection
                        focus_score = 1.0 / (1.0 + np.exp(-focus_deviation * 3.0))
                        relax_score = 1.0 / (1.0 + np.exp(-relax_deviation * 3.0))
                        
                        # Neutral is high when both focus and relax are near baseline
                        # Use product of inverse deviations
                        neutral_score = np.exp(-(focus_deviation**2 + relax_deviation**2))
                        
                        # Normalize to sum to 1.0
                        total = focus_score + relax_score + neutral_score
                        focus_norm = focus_score / total
                        relax_norm = relax_score / total
                        neutral_norm = neutral_score / total
                        
                        brain_state = {
                            'focus': focus_norm,
                            'relax': relax_norm,
                            'neutral': neutral_norm,
                            'stability': 0.5,
                            **features
                        }
                    
                    # 7. Generate visual parameters
                    visual_params = self.visual_generator.generate_params(brain_state)
                    
                    # Cache for continuous updates
                    last_brain_state = brain_state
                    last_visual_params = visual_params
                    should_update = True
                
                # 8. Send updates every iteration (continuous stream)
                # Use latest computed state or previous state if not yet computed
                if last_brain_state is not None:
                    if self.on_brain_state:
                        await self.on_brain_state(last_brain_state)
                    
                    if self.on_visual_params and last_visual_params is not None:
                        await self.on_visual_params(last_visual_params)
                    
                    # Send empty music events for compatibility
                    if self.on_music_events:
                        await self.on_music_events({})
                    
                    # Track updates sent
                    self.update_count += 1
                    self.metrics['updates_sent'] = self.update_count
                    
                    # Log every 50 updates to verify continuous stream
                    if self.update_count % 50 == 0:
                        logger.debug(
                            "continuous_updates",
                            updates_sent=self.update_count,
                            samples_processed=self.sample_count
                        )
                    
                    # Track latency only when we compute new features
                    if should_update:
                        total_latency = time.time() - loop_start
                        self.metrics['total_latency'].append(total_latency)
                
                # Sleep to maintain consistent timing
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, update_interval - elapsed)  # Minimum 1ms sleep
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception("processing_loop_error", error=str(e))
                if self.on_error:
                    await self.on_error(str(e))
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dict to array for ML."""
        return np.array([
            features['delta_power'],
            features['theta_power'],
            features['alpha_power'],
            features['beta_power'],
            features['gamma_power'],
            features['hemispheric_asymmetry'],
            features['focus_metric'],
            features['relax_metric']
        ])
    
    def calibrate(self, calibration_data: Dict[str, np.ndarray]):
        """
        Calibrate the pipeline with user data (legacy method).
        
        Args:
            calibration_data: Dictionary with keys 'neutral', 'focus', 'relax'
                             Each value is array of shape (n_samples, n_features)
        """
        # Prepare training data
        X = np.vstack([
            calibration_data['neutral'],
            calibration_data['focus'],
            calibration_data['relax']
        ])
        
        y = np.array(
            [0] * len(calibration_data['neutral']) +
            [1] * len(calibration_data['focus']) +
            [2] * len(calibration_data['relax'])
        )
        
        # Train classifier
        self.state_classifier.train(X, y)
        self.is_calibrated = True
        
        logger.info("pipeline_calibrated", n_samples=len(X))
    
    def set_calibration_mode(self, enabled: bool) -> None:
        """
        Enable or disable calibration mode.
        
        When enabled, the pipeline sends feature vectors via on_features callback
        for collection during calibration stages.
        
        Args:
            enabled: Whether to enable calibration mode
        """
        self.calibration_mode = enabled
        logger.info("calibration_mode_set", enabled=enabled)
    
    def apply_calibration(self, calibration: UserCalibration) -> None:
        """
        Apply a trained user calibration to the pipeline.
        
        Args:
            calibration: Trained UserCalibration instance
        """
        if not calibration.is_trained():
            raise ValueError("Calibration model is not trained")
        
        self.user_calibration = calibration
        self.is_calibrated = True
        self.calibration_mode = False  # Exit calibration mode
        
        logger.info(
            "calibration_applied",
            user_id=calibration.user_id
        )
    
    def apply_calibration_session(self, session: CalibrationSession) -> None:
        """
        Apply a saved calibration session to the pipeline.
        
        Args:
            session: Saved CalibrationSession to apply
        """
        # Create a new UserCalibration and load the session
        calibration = UserCalibration(user_id=session.user_id)
        calibration.load_session(session)
        
        self.apply_calibration(calibration)
        
        logger.info(
            "calibration_session_applied",
            user_id=session.user_id,
            accuracy=session.validation_accuracy
        )
    
    def set_mental_state(self, state: str, intensity: float = 1.0):
        """
        Set simulator mental state (for testing).
        
        Args:
            state: Mental state ('neutral', 'focus', 'relax')
            intensity: State intensity (0-1)
        """
        if self.device_type == "simulator" and self.eeg_source:
            self.eeg_source.set_mental_state(state, intensity)
            logger.debug("simulator_state_set", state=state, intensity=intensity)
    
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        if not self.metrics['total_latency']:
            return {
                'total_updates_sent': self.update_count,
                'total_samples_processed': self.sample_count
            }
        
        return {
            'avg_total_latency_ms': np.mean(self.metrics['total_latency']) * 1000,
            'avg_preprocessing_ms': np.mean(self.metrics['preprocessing_time']) * 1000,
            'avg_feature_extraction_ms': np.mean(self.metrics['feature_extraction_time']) * 1000,
            'avg_ml_inference_ms': np.mean(self.metrics['ml_inference_time']) * 1000 if self.metrics['ml_inference_time'] else 0,
            'total_samples_processed': self.sample_count,
            'total_updates_sent': self.update_count,
            'session_duration_s': time.time() - self.session_start_time if self.session_start_time else 0
        }
    
    def get_current_state(self) -> Dict:
        """Get current pipeline state."""
        return {
            'is_running': self.is_running,
            'is_calibrated': self.is_calibrated,
            'sample_count': self.sample_count,
            'metrics': self.get_metrics()
        }
