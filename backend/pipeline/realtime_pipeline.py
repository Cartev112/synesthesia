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
from backend.signal_processing import SignalPreprocessor, FeatureExtractor, CircularBuffer
from backend.ml import ArtifactClassifier, MentalStateTracker, MentalStateClassifier
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
        use_simulator: bool = True,
        on_brain_state: Optional[Callable] = None,
        on_music_events: Optional[Callable] = None,
        on_visual_params: Optional[Callable] = None,
        on_audio_buffer: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Initialize real-time pipeline.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            use_simulator: Use simulator instead of real hardware
            on_brain_state: Callback for brain state updates
            on_music_events: Callback for music events
            on_visual_params: Callback for visual parameters
            on_error: Callback for errors
        """
        self.sampling_rate = sampling_rate
        self.use_simulator = use_simulator
        
        # Callbacks
        self.on_brain_state = on_brain_state
        self.on_music_events = on_music_events
        self.on_visual_params = on_visual_params
        self.on_audio_buffer = on_audio_buffer
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
        
        # Music generator (drives visual/brain-state events; audio now handled fully on frontend)
        # Visual parameter generator
        self.visual_generator = VisualParameterGenerator()
        
        # State
        self.is_running = False
        self.is_calibrated = False
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
            use_simulator=use_simulator
        )
    
    async def start(self):
        """Start the real-time pipeline."""
        if self.is_running:
            logger.warning("pipeline_already_running")
            return
        
        # Initialize EEG source
        if self.use_simulator:
            self.eeg_source = StreamingEEGSimulator(
                n_channels=8,
                sampling_rate=self.sampling_rate
            )
            self.eeg_source.start_stream()
            logger.info("using_eeg_simulator")
        else:
            # TODO: Initialize real hardware
            raise NotImplementedError("Real hardware not yet implemented")
        
        self.is_running = True
        self.session_start_time = time.time()
        self.sample_count = 0
        
        logger.info("pipeline_started")
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
    
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
    
    async def _processing_loop(self):
        """Main processing loop."""
        chunk_size = 32  # Process 32 samples at a time (~125ms at 256Hz)
        update_interval = chunk_size / self.sampling_rate  # Target time per iteration
        
        # Track last brain state for continuous updates
        last_brain_state = None
        last_visual_params = None
        
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
                if self.buffer.current_samples >= 256 and not is_artifact:
                    t0 = time.time()
                    feature_window = self.buffer.get_latest(duration=1.0)
                    features = self.extractor.extract_features(feature_window)
                    self.metrics['feature_extraction_time'].append(time.time() - t0)
                    
                    # 6. ML inference
                    if self.is_calibrated:
                        t0 = time.time()
                        feature_array = self._features_to_array(features)
                        brain_state = self.state_tracker.update(feature_array)
                        self.metrics['ml_inference_time'].append(time.time() - t0)
                    else:
                        # Use raw features without classification
                        # Normalize metrics to 0-1 range
                        focus_raw = features['focus_metric']
                        relax_raw = features['relax_metric']
                        
                        # Better normalization based on typical ranges:
                        # focus_metric: 0.1-1.5 (neutral=0.3, focus=1.3)
                        # relax_metric: 0.1-5.0 (neutral=1.3, relax=5.0)
                        focus_norm = min(max((focus_raw - 0.2) / 1.2, 0.0), 1.0)
                        relax_norm = min(max((relax_raw - 0.5) / 4.0, 0.0), 1.0)
                        
                        # Normalize to sum to 1.0
                        total = focus_norm + relax_norm
                        if total > 0:
                            focus_norm = focus_norm / (total + 0.5)  # Add baseline for neutral
                            relax_norm = relax_norm / (total + 0.5)
                        
                        neutral_norm = max(0.0, 1.0 - focus_norm - relax_norm)
                        
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
        Calibrate the pipeline with user data.
        
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
    
    def set_mental_state(self, state: str, intensity: float = 1.0):
        """
        Set simulator mental state (for testing).
        
        Args:
            state: Mental state ('neutral', 'focus', 'relax')
            intensity: State intensity (0-1)
        """
        if self.use_simulator and self.eeg_source:
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
