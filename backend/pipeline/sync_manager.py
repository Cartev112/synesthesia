"""
Sync Manager for multi-user BCI sessions.

Coordinates two RealtimePipeline instances and computes
inter-brain synchrony for music/visual modulation.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from backend.pipeline.realtime_pipeline import RealtimePipeline
from backend.pipeline.sync_metrics import FeatureFrame, SyncState, SyncMetricsCalculator
from backend.core.logging import get_logger

logger = get_logger(__name__)


class SyncSessionPhase(Enum):
    """Phases of a sync session."""
    IDLE = "idle"
    CONNECTING = "connecting"
    BASELINE = "baseline"
    ACTIVE = "active"
    STOPPED = "stopped"


@dataclass
class SyncSessionConfig:
    """Configuration for a sync session."""
    session_id: str
    baseline_duration: float = 30.0  # seconds
    sync_update_rate: float = 4.0    # Hz (updates per second)
    device_type_a: str = "simulator"
    device_type_b: str = "simulator"
    device_address_a: Optional[str] = None
    device_address_b: Optional[str] = None


@dataclass
class SyncSessionState:
    """Current state of a sync session."""
    phase: SyncSessionPhase = SyncSessionPhase.IDLE
    user_a_connected: bool = False
    user_b_connected: bool = False
    baseline_complete: bool = False
    baseline_progress: float = 0.0
    current_sync: Optional[SyncState] = None
    start_time: Optional[float] = None
    error: Optional[str] = None


class SyncSession:
    """
    Manages a synchronized BCI session between two users.
    
    Coordinates two RealtimePipeline instances, collects feature
    frames, computes synchrony metrics, and outputs sync state
    for music/visual generation.
    """
    
    def __init__(
        self,
        config: SyncSessionConfig,
        on_sync_state: Optional[Callable] = None,
        on_phase_change: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Initialize sync session.
        
        Args:
            config: Session configuration
            on_sync_state: Async callback for sync state updates
            on_phase_change: Async callback for phase changes
            on_error: Async callback for errors
        """
        self.config = config
        self.on_sync_state = on_sync_state
        self.on_phase_change = on_phase_change
        self.on_error = on_error
        
        # Pipelines for each user
        self.pipeline_a: Optional[RealtimePipeline] = None
        self.pipeline_b: Optional[RealtimePipeline] = None
        
        # Sync metrics calculator
        self.metrics_calculator = SyncMetricsCalculator(
            window_size=10,
            ema_alpha=0.3,
            hysteresis_threshold=0.08
        )
        
        # Session state
        self.state = SyncSessionState()
        
        # Processing task
        self._sync_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Feature collection queues
        self._frame_queue_a: asyncio.Queue[FeatureFrame] = asyncio.Queue(maxsize=50)
        self._frame_queue_b: asyncio.Queue[FeatureFrame] = asyncio.Queue(maxsize=50)
        
        # Baseline timing
        self._baseline_start_time: Optional[float] = None
        
        logger.info(
            "sync_session_created",
            session_id=config.session_id,
            device_a=config.device_type_a,
            device_b=config.device_type_b
        )
    
    async def start(self) -> None:
        """Start the sync session and both pipelines."""
        if self._is_running:
            logger.warning("sync_session_already_running")
            return
        
        self._is_running = True
        self.state.phase = SyncSessionPhase.CONNECTING
        self.state.start_time = time.time()
        
        await self._notify_phase_change()
        
        try:
            # Create pipelines for both users
            await self._create_pipelines()
            
            # Start pipelines
            await self.pipeline_a.start()
            self.state.user_a_connected = True
            logger.info("pipeline_a_started")
            
            await self.pipeline_b.start()
            self.state.user_b_connected = True
            logger.info("pipeline_b_started")
            
            # Start sync processing loop
            self._sync_task = asyncio.create_task(self._sync_processing_loop())
            
            # Transition to baseline phase
            await self._start_baseline_phase()
            
        except Exception as e:
            self.state.error = str(e)
            self.state.phase = SyncSessionPhase.STOPPED
            await self._notify_error(str(e))
            logger.exception("sync_session_start_failed")
            raise
    
    async def stop(self) -> Dict:
        """
        Stop the sync session and cleanup.
        
        Returns:
            Session metrics and statistics
        """
        if not self._is_running:
            return {}
        
        self._is_running = False
        self.state.phase = SyncSessionPhase.STOPPED
        
        # Stop sync task
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        # Stop pipelines
        metrics_a = {}
        metrics_b = {}
        
        if self.pipeline_a:
            await self.pipeline_a.stop()
            metrics_a = self.pipeline_a.get_metrics()
        
        if self.pipeline_b:
            await self.pipeline_b.stop()
            metrics_b = self.pipeline_b.get_metrics()
        
        await self._notify_phase_change()
        
        session_duration = time.time() - self.state.start_time if self.state.start_time else 0
        
        logger.info(
            "sync_session_stopped",
            session_id=self.config.session_id,
            duration=session_duration
        )
        
        return {
            'session_id': self.config.session_id,
            'duration': session_duration,
            'pipeline_a_metrics': metrics_a,
            'pipeline_b_metrics': metrics_b,
            'baseline_complete': self.state.baseline_complete
        }
    
    async def _create_pipelines(self) -> None:
        """Create RealtimePipeline instances for both users."""
        
        # Feature callbacks that route to our queues
        async def on_features_a(features: NDArray[np.float64], feature_dict: Dict, is_artifact: bool):
            frame = FeatureFrame(
                timestamp=time.time(),
                user_id='user_a',
                features=feature_dict,
                feature_array=features,
                is_artifact=is_artifact
            )
            try:
                self._frame_queue_a.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop oldest frame
                try:
                    self._frame_queue_a.get_nowait()
                    self._frame_queue_a.put_nowait(frame)
                except:
                    pass
        
        async def on_features_b(features: NDArray[np.float64], feature_dict: Dict, is_artifact: bool):
            frame = FeatureFrame(
                timestamp=time.time(),
                user_id='user_b',
                features=feature_dict,
                feature_array=features,
                is_artifact=is_artifact
            )
            try:
                self._frame_queue_b.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    self._frame_queue_b.get_nowait()
                    self._frame_queue_b.put_nowait(frame)
                except:
                    pass
        
        # Pipeline A
        self.pipeline_a = RealtimePipeline(
            sampling_rate=256,
            device_type=self.config.device_type_a,
            device_address=self.config.device_address_a,
            on_sync_features=on_features_a
        )
        
        # Pipeline B
        self.pipeline_b = RealtimePipeline(
            sampling_rate=256,
            device_type=self.config.device_type_b,
            device_address=self.config.device_address_b,
            on_sync_features=on_features_b
        )
    
    async def _start_baseline_phase(self) -> None:
        """Start the baseline collection phase."""
        self.state.phase = SyncSessionPhase.BASELINE
        self.state.baseline_progress = 0.0
        self._baseline_start_time = time.time()
        
        self.metrics_calculator.start_baseline()
        
        await self._notify_phase_change()
        
        logger.info(
            "baseline_phase_started",
            duration=self.config.baseline_duration
        )
    
    async def _finish_baseline_phase(self) -> None:
        """Finish baseline and transition to active phase."""
        baseline_stats = self.metrics_calculator.finish_baseline()
        self.state.baseline_complete = True
        self.state.baseline_progress = 1.0
        self.state.phase = SyncSessionPhase.ACTIVE
        
        await self._notify_phase_change()
        
        logger.info(
            "baseline_phase_complete",
            stats=baseline_stats
        )
    
    async def _sync_processing_loop(self) -> None:
        """Main sync processing loop."""
        update_interval = 1.0 / self.config.sync_update_rate
        
        while self._is_running:
            try:
                loop_start = time.time()
                
                # Collect frames from queues
                await self._collect_frames()
                
                # Update baseline progress if in baseline phase
                if self.state.phase == SyncSessionPhase.BASELINE:
                    elapsed = time.time() - self._baseline_start_time
                    self.state.baseline_progress = min(elapsed / self.config.baseline_duration, 1.0)
                    
                    if elapsed >= self.config.baseline_duration:
                        await self._finish_baseline_phase()
                
                # Compute sync metrics
                sync_state = self.metrics_calculator.compute_sync()
                
                if sync_state is not None:
                    self.state.current_sync = sync_state
                    
                    # Only emit during active phase
                    if self.state.phase == SyncSessionPhase.ACTIVE:
                        await self._notify_sync_state(sync_state)
                
                # Maintain update rate
                elapsed = time.time() - loop_start
                sleep_time = max(0.01, update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("sync_processing_error", error=str(e))
                await self._notify_error(str(e))
    
    async def _collect_frames(self) -> None:
        """Collect frames from both user queues."""
        # Collect from A
        while not self._frame_queue_a.empty():
            try:
                frame = self._frame_queue_a.get_nowait()
                self.metrics_calculator.add_frame(frame)
            except asyncio.QueueEmpty:
                break
        
        # Collect from B
        while not self._frame_queue_b.empty():
            try:
                frame = self._frame_queue_b.get_nowait()
                self.metrics_calculator.add_frame(frame)
            except asyncio.QueueEmpty:
                break
    
    async def _notify_sync_state(self, sync_state: SyncState) -> None:
        """Notify callback of sync state update."""
        if self.on_sync_state:
            try:
                await self.on_sync_state(sync_state)
            except Exception as e:
                logger.error("sync_state_callback_error", error=str(e))
    
    async def _notify_phase_change(self) -> None:
        """Notify callback of phase change."""
        if self.on_phase_change:
            try:
                await self.on_phase_change(self.state.phase.value, self.state)
            except Exception as e:
                logger.error("phase_change_callback_error", error=str(e))
    
    async def _notify_error(self, error: str) -> None:
        """Notify callback of error."""
        if self.on_error:
            try:
                await self.on_error(error)
            except Exception as e:
                logger.error("error_callback_error", error=str(e))
    
    def get_state(self) -> Dict:
        """Get current session state as dictionary."""
        return {
            'session_id': self.config.session_id,
            'phase': self.state.phase.value,
            'user_a_connected': self.state.user_a_connected,
            'user_b_connected': self.state.user_b_connected,
            'baseline_complete': self.state.baseline_complete,
            'baseline_progress': self.state.baseline_progress,
            'current_sync': self.state.current_sync.to_dict() if self.state.current_sync else None,
            'start_time': self.state.start_time,
            'error': self.state.error
        }
    
    def set_simulator_states(self, state_a: str, state_b: str, intensity: float = 1.0) -> None:
        """
        Set mental states for both simulators (for testing).
        
        Args:
            state_a: Mental state for user A ('neutral', 'focus', 'relax')
            state_b: Mental state for user B
            intensity: State intensity (0-1)
        """
        if self.pipeline_a and self.config.device_type_a == "simulator":
            self.pipeline_a.set_mental_state(state_a, intensity)
        
        if self.pipeline_b and self.config.device_type_b == "simulator":
            self.pipeline_b.set_mental_state(state_b, intensity)


class SyncManager:
    """
    Manager for multiple sync sessions.
    
    Handles session lifecycle and provides session lookup.
    """
    
    def __init__(self):
        self.sessions: Dict[str, SyncSession] = {}
        logger.info("sync_manager_initialized")
    
    async def create_session(
        self,
        session_id: str,
        config: Optional[SyncSessionConfig] = None,
        on_sync_state: Optional[Callable] = None,
        on_phase_change: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> SyncSession:
        """
        Create a new sync session.
        
        Args:
            session_id: Unique session identifier
            config: Session configuration (uses defaults if None)
            on_sync_state: Callback for sync state updates
            on_phase_change: Callback for phase changes
            on_error: Callback for errors
            
        Returns:
            Created SyncSession
        """
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        if config is None:
            config = SyncSessionConfig(session_id=session_id)
        
        session = SyncSession(
            config=config,
            on_sync_state=on_sync_state,
            on_phase_change=on_phase_change,
            on_error=on_error
        )
        
        self.sessions[session_id] = session
        
        logger.info("sync_session_created_by_manager", session_id=session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SyncSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    async def stop_session(self, session_id: str) -> Optional[Dict]:
        """
        Stop and remove a session.
        
        Returns:
            Session metrics, or None if session doesn't exist
        """
        session = self.sessions.pop(session_id, None)
        
        if session:
            return await session.stop()
        
        return None
    
    async def stop_all(self) -> None:
        """Stop all active sessions."""
        for session_id in list(self.sessions.keys()):
            await self.stop_session(session_id)
        
        logger.info("all_sync_sessions_stopped")
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())

