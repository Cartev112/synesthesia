"""
WebSocket endpoint for real-time EEG streaming.
"""

from typing import Dict, Optional
import asyncio
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from backend.pipeline.realtime_pipeline import RealtimePipeline
from backend.ml.calibration import CalibrationProtocol, CalibrationSession
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time streaming.
    """
    
    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.pipelines: Dict[str, RealtimePipeline] = {}
        self.calibration_protocols: Dict[str, CalibrationProtocol] = {}
        self.calibration_pipelines: Dict[str, RealtimePipeline] = {}  # Separate pipelines for calibration
    
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("websocket_connected", session_id=session_id)
    
    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.pop(session_id, None)
        logger.info("websocket_disconnected", session_id=session_id)
    
    async def send_message(self, session_id: str, message: dict) -> bool:
        """
        Send message to a specific session.
        
        Returns:
            True if sent successfully, False if connection doesn't exist
        """
        websocket = self.active_connections.get(session_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(message)
            return True
        return False
    
    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        for session_id, websocket in list(self.active_connections.items()):
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        "broadcast_failed",
                        session_id=session_id,
                        error=str(e)
                    )


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time EEG streaming.
    
    Client sends:
    - start_session: Initialize streaming
    - stop_session: Stop streaming
    - calibration_start: Start calibration protocol and pipeline
    - calibration_start_stage: Start a specific calibration stage
    - calibration_stop_stage: Stop current calibration stage
    - calibration_train: Train model from collected data
    - calibration_cancel: Cancel calibration
    
    Server sends:
    - brain_state: Current brain state features
    - music_events: Musical events to play
    - visual_params: Visual parameters
    - calibration_started: Calibration initialized
    - calibration_stage_started: Stage started with instructions
    - calibration_progress: Progress updates during stages
    - calibration_complete: Model trained successfully
    - error: Error messages
    """
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            # Handle different message types
            if message_type == "start_session":
                await handle_start_session(session_id, message)
                
            elif message_type == "stop_session":
                await handle_stop_session(session_id)
                # Don't break - keep connection open for potential restart
                
            elif message_type == "calibration_start":
                await handle_calibration_start(session_id, message)
                
            elif message_type == "calibration_start_stage":
                await handle_calibration_start_stage(session_id, message)
                
            elif message_type == "calibration_stop_stage":
                await handle_calibration_stop_stage(session_id, message)
                
            elif message_type == "calibration_train":
                await handle_calibration_train(session_id, message)
                
            elif message_type == "calibration_cancel":
                await handle_calibration_cancel(session_id, message)
                
            elif message_type == "calibration_progress":
                await handle_calibration_progress(session_id, message)
                
            else:
                await manager.send_message(session_id, {
                    "type": "error",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected", session_id=session_id)
    
    except Exception as e:
        logger.exception(
            "websocket_error",
            session_id=session_id,
            error=str(e)
        )
        await manager.send_message(session_id, {
            "type": "error",
            "code": "WEBSOCKET_ERROR",
            "message": str(e)
        })
    
    finally:
        manager.disconnect(session_id)


async def handle_start_session(session_id: str, message: dict) -> None:
    """Handle start_session message."""
    # Check if session is already running
    if session_id in manager.pipelines:
        logger.warning("session_already_active", session_id=session_id)
        await manager.send_message(session_id, {
            "type": "error",
            "code": "SESSION_ALREADY_ACTIVE",
            "message": "Session is already running"
        })
        return
    
    # Create pipeline with callbacks
    async def on_brain_state(brain_state: dict):
        await manager.send_message(session_id, {
            "type": "brain_state",
            "data": brain_state,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def on_music_events(events: dict):
        # Music generation is handled on the frontend now; keep callback for compatibility
        await manager.send_message(session_id, {
            "type": "music_events",
            "data": events,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def on_visual_params(params: dict):
        await manager.send_message(session_id, {
            "type": "visual_params",
            "data": params,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def on_error(error: str):
        await manager.send_message(session_id, {
            "type": "error",
            "code": "PIPELINE_ERROR",
            "message": error
        })
    
    # Get device configuration from message
    device_type = message.get("device_type", "simulator")
    device_address = message.get("device_address")
    device_preset = message.get("device_preset", "full_research")
    
    pipeline = RealtimePipeline(
        sampling_rate=256,
        device_type=device_type,
        device_address=device_address,
        device_preset=device_preset,
        on_brain_state=on_brain_state,
        on_music_events=on_music_events,
        on_visual_params=on_visual_params,
        on_error=on_error
    )
    
    # Check if there's a trained calibration to apply
    calibration_protocol = manager.calibration_protocols.get(session_id)
    if calibration_protocol and calibration_protocol.calibration.is_trained():
        pipeline.apply_calibration(calibration_protocol.calibration)
        logger.info("applied_calibration_to_session", session_id=session_id)
    
    manager.pipelines[session_id] = pipeline
    await pipeline.start()
    
    await manager.send_message(session_id, {
        "type": "session_started",
        "session_id": session_id,
        "message": "Session started successfully",
        "is_calibrated": pipeline.is_calibrated
    })
    
    logger.info("session_started", session_id=session_id, is_calibrated=pipeline.is_calibrated)


async def handle_stop_session(session_id: str) -> None:
    """Handle stop_session message."""
    pipeline = manager.pipelines.get(session_id)
    
    if pipeline:
        await pipeline.stop()
        metrics = pipeline.get_metrics()
        manager.pipelines.pop(session_id, None)
        
        await manager.send_message(session_id, {
            "type": "session_stopped",
            "session_id": session_id,
            "message": "Session stopped successfully",
            "metrics": metrics
        })
    else:
        await manager.send_message(session_id, {
            "type": "session_stopped",
            "session_id": session_id,
            "message": "No active session found"
        })
    
    logger.info("session_stopped", session_id=session_id)


async def handle_calibration_start(session_id: str, message: dict) -> None:
    """
    Handle calibration_start message - initialize calibration protocol and pipeline.
    
    This creates the calibration protocol and starts a dedicated pipeline
    for collecting EEG features during calibration.
    """
    user_id = message.get("user_id", session_id)
    device_type = message.get("device_type", "simulator")
    device_address = message.get("device_address")
    device_preset = message.get("device_preset", "full_research")
    
    # Check if calibration is already in progress
    if session_id in manager.calibration_protocols:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "CALIBRATION_ALREADY_ACTIVE",
            "message": "Calibration is already in progress"
        })
        return
    
    # Create calibration protocol
    protocol = CalibrationProtocol(user_id=user_id)
    manager.calibration_protocols[session_id] = protocol
    
    # Create feature callback that feeds into the calibration protocol
    sample_count = {"value": 0}  # Use dict to allow mutation in closure
    
    async def on_features(features: np.ndarray):
        """Callback to collect features during calibration."""
        try:
            # Only add sample if a stage is active (not validation)
            if protocol.current_stage and protocol.current_stage != 'validation':
                protocol.add_sample(features)
                sample_count["value"] += 1
                
                # Send progress update periodically (every 5 samples)
                if sample_count["value"] % 5 == 0:
                    progress = protocol.get_stage_progress()
                    progress['samples_collected'] = sample_count["value"]
                    await manager.send_message(session_id, {
                        "type": "calibration_progress",
                        "progress": progress
                    })
        except Exception as e:
            logger.error("calibration_feature_callback_error", error=str(e))
    
    async def on_error(error: str):
        await manager.send_message(session_id, {
            "type": "error",
            "code": "CALIBRATION_PIPELINE_ERROR",
            "message": error
        })
    
    # Create calibration pipeline
    pipeline = RealtimePipeline(
        sampling_rate=256,
        device_type=device_type,
        device_address=device_address,
        device_preset=device_preset,
        on_features=on_features,
        on_error=on_error
    )
    pipeline.set_calibration_mode(True)
    
    manager.calibration_pipelines[session_id] = pipeline
    
    # Start the pipeline
    try:
        await pipeline.start()
    except Exception as e:
        logger.exception("calibration_pipeline_start_failed", session_id=session_id)
        manager.calibration_protocols.pop(session_id, None)
        manager.calibration_pipelines.pop(session_id, None)
        await manager.send_message(session_id, {
            "type": "error",
            "code": "CALIBRATION_START_FAILED",
            "message": str(e)
        })
        return
    
    await manager.send_message(session_id, {
        "type": "calibration_started",
        "user_id": user_id,
        "device_type": device_type,
        "stages": ["baseline", "focus", "relax"],
        "stage_durations": {
            "baseline": protocol.BASELINE_DURATION,
            "focus": protocol.FOCUS_DURATION,
            "relax": protocol.RELAX_DURATION
        }
    })
    
    logger.info(
        "calibration_started",
        session_id=session_id,
        user_id=user_id,
        device_type=device_type
    )


async def handle_calibration_start_stage(session_id: str, message: dict) -> None:
    """Handle calibration_start_stage message - start a specific calibration stage."""
    protocol = manager.calibration_protocols.get(session_id)
    pipeline = manager.calibration_pipelines.get(session_id)
    
    if not protocol or not pipeline:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "NO_CALIBRATION_SESSION",
            "message": "No active calibration session. Call calibration_start first."
        })
        return
    
    stage = message.get("stage", "baseline")
    
    try:
        # Start the specified stage
        stage_info = protocol.start_stage(stage)
        
        # Set simulator state if applicable (for demo/testing)
        if pipeline.device_type == "simulator" and stage != 'validation':
            # Map stage to simulator mental state
            sim_state = 'neutral' if stage == 'baseline' else stage
            pipeline.set_mental_state(sim_state, intensity=1.0)
        
        await manager.send_message(session_id, {
            "type": "calibration_stage_started",
            "stage": stage_info['stage'],
            "duration": stage_info['duration'],
            "instructions": stage_info['instructions'],
            "state_label": stage_info['state_label']
        })
        
        logger.info(
            "calibration_stage_started",
            session_id=session_id,
            stage=stage
        )
        
    except ValueError as e:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "INVALID_STAGE",
            "message": str(e)
        })


async def handle_calibration_stop_stage(session_id: str, message: dict) -> None:
    """Handle calibration_stop_stage message - end current calibration stage."""
    protocol = manager.calibration_protocols.get(session_id)
    
    if not protocol:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "NO_CALIBRATION_SESSION",
            "message": "No active calibration session"
        })
        return
    
    # Get final progress for the stage
    progress = protocol.get_stage_progress()
    current_stage = protocol.current_stage
    
    # Clear current stage
    protocol.current_stage = None
    protocol.stage_start_time = None
    
    # Get sample counts
    sample_counts = {
        state: len(samples)
        for state, samples in protocol.calibration.calibration_data.items()
    }
    
    await manager.send_message(session_id, {
        "type": "calibration_stage_stopped",
        "stage": current_stage,
        "final_progress": progress,
        "sample_counts": sample_counts
    })
    
    logger.info(
        "calibration_stage_stopped",
        session_id=session_id,
        stage=current_stage,
        sample_counts=sample_counts
    )


async def handle_calibration_train(session_id: str, message: dict) -> None:
    """Handle calibration_train message - train the model and apply to session."""
    protocol = manager.calibration_protocols.get(session_id)
    calibration_pipeline = manager.calibration_pipelines.get(session_id)
    
    if not protocol:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "NO_CALIBRATION_SESSION",
            "message": "No active calibration session"
        })
        return
    
    try:
        # Stop the calibration pipeline
        if calibration_pipeline:
            await calibration_pipeline.stop()
            manager.calibration_pipelines.pop(session_id, None)
        
        # Train the model
        results = protocol.train_model()
        
        # Save the session (keeps the calibration in manager.calibration_protocols for later use)
        calibration_session = protocol.save()
        
        await manager.send_message(session_id, {
            "type": "calibration_complete",
            "validation_accuracy": results['validation_accuracy'],
            "sample_counts": results['sample_counts'],
            "training_time": results['training_time'],
            "feature_importance": results['feature_importance']
        })
        
        logger.info(
            "calibration_trained",
            session_id=session_id,
            accuracy=results['validation_accuracy']
        )
        
    except Exception as e:
        logger.exception("calibration_training_failed", session_id=session_id)
        await manager.send_message(session_id, {
            "type": "error",
            "code": "CALIBRATION_TRAINING_FAILED",
            "message": str(e)
        })


async def handle_calibration_cancel(session_id: str, message: dict) -> None:
    """Handle calibration_cancel message - cancel and cleanup calibration."""
    protocol = manager.calibration_protocols.get(session_id)
    pipeline = manager.calibration_pipelines.get(session_id)
    
    # Stop pipeline if running
    if pipeline:
        await pipeline.stop()
        manager.calibration_pipelines.pop(session_id, None)
    
    # Remove protocol
    if protocol:
        manager.calibration_protocols.pop(session_id, None)
    
    await manager.send_message(session_id, {
        "type": "calibration_cancelled",
        "message": "Calibration cancelled"
    })
    
    logger.info("calibration_cancelled", session_id=session_id)


async def handle_calibration_progress(session_id: str, message: dict) -> None:
    """Handle calibration_progress request - get current progress."""
    protocol = manager.calibration_protocols.get(session_id)
    
    if not protocol:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "NO_CALIBRATION_SESSION",
            "message": "No active calibration session"
        })
        return
    
    progress = protocol.get_stage_progress()
    
    # Add sample counts
    sample_counts = {
        state: len(samples)
        for state, samples in protocol.calibration.calibration_data.items()
    }
    progress['sample_counts'] = sample_counts
    
    await manager.send_message(session_id, {
        "type": "calibration_progress",
        "progress": progress
    })
