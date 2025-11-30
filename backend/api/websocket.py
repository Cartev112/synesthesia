"""
WebSocket endpoint for real-time EEG streaming.
"""

from typing import Dict, Optional
import asyncio

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
    - calibration_start: Start calibration
    - calibration_label: Label calibration data
    
    Server sends:
    - brain_state: Current brain state features
    - music_events: Musical events to play
    - visual_params: Visual parameters
    - calibration_progress: Calibration progress
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
                
            elif message_type == "calibration_label":
                await handle_calibration_label(session_id, message)
                
            elif message_type == "calibration_train":
                await handle_calibration_train(session_id, message)
                
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
    
    pipeline = RealtimePipeline(
        sampling_rate=256,
        use_simulator=True,
        on_brain_state=on_brain_state,
        on_music_events=on_music_events,
        on_visual_params=on_visual_params,
        on_error=on_error
    )
    
    manager.pipelines[session_id] = pipeline
    await pipeline.start()
    
    await manager.send_message(session_id, {
        "type": "session_started",
        "session_id": session_id,
        "message": "Session started successfully"
    })
    
    logger.info("session_started", session_id=session_id)


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
    """Handle calibration_start message."""
    user_id = message.get("user_id", session_id)
    stage = message.get("stage", "baseline")
    
    # Create or get calibration protocol
    if session_id not in manager.calibration_protocols:
        manager.calibration_protocols[session_id] = CalibrationProtocol(user_id=user_id)
    
    protocol = manager.calibration_protocols[session_id]
    
    # Start the specified stage
    stage_info = protocol.start_stage(stage)
    
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
        user_id=user_id,
        stage=stage
    )


async def handle_calibration_label(session_id: str, message: dict) -> None:
    """Handle calibration_label message - add feature sample."""
    protocol = manager.calibration_protocols.get(session_id)
    
    if not protocol:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "NO_CALIBRATION_SESSION",
            "message": "No active calibration session"
        })
        return
    
    # Get features from message
    features = message.get("features")
    if features is None:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "MISSING_FEATURES",
            "message": "Features required for calibration sample"
        })
        return
    
    # Add sample to protocol
    import numpy as np
    features_array = np.array(features)
    protocol.add_sample(features_array)
    
    # Get progress
    progress = protocol.get_stage_progress()
    
    await manager.send_message(session_id, {
        "type": "calibration_progress",
        "progress": progress
    })
    


async def handle_calibration_train(session_id: str, message: dict) -> None:
    """Handle calibration_train message - train the model."""
    protocol = manager.calibration_protocols.get(session_id)
    
    if not protocol:
        await manager.send_message(session_id, {
            "type": "error",
            "code": "NO_CALIBRATION_SESSION",
            "message": "No active calibration session"
        })
        return
    
    try:
        # Train the model
        results = protocol.train_model()
        
        # Save the session
        session = protocol.save()
        
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
    
    await manager.send_message(session_id, {
        "type": "calibration_progress",
        "progress": progress
    })




