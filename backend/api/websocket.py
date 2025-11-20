"""
WebSocket endpoint for real-time EEG streaming.
"""

from typing import Dict, Optional
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from backend.pipeline.realtime_pipeline import RealtimePipeline
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
            
            logger.debug(
                "websocket_message_received",
                session_id=session_id,
                type=message_type
            )
            
            # Handle different message types
            if message_type == "start_session":
                await handle_start_session(session_id, message)
                
            elif message_type == "stop_session":
                await handle_stop_session(session_id)
                break
                
            elif message_type == "calibration_start":
                await handle_calibration_start(session_id, message)
                
            elif message_type == "calibration_label":
                await handle_calibration_label(session_id, message)
                
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
    # Create pipeline with callbacks
    async def on_brain_state(brain_state: dict):
        await manager.send_message(session_id, {
            "type": "brain_state",
            "data": brain_state,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def on_music_events(events: dict):
        # Convert MidiEvent objects to dicts
        serializable_events = {}
        for layer, layer_events in events.items():
            serializable_events[layer] = [
                {
                    "note": e.note,
                    "velocity": e.velocity,
                    "duration": e.duration,
                    "time": e.time
                }
                for e in layer_events
            ]
        
        await manager.send_message(session_id, {
            "type": "music_events",
            "data": serializable_events,
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
    protocol = message.get("protocol", "standard")
    
    # TODO: Initialize calibration session
    # TODO: Send calibration instructions
    
    await manager.send_message(session_id, {
        "type": "calibration_started",
        "protocol": protocol,
        "message": "Calibration started. Please follow instructions."
    })
    
    logger.info(
        "calibration_started",
        session_id=session_id,
        protocol=protocol
    )


async def handle_calibration_label(session_id: str, message: dict) -> None:
    """Handle calibration_label message."""
    state = message.get("state")
    
    # TODO: Mark current data with label
    # TODO: Update calibration progress
    
    await manager.send_message(session_id, {
        "type": "calibration_label_received",
        "state": state,
        "message": f"Label '{state}' recorded"
    })
    
    logger.debug(
        "calibration_label_received",
        session_id=session_id,
        state=state
    )



