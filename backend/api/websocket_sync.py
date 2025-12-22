"""
WebSocket endpoint for multi-user sync sessions.
"""

from typing import Dict, Optional
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from backend.pipeline.sync_manager import (
    SyncManager,
    SyncSession,
    SyncSessionConfig,
    SyncSessionPhase,
)
from backend.pipeline.sync_metrics import SyncState
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SyncConnectionManager:
    """
    Manages WebSocket connections for sync sessions.
    """
    
    def __init__(self) -> None:
        # Map of sync_session_id -> {user_id: websocket}
        self.connections: Dict[str, Dict[str, WebSocket]] = {}
        self.sync_manager = SyncManager()
    
    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str
    ) -> None:
        """Accept and register a WebSocket connection for a sync session."""
        await websocket.accept()
        
        if session_id not in self.connections:
            self.connections[session_id] = {}
        
        self.connections[session_id][user_id] = websocket
        
        logger.info(
            "sync_websocket_connected",
            session_id=session_id,
            user_id=user_id
        )
    
    def disconnect(self, session_id: str, user_id: str) -> None:
        """Remove a WebSocket connection."""
        if session_id in self.connections:
            self.connections[session_id].pop(user_id, None)
            
            # Clean up empty sessions
            if not self.connections[session_id]:
                del self.connections[session_id]
        
        logger.info(
            "sync_websocket_disconnected",
            session_id=session_id,
            user_id=user_id
        )
    
    async def send_to_user(
        self,
        session_id: str,
        user_id: str,
        message: dict
    ) -> bool:
        """Send message to a specific user in a session."""
        if session_id not in self.connections:
            return False
        
        websocket = self.connections[session_id].get(user_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(message)
            return True
        return False
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message: dict
    ) -> None:
        """Broadcast message to all users in a sync session."""
        if session_id not in self.connections:
            return
        
        for user_id, websocket in list(self.connections[session_id].items()):
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        "sync_broadcast_failed",
                        session_id=session_id,
                        user_id=user_id,
                        error=str(e)
                    )
    
    def get_session_users(self, session_id: str) -> list[str]:
        """Get list of connected user IDs for a session."""
        if session_id not in self.connections:
            return []
        return list(self.connections[session_id].keys())


# Global connection manager
sync_manager = SyncConnectionManager()


@router.websocket("/ws/sync/{session_id}/{user_id}")
async def sync_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    user_id: str
):
    """
    WebSocket endpoint for multi-user sync sessions.
    
    Each user connects with their user_id (user_a or user_b).
    
    Client sends:
    - sync_create: Create a new sync session (host only)
    - sync_join: Join an existing session
    - sync_start: Start the sync session (after both users join)
    - sync_stop: Stop the sync session
    - set_simulator_state: Set simulator states (for testing)
    
    Server sends:
    - sync_created: Session created successfully
    - sync_user_joined: A user joined the session
    - sync_started: Session started (baseline phase begins)
    - sync_phase_changed: Session phase changed
    - sync_state: Real-time sync state updates
    - sync_stopped: Session stopped
    - error: Error messages
    """
    await sync_manager.connect(websocket, session_id, user_id)
    
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            if message_type == "sync_create":
                await handle_sync_create(session_id, user_id, message)
                
            elif message_type == "sync_join":
                await handle_sync_join(session_id, user_id, message)
                
            elif message_type == "sync_start":
                await handle_sync_start(session_id, user_id, message)
                
            elif message_type == "sync_stop":
                await handle_sync_stop(session_id, user_id, message)
                
            elif message_type == "set_simulator_state":
                await handle_set_simulator_state(session_id, message)
                
            else:
                await sync_manager.send_to_user(session_id, user_id, {
                    "type": "error",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(
            "sync_client_disconnected",
            session_id=session_id,
            user_id=user_id
        )
    
    except Exception as e:
        logger.exception(
            "sync_websocket_error",
            session_id=session_id,
            user_id=user_id,
            error=str(e)
        )
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "WEBSOCKET_ERROR",
            "message": str(e)
        })
    
    finally:
        sync_manager.disconnect(session_id, user_id)
        
        # Stop session if both users disconnect
        if not sync_manager.get_session_users(session_id):
            session = sync_manager.sync_manager.get_session(session_id)
            if session:
                await sync_manager.sync_manager.stop_session(session_id)


async def handle_sync_create(
    session_id: str,
    user_id: str,
    message: dict
) -> None:
    """Handle sync_create message - create a new sync session."""
    device_type_a = message.get("device_type_a", "simulator")
    device_type_b = message.get("device_type_b", "simulator")
    device_address_a = message.get("device_address_a")
    device_address_b = message.get("device_address_b")
    baseline_duration = message.get("baseline_duration", 30.0)
    
    # Check if session already exists
    existing = sync_manager.sync_manager.get_session(session_id)
    if existing:
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "SESSION_EXISTS",
            "message": f"Sync session {session_id} already exists"
        })
        return
    
    # Create callbacks for this session
    async def on_sync_state(state: SyncState):
        await sync_manager.broadcast_to_session(session_id, {
            "type": "sync_state",
            "data": state.to_dict(),
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def on_phase_change(phase: str, state):
        await sync_manager.broadcast_to_session(session_id, {
            "type": "sync_phase_changed",
            "phase": phase,
            "baseline_progress": state.baseline_progress,
            "baseline_complete": state.baseline_complete
        })
    
    async def on_error(error: str):
        await sync_manager.broadcast_to_session(session_id, {
            "type": "error",
            "code": "SYNC_SESSION_ERROR",
            "message": error
        })
    
    # Create session config
    config = SyncSessionConfig(
        session_id=session_id,
        baseline_duration=baseline_duration,
        device_type_a=device_type_a,
        device_type_b=device_type_b,
        device_address_a=device_address_a,
        device_address_b=device_address_b,
    )
    
    try:
        session = await sync_manager.sync_manager.create_session(
            session_id=session_id,
            config=config,
            on_sync_state=on_sync_state,
            on_phase_change=on_phase_change,
            on_error=on_error
        )
        
        await sync_manager.broadcast_to_session(session_id, {
            "type": "sync_created",
            "session_id": session_id,
            "config": {
                "baseline_duration": config.baseline_duration,
                "device_type_a": config.device_type_a,
                "device_type_b": config.device_type_b,
            },
            "users_connected": sync_manager.get_session_users(session_id)
        })
        
        logger.info(
            "sync_session_created",
            session_id=session_id,
            created_by=user_id
        )
        
    except Exception as e:
        logger.exception("sync_create_failed", session_id=session_id)
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "SYNC_CREATE_FAILED",
            "message": str(e)
        })


async def handle_sync_join(
    session_id: str,
    user_id: str,
    message: dict
) -> None:
    """Handle sync_join message - join an existing session."""
    session = sync_manager.sync_manager.get_session(session_id)
    
    if not session:
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "SESSION_NOT_FOUND",
            "message": f"Sync session {session_id} not found"
        })
        return
    
    # Notify all users that someone joined
    await sync_manager.broadcast_to_session(session_id, {
        "type": "sync_user_joined",
        "user_id": user_id,
        "users_connected": sync_manager.get_session_users(session_id),
        "session_state": session.get_state()
    })
    
    logger.info(
        "user_joined_sync_session",
        session_id=session_id,
        user_id=user_id
    )


async def handle_sync_start(
    session_id: str,
    user_id: str,
    message: dict
) -> None:
    """Handle sync_start message - start the sync session."""
    session = sync_manager.sync_manager.get_session(session_id)
    
    if not session:
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "SESSION_NOT_FOUND",
            "message": f"Sync session {session_id} not found"
        })
        return
    
    # Check if both users are connected
    users = sync_manager.get_session_users(session_id)
    if len(users) < 2:
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "WAITING_FOR_PARTNER",
            "message": "Waiting for partner to connect",
            "users_connected": users
        })
        return
    
    try:
        await session.start()
        
        await sync_manager.broadcast_to_session(session_id, {
            "type": "sync_started",
            "session_id": session_id,
            "phase": session.state.phase.value,
            "baseline_duration": session.config.baseline_duration
        })
        
        logger.info(
            "sync_session_started",
            session_id=session_id,
            started_by=user_id
        )
        
    except Exception as e:
        logger.exception("sync_start_failed", session_id=session_id)
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "error",
            "code": "SYNC_START_FAILED",
            "message": str(e)
        })


async def handle_sync_stop(
    session_id: str,
    user_id: str,
    message: dict
) -> None:
    """Handle sync_stop message - stop the sync session."""
    metrics = await sync_manager.sync_manager.stop_session(session_id)
    
    if metrics:
        await sync_manager.broadcast_to_session(session_id, {
            "type": "sync_stopped",
            "session_id": session_id,
            "metrics": metrics
        })
    else:
        await sync_manager.send_to_user(session_id, user_id, {
            "type": "sync_stopped",
            "session_id": session_id,
            "message": "Session not found or already stopped"
        })
    
    logger.info(
        "sync_session_stopped",
        session_id=session_id,
        stopped_by=user_id
    )


async def handle_set_simulator_state(
    session_id: str,
    message: dict
) -> None:
    """Handle set_simulator_state message - set simulator states for testing."""
    session = sync_manager.sync_manager.get_session(session_id)
    
    if not session:
        return
    
    state_a = message.get("state_a", "neutral")
    state_b = message.get("state_b", "neutral")
    intensity = message.get("intensity", 1.0)
    
    session.set_simulator_states(state_a, state_b, intensity)
    
    logger.debug(
        "simulator_states_set",
        session_id=session_id,
        state_a=state_a,
        state_b=state_b
    )

