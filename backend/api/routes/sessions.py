"""
Session management endpoints.
"""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SessionCreate(BaseModel):
    """Session creation request."""
    user_id: str
    calibration_id: str | None = None


class SessionResponse(BaseModel):
    """Session response."""
    id: str
    user_id: str
    calibration_id: str | None
    started_at: str
    duration_seconds: int | None
    stats: dict | None


@router.get("/sessions")
async def list_sessions(user_id: str | None = None):
    """List all sessions, optionally filtered by user."""
    # TODO: Query sessions from database
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    # TODO: Query session from database
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/sessions", response_model=SessionResponse)
async def create_session(session: SessionCreate):
    """Create a new session."""
    # TODO: Create session in database
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/sessions/{session_id}/recording")
async def get_session_recording(session_id: str):
    """Download session recording (HDF5 file)."""
    # TODO: Stream HDF5 file
    raise HTTPException(status_code=501, detail="Not implemented")



