"""
SQLAlchemy database models.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    LargeBinary,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.data.database import Base


def generate_uuid():
    """Generate UUID for primary keys."""
    return str(uuid.uuid4())


class User(Base):
    """User model."""
    
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    settings = Column(JSON, default=dict)
    
    # Relationships
    calibrations = relationship("Calibration", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class Calibration(Base):
    """Calibration model - stores user-specific ML models."""
    
    __tablename__ = "calibrations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    validation_accuracy = Column(Float, nullable=True)
    num_samples = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Pickled model data
    model_data = Column(LargeBinary, nullable=True)
    
    # Baseline statistics (JSON)
    baseline_stats = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="calibrations")
    sessions = relationship("Session", back_populates="calibration")
    
    def __repr__(self):
        return f"<Calibration(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class Session(Base):
    """Session model - stores BCI session metadata."""
    
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    calibration_id = Column(String, ForeignKey("calibrations.id"), nullable=True)
    
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Aggregated statistics (JSON)
    stats = Column(JSON, default=dict)
    
    # Path to HDF5 recording file
    recording_path = Column(String(512), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    calibration = relationship("Calibration", back_populates="sessions")
    events = relationship("SessionEvent", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, started={self.started_at})>"


class SessionEvent(Base):
    """Session events - artifacts, state changes, user actions."""
    
    __tablename__ = "session_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # 'artifact', 'state_change', 'user_action'
    data = Column(JSON, default=dict)
    
    # Relationships
    session = relationship("Session", back_populates="events")
    
    def __repr__(self):
        return f"<SessionEvent(id={self.id}, session_id={self.session_id}, type={self.event_type})>"



