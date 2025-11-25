"""
Database setup and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI routes to get database session.
    
    Usage:
        @app.get("/users")
        async def get_users(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database - create all tables.
    
    Should be called on application startup or via migration tool.
    """
    logger.info("initializing_database")
    Base.metadata.create_all(bind=engine)
    logger.info("database_initialized")


def drop_db() -> None:
    """
    Drop all tables.
    
    WARNING: This will delete all data!
    Should only be used in development/testing.
    """
    if not settings.is_development:
        raise RuntimeError("Cannot drop database in production!")
    
    logger.warning("dropping_all_tables")
    Base.metadata.drop_all(bind=engine)
    logger.warning("all_tables_dropped")





