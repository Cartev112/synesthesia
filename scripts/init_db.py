"""
Initialize database - create all tables.

Run this script to set up the database schema:
    python scripts/init_db.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from backend.data.database import init_db
from backend.core.logging import get_logger

logger = get_logger(__name__)


def main():
    """Initialize the database."""
    logger.info("Starting database initialization...")
    
    try:
        init_db()
        logger.info("Database initialization complete!")
        print("✓ Database initialized successfully")
        
    except Exception as e:
        logger.exception("Database initialization failed", error=str(e))
        print(f"✗ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



