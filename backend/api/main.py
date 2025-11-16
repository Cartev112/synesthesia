"""
Main FastAPI application for Synesthesia BCI system.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import health, sessions, users
from backend.api.websocket import router as websocket_router
from backend.core.config import settings
from backend.core.exceptions import SynesthesiaError
from backend.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        env=settings.env
    )
    
    # TODO: Initialize database connection pool
    # TODO: Initialize Redis connection
    # TODO: Load ML models
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    
    # TODO: Close database connections
    # TODO: Close Redis connection
    # TODO: Cleanup resources


# Create FastAPI application
app = FastAPI(
    title="Synesthesia BCI API",
    description="Real-time Brain-Computer Interface for Music and Art Generation",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware (configure based on frontend needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(SynesthesiaError)
async def synesthesia_error_handler(request, exc: SynesthesiaError) -> JSONResponse:
    """Handle custom Synesthesia errors."""
    logger.error(
        "synesthesia_error",
        error_code=exc.code,
        message=exc.message,
        path=request.url.path
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected errors."""
    logger.exception(
        "unexpected_error",
        error_type=type(exc).__name__,
        path=request.url.path
    )
    
    if settings.is_development:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "type": type(exc).__name__
                }
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred"
                }
            }
        )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(users.router, prefix="/api/v1", tags=["users"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
app.include_router(websocket_router, tags=["websocket"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )



