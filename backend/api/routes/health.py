"""
Health check endpoints.
"""

from fastapi import APIRouter

from backend.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    System health check endpoint.
    
    Returns system status and basic diagnostics.
    """
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.env,
        "components": {
            "eeg_device": "not_implemented",  # TODO: Check device status
            "ml_models": "not_implemented",   # TODO: Check ML models loaded
            "database": "not_implemented",    # TODO: Check database connection
            "redis": "not_implemented"        # TODO: Check Redis connection
        }
    }


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint (placeholder).
    
    TODO: Implement Prometheus metrics export
    """
    return {
        "active_sessions": 0,
        "total_sessions": 0,
        "avg_latency_ms": 0.0,
        "error_count": 0
    }



