from typing import Any
from fastapi import APIRouter
from src.mlops.gpu_monitor import GPUMonitor

router = APIRouter(tags=["System Health & Metrics"])

gpu_monitor = GPUMonitor()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Returns platform health status and operational state.
    """
    return {
        "status": "healthy",
        "service": "Clinical Intelligence Platform",
        "version": "1.0.0"
    }


@router.get("/metrics")
async def get_system_metrics() -> dict[str, Any]:
    """
    Returns GPU/CPU system telemetry metrics.
    """
    gpu_stats = gpu_monitor.collect()
    return {
        "gpu_metrics": gpu_stats,
        "active": True
    }
