from typing import Any

from src.mlops.health_monitor import HealthMonitor


def create_metrics_app():
    from fastapi import FastAPI

    app = FastAPI(
        title="Medical AI Agent MLOps Metrics",
        description="Infrastructure health and runtime metrics for the Medical AI Agent stack.",
        version="1.0.0",
    )
    monitor = HealthMonitor()

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return monitor.collect()

    return app
