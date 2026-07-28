import time
from typing import Any

import requests

from src.mlops.gpu_monitor import GPUMonitor
from src.utils.config import Config


class HealthMonitor:
    def __init__(
        self,
        gpu_monitor: GPUMonitor | None = None,
        redis_url: str | None = None,
        postgres_dsn: str | None = None,
        latency_probe_url: str | None = None,
    ) -> None:
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self.redis_url = redis_url or Config.REDIS_URL
        self.postgres_dsn = postgres_dsn or Config.POSTGRES_DSN
        self.latency_probe_url = latency_probe_url or Config.METRICS_LATENCY_PROBE_URL

    def collect(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "system": self._system_metrics(),
            "gpu": self.gpu_monitor.collect(),
            "redis": self._redis_health(),
            "postgres": self._postgres_health(),
            "latency": self._latency_health(),
        }

    def _system_metrics(self) -> dict[str, Any]:
        try:
            import psutil
        except ImportError:
            return {"error": "psutil_unavailable"}

        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "ram_total_mb": round(memory.total / (1024 * 1024), 2),
            "ram_used_mb": round(memory.used / (1024 * 1024), 2),
            "ram_percent": memory.percent,
        }

    def _redis_health(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            import redis

            client = redis.Redis.from_url(self.redis_url, socket_connect_timeout=1, socket_timeout=1)
            ok = bool(client.ping())
            return {"available": ok, "latency_ms": self._elapsed_ms(start)}
        except Exception as exc:
            return {"available": False, "latency_ms": self._elapsed_ms(start), "error": str(exc)}

    def _postgres_health(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            import psycopg

            with psycopg.connect(self.postgres_dsn, connect_timeout=1) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            return {"available": True, "latency_ms": self._elapsed_ms(start)}
        except Exception as exc:
            return {"available": False, "latency_ms": self._elapsed_ms(start), "error": str(exc)}

    def _latency_health(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            response = requests.get(self.latency_probe_url, timeout=2)
            return {
                "available": response.ok,
                "status_code": response.status_code,
                "latency_ms": self._elapsed_ms(start),
            }
        except Exception as exc:
            return {"available": False, "latency_ms": self._elapsed_ms(start), "error": str(exc)}

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000.0, 4)
