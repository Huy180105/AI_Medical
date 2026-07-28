from src.mlops.health_monitor import HealthMonitor


class FakeGPUMonitor:
    def collect(self):
        return [{"available": False}]


def test_health_monitor_payload_shape():
    monitor = HealthMonitor(gpu_monitor=FakeGPUMonitor(), redis_url="redis://localhost:1/0", postgres_dsn="bad")

    payload = monitor.collect()

    assert "gpu" in payload
    assert "system" in payload
    assert "redis" in payload
    assert "postgres" in payload
    assert "latency" in payload
