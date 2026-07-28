from src.mlops.gpu_monitor import GPUMonitor


def test_gpu_monitor_returns_list_payload():
    metrics = GPUMonitor().collect()

    assert isinstance(metrics, list)
    assert "available" in metrics[0]
