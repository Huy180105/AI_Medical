from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class GPUMetrics:
    available: bool
    name: str | None = None
    cuda_version: str | None = None
    memory_total_mb: float | None = None
    memory_used_mb: float | None = None
    memory_free_mb: float | None = None
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUMonitor:
    def collect(self) -> list[dict[str, Any]]:
        try:
            import pynvml
        except ImportError:
            return [GPUMetrics(available=False, error="pynvml_unavailable").to_dict()]

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            cuda_version = str(pynvml.nvmlSystemGetCudaDriverVersion())
            metrics: list[dict[str, Any]] = []
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                metrics.append(
                    GPUMetrics(
                        available=True,
                        name=str(name),
                        cuda_version=cuda_version,
                        memory_total_mb=round(memory.total / (1024 * 1024), 2),
                        memory_used_mb=round(memory.used / (1024 * 1024), 2),
                        memory_free_mb=round(memory.free / (1024 * 1024), 2),
                        utilization_percent=float(utilization.gpu),
                        temperature_c=float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                        power_watts=round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 2),
                    ).to_dict()
                )
            return metrics or [GPUMetrics(available=False, error="no_cuda_device").to_dict()]
        except Exception as exc:
            return [GPUMetrics(available=False, error=str(exc)).to_dict()]
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
