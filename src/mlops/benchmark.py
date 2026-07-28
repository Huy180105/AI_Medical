import statistics
import time
from pathlib import Path
from typing import Any, Protocol

from src.mlops.gpu_monitor import GPUMonitor
from src.mlops.report_exporter import BenchmarkReportExporter
from src.utils.config import Config


class AgentLike(Protocol):
    def process(self, text: str) -> dict[str, Any]:
        ...


class BenchmarkRunner:
    def __init__(
        self,
        gpu_monitor: GPUMonitor | None = None,
        exporter: BenchmarkReportExporter | None = None,
    ) -> None:
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self.exporter = exporter or BenchmarkReportExporter()

    def run_agent_benchmark(
        self,
        agent: AgentLike,
        samples: list[str],
        warmup_runs: int = 1,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        for sample in samples[:warmup_runs]:
            agent.process(sample)

        runs: list[dict[str, Any]] = []
        for sample in samples:
            start = time.perf_counter()
            result = agent.process(sample)
            end_to_end_ms = (time.perf_counter() - start) * 1000.0
            processing_time = result.get("processing_time", {})
            runs.append(
                {
                    "end_to_end_ms": round(end_to_end_ms, 4),
                    "ner_ms": processing_time.get("ner_ms"),
                    "retriever_ms": processing_time.get("retriever_ms"),
                    "reasoner_ms": processing_time.get("reasoner_ms"),
                    "agent_ms": processing_time.get("total_ms", end_to_end_ms),
                    "confidence": result.get("confidence"),
                }
            )

        report = {
            "sample_count": len(samples),
            "latency": self._latency_summary(runs),
            "gpu": self.gpu_monitor.collect(),
            "runs": runs,
        }

        if output_dir:
            output = Path(output_dir)
            self.exporter.export_json(report, str(output / "benchmark.json"))
            self.exporter.export_csv(report, str(output / "benchmark.csv"))
            self.exporter.export_html(report, str(output / "benchmark.html"))

        return report

    def default_output_dir(self) -> str:
        return str(Path(Config.EXPERIMENTS_DIR) / "benchmarks")

    def _latency_summary(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in ["end_to_end_ms", "ner_ms", "retriever_ms", "reasoner_ms", "agent_ms"]:
            values = [float(run[key]) for run in runs if run.get(key) is not None]
            if not values:
                continue
            summary[key] = {
                "avg": round(statistics.mean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "p95": round(self._percentile(values, 95), 4),
            }
        return summary

    def _percentile(self, values: list[float], percentile: int) -> float:
        if len(values) == 1:
            return values[0]
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
