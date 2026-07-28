from src.mlops.benchmark import BenchmarkRunner


class FakeAgent:
    def process(self, text: str):
        return {
            "confidence": 0.8,
            "processing_time": {
                "ner_ms": 1.0,
                "retriever_ms": 2.0,
                "reasoner_ms": 3.0,
                "total_ms": 6.0,
            },
        }


class FakeGPUMonitor:
    def collect(self):
        return [{"available": False}]


def test_benchmark_runner_summarizes_agent_latency(tmp_path):
    report = BenchmarkRunner(gpu_monitor=FakeGPUMonitor()).run_agent_benchmark(
        agent=FakeAgent(),
        samples=["sot ho", "dau dau"],
        warmup_runs=0,
        output_dir=str(tmp_path),
    )

    assert report["sample_count"] == 2
    assert report["latency"]["ner_ms"]["avg"] == 1.0
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "benchmark.csv").exists()
    assert (tmp_path / "benchmark.html").exists()
