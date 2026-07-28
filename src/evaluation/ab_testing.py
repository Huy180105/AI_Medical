import random
import time
from typing import Any


class ABTestingSimulator:
    """
    Simulates A/B testing traffic splits between model variants (e.g. Variant A: PhoBERT RAG vs Variant B: Multi-Agent Swarm).
    """

    def __init__(self, split_ratio: float = 0.5) -> None:
        self.split_ratio = split_ratio  # Portion of traffic routed to Variant B
        self.stats_a = {"requests": 0, "total_latency_ms": 0.0, "total_confidence": 0.0}
        self.stats_b = {"requests": 0, "total_latency_ms": 0.0, "total_confidence": 0.0}

    def route_request(self, payload: dict[str, Any]) -> str:
        """
        Decides whether to route request to Variant A or Variant B based on split ratio.
        """
        rnd = random.random()
        return "Variant_B" if rnd < self.split_ratio else "Variant_A"

    def record_result(self, variant: str, latency_ms: float, confidence: float) -> None:
        """
        Records latency and confidence telemetry for the executed variant.
        """
        stats = self.stats_b if variant == "Variant_B" else self.stats_a
        stats["requests"] += 1
        stats["total_latency_ms"] += latency_ms
        stats["total_confidence"] += confidence

    def get_experiment_report(self) -> dict[str, Any]:
        """
        Generates an A/B test summary report.
        """
        total_reqs = self.stats_a["requests"] + self.stats_b["requests"]
        
        avg_lat_a = self.stats_a["total_latency_ms"] / self.stats_a["requests"] if self.stats_a["requests"] > 0 else 0.0
        avg_lat_b = self.stats_b["total_latency_ms"] / self.stats_b["requests"] if self.stats_b["requests"] > 0 else 0.0

        avg_conf_a = self.stats_a["total_confidence"] / self.stats_a["requests"] if self.stats_a["requests"] > 0 else 0.0
        avg_conf_b = self.stats_b["total_confidence"] / self.stats_b["requests"] if self.stats_b["requests"] > 0 else 0.0

        return {
            "total_evaluated_requests": total_reqs,
            "variant_a_stats": {
                "requests": self.stats_a["requests"],
                "avg_latency_ms": round(avg_lat_a, 2),
                "avg_confidence": round(avg_conf_a, 4)
            },
            "variant_b_stats": {
                "requests": self.stats_b["requests"],
                "avg_latency_ms": round(avg_lat_b, 2),
                "avg_confidence": round(avg_conf_b, 4)
            },
            "split_ratio": self.split_ratio
        }
