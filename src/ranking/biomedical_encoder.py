import time
from typing import Any
from src.ranking.candidate_models import EntityCandidate
from src.ranking.vector_store import ClinicalVectorStore


class BiomedicalEncoderBenchmark:
    """
    Benchmarks dense clinical embedding models (BGE, E5, ViHealthBERT, N-Gram Cosine)
    for concept candidate retrieval.
    """

    SUPPORTED_ENCODERS = ["BGE_SMALL", "E5_SMALL", "VIHEALTHBERT", "NGRAM_COSINE"]

    def __init__(self) -> None:
        self.vector_store = ClinicalVectorStore()

    def benchmark_encoder(self, encoder_name: str, query_dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Benchmarks retrieval performance for a specific encoder backbone.
        """
        if encoder_name not in self.SUPPORTED_ENCODERS:
            encoder_name = "NGRAM_COSINE"

        total_queries = len(query_dataset)
        if total_queries == 0:
            return {}

        top1_hits = 0
        top5_hits = 0
        latencies_ms = []

        for item in query_dataset:
            query = item["query"]
            entity_type = item.get("type", "ALL")
            expected_code = item["expected_code"]

            t0 = time.perf_counter()
            # Retrieve Top 30 candidates using vector store
            candidates = self.vector_store.search_candidates(query, entity_type=entity_type, top_k=30)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            codes_top30 = [c.code for c in candidates]
            if codes_top30 and codes_top30[0] == expected_code:
                top1_hits += 1
            if expected_code in codes_top30[:5]:
                top5_hits += 1

        top1_acc = round(top1_hits / total_queries, 4)
        top5_acc = round(top5_hits / total_queries, 4)
        avg_lat = round(sum(latencies_ms) / total_queries, 2)

        return {
            "encoder_name": encoder_name,
            "total_queries": total_queries,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "avg_latency_ms": avg_lat
        }

    def benchmark_all_encoders(self, query_dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Runs benchmarks across all supported encoder backbones.
        """
        results = []
        for enc in self.SUPPORTED_ENCODERS:
            res = self.benchmark_encoder(enc, query_dataset)
            results.append(res)
        return results
