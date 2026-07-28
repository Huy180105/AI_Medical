import time
from typing import Any
from src.ranking.candidate_ranker import CandidateRetrievalRanker


class RankingBenchmarkSuite:
    """
    Evaluates concept normalization quality across Top-1, Top-5, MRR, Recall@K, and query latency.
    """

    def __init__(self) -> None:
        self.ranker = CandidateRetrievalRanker()

    def evaluate_benchmark(self, eval_dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Runs candidate ranking evaluation over an annotated dataset of entity queries and expected codes.
        """
        total_queries = len(eval_dataset)
        if total_queries == 0:
            return {}

        top1_hits = 0
        top5_hits = 0
        reciprocal_ranks = []
        latencies_ms = []

        for item in eval_dataset:
            query = item["query"]
            entity_type = item.get("type", "ALL")
            expected_code = item["expected_code"]

            t0 = time.perf_counter()
            res = self.ranker.rank_entity(query, entity_type=entity_type, top_k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            codes_in_top5 = [c.code for c in res.top_candidates]

            # Top-1 Hit
            if codes_in_top5 and codes_in_top5[0] == expected_code:
                top1_hits += 1

            # Top-5 Hit
            if expected_code in codes_in_top5:
                top5_hits += 1
                rank_idx = codes_in_top5.index(expected_code) + 1
                reciprocal_ranks.append(1.0 / rank_idx)
            else:
                reciprocal_ranks.append(0.0)

        top1_acc = round(top1_hits / total_queries, 4)
        top5_acc = round(top5_hits / total_queries, 4)
        mrr = round(sum(reciprocal_ranks) / total_queries, 4)
        avg_lat = round(sum(latencies_ms) / total_queries, 2)

        return {
            "total_queries": total_queries,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "mrr": mrr,
            "recall_at_5": top5_acc,
            "avg_latency_ms": avg_lat
        }
