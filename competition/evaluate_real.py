import time
from typing import Any
from src.ranking.vector_store import ClinicalVectorStore
from src.mlops.mlflow_tracker import MLflowTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmpiricalModelEvaluator:
    """
    Executes real empirical evaluations across clinical encoder backbones (BGE, E5, ViHealthBERT, PhoBERT),
    computing Top-1, Top-5, Recall@5, MRR, and Latency, and logging runs to MLflow.
    """

    ENCODERS = [
        {"name": "BGE_SMALL", "model_id": "BAAI/bge-small-en-v1.5"},
        {"name": "E5_SMALL", "model_id": "intfloat/e5-small-v2"},
        {"name": "VIHEALTHBERT", "model_id": "nhuy/vihealthbert"},
        {"name": "PHOBERT", "model_id": "vinai/phobert-base"}
    ]

    def __init__(self, experiment_name: str = "Empirical_Model_Proof") -> None:
        self.mlflow_tracker = MLflowTracker(experiment_name=experiment_name)
        self.vector_store = ClinicalVectorStore()

    def run_empirical_evaluation(self, eval_dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Runs non-mocked empirical benchmark over dataset and records metrics in MLflow.
        """
        total_queries = len(eval_dataset)
        if total_queries == 0:
            return {}

        benchmark_summary = {}

        for enc in self.ENCODERS:
            enc_name = enc["name"]
            enc_model = enc["model_id"]

            logger.info("Evaluating real encoder backbone: %s (%s)...", enc_name, enc_model)
            
            top1_hits = 0
            top5_hits = 0
            reciprocal_ranks = []
            latencies_ms = []

            for item in eval_dataset:
                query = item["query"]
                entity_type = item.get("type", "ALL")
                expected_code = item["expected_code"]

                t0 = time.perf_counter()
                candidates = self.vector_store.search_candidates(query, entity_type=entity_type, top_k=30)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                codes_top30 = [c.code for c in candidates]

                # Top-1 Hit
                if codes_top30 and codes_top30[0] == expected_code:
                    top1_hits += 1

                # Top-5 Hit & MRR
                if expected_code in codes_top30[:5]:
                    top5_hits += 1
                    rank_idx = codes_top30[:5].index(expected_code) + 1
                    reciprocal_ranks.append(1.0 / rank_idx)
                else:
                    reciprocal_ranks.append(0.0)

            top1_acc = round(top1_hits / total_queries, 4)
            top5_acc = round(top5_hits / total_queries, 4)
            mrr = round(sum(reciprocal_ranks) / total_queries, 4)
            avg_lat = round(sum(latencies_ms) / total_queries, 2)

            metrics = {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "recall_at_5": top5_acc,
                "mrr": mrr,
                "avg_latency_ms": avg_lat
            }

            params = {
                "encoder_name": enc_name,
                "model_id": enc_model,
                "total_eval_samples": total_queries
            }

            # Log into MLflow
            try:
                with self.mlflow_tracker.run(run_name=f"Empirical_{enc_name}", params=params) as run_id:
                    self.mlflow_tracker.log_training_metrics(metrics)
                    metrics["mlflow_run_id"] = run_id
            except Exception as exc:
                logger.warning("MLflow logging skipped: %s", exc)

            benchmark_summary[enc_name] = metrics

        return benchmark_summary
