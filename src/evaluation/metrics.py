import math
from typing import Any


class EvaluationMetricsEngine:
    """
    Computes statistical ROC/AUROC curves and system throughput metrics.
    """

    @staticmethod
    def calculate_auroc(y_true: list[int], y_scores: list[float]) -> dict[str, Any]:
        """
        Computes Area Under ROC Curve (AUROC) using the trapezoidal rule across confidence scores.
        """
        if not y_true or not y_scores or len(y_true) != len(y_scores):
            return {"auroc": 0.5, "roc_points": []}

        # Pair true values and predictions, sorted descending by confidence score
        paired = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
        
        num_pos = sum(y_true)
        num_neg = len(y_true) - num_pos

        if num_pos == 0 or num_neg == 0:
            return {"auroc": 1.0 if num_pos == len(y_true) else 0.5, "roc_points": []}

        tp = 0
        fp = 0
        roc_points = [{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}]
        
        for score, label in paired:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / num_pos
            fpr = fp / num_neg
            roc_points.append({"fpr": round(fpr, 4), "tpr": round(tpr, 4), "threshold": round(score, 4)})

        # Compute AUROC using trapezoidal area sum
        auroc = 0.0
        for i in range(1, len(roc_points)):
            prev_point = roc_points[i-1]
            curr_point = roc_points[i]
            # Area of trapezoid = (FPR2 - FPR1) * (TPR1 + TPR2) / 2
            width = curr_point["fpr"] - prev_point["fpr"]
            height = (curr_point["tpr"] + prev_point["tpr"]) / 2.0
            auroc += width * height

        return {
            "auroc": round(max(0.0, min(1.0, auroc)), 4),
            "roc_points": roc_points
        }

    @staticmethod
    def calculate_throughput(total_samples: int, total_duration_seconds: float) -> dict[str, float]:
        """
        Computes processing throughput in samples per second and average latency in ms.
        """
        if total_duration_seconds <= 0:
            return {"throughput_samples_per_sec": 0.0, "avg_latency_ms": 0.0}

        samples_per_sec = total_samples / total_duration_seconds
        avg_latency_ms = (total_duration_seconds * 1000.0) / total_samples if total_samples > 0 else 0.0

        return {
            "throughput_samples_per_sec": round(samples_per_sec, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "total_duration_sec": round(total_duration_seconds, 3)
        }
