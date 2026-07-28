from typing import Any


class ClinicalMetricsCalculator:
    """
    Calculates statistical and clinical metrics including Precision, Recall, F1-score,
    Sensitivity, Specificity, and Clinical Safety Violation Rates.
    """

    @staticmethod
    def calculate_precision_recall_f1(true_entities: list[dict[str, Any]], pred_entities: list[dict[str, Any]]) -> dict[str, float]:
        """
        Computes entity-level Precision, Recall, and F1-score.
        Matches entity text and entity type.
        """
        if not true_entities and not pred_entities:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        true_tuples = {(e.get("text", "").lower(), e.get("type", "").upper()) for e in true_entities}
        pred_tuples = {(e.get("text", "").lower(), e.get("type", "").upper()) for e in pred_entities}

        true_positives = len(true_tuples.intersection(pred_tuples))
        false_positives = len(pred_tuples - true_tuples)
        false_negatives = len(true_tuples - pred_tuples)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

    @staticmethod
    def calculate_sensitivity_specificity(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
        """
        Computes Clinical Sensitivity (Recall for positive disease/anomaly condition)
        and Specificity (True Negative Rate for normal control state).
        """
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4)
        }

    @staticmethod
    def calculate_safety_violation_rate(violations: int, total_cases: int) -> dict[str, float]:
        """
        Computes the clinical safety violation rate (e.g. prescribing a contraindicated drug).
        """
        rate = violations / total_cases if total_cases > 0 else 0.0
        return {
            "safety_violation_count": violations,
            "total_evaluated_cases": total_cases,
            "safety_violation_rate": round(rate, 4),
            "safety_compliance_rate": round(1.0 - rate, 4)
        }
