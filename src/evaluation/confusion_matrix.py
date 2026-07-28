from typing import Any


class ConfusionMatrixGenerator:
    """
    Generates multi-class confusion matrices for entity extraction and wearable anomaly detection.
    """

    @staticmethod
    def generate_matrix(classes: list[str], y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
        """
        Builds a multi-class confusion matrix dictionary.
        """
        labels = sorted(list(set(classes + y_true + y_pred)))
        matrix = {t: {p: 0 for p in labels} for t in labels}

        for true_label, pred_label in zip(y_true, y_pred):
            t_key = true_label if true_label in labels else "OTHER"
            p_key = pred_label if pred_label in labels else "OTHER"
            if t_key not in matrix:
                matrix[t_key] = {p: 0 for p in labels}
            if p_key not in matrix[t_key]:
                matrix[t_key][p_key] = 0
            matrix[t_key][p_key] += 1

        return {
            "labels": labels,
            "matrix": matrix
        }
