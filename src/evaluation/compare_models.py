from typing import Any


class ModelComparisonEngine:
    """
    Compares metrics side-by-side between baseline and candidate models,
    calculating percentage deltas and performance trade-offs.
    """

    @staticmethod
    def compare(
        model_a_name: str,
        model_a_metrics: dict[str, Any],
        model_b_name: str,
        model_b_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generates a comparative analysis dictionary between Model A and Model B.
        """
        comparison = {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "metrics_comparison": {},
            "summary": ""
        }

        keys_to_compare = ["overall_score_percentage", "f1", "accuracy", "sensitivity", "specificity", "avg_latency_ms"]

        for k in keys_to_compare:
            val_a = model_a_metrics.get(k)
            val_b = model_b_metrics.get(k)

            if val_a is not None and val_b is not None:
                delta = round(val_b - val_a, 4)
                pct_change = round((delta / val_a) * 100.0, 2) if val_a != 0 else 0.0
                
                comparison["metrics_comparison"][k] = {
                    model_a_name: val_a,
                    model_b_name: val_b,
                    "delta": delta,
                    "percentage_change": pct_change
                }

        score_a = model_a_metrics.get("overall_score_percentage", 0.0)
        score_b = model_b_metrics.get("overall_score_percentage", 0.0)

        if score_b > score_a:
            winner = model_b_name
            margin = round(score_b - score_a, 2)
            comparison["summary"] = f"Winner: '{winner}' outperforming '{model_a_name}' by +{margin}% overall score."
        elif score_a > score_b:
            winner = model_a_name
            margin = round(score_a - score_b, 2)
            comparison["summary"] = f"Winner: '{winner}' outperforming '{model_b_name}' by +{margin}% overall score."
        else:
            comparison["summary"] = f"Models '{model_a_name}' and '{model_b_name}' performed identically."

        return comparison
