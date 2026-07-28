from typing import Any


class ClinicalRiskEngine:
    """
    Evaluates patient risk level (Low, Medium, High, Emergency) based on
    symptoms, candidate diseases, red flags, and drug contraindications.
    """

    @classmethod
    def assess_risk(
        cls,
        entities: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        red_flags: list[str],
        contraindications: list[dict[str, Any]],
    ) -> str:
        """
        Calculates the risk level based on rules.
        - Emergency: If red flags are detected.
        - High: If any active contraindications are detected.
        - Medium: If there are active diseases with moderate confidence (>0.5) but no emergency indicators.
        - Low: Otherwise (mild symptoms, no active contraindications).
        """
        if red_flags:
            return "Emergency"

        if contraindications:
            return "High"

        # Check if we have high-confidence candidate diseases
        has_moderate_disease = any(
            candidate.get("confidence", 0.0) >= 0.5 for candidate in graph_results
        )

        # Check if we have multiple symptoms or complex clinical picture
        symptoms_count = sum(
            1 for ent in entities if ent.get("type", "").upper() == "SYMPTOM"
        )

        if has_moderate_disease or symptoms_count >= 2:
            return "Medium"

        return "Low"
