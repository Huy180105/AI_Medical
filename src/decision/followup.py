from typing import Any


class ClinicalFollowUpEngine:
    """
    Determines patient follow-up actions, timelines, and monitoring instructions.
    """

    @classmethod
    def determine_followup(
        cls,
        risk_level: str,
        entities: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Determines the clinical follow-up disposition based on the risk level.
        Dispositions: Revisit, Emergency, Hospitalization, Observation.
        """
        if risk_level == "Emergency":
            return {
                "action": "Emergency",
                "timeframe": "Immediate",
                "instructions": "Seek immediate emergency medical attention. Call emergency services or go to the nearest ER.",
            }
        elif risk_level == "High":
            return {
                "action": "Hospitalization",
                "timeframe": "Within 12-24 hours",
                "instructions": "Report to a hospital or clinical facility for active observation and specialized review.",
            }
        elif risk_level == "Medium":
            return {
                "action": "Revisit",
                "timeframe": "Within 48-72 hours",
                "instructions": "Schedule an outpatient visit. Revisit immediately if you experience shortness of breath, high persistent fever, or worsening chest pain.",
            }
        else:  # Low risk
            return {
                "action": "Observation",
                "timeframe": "Self-monitoring / As needed",
                "instructions": "Monitor symptoms at home. Take temperature twice daily. Seek advice if symptoms do not resolve within 5-7 days.",
            }
        
