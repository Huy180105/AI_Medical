from typing import Any, Optional
from src.wearable.patient_state import HealthPacket


class WearableRiskPredictor:
    """
    Evaluates composite patient risk based on active anomalies and trends.
    Integrates directly with the Clinical Decision Engine (CDSS) to generate alerts.
    """

    def __init__(self, decision_engine: Optional[Any] = None) -> None:
        self.decision_engine = decision_engine

    def assess_risk(self, anomalies: list[dict[str, Any]]) -> str:
        """
        Classifies risk levels based on active physiological anomalies.
        """
        if not anomalies:
            return "Low"

        severities = [a.get("severity", "Medium") for a in anomalies]

        if "Emergency" in severities:
            return "Emergency"
        if "High" in severities or severities.count("Medium") >= 2:
            return "High"
        return "Medium"

    def trigger_clinical_decision(
        self,
        packet: HealthPacket,
        anomalies: list[dict[str, Any]],
        risk_level: str
    ) -> Optional[dict[str, Any]]:
        """
        If the risk level is High or Emergency, maps anomalies to clinical entities
        and triggers the Clinical Decision Engine (CDSS) to formulate recommendation outputs.
        """
        if not self.decision_engine or risk_level not in ("High", "Emergency"):
            return None

        # Map wearable anomalies to clinical symptom entities
        entities = []
        for anomaly in anomalies:
            anomaly_type = anomaly.get("type", "")
            if anomaly_type == "Hypoxia":
                entities.append({"text": "thiếu oxy", "type": "SYMPTOM"})
            elif anomaly_type == "Tachycardia":
                entities.append({"text": "tim đập nhanh", "type": "SYMPTOM"})
            elif anomaly_type == "Bradycardia":
                entities.append({"text": "nhịp tim chậm", "type": "SYMPTOM"})
            elif anomaly_type == "Fever Spike":
                entities.append({"text": "sốt cao", "type": "SYMPTOM"})
            elif anomaly_type == "Arrhythmia (AFib)":
                entities.append({"text": "loạn nhịp tim", "type": "SYMPTOM"})
            elif anomaly_type == "Sleep Apnea Indicator":
                entities.append({"text": "ngưng thở khi ngủ", "type": "SYMPTOM"})

        # If we have a history of medication or other context, we can extract it.
        # Run CDSS decision
        # We pass empty graph results or we can pre-simulate a graph context for these symptoms if needed
        # In a real pipeline, the entities would be linked to the KG.
        decision = self.decision_engine.make_decision(entities, [])
        
        # Override the risk level in CDSS with the watch risk level
        decision["risk_level"] = risk_level
        
        return decision
