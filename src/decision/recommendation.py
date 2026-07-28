from typing import Any
from src.graph.ontology import MedicalOntology, NodeType


class ClinicalRecommendationEngine:
    """
    Generates diagnosis candidates, lab tests, medication recommendations,
    referrals, and lifestyle advice based on clinical input.
    """

    @classmethod
    def generate_recommendations(
        cls,
        entities: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        risk_level: str,
        contraindications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Synthesizes recommendations from entities, graph findings, and risk assessment.
        """
        # 1. Collect diagnosis candidates
        diagnosis_candidates = []
        for candidate in graph_results:
            diagnosis_candidates.append({
                "disease": candidate.get("disease", ""),
                "confidence": candidate.get("confidence", 0.0),
                "explanation": candidate.get("explanation", ""),
            })

        # 2. Recommended laboratory tests
        recommended_labs = []
        # Pull required labs directly from graph evidence
        for candidate in graph_results:
            evidence = candidate.get("evidence", {})
            matched_labs = evidence.get("matched_labs", [])
            for lab in matched_labs:
                if lab not in recommended_labs:
                    recommended_labs.append(lab)

        # Fallback recommendations based on symptoms if no graph lab matches
        symptom_texts = {
            ent.get("text", "").lower()
            for ent in entities
            if ent.get("type", "").upper() == "SYMPTOM"
        }
        
        # Check canonical symptom aliases
        has_cough_or_dyspnea = any(
            any(alias in sym for alias in ("cough", "ho", "shortness of breath", "dyspnea", "kho tho", "khó thở"))
            for sym in symptom_texts
        )
        has_fever = any(
            any(alias in sym for alias in ("fever", "sot", "sốt", "temperature"))
            for sym in symptom_texts
        )

        if has_cough_or_dyspnea and "Chest X-ray" not in recommended_labs:
            recommended_labs.append("Chest X-ray")
            recommended_labs.append("Oxygen saturation (SpO2) monitoring")
        if (has_fever or has_cough_or_dyspnea) and "Complete blood count" not in recommended_labs:
            recommended_labs.append("Complete blood count (CBC)")

        # 3. Recommended medication categories
        medication_categories = []
        avoid_nsaids = False

        # If kidney disease is suspected or confirmed in entities/contraindications, avoid NSAIDs
        for ci in contraindications:
            if "kidney" in ci.get("disease", "").lower() or "renal" in ci.get("disease", "").lower():
                avoid_nsaids = True
            if "gastric" in ci.get("disease", "").lower() or "ulcer" in ci.get("disease", "").lower():
                avoid_nsaids = True

        # Suggest paracetamol for pain/fever if not contraindicated
        if has_fever or any(s in " ".join(symptom_texts) for s in ("pain", "dau", "đau", "headache", "đầu", "dau dau", "đau đầu")):
            medication_categories.append({
                "category": "Antipyretics and Analgesics (e.g. Paracetamol)",
                "note": "Preferred choice for fever and mild pain. Monitor daily dosage."
            })

        if avoid_nsaids:
            medication_categories.append({
                "category": "NSAIDs (e.g. Ibuprofen)",
                "status": "Contraindicated",
                "note": "AVOID. Contraindicated due to renal risk or gastric ulcer history."
            })
        elif has_fever or "inflammation" in " ".join(symptom_texts):
            medication_categories.append({
                "category": "NSAIDs (e.g. Ibuprofen)",
                "status": "Indicated",
                "note": "Can be used if no renal, gastric, or late pregnancy contraindications exist."
            })

        # 4. Referral suggestion
        if risk_level == "Emergency":
            referral = "Immediate referral to the nearest Emergency Department (ER)."
        elif risk_level == "High":
            referral = "Referral to a specialist (e.g., Pulmonologist, Nephrologist, or Internist)."
        elif risk_level == "Medium":
            referral = "Consult a General Practitioner (GP) or outpatient clinic for evaluation."
        else:
            referral = "Routine primary care follow-up or pharmacist consultation if symptoms persist."

        # 5. Lifestyle advice
        lifestyle = ["Ensure adequate rest and hydration."]
        
        # Add infection control if respiratory issues are present
        if has_cough_or_dyspnea:
            lifestyle.append("Wear face masks and isolate to prevent potential transmission of viral pathoghens.")
            lifestyle.append("Keep the living environment well-ventilated.")

        if contraindications:
            lifestyle.append("Stop taking contraindicated medications immediately.")

        return {
            "diagnosis_candidates": diagnosis_candidates,
            "recommended_labs": recommended_labs,
            "recommended_medication_categories": medication_categories,
            "referral_suggestion": referral,
            "lifestyle_advice": lifestyle,
        }
