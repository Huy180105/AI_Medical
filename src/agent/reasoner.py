from typing import Any


class RuleBasedClinicalReasoner:
    def reason(
        self,
        text: str,
        entities: list[dict[str, Any]],
        normalized_entities: list[dict[str, Any]],
        knowledge: list[dict[str, Any]],
    ) -> dict[str, Any]:
        terms = {item["original"].lower() for item in normalized_entities}
        source_text = text.lower()

        has_fever = self._contains_any(terms, source_text, {"sot", "fever", "r50.9"})
        has_cough = self._contains_any(terms, source_text, {"ho", "cough", "r05"})
        has_paracetamol = self._contains_any(terms, source_text, {"paracetamol", "acetaminophen"})

        possible_diseases: list[dict[str, Any]] = []
        recommendations: list[str] = []
        red_flags: list[str] = []

        if has_fever and has_cough:
            possible_diseases.append(
                {
                    "name": "Respiratory infection",
                    "evidence": ["fever", "cough"],
                    "confidence": 0.78,
                }
            )
            recommendations.append(
                "Assess respiratory rate, oxygen saturation, chest pain, dyspnea, fever duration, and hydration."
            )
            recommendations.append("Use supportive care and follow local respiratory infection guideline.")
        elif has_fever:
            possible_diseases.append(
                {
                    "name": "Febrile illness",
                    "evidence": ["fever"],
                    "confidence": 0.55,
                }
            )
            recommendations.append("Assess infection source, duration, temperature, hydration, and warning signs.")

        if has_paracetamol:
            recommendations.append(
                "Review paracetamol dose, duplicate acetaminophen exposure, liver disease, alcohol use, age, and weight."
            )

        if any(flag in source_text for flag in ["shortness of breath", "dyspnea", "kho tho", "chest pain", "dau nguc"]):
            red_flags.append("Respiratory or chest red flag requires urgent clinical assessment.")

        guideline_titles = [
            item.get("metadata", {}).get("title", "")
            for item in knowledge
            if item.get("metadata", {}).get("source_type") == "medical_guideline"
        ]

        if not recommendations:
            recommendations.append("Insufficient deterministic evidence. Recommend clinician review and more context.")

        confidence = self._confidence(possible_diseases, knowledge)
        return {
            "possible_diseases": possible_diseases,
            "recommended_guidelines": [title for title in guideline_titles if title],
            "recommendations": recommendations,
            "red_flags": red_flags,
            "is_deterministic": True,
            "llm_used": False,
            "evidence": {
                "entities": entities,
                "knowledge_count": len(knowledge),
            },
            "confidence": confidence,
        }

    def _contains_any(self, terms: set[str], text: str, candidates: set[str]) -> bool:
        return bool(terms.intersection(candidates)) or any(candidate in text for candidate in candidates)

    def _confidence(self, possible_diseases: list[dict[str, Any]], knowledge: list[dict[str, Any]]) -> float:
        disease_confidence = max((item["confidence"] for item in possible_diseases), default=0.35)
        retrieval_bonus = min(len(knowledge), 5) * 0.03
        return round(min(disease_confidence + retrieval_bonus, 0.95), 4)
