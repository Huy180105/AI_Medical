from typing import Any
from src.graph.ontology import MedicalOntology


class ClinicalRuleEngine:
    """
    Evaluates clinical rules on patient presentations (NER entities)
    and graph reasoning results.
    """

    RED_FLAGS = {
        "shortness of breath",
        "dyspnea",
        "kho tho",
        "khó thở",
        "chest pain",
        "dau nguc",
        "đau ngực",
        "confusion",
        "cyanosis",
        "low oxygen",
        "oxygen saturation below",
    }

    @classmethod
    def check_red_flags(cls, entities: list[dict[str, Any]]) -> list[str]:
        """
        Scans symptoms and original entity texts to identify red flag triggers.
        """
        flags_detected = []
        for ent in entities:
            ent_text = ent.get("text", "").lower()
            ent_type = ent.get("type", "").upper()

            # Check if any red flag phrase is a substring of the entity text
            for flag in cls.RED_FLAGS:
                if flag in ent_text:
                    flags_detected.append(f"Critical symptom/sign detected: '{ent['text']}'")
                    break
        return flags_detected

    @classmethod
    def check_contraindications(
        cls, entities: list[dict[str, Any]], graph_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Compares medications in entities against contraindicated drugs in graph results.
        Returns a list of detected contraindication warnings.
        """
        contraindications = []
        
        # Extract all medicine names from entities
        input_meds = [
            ent.get("text", "")
            for ent in entities
            if ent.get("type", "").upper() in ("MEDICINE", "DRUG")
        ]

        # For each candidate disease in graph_results, check if input meds are contraindicated
        for candidate in graph_results:
            disease_name = candidate.get("disease", "")
            evidence = candidate.get("evidence", {})
            contra_in_graph = evidence.get("contraindicated_drugs", [])

            for med in input_meds:
                # Compare case-insensitively or via canonical forms
                med_norm = MedicalOntology.normalize_text(med)
                for ci_drug in contra_in_graph:
                    if med_norm == MedicalOntology.normalize_text(ci_drug):
                        contraindications.append({
                            "medication": med,
                            "disease": disease_name,
                            "reason": f"Medication '{med}' is contraindicated for patient with '{disease_name}' according to clinical knowledge graph."
                        })
        return contraindications
