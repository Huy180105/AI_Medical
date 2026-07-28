from typing import Any
from src.relation.relation_models import ClinicalEntityRef, ClinicalRelation
from src.relation.relation_rules import (
    DOMAIN_CONSTRAINTS,
    REGEX_TREATS,
    REGEX_CAUSED_BY,
    REGEX_HAS_SYMPTOM,
    REGEX_HAS_TEST_RESULT,
    REGEX_CONTRAINDICATED_FOR
)


class ClinicalRelationDetector:
    """
    Extracts directed semantic relations (TREATS, CAUSED_BY, HAS_SYMPTOM, HAS_TEST_RESULT, CONTRAINDICATED_FOR)
    between pairs of extracted medical entities.
    """

    def __init__(self, max_char_distance: int = 120) -> None:
        self.max_char_distance = max_char_distance

    def extract_relations(self, text: str, entities: list[dict[str, Any]]) -> list[ClinicalRelation]:
        """
        Scans all pairwise entity combinations in text and returns extracted ClinicalRelation objects.
        """
        relations = []
        if len(entities) < 2:
            return relations

        # Normalize entity representations
        norm_entities = self._normalize_entities(text, entities)

        for i in range(len(norm_entities)):
            for j in range(len(norm_entities)):
                if i == j:
                    continue

                e1 = norm_entities[i]
                e2 = norm_entities[j]

                # Check character span distance
                dist = abs(e1["start"] - e2["start"])
                if dist > self.max_char_distance:
                    continue

                # Check if sentence boundary exists between e1 and e2
                min_idx = min(e1["end"], e2["end"])
                max_idx = max(e1["start"], e2["start"])
                between_text = text[min_idx:max_idx]

                if between_text.count(".") > 1:
                    continue  # Skip cross-sentence relations spanning multiple periods

                rel = self._detect_pair_relation(between_text, e1, e2)
                if rel:
                    relations.append(rel)

        return relations

    def _detect_pair_relation(self, between_text: str, e1: dict[str, Any], e2: dict[str, Any]) -> ClinicalRelation | None:
        """
        Checks relation patterns against text span between e1 and e2.
        """
        t1, t2 = e1["type"].upper(), e2["type"].upper()

        ref1 = ClinicalEntityRef(text=e1["text"], type=e1["type"], start=e1["start"], end=e1["end"])
        ref2 = ClinicalEntityRef(text=e2["text"], type=e2["type"], start=e2["start"], end=e2["end"])

        # 1. CONTRAINDICATED_FOR (Medication -> Disease)
        if t1 in DOMAIN_CONSTRAINTS["CONTRAINDICATED_FOR"]["subject_types"] and t2 in DOMAIN_CONSTRAINTS["CONTRAINDICATED_FOR"]["object_types"]:
            if REGEX_CONTRAINDICATED_FOR.search(between_text):
                return ClinicalRelation(subject=ref1, relation_type="CONTRAINDICATED_FOR", object=ref2, confidence=0.95)

        # 2. TREATS (Medication -> Disease/Symptom)
        if t1 in DOMAIN_CONSTRAINTS["TREATS"]["subject_types"] and t2 in DOMAIN_CONSTRAINTS["TREATS"]["object_types"]:
            if REGEX_TREATS.search(between_text) or e1["start"] < e2["start"]:
                return ClinicalRelation(subject=ref1, relation_type="TREATS", object=ref2, confidence=0.90)

        # 3. CAUSED_BY (Symptom/Disease -> Disease)
        if t1 in DOMAIN_CONSTRAINTS["CAUSED_BY"]["subject_types"] and t2 in DOMAIN_CONSTRAINTS["CAUSED_BY"]["object_types"]:
            if REGEX_CAUSED_BY.search(between_text):
                return ClinicalRelation(subject=ref1, relation_type="CAUSED_BY", object=ref2, confidence=0.92)

        # 4. HAS_SYMPTOM (Disease -> Symptom)
        if t1 in DOMAIN_CONSTRAINTS["HAS_SYMPTOM"]["subject_types"] and t2 in DOMAIN_CONSTRAINTS["HAS_SYMPTOM"]["object_types"]:
            if REGEX_HAS_SYMPTOM.search(between_text) or e1["start"] < e2["start"]:
                return ClinicalRelation(subject=ref1, relation_type="HAS_SYMPTOM", object=ref2, confidence=0.88)

        # 5. HAS_TEST_RESULT (Test -> Result/Value)
        if t1 in DOMAIN_CONSTRAINTS["HAS_TEST_RESULT"]["subject_types"] and t2 in DOMAIN_CONSTRAINTS["HAS_TEST_RESULT"]["object_types"]:
            if REGEX_HAS_TEST_RESULT.search(between_text) or e1["start"] < e2["start"]:
                return ClinicalRelation(subject=ref1, relation_type="HAS_TEST_RESULT", object=ref2, confidence=0.95)

        return None

    def _normalize_entities(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        norm = []
        for ent in entities:
            ent_text = ent.get("text", "")
            start = ent.get("start", -1)
            end = ent.get("end", -1)

            if start == -1 or end == -1:
                pos = text.find(ent_text)
                if pos != -1:
                    start = pos
                    end = pos + len(ent_text)
                else:
                    start = 0
                    end = len(ent_text)

            norm.append({
                "text": ent_text,
                "type": ent.get("type", "UNKNOWN"),
                "start": start,
                "end": end
            })
        return norm
