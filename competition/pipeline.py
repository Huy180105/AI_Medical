from typing import Any
from src.assertion.assertion_detector import ClinicalAssertionDetector
from src.relation.relation_detector import ClinicalRelationDetector
from src.ranking.candidate_ranker import CandidateRetrievalRanker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CompetitionPipeline:
    """
    End-to-end Viettel AI Race Competition Pipeline processing raw clinical text (.txt)
    into competition-compliant JSON records.
    """

    def __init__(self) -> None:
        self.assertion_detector = ClinicalAssertionDetector()
        self.relation_detector = ClinicalRelationDetector()
        self.candidate_ranker = CandidateRetrievalRanker()

    def process_text(self, document_id: str, raw_text: str, ner_entities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """
        Executes end-to-end extraction and mapping over raw clinical text.
        """
        if ner_entities is None:
            # Fallback simple entity extractor if model is offline during unit testing
            ner_entities = self._extract_fallback_entities(raw_text)

        # 1. Assertion Detection
        assertions = self.assertion_detector.detect_assertions(raw_text, ner_entities)
        assertion_dicts = [a.to_dict() for a in assertions]

        # 2. Relation Extraction
        relations = self.relation_detector.extract_relations(raw_text, ner_entities)
        relation_dicts = [r.to_dict() for r in relations]

        # 3. Candidate Normalization (ICD-10 & RxNorm)
        icd10_codes = []
        rxnorm_codes = []

        for ent in ner_entities:
            ent_text = ent.get("text", "")
            ent_type = ent.get("type", "ALL")

            ranking_res = self.candidate_ranker.rank_entity(ent_text, entity_type=ent_type, top_k=1)
            if ranking_res and ranking_res.top_candidates:
                top_cand = ranking_res.top_candidates[0]
                code_entry = {
                    "entity_text": ent_text,
                    "code": top_cand.code,
                    "display": top_cand.display,
                    "confidence": top_cand.score
                }
                if "RxNorm" in top_cand.code:
                    rxnorm_codes.append(code_entry)
                else:
                    icd10_codes.append(code_entry)

        return {
            "document_id": document_id,
            "text": raw_text,
            "entities": ner_entities,
            "assertions": assertion_dicts,
            "relations": relation_dicts,
            "icd10_codes": icd10_codes,
            "rxnorm_codes": rxnorm_codes
        }

    def _extract_fallback_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Simple rule-based fallback entity extractor for dry-run testing.
        """
        entities = []
        keywords = [
            ("ho kéo dài", "SYMPTOM"),
            ("ho đờm", "SYMPTOM"),
            ("sốt cao", "SYMPTOM"),
            ("đau ngực", "SYMPTOM"),
            ("khó thở", "SYMPTOM"),
            ("viêm phổi", "DISEASE"),
            ("trào ngược dạ dày", "DISEASE"),
            ("nhồi máu cơ tim", "DISEASE"),
            ("Paracetamol", "MEDICINE"),
            ("Aspirin", "MEDICINE"),
            ("Amoxicillin", "MEDICINE"),
            ("X-quang ngực", "TEST")
        ]

        for kw, etype in keywords:
            start = text.find(kw)
            if start != -1:
                entities.append({
                    "text": kw,
                    "type": etype,
                    "start": start,
                    "end": start + len(kw)
                })

        return entities
