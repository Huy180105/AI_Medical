import time
from typing import Any, Protocol

from src.agent.reasoner import RuleBasedClinicalReasoner
from src.rag.pipeline import MedicalRAGPipeline
from src.services.icd10_service import ICD10Service
from src.services.rxnorm_service import RxNormService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NERPredictor(Protocol):
    def predict(self, text: str) -> list[dict[str, Any]]:
        ...


class MedicalWorkflow:
    def __init__(
        self,
        ner_predictor: NERPredictor,
        rag_pipeline: MedicalRAGPipeline | None = None,
        reasoner: RuleBasedClinicalReasoner | None = None,
        icd10_service: ICD10Service | None = None,
        rxnorm_service: RxNormService | None = None,
    ) -> None:
        self.ner_predictor = ner_predictor
        self.rag_pipeline = rag_pipeline or MedicalRAGPipeline()
        self.reasoner = reasoner or RuleBasedClinicalReasoner()
        self.icd10_service = icd10_service or ICD10Service()
        self.rxnorm_service = rxnorm_service or RxNormService()

    def run(self, text: str) -> dict[str, Any]:
        total_start = time.perf_counter()

        ner_start = time.perf_counter()
        entities = self.ner_predictor.predict(text)
        ner_latency = self._elapsed_ms(ner_start)

        normalized_entities = self._normalize_entities(entities)
        retrieval_query = self._build_retrieval_query(text, normalized_entities)

        retriever_start = time.perf_counter()
        knowledge = self.rag_pipeline.run(retrieval_query)
        retriever_latency = self._elapsed_ms(retriever_start)

        reasoner_start = time.perf_counter()
        clinical_reasoning = self.reasoner.reason(text, entities, normalized_entities, knowledge)
        reasoner_latency = self._elapsed_ms(reasoner_start)

        total_latency = self._elapsed_ms(total_start)
        confidence = float(clinical_reasoning.get("confidence", 0.0))
        latency = {
            "ner_ms": ner_latency,
            "retriever_ms": retriever_latency,
            "reasoner_ms": reasoner_latency,
            "total_ms": total_latency,
        }
        gpu_usage = self._gpu_usage()

        logger.info(
            "Agent latency ner=%.2fms retriever=%.2fms reasoner=%.2fms total=%.2fms gpu=%s",
            ner_latency,
            retriever_latency,
            reasoner_latency,
            total_latency,
            gpu_usage,
        )

        return {
            "entities": entities,
            "normalized_entities": normalized_entities,
            "knowledge": knowledge,
            "clinical_reasoning": clinical_reasoning,
            "confidence": confidence,
            "processing_time": latency,
            "gpu_usage": gpu_usage,
        }

    def _normalize_entities(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for entity in entities:
            entity_type = str(entity.get("type", "")).upper()
            if entity_type == "MEDICINE":
                normalized.append(self.rxnorm_service.normalize(entity))
            elif entity_type in {"SYMPTOM", "DISEASE"}:
                normalized.append(self.icd10_service.normalize(entity))
            else:
                normalized.append(
                    {
                        "original": entity.get("text", ""),
                        "type": entity.get("type", ""),
                        "code_system": "",
                        "code": "",
                        "name": entity.get("text", ""),
                        "confidence": entity.get("score", 0.0),
                    }
                )
        return normalized

    def _build_retrieval_query(self, text: str, normalized_entities: list[dict[str, Any]]) -> str:
        normalized_terms = " ".join(
            str(item.get("name") or item.get("original") or "") for item in normalized_entities
        )
        return f"{text} {normalized_terms}".strip()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000.0, 4)

    def _gpu_usage(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError:
            return {"available": False, "device": "cpu", "error": "torch_unavailable"}

        if not torch.cuda.is_available():
            return {"available": False, "device": "cpu"}
        device_index = torch.cuda.current_device()
        return {
            "available": True,
            "device": torch.cuda.get_device_name(device_index),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(device_index) / (1024 * 1024), 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(device_index) / (1024 * 1024), 2),
        }
