from typing import Any
from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class KnowledgeAgent(BaseAgent):
    """
    Normalizes raw NER entities to standard coding systems (ICD-10, RxNorm)
    using the available dictionary mappings.
    """

    def __init__(self, icd10_service: Any, rxnorm_service: Any) -> None:
        self.icd10_service = icd10_service
        self.rxnorm_service = rxnorm_service
        self.confidence = 0.90

    @property
    def name(self) -> str:
        return "knowledge_agent"

    def process(self, context: AgentContext) -> AgentContext:
        normalized: list[dict[str, Any]] = []
        
        for entity in context.entities:
            ent_type = str(entity.get("type", "")).upper()
            if ent_type == "MEDICINE":
                norm_val = self.rxnorm_service.normalize(entity)
                normalized.append(norm_val)
            elif ent_type in ("SYMPTOM", "DISEASE"):
                norm_val = self.icd10_service.normalize(entity)
                normalized.append(norm_val)
            else:
                normalized.append({
                    "original": entity.get("text", ""),
                    "type": entity.get("type", ""),
                    "code_system": "",
                    "code": "",
                    "name": entity.get("text", ""),
                    "confidence": entity.get("score", 0.0),
                })
                
        context.normalized_entities = normalized
        return context
