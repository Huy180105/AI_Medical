from typing import Any
from pydantic import BaseModel, Field


class ClinicalEntityRef(BaseModel):
    """
    Reference to a medical entity involved in a relation.
    """
    text: str
    type: str
    start: int = 0
    end: int = 0


class ClinicalRelation(BaseModel):
    """
    Directed semantic relation between a subject entity and object entity.
    Types: TREATS, CAUSED_BY, HAS_SYMPTOM, HAS_TEST_RESULT, CONTRAINDICATED_FOR
    """
    subject: ClinicalEntityRef
    relation_type: str = Field(description="Relation category (e.g. TREATS, CAUSED_BY, HAS_SYMPTOM).")
    object: ClinicalEntityRef
    confidence: float = Field(default=1.0, description="Extraction confidence score (0.0 to 1.0).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject.model_dump(),
            "relation_type": self.relation_type,
            "object": self.object.model_dump(),
            "confidence": round(self.confidence, 4)
        }
