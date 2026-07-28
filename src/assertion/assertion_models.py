from typing import Any
from pydantic import BaseModel, Field


class ClinicalAssertion(BaseModel):
    """
    Assertion context status flags for extracted medical entities.
    """
    is_negated: bool = Field(default=False, description="Entity is negated/absent (e.g. không sốt).")
    is_family: bool = Field(default=False, description="Entity relates to family history (e.g. mẹ bị hen).")
    is_historical: bool = Field(default=False, description="Entity occurred in past history (e.g. tiền sử lao).")
    is_uncertain: bool = Field(default=False, description="Entity is uncertain/hypothetical (e.g. nghi ngờ viêm phổi).")
    is_conditional: bool = Field(default=False, description="Entity is conditional (e.g. nếu đau ngực thì...).")


class AssertionTaggedEntity(BaseModel):
    """
    Medical entity enriched with start/end character offsets, type, and clinical assertion status.
    """
    text: str
    type: str
    start: int = 0
    end: int = 0
    assertion: ClinicalAssertion = Field(default_factory=ClinicalAssertion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "assertion": self.assertion.model_dump()
        }
