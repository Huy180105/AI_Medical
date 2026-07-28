from typing import Any
from pydantic import BaseModel, Field


class EntityCandidate(BaseModel):
    """
    Standard concept candidate (ICD-10 / RxNorm).
    """
    code: str = Field(description="Standard clinical code (e.g. J18.9 or RxNorm 161).")
    display: str = Field(description="Concept display text.")
    score: float = Field(description="Raw similarity or reranker relevance score.")
    confidence: float = Field(default=1.0, description="Normalized calibrated confidence score (0.0 to 1.0).")
    rank: int = Field(default=1, description="Candidate rank (1 to K).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "display": self.display,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "rank": self.rank
        }


class CandidateRankingResult(BaseModel):
    """
    Top-K candidate ranking results for an entity text.
    """
    entity_text: str
    entity_type: str
    top_candidates: list[EntityCandidate] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            "top_candidates": [c.to_dict() for c in self.top_candidates]
        }
