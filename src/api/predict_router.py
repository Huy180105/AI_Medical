from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.utils.logger import get_logger
from src.inference.predict_ner import MedicalNERPredictor

logger = get_logger(__name__)

router = APIRouter(tags=["Clinical NER Prediction"])

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="Vietnamese clinical text for medical entity extraction.",
        json_schema_extra={"example": "Bệnh nhân sốt cao và đau ngực, được bác sĩ chỉ định dùng Paracetamol."}
    )

class EntityResponse(BaseModel):
    text: str
    type: str
    score: float = 1.0
    start: int = 0
    end: int = 0

class PredictResponse(BaseModel):
    text: str
    entities: list[EntityResponse]
    entity_count: int

@router.post("/predict", response_model=PredictResponse)
@router.post("/api/v1/predict", response_model=PredictResponse)
async def predict_entities(fastapi_req: Request, payload: PredictRequest) -> PredictResponse:
    """
    Extracts high-precision clinical medical entities (SYMPTOM, DISEASE, MEDICINE, TEST) from Vietnamese text.
    """
    try:
        predictor = getattr(fastapi_req.app.state, "predictor", None)
        if predictor is None:
            predictor = MedicalNERPredictor()

        raw_entities = predictor.predict(payload.text)
        
        filtered_entities = []
        for e in raw_entities:
            filtered_entities.append(
                EntityResponse(
                    text=e.get("text", "").strip(),
                    type=e.get("type", "UNKNOWN"),
                    score=float(e.get("score", 1.0)),
                    start=int(e.get("start", 0)),
                    end=int(e.get("end", 0))
                )
            )

        return PredictResponse(
            text=payload.text,
            entities=filtered_entities,
            entity_count=len(filtered_entities)
        )
    except Exception as exc:
        logger.error("Error during entity extraction: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

