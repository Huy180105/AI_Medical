from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

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
    start: int = 0
    end: int = 0


class PredictResponse(BaseModel):
    text: str
    entities: list[EntityResponse]
    entity_count: int


@router.post("/predict", response_model=PredictResponse)
async def predict_entities(fastapi_req: Request, payload: PredictRequest) -> PredictResponse:
    """
    Extracts clinical medical entities (SYMPTOM, DISEASE, MEDICINE, TEST) from Vietnamese text.
    """
    try:
        predictor = getattr(fastapi_req.app.state, "predictor", None)
        if not predictor:
            raise HTTPException(status_code=503, detail="NER Predictor model not initialized.")

        raw_entities = predictor.predict(payload.text)
        entities = [
            EntityResponse(
                text=ent.get("text", ""),
                type=ent.get("type", "UNKNOWN"),
                start=ent.get("start", 0),
                end=ent.get("end", 0)
            )
            for ent in raw_entities
        ]
        return PredictResponse(
            text=payload.text,
            entities=entities,
            entity_count=len(entities)
        )
    except Exception as exc:
        logger.error("Error during entity extraction: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
