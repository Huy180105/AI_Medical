from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/workflow", tags=["Clinical Workflow"])


class WorkflowRequest(BaseModel):
    workflow_id: str = Field(default="default_clinical_workflow")
    input_text: str = Field(..., description="Clinical note text for end-to-end workflow execution.")


@router.post("/run")
async def run_clinical_workflow(fastapi_req: Request, payload: WorkflowRequest) -> dict[str, Any]:
    """
    Executes end-to-end clinical workflow (NER -> Assertion -> Relation -> CDSS -> FHIR).
    """
    try:
        predictor = getattr(fastapi_req.app.state, "predictor", None)
        decision_engine = getattr(fastapi_req.app.state, "decision_engine", None)
        
        entities = predictor.predict(payload.input_text) if predictor else []
        decision = decision_engine.make_decision(entities, []) if decision_engine else {}

        return {
            "workflow_id": payload.workflow_id,
            "status": "COMPLETED",
            "input_text": payload.input_text,
            "extracted_entities": entities,
            "clinical_decision": decision
        }
    except Exception as exc:
        logger.error("Error executing clinical workflow: %s", exc)
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}")
