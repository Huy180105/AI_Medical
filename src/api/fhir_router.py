from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.interop.fhir_exporter import FHIRExporter
from src.interop.patient import FHIRPatientBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/fhir", tags=["FHIR Interoperability"])


class FHIRBundleRequest(BaseModel):
    """Payload format to convert into a FHIR R4 Bundle."""
    patient_id: str = Field(default="patient_001", description="Target patient MRN or ID.")
    session_id: str = Field(default="session_001")
    telemetry: dict[str, Any] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: str = Field(default="Low")
    clinical_decision: dict[str, Any] = Field(default_factory=dict)
    recommendations: dict[str, Any] = Field(default_factory=dict)


@router.post("/bundle")
async def export_fhir_bundle(request: FHIRBundleRequest) -> dict[str, Any]:
    """
    Transforms internal clinical decisions and telemetry states into a standardized FHIR R4 Bundle JSON.
    """
    try:
        payload = request.model_dump()
        bundle = FHIRExporter.export_bundle(payload)
        return bundle
    except Exception as exc:
        logger.error("Failed to generate FHIR Bundle: %s", exc)
        raise HTTPException(status_code=500, detail=f"FHIR Bundle export failed: {exc}")


@router.get("/patient/{patient_id}")
async def get_fhir_patient(patient_id: str) -> dict[str, Any]:
    """
    Returns a FHIR R4 Patient resource representation for the given patient ID.
    """
    try:
        return FHIRPatientBuilder.build_patient(patient_id=patient_id)
    except Exception as exc:
        logger.error("Failed to generate FHIR Patient: %s", exc)
        raise HTTPException(status_code=500, detail=f"FHIR Patient retrieval failed: {exc}")
