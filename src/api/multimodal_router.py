from datetime import datetime, timezone
import json
import os
from typing import Any, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.agents.context import AgentContext
from src.agents.cxr_agent import CXRDiagnosticAgent
from src.agents.executor import AgentExecutor
from src.agents.guideline_agent import GuidelineAgent
from src.agents.lab_vital_agent import VitalLabDynamicsAgent
from src.agents.multimodal_orchestrator import MultimodalConsensusOrchestrator
from src.agents.temporal_guard import TemporalLeakageGuardAgent
from src.agents.triage_agent import TriageTimelineAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/multimodal",
    tags=["Multimodal Multi-Agent Clinical Platform"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOAD_TEMP_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

_cached_cxr_agent: Optional[CXRDiagnosticAgent] = None


def get_cxr_agent() -> CXRDiagnosticAgent:
    global _cached_cxr_agent
    if _cached_cxr_agent is None:
        _cached_cxr_agent = CXRDiagnosticAgent()
    return _cached_cxr_agent


def build_multimodal_swarm(ner_predictor: Optional[Any] = None) -> AgentExecutor:
    """Instantiates the 6-stage Multimodal Clinical Multi-Agent Swarm."""
    agents = [
        TemporalLeakageGuardAgent(),
        TriageTimelineAgent(ner_predictor=ner_predictor),
        VitalLabDynamicsAgent(),
        get_cxr_agent(),
        GuidelineAgent(),
        MultimodalConsensusOrchestrator(),
    ]
    return AgentExecutor(agents)


class MultimodalConsultJSONRequest(BaseModel):
    clinical_text: str = Field(
        ...,
        description="Patient clinical notes, chief complaint, or history of illness.",
        json_schema_extra={
            "example": "Bệnh nhân sốt cao rét run, ho đờm vàng và khó thở tăng dần 2 ngày nay."
        },
    )
    vitals: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        json_schema_extra={
            "example": {"heart_rate": 105, "respiratory_rate": 24, "systolic_bp": 95, "spo2": 92}
        },
    )
    labs: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        json_schema_extra={"example": {"wbc": 15.2, "crp": 48.0}},
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Local path to Chest X-Ray image (PNG, JPG, DICOM).",
    )
    patient_id: Optional[str] = Field(default="PATIENT_DEFAULT")
    prediction_anchor_time: Optional[str] = Field(default=None)
    save_output: bool = Field(default=True)


@router.get("/nih-benchmark")
async def get_nih_cxr_benchmark() -> dict[str, Any]:
    """
    Returns Experiment E1 (Pretrained DenseNet-121 on NIH ChestX-ray14) benchmark metrics
    and dataset split accounting for dashboard visualization.
    """
    bench_path = os.path.join(REPORTS_DIR, "nih_benchmark_report.json")
    manifest_summary_path = os.path.join(REPORTS_DIR, "nih_manifest_summary.json")

    benchmark_data: Optional[dict[str, Any]] = None
    manifest_summary: Optional[dict[str, Any]] = None

    if os.path.exists(bench_path):
        try:
            with open(bench_path, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception as exc:
            logger.warning("Could not parse nih_benchmark_report.json: %s", exc)

    if os.path.exists(manifest_summary_path):
        try:
            with open(manifest_summary_path, "r", encoding="utf-8") as f:
                manifest_summary = json.load(f)
        except Exception as exc:
            logger.warning("Could not parse nih_manifest_summary.json: %s", exc)

    return {
        "status": "success",
        "experiment_id": "E1_VISION_NIH_CHESTXRAY14",
        "has_benchmark_report": benchmark_data is not None,
        "benchmark_report": benchmark_data,
        "manifest_summary": manifest_summary,
    }


@router.post("/consult")
async def consult_multimodal_case_multipart(
    fastapi_req: Request,
    clinical_text: str = Form(..., description="Chief complaint, triage notes, and patient history"),
    patient_id: Optional[str] = Form("PATIENT_UNKNOWN"),
    prediction_anchor_time: Optional[str] = Form(None),
    heart_rate: Optional[float] = Form(None),
    respiratory_rate: Optional[float] = Form(None),
    systolic_bp: Optional[float] = Form(None),
    diastolic_bp: Optional[float] = Form(None),
    spo2: Optional[float] = Form(None),
    temperature_c: Optional[float] = Form(None),
    wbc: Optional[float] = Form(None),
    crp: Optional[float] = Form(None),
    troponin: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
    save_output: bool = Form(True),
) -> dict[str, Any]:
    """
    Multimodal Clinical Consultation Endpoint (Multipart Form + Image Upload).
    """
    try:
        saved_image_path = None
        if image and image.filename:
            safe_filename = f"upload_{uuid.uuid4().hex[:8]}_{image.filename}"
            saved_image_path = os.path.join(UPLOAD_TEMP_DIR, safe_filename)
            content = await image.read()
            with open(saved_image_path, "wb") as f:
                f.write(content)

        vitals = {
            k: v
            for k, v in {
                "heart_rate": heart_rate,
                "respiratory_rate": respiratory_rate,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "spo2": spo2,
                "temperature_c": temperature_c,
            }.items()
            if v is not None
        }
        labs = {
            k: v
            for k, v in {"wbc": wbc, "crp": crp, "troponin": troponin}.items()
            if v is not None
        }

        predictor = getattr(fastapi_req.app.state, "predictor", None)
        executor = build_multimodal_swarm(ner_predictor=predictor)

        now_iso = datetime.now(timezone.utc).isoformat()
        context = AgentContext(text=clinical_text)
        context.patient_id = patient_id
        context.metadata["prediction_anchor_time"] = prediction_anchor_time or now_iso
        context.metadata["vitals"] = vitals
        context.metadata["labs"] = labs
        if saved_image_path:
            context.metadata["image_path"] = saved_image_path

        pipeline = [
            "temporal_guard",
            "triage_agent",
            "lab_vital_agent",
            "cxr_agent",
            "guideline_agent",
            "multimodal_orchestrator",
        ]
        context = executor.execute_pipeline(pipeline, context)

        consultation_id = f"consultation_{uuid.uuid4().hex[:10]}"
        result = {
            "consultation_id": consultation_id,
            "patient_id": patient_id,
            "timestamp": now_iso,
            "clinical_decision": context.clinical_decision,
            "trace": context.trace,
            "status": "success",
        }

        if save_output:
            out_file = os.path.join(OUTPUT_DIR, f"{consultation_id}.json")
            with open(out_file, "w", encoding="utf-8") as of:
                json.dump(result, of, indent=2, ensure_ascii=False)
            result["output_file"] = os.path.relpath(out_file, BASE_DIR)

        return result
    except Exception as exc:
        logger.error("Multimodal consultation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Multimodal consultation failed: {exc}")


@router.post("/consult-json")
async def consult_multimodal_case_json(
    fastapi_req: Request,
    payload: MultimodalConsultJSONRequest,
) -> dict[str, Any]:
    """
    Multimodal Clinical Consultation Endpoint (JSON Payload).
    """
    try:
        predictor = getattr(fastapi_req.app.state, "predictor", None)
        executor = build_multimodal_swarm(ner_predictor=predictor)

        now_iso = datetime.now(timezone.utc).isoformat()
        context = AgentContext(text=payload.clinical_text)
        context.patient_id = payload.patient_id
        context.metadata["prediction_anchor_time"] = payload.prediction_anchor_time or now_iso
        context.metadata["vitals"] = payload.vitals
        context.metadata["labs"] = payload.labs
        if payload.image_path:
            full_img_path = payload.image_path
            if not os.path.isabs(full_img_path):
                full_img_path = os.path.join(BASE_DIR, full_img_path)
            context.metadata["image_path"] = full_img_path

        pipeline = [
            "temporal_guard",
            "triage_agent",
            "lab_vital_agent",
            "cxr_agent",
            "guideline_agent",
            "multimodal_orchestrator",
        ]
        context = executor.execute_pipeline(pipeline, context)

        consultation_id = f"consultation_{uuid.uuid4().hex[:10]}"
        result = {
            "consultation_id": consultation_id,
            "patient_id": payload.patient_id,
            "timestamp": now_iso,
            "clinical_decision": context.clinical_decision,
            "trace": context.trace,
            "status": "success",
        }

        if payload.save_output:
            out_file = os.path.join(OUTPUT_DIR, f"{consultation_id}.json")
            with open(out_file, "w", encoding="utf-8") as of:
                json.dump(result, of, indent=2, ensure_ascii=False)
            result["output_file"] = os.path.relpath(out_file, BASE_DIR)

        return result
    except Exception as exc:
        logger.error("Multimodal consultation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Multimodal consultation failed: {exc}")
