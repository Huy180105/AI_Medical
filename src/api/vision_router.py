from typing import Any
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from src.vision.xray_classifier import ChestXRayClassifier
from src.vision.multimodal_fusion import MultimodalClinicalFusion
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/vision", tags=["Medical Vision & Multimodal CXR"])

_xray_classifier: ChestXRayClassifier | None = None
_multimodal_fusion: MultimodalClinicalFusion | None = None


def get_xray_classifier() -> ChestXRayClassifier:
    global _xray_classifier
    if _xray_classifier is None:
        _xray_classifier = ChestXRayClassifier(allow_untrained=True)
    return _xray_classifier


def get_multimodal_fusion() -> MultimodalClinicalFusion:
    global _multimodal_fusion
    if _multimodal_fusion is None:
        classifier = get_xray_classifier()
        _multimodal_fusion = MultimodalClinicalFusion(xray_classifier=classifier)
    return _multimodal_fusion


@router.get("/health")
async def vision_health() -> dict[str, Any]:
    classifier = get_xray_classifier()
    return {
        "status": "healthy",
        "service": "Chest X-Ray Deep Vision & Multimodal Fusion",
        "backbone": "DenseNet-121 (CheXNet architecture)",
        "supported_pathologies": classifier.labels,
        "device": str(classifier.device)
    }


@router.post("/xray/predict")
async def predict_chest_xray(
    file: UploadFile = File(..., description="Chest X-Ray image file (.png, .jpg, .jpeg)")
) -> dict[str, Any]:
    """
    Analyzes uploaded Chest X-Ray (CXR) radiograph using DenseNet-121.
    Returns detected pathologies, probabilities, and Grad-CAM spatial activation hotspots.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        # Still attempt processing if standard image extension
        ext = file.filename.split(".")[-1].lower() if file.filename else ""
        if ext not in ["png", "jpg", "jpeg", "bmp", "dcm"]:
            raise HTTPException(status_code=400, detail="File must be an image (.png, .jpg, .jpeg)")

    try:
        image_bytes = await file.read()
        classifier = get_xray_classifier()
        prediction = classifier.predict(image_bytes)
        cam_metadata = classifier.generate_heatmap_cam(image_bytes)

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": prediction,
            "explainability_cam": cam_metadata
        }
    except Exception as exc:
        logger.error("Error processing X-ray image: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to analyze X-ray: {str(exc)}")


@router.post("/multimodal/predict")
async def predict_multimodal_clinical(
    file: UploadFile = File(..., description="Chest X-Ray image file"),
    clinical_text: str = Form(..., description="Patient clinical note or symptoms in Vietnamese")
) -> dict[str, Any]:
    """
    Synthesizes Chest X-Ray radiograph + Unstructured Clinical Text.
    Executes cross-modal gated attention fusion to output a unified clinical decision.
    """
    if not clinical_text.strip():
        raise HTTPException(status_code=400, detail="Clinical text cannot be empty.")

    try:
        image_bytes = await file.read()
        fusion_engine = get_multimodal_fusion()
        
        multimodal_result = fusion_engine.analyze_multimodal(
            image_input=image_bytes,
            clinical_text=clinical_text
        )

        return {
            "status": "success",
            "filename": file.filename,
            "result": multimodal_result
        }
    except Exception as exc:
        logger.error("Error executing multimodal fusion: %s", exc)
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(exc)}")
