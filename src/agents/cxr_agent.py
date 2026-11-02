import io
import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.utils.logger import get_logger
from src.vision.pretrained_cxr import PretrainedNihDenseNet121
from src.vision.xray_classifier import ChestXRayClassifier

logger = get_logger(__name__)


class CXRDiagnosticAgent(BaseAgent):
    """
    Diagnostic Imaging Specialist Agent for Chest X-Ray (CXR) interpretation.
    Prioritizes the provenance-verified PretrainedNihDenseNet121 (14 NIH labels),
    enforces image verification, quarantines synthetic test fixtures, and returns
    standardized inference status codes.
    """

    def __init__(self, classifier: Optional[Any] = None) -> None:
        self.classifier = classifier
        self._nih_adapter: Optional[PretrainedNihDenseNet121] = None

    @property
    def name(self) -> str:
        return "cxr_agent"

    def _ensure_classifier(self) -> Optional[Any]:
        if self.classifier is not None:
            return self.classifier
        if self._nih_adapter is not None:
            return self._nih_adapter

        nih_ckpt = Path("models/nih-densenet121-nih14.pt")
        if nih_ckpt.exists():
            try:
                self._nih_adapter = PretrainedNihDenseNet121(
                    weights_path=nih_ckpt,
                    checkpoint_profile="nih14",
                    verify_sha256=True,
                )
                return self._nih_adapter
            except Exception as exc:
                logger.warning("PretrainedNihDenseNet121 load fallback in CXR agent: %s", exc)

        try:
            self.classifier = ChestXRayClassifier(allow_untrained=True)
        except Exception as exc:
            logger.error("Failed to initialize ChestXRayClassifier in CXR agent: %s", exc)
        return self.classifier

    def _validate_image_input(self, image_input: Any) -> tuple[bool, str, bool]:
        """
        Validates input format and detects whether image is a synthetic test fixture.
        Returns: (is_valid: bool, reason: str, is_synthetic: bool)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return False, f"File not found: {image_input}", False
            norm_p = image_input.replace("\\", "/").lower()
            is_synthetic = "synthetic" in norm_p or "tests/fixtures" in norm_p
            try:
                with Image.open(image_input) as img:
                    img.verify()
                return True, "VALID_PATH", is_synthetic
            except Exception as exc:
                return False, f"Corrupt image file: {exc}", is_synthetic

        if isinstance(image_input, (bytes, bytearray)):
            try:
                with Image.open(io.BytesIO(image_input)) as img:
                    img.verify()
                return True, "VALID_BYTES", False
            except Exception as exc:
                return False, f"Invalid image bytes: {exc}", False

        return False, "UNSUPPORTED_INPUT_TYPE", False

    def process(self, context: AgentContext) -> AgentContext:
        image_input = context.metadata.get("image_path") or context.metadata.get("image_bytes")

        if not image_input:
            logger.info("CXR Diagnostic Agent: No image input provided. Status: NOT_PROVIDED.")
            context.metadata["cxr_status"] = "NOT_PROVIDED"
            context.metadata["cxr_available"] = False
            return context

        is_valid, validation_reason, is_synthetic = self._validate_image_input(image_input)
        if not is_valid:
            logger.warning("CXR Diagnostic Agent: Invalid image input: %s", validation_reason)
            context.metadata["cxr_status"] = "INVALID_INPUT"
            context.metadata["cxr_available"] = False
            context.metadata["cxr_error"] = validation_reason
            return context

        if is_synthetic:
            context.metadata["cxr_is_synthetic_warning"] = (
                "SYNTHETIC_FIXTURE_WARNING: Input is a synthetic graphic simulation. "
                "Output cannot be used for clinical evaluation or benchmarking."
            )
            logger.warning("CXR Diagnostic Agent: Processing synthetic test fixture (%s).", image_input)

        model = self._ensure_classifier()
        if not model:
            context.metadata["cxr_status"] = "MODEL_UNAVAILABLE"
            context.metadata["cxr_available"] = False
            context.metadata["cxr_error"] = "Vision classifier backbone could not be loaded."
            return context

        try:
            prediction = model.predict(image_input)
            if isinstance(model, PretrainedNihDenseNet121):
                op_probs = {
                    k: float(v)
                    for k, v in prediction.get("op_calibrated_probabilities", {}).items()
                    if v is not None
                }
                if "Effusion" in op_probs and "Pleural Effusion" not in op_probs:
                    op_probs["Pleural Effusion"] = op_probs["Effusion"]
                raw_predictions = op_probs
                top_lbl = max(op_probs, key=op_probs.get) if op_probs else "Normal"
                top_conf = float(op_probs.get(top_lbl, 0.0))
                primary_finding = top_lbl if top_conf >= 0.50 else "Normal"
            else:
                raw_predictions = (
                    prediction.get("predictions")
                    or prediction.get("pathology_probabilities")
                    or {}
                )
                primary_finding = prediction.get("primary_finding", "Normal")
                top_conf = float(prediction.get("confidence", 0.0))

            for k, val in raw_predictions.items():
                if val != val or val < 0.0 or val > 1.0:
                    raise ValueError(f"Invalid probability value for label '{k}': {val}")

            status = "SUCCESS_SYNTHETIC" if is_synthetic else "SUCCESS"
            context.metadata["cxr_status"] = status
            context.metadata["cxr_available"] = True
            context.metadata["cxr_is_synthetic"] = is_synthetic
            context.metadata["cxr_findings"] = raw_predictions
            context.metadata["cxr_primary_finding"] = primary_finding
            context.metadata["cxr_confidence"] = top_conf

            context.trace.append(
                {
                    "agent": self.name,
                    "status": status,
                    "is_synthetic": is_synthetic,
                    "findings": context.metadata["cxr_findings"],
                    "primary": context.metadata["cxr_primary_finding"],
                }
            )
        except Exception as exc:
            logger.error("CXR Diagnostic Agent: Inference failed: %s", exc)
            context.metadata["cxr_status"] = "INFERENCE_FAILED"
            context.metadata["cxr_available"] = False
            context.metadata["cxr_error"] = str(exc)

        return context
