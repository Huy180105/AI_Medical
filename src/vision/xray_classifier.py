import io
import os
from typing import Any, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Canonical radiological label taxonomies
CHEXPERT_14_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

CHEXPERT_5_COMPETITION_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

LEGACY_NIH_6_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Infiltration",
    "Atelectasis",
    "Cardiomegaly",
    "Normal",
]


class UntrainedModelError(ValueError):
    """Raised when clinical inference or benchmark is attempted on an untrained model."""
    pass


class CheckpointMismatchError(ValueError):
    """Raised when loaded checkpoint dimensions do not match the expected label taxonomy."""
    pass


class ChestXRayClassifier:
    """
    Deep Learning Chest X-Ray (CXR) Classifier based on DenseNet-121 (CheXNet architecture).
    
    CRITICAL METHODOLOGICAL GUARANTEES:
    1. Checkpoint Verification: Must explicitly load a provenance-verified checkpoint
       for clinical benchmarking. Without a valid checkpoint, the model is marked
       as UNTRAINED and blocked from clinical claims unless allow_untrained=True is set.
    2. Label Taxonomy: Differentiates CheXpert 14, CheXpert 5 Competition, and Legacy NIH 6.
       'Normal' and 'Infiltration' are NOT 1-to-1 mappings for CheXpert labels.
    3. Explainability: generate_heatmap_cam() produces a Class-Agnostic Channel-Mean
       Activation Map, NOT a class-specific Grad-CAM.
    """

    def __init__(
        self,
        device: str | None = None,
        weights_path: str | None = None,
        labels: Optional[List[str]] = None,
        allow_untrained: bool = False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.labels = labels if labels is not None else Config.XRAY_LABELS
        self.num_classes = len(self.labels)
        self.allow_untrained = allow_untrained

        self.is_trained = False
        self.weights_source = "UNTRAINED_KAIMING_INITIALIZED"

        logger.info(
            "Initializing DenseNet-121 CXR model with %d target classes on %s",
            self.num_classes,
            self.device,
        )
        self.model = self._build_densenet_model()

        # Load and verify checkpoint if provided
        if weights_path:
            self._load_checkpoint(weights_path)
        else:
            if not self.allow_untrained:
                raise UntrainedModelError(
                    "A trained checkpoint is required for clinical benchmarking. "
                    "Provide weights_path or explicitly set allow_untrained=True for software testing only."
                )
            logger.warning(
                "WARNING: ChestXRayClassifier initialized WITHOUT a trained checkpoint. "
                "Output probabilities are random and cannot be used for clinical evaluation."
            )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((Config.XRAY_IMG_SIZE, Config.XRAY_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_densenet_model(self) -> nn.Module:
        """
        Builds DenseNet-121 with custom multi-label classification projection.
        """
        densenet = models.densenet121(weights=None)
        num_features = densenet.classifier.in_features  # 1024
        
        densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )

        torch.manual_seed(42)
        with torch.no_grad():
            for m in densenet.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        return densenet

    def _load_checkpoint(self, weights_path: str) -> None:
        """
        Loads and validates checkpoint format, tensor dimensions, and state_dict keys.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

        logger.info("Loading CXR checkpoint from: %s", weights_path)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Strip DataParallel 'module.' prefix if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k[7:] if k.startswith("module.") else k
            clean_state_dict[clean_key] = v

        try:
            self.model.load_state_dict(clean_state_dict, strict=True)
        except RuntimeError as e:
            raise CheckpointMismatchError(
                f"Checkpoint architecture mismatch with DenseNet-121 (num_classes={self.num_classes}): {e}"
            )

        self.is_trained = True
        self.weights_source = str(weights_path)
        logger.info("Successfully loaded and verified checkpoint: %s", weights_path)

    def _load_image(self, image_input: bytes | str | Image.Image) -> Image.Image:
        """
        Standardizes image input into a RGB PIL Image.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def extract_features(self, image_input: bytes | str | Image.Image) -> torch.Tensor:
        """
        Extracts 1024-dimensional dense visual feature embedding from the penultimate layer.
        """
        img = self._load_image(image_input)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.features(tensor)
            out = nn.functional.relu(features, inplace=True)
            pooled = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            flat = torch.flatten(pooled, 1)

        return flat

    def predict(self, image_input: bytes | str | Image.Image, threshold: float = 0.40) -> dict[str, Any]:
        """
        Infers thoracic pathologies from Chest X-Ray image.
        Returns confidence scores, detected pathologies, model provenance status, and metadata.
        """
        img = self._load_image(image_input)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        results_by_label = {
            label: round(float(probs[i]), 4) for i, label in enumerate(self.labels)
        }

        # Pathologies exceeding threshold (excluding negative 'Normal' or 'No Finding')
        detected = [
            label for label, score in results_by_label.items()
            if score >= threshold and label not in {"Normal", "No Finding"}
        ]

        if not detected and results_by_label.get("Normal", 0.0) >= 0.35:
            primary_finding = "Normal"
            primary_confidence = results_by_label["Normal"]
        elif not detected and results_by_label.get("No Finding", 0.0) >= 0.35:
            primary_finding = "No Finding"
            primary_confidence = results_by_label["No Finding"]
        elif detected:
            primary_finding = max(detected, key=lambda k: results_by_label[k])
            primary_confidence = results_by_label[primary_finding]
        else:
            primary_finding = max(results_by_label, key=results_by_label.get)
            primary_confidence = results_by_label[primary_finding]

        return {
            "primary_finding": primary_finding,
            "confidence": primary_confidence,
            "detected_pathologies": detected,
            "pathology_probabilities": results_by_label,
            "image_size": f"{img.width}x{img.height}",
            "model_backbone": "DenseNet-121",
            "model_training_status": "TRAINED" if self.is_trained else "UNTRAINED_SOFTWARE_TEST_ONLY",
            "weights_source": self.weights_source,
            "device": str(self.device),
        }

    def generate_heatmap_cam(self, image_input: bytes | str | Image.Image) -> dict[str, Any]:
        """
        Generates Class-Agnostic Channel-Mean Activation Map (CAM) heatmap metadata.
        NOTE: This computes spatial mean activation across channels, NOT class-specific Grad-CAM.
        """
        img = self._load_image(image_input)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.features(tensor)
            act = nn.functional.relu(features, inplace=True)
            # Spatial mean activation across channels
            heatmap = torch.mean(act, dim=1).squeeze(0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

        hotspots = []
        h, w = heatmap.shape
        for r in range(h):
            for c in range(w):
                val = float(heatmap[r, c])
                if val >= 0.6:
                    hotspots.append({
                        "grid_y": r,
                        "grid_x": c,
                        "relative_y": round(r / h, 3),
                        "relative_x": round(c / w, 3),
                        "intensity": round(val, 3),
                    })

        return {
            "method": "Class-Agnostic Channel-Mean Activation Map",
            "grid_resolution": f"{w}x{h}",
            "hotspot_count": len(hotspots),
            "hotspots": hotspots[:15],
            "max_activation": round(float(np.max(heatmap)), 4),
        }
