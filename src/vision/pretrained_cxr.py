"""
Pretrained Chest X-Ray Baseline Adapter (DenseNet-121 / CheXpert).

MODEL CARD & PROVENANCE METADATA:
- Model Architecture: DenseNet-121 (1-channel input Conv2d(1, 64, kernel_size=7, stride=2, padding=3))
- Checkpoint Origin: TorchXRayVision v1.5.5 (Cohen et al., MIDL 2022, https://arxiv.org/abs/2002.02497)
- Checkpoint Filename: chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
- Checkpoint SHA-256: 1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4
- Training Dataset: Stanford CheXpert Dataset ONLY (single-dataset 'chex' weights; NOT multi-dataset 'all')
- Release URL: https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
- License: Apache License 2.0 (OSI Approved)
- Input Preprocessing:
  1. 8-bit grayscale conversion [0, 255] -> linear scaling to [-1024.0, 1024.0] via (2 * (img / 255.0) - 1.0) * 1024.0
  2. Square Center Crop (XRayCenterCrop: min(H, W) x min(H, W)) to prevent aspect-ratio distortion
  3. Bilinear interpolation resize to 224 x 224 (1 channel)
- Output Activation:
  - Raw probabilities: torch.sigmoid(logits)
  - Operating-point calibrated probabilities: piecewise-linear rescaling via CheXpert op_threshs

18-Slot Classifier Head Verification (nn.Linear(1024, 18)):
TorchXRayVision uses a universal 18-class output layer across all datasets. In the CheXpert-only ('chex')
checkpoint, 11 CheXpert slots have trained weights (||w||_2 > 0), whereas 7 non-CheXpert slots
(indices 2, 5, 6, 9, 11, 12, 13) were masked during training and have exact zero weights (||w||_2 == 0, bias == 0).
Those 7 slots are explicitly quarantined as None and blocked from downstream use.

Explicit Target Mapping (CheXpert 5 Competition):
1. Atelectasis      -> Raw index 0  ('Atelectasis', active CheXpert slot)
2. Consolidation    -> Raw index 1  ('Consolidation', active CheXpert slot)
3. Edema            -> Raw index 4  ('Edema', active CheXpert slot)
4. Pleural Effusion -> Raw index 7  ('Effusion' in TorchXRayVision ontology; mapped to MIMIC-CXR 'Pleural Effusion')
5. Cardiomegaly     -> Raw index 10 ('Cardiomegaly', active CheXpert slot)
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_CHEX_DENSENET121_SHA256 = (
    "1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4"
)
EXPECTED_NIH_DENSENET121_SHA256 = (
    "31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf"
)
NIH_DENSENET121_RELEASE_URL = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)

# Universal 18 target names stored in serialized TorchXRayVision DenseNet.targets
RAW_CHECKPOINT_TARGETS = [
    "Atelectasis",                  # 0  - Trained on CheXpert (w_norm=1.7562, bias=+0.0583)
    "Consolidation",                # 1  - Trained on CheXpert (w_norm=2.2071, bias=-0.0275)
    "Infiltration",                 # 2  - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Pneumothorax",                 # 3  - Trained on CheXpert (w_norm=2.6213, bias=-0.0733)
    "Edema",                        # 4  - Trained on CheXpert (w_norm=2.4843, bias=+0.0385)
    "Emphysema",                    # 5  - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Fibrosis",                     # 6  - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Effusion",                     # 7  - Trained on CheXpert (w_norm=2.5101, bias=+0.0799) -> Maps to 'Pleural Effusion'
    "Pneumonia",                    # 8  - Trained on CheXpert (w_norm=1.9007, bias=-0.0603)
    "Pleural_Thickening",           # 9  - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Cardiomegaly",                 # 10 - Trained on CheXpert (w_norm=2.2760, bias=+0.0280)
    "Nodule",                       # 11 - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Mass",                         # 12 - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Hernia",                       # 13 - UNTRAINED in 'chex' (w_norm=0.0000, bias=0.0000)
    "Lung Lesion",                  # 14 - Trained on CheXpert (w_norm=1.9358, bias=+0.0207)
    "Fracture",                     # 15 - Trained on CheXpert (w_norm=1.8849, bias=-0.0109)
    "Lung Opacity",                 # 16 - Trained on CheXpert (w_norm=2.0103, bias=+0.1996)
    "Enlarged Cardiomediastinum",   # 17 - Trained on CheXpert (w_norm=2.1056, bias=-0.0262)
]

# Active labels in TorchXRayVision model_urls['chex']['labels'] (empty string = masked/untrained)
CHEX_ACTIVE_SLOT_LABELS = [
    "Atelectasis",                  # 0
    "Consolidation",                # 1
    "",                             # 2  (Masked)
    "Pneumothorax",                 # 3
    "Edema",                        # 4
    "",                             # 5  (Masked)
    "",                             # 6  (Masked)
    "Effusion",                     # 7
    "Pneumonia",                    # 8
    "",                             # 9  (Masked)
    "Cardiomegaly",                 # 10
    "",                             # 11 (Masked)
    "",                             # 12 (Masked)
    "",                             # 13 (Masked)
    "Lung Lesion",                  # 14
    "Fracture",                     # 15
    "Lung Opacity",                 # 16
    "Enlarged Cardiomediastinum",   # 17
]

# Published TorchXRayVision operating point thresholds for model_urls['chex']
CHEX_OP_THRESHS = [
    0.1988969,
    0.05710573,
    np.nan,
    0.0531293,
    0.1435217,
    np.nan,
    np.nan,
    0.27212676,
    0.07749717,
    np.nan,
    0.19712369,
    np.nan,
    np.nan,
    np.nan,
    0.09932402,
    0.09273402,
    0.3270967,
    0.10888247,
]

ACTIVE_CHEXPERT_INDICES = [0, 1, 3, 4, 7, 8, 10, 14, 15, 16, 17]
INACTIVE_MASKED_INDICES = [2, 5, 6, 9, 11, 12, 13]
ACTIVE_CHEXPERT_11_INDICES = ACTIVE_CHEXPERT_INDICES
INACTIVE_CHEXPERT_7_INDICES = INACTIVE_MASKED_INDICES

# Explicit, tested mapping from canonical CheXpert 5 Competition names to checkpoint raw target indices
CHEXPERT_5_MAPPING = {
    "Atelectasis": 0,
    "Consolidation": 1,
    "Edema": 4,
    "Pleural Effusion": 7,  # Explicit terminology mapping from TorchXRayVision 'Effusion' (index 7)
    "Cardiomegaly": 10,
}

ADDITIONAL_CHEXPERT_MAPPING = {
    "Pneumothorax": 3,
    "Pneumonia": 8,
    "Lung Lesion": 14,
    "Fracture": 15,
    "Lung Opacity": 16,
    "Enlarged Cardiomediastinum": 17,
}

# ==============================================================================
# NIH ChestX-ray14 (14-Label Multi-Label Ontology & Checkpoint Definitions)
# ==============================================================================
NIH_14_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# Exact mapping from canonical NIH ChestX-ray14 label names to TorchXRayVision 18-slot head indices (0..13)
NIH_14_MAPPING: Dict[str, int] = {
    "Atelectasis": 0,
    "Cardiomegaly": 10,
    "Consolidation": 1,
    "Edema": 4,
    "Effusion": 7,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Hernia": 13,
    "Infiltration": 2,
    "Mass": 12,
    "Nodule": 11,
    "Pleural_Thickening": 9,
    "Pneumonia": 8,
    "Pneumothorax": 3,
}

# Active labels in TorchXRayVision model_urls['densenet121-res224-nih']['labels']
NIH_ACTIVE_SLOT_LABELS: List[str] = [
    "Atelectasis",         # 0  (w_norm=1.9416, bias=-0.067342)
    "Consolidation",       # 1  (w_norm=2.2840, bias=-0.052101)
    "Infiltration",        # 2  (w_norm=1.9286, bias=-0.077933)
    "Pneumothorax",        # 3  (w_norm=2.7955, bias=-0.067032)
    "Edema",               # 4  (w_norm=4.3826, bias=-0.003064)
    "Emphysema",           # 5  (w_norm=2.9448, bias=-0.069527)
    "Fibrosis",            # 6  (w_norm=2.2399, bias=-0.029679)
    "Effusion",            # 7  (w_norm=2.1978, bias=-0.055370)
    "Pneumonia",           # 8  (w_norm=2.7563, bias=-0.017621)
    "Pleural_Thickening",  # 9  (w_norm=2.0202, bias=-0.034995)
    "Cardiomegaly",        # 10 (w_norm=2.8692, bias=-0.049546)
    "Nodule",              # 11 (w_norm=1.7309, bias=-0.067007)
    "Mass",                # 12 (w_norm=2.1821, bias=-0.033516)
    "Hernia",              # 13 (w_norm=3.4107, bias=-0.100636)
    "",                    # 14 (Masked in NIH checkpoint: w_norm=0.0000)
    "",                    # 15 (Masked in NIH checkpoint: w_norm=0.0000)
    "",                    # 16 (Masked in NIH checkpoint: w_norm=0.0000)
    "",                    # 17 (Masked in NIH checkpoint: w_norm=0.0000)
]

# Published TorchXRayVision operating point thresholds for model_urls['densenet121-res224-nih']
NIH_OP_THRESHS: List[float] = [
    0.039117552,
    0.0034529066,
    0.11396341,
    0.0057298196,
    0.00045666535,
    0.0018880932,
    0.012037827,
    0.038744126,
    0.0037213727,
    0.014730946,
    0.016149804,
    0.054241467,
    0.037198864,
    0.0004403434,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
]

ACTIVE_NIH_14_INDICES: List[int] = list(range(14))
INACTIVE_NIH_4_INDICES: List[int] = [14, 15, 16, 17]
SHARED_CHEX_NIH_7_INDICES: List[int] = [0, 1, 3, 4, 7, 8, 10]
UNTRAINED_CHEX_ON_NIH_7_INDICES: List[int] = [2, 5, 6, 9, 11, 12, 13]


class PretrainedCheXpertDenseNet121:
    """
    Standardized, reproducible wrapper for pretrained CheXpert DenseNet-121.
    Guarantees exact TorchXRayVision preprocessing ([-1024, 1024] + XRayCenterCrop + 224x224 resize),
    SHA-256 and per-slot weight norm verification, and masking of non-CheXpert output slots.
    """

    def __init__(
        self,
        weights_path: str = "models/chex-densenet121-chexpert.pt",
        device: Optional[str] = None,
        verify_sha256: bool = True,
        center_crop: bool = True,
    ):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Verified pretrained checkpoint not found at: {self.weights_path}. "
                "Download it first from: https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
            )

        self.verify_sha256 = verify_sha256
        self.center_crop = center_crop
        self.checkpoint_sha256 = self._compute_sha256(self.weights_path)

        if self.verify_sha256 and self.checkpoint_sha256 != EXPECTED_CHEX_DENSENET121_SHA256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {self.weights_path}! "
                f"Expected {EXPECTED_CHEX_DENSENET121_SHA256}, got {self.checkpoint_sha256}."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "Loading pretrained CheXpert DenseNet-121 on %s (SHA-256: %s...)",
            self.device,
            self.checkpoint_sha256[:12],
        )
        self.model = self._load_verified_model()
        self.model.to(self.device)
        self.model.eval()

        self.target_mapping = CHEXPERT_5_MAPPING
        self.target_names = list(self.target_mapping.keys())

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest().lower()

    def _load_verified_model(self) -> nn.Module:
        """
        Loads the torchxrayvision DenseNet-121 checkpoint safely and verifies:
        1. 18-slot target ordering matches RAW_CHECKPOINT_TARGETS.
        2. All 11 active CheXpert slots have trained non-zero classifier weights (||w_i||_2 > 0).
        3. All 7 non-CheXpert slots have zero classifier weights (||w_i||_2 == 0) and are quarantined.
        """
        try:
            import torchxrayvision as xrv
            with torch.serialization.safe_globals([xrv.models.DenseNet]):
                model = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        except Exception:
            model = torch.load(self.weights_path, map_location=self.device, weights_only=False)

        for mod in model.modules():
            if not hasattr(mod, "_non_persistent_buffers_set"):
                mod._non_persistent_buffers_set = set()

        checkpoint_targets = getattr(model, "targets", None)
        if checkpoint_targets != RAW_CHECKPOINT_TARGETS:
            raise ValueError(
                f"Checkpoint targets mismatch! Expected {RAW_CHECKPOINT_TARGETS[:5]}..., got {checkpoint_targets}"
            )

        # Populate TorchXRayVision active pathologies metadata onto loaded module
        model.pathologies = list(CHEX_ACTIVE_SLOT_LABELS)

        # Verify classifier weight norms for active vs masked slots
        with torch.no_grad():
            weight_norms = model.classifier.weight.detach().cpu().norm(dim=1).numpy()
            for idx in ACTIVE_CHEXPERT_INDICES:
                if weight_norms[idx] <= 0.0:
                    raise ValueError(
                        f"Active CheXpert slot {idx} ({RAW_CHECKPOINT_TARGETS[idx]}) has zero weight norm!"
                    )
            for idx in INACTIVE_MASKED_INDICES:
                if weight_norms[idx] > 1e-6:
                    raise ValueError(
                        f"Expected masked non-CheXpert slot {idx} ({RAW_CHECKPOINT_TARGETS[idx]}) to have zero weights, "
                        f"got norm={weight_norms[idx]}"
                    )

        return model

    def preprocess_image(self, image_input: Union[bytes, str, Path, Image.Image]) -> torch.Tensor:
        """
        Implements TorchXRayVision canonical preprocessing:
        1. Grayscale 2D image conversion (L mode, [0, 255]).
        2. Linear normalization from [0, 255] -> [-1024.0, 1024.0]: (2 * (img / 255.0) - 1.0) * 1024.0.
        3. Square Center Crop (XRayCenterCrop) along the longer dimension so aspect ratio is preserved.
        4. Bilinear interpolation resize to (1, 1, 224, 224).
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input)).convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_arr = np.array(img, dtype=np.float32)
        if img_arr.ndim != 2:
            raise ValueError(f"Expected 2D grayscale array after conversion, got shape {img_arr.shape}")

        # 1. Scale [0, 255] -> [-1024.0, 1024.0]
        norm_arr = (2.0 * (img_arr / 255.0) - 1.0) * 1024.0

        # 2. Square Center Crop (XRayCenterCrop equivalent)
        if self.center_crop:
            h, w = norm_arr.shape
            crop_size = min(h, w)
            start_x = w // 2 - (crop_size // 2)
            start_y = h // 2 - (crop_size // 2)
            norm_arr = norm_arr[start_y : start_y + crop_size, start_x : start_x + crop_size]

        # 3. Resize to (1, 1, 224, 224) using bilinear interpolation
        tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0)
        if tensor.shape[2] != 224 or tensor.shape[3] != 224:
            tensor = F.interpolate(
                tensor,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        return tensor.to(self.device)

    @staticmethod
    def _apply_op_norm(prob: float, op_thresh: float) -> Optional[float]:
        """Piecewise-linear operating-point normalization from TorchXRayVision op_norm."""
        if np.isnan(op_thresh):
            return None
        if prob < op_thresh:
            val = prob / (op_thresh * 2.0)
        else:
            val = 1.0 - ((1.0 - prob) / ((1.0 - op_thresh) * 2.0))
        return round(float(np.clip(val, 0.0, 1.0)), 4)

    def get_slot_audit_table(self) -> List[Dict[str, Any]]:
        """
        Returns the complete 18-slot verification table documenting index, raw name,
        CheXpert active status, weight L2 norm, bias, and operating threshold.
        """
        with torch.no_grad():
            w_norms = self.model.classifier.weight.detach().cpu().norm(dim=1).numpy()
            biases = self.model.classifier.bias.detach().cpu().numpy()

        table = []
        for i in range(len(RAW_CHECKPOINT_TARGETS)):
            is_active = i in ACTIVE_CHEXPERT_INDICES
            table.append({
                "slot_index": i,
                "raw_target_name": RAW_CHECKPOINT_TARGETS[i],
                "chexpert_active_label": CHEX_ACTIVE_SLOT_LABELS[i] if is_active else None,
                "mapped_mimic_cxr_label": (
                    "Pleural Effusion"
                    if RAW_CHECKPOINT_TARGETS[i] == "Effusion"
                    else (RAW_CHECKPOINT_TARGETS[i] if is_active else None)
                ),
                "is_trained_in_chex_checkpoint": is_active,
                "is_active_in_chex_checkpoint": is_active,
                "in_chexpert_5_competition": i in CHEXPERT_5_MAPPING.values(),
                "weight_l2_norm": round(float(w_norms[i]), 4),
                "bias": round(float(biases[i]), 6),
                "op_thresh": None if np.isnan(CHEX_OP_THRESHS[i]) else round(float(CHEX_OP_THRESHS[i]), 6),
            })
        return table

    def predict(self, image_input: Union[bytes, str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Infers CheXpert competition probabilities from chest radiograph.
        Returns:
            - target_probabilities: Dict[str, float] for the 5 competition pathologies (sigmoid).
            - op_calibrated_probabilities: Dict[str, float] rescaled by TorchXRayVision op_threshs.
            - active_11_probabilities: Dict[str, float] for all 11 active CheXpert slots.
            - raw_18_probabilities: Dict[str, Optional[float]] where the 7 untrained slots are explicitly None.
            - inactive_untrained_slots: List[str] of the 7 masked slots in the 'chex' checkpoint.
            - model_metadata: Provenance, SHA-256, license, and label equivalence details.
        """
        tensor = self.preprocess_image(image_input)

        with torch.no_grad():
            # Use features2 + classifier directly so we control sigmoid & op_norm explicitly
            features = self.model.features2(tensor)
            logits = self.model.classifier(features)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        if np.isnan(probs).any() or np.isinf(probs).any():
            raise ValueError("Model produced NaN or Inf probabilities")
        probs = np.clip(probs, 0.0, 1.0)

        # 1. CheXpert 5 Competition targets (raw sigmoid probabilities)
        chexpert_5_probs = {
            target: round(float(probs[idx]), 4)
            for target, idx in self.target_mapping.items()
        }

        # 2. Operating-point calibrated probabilities for CheXpert 5 Competition targets
        op_calibrated_5_probs = {
            target: self._apply_op_norm(float(probs[idx]), CHEX_OP_THRESHS[idx])
            for target, idx in self.target_mapping.items()
        }

        # 3. All 11 active CheXpert slots
        active_11_probs = {
            RAW_CHECKPOINT_TARGETS[idx]: round(float(probs[idx]), 4)
            for idx in ACTIVE_CHEXPERT_INDICES
        }

        # 4. Raw 18 slots with untrained non-CheXpert slots explicitly masked as None
        raw_18_probs: Dict[str, Optional[float]] = {}
        for i, name in enumerate(RAW_CHECKPOINT_TARGETS):
            if i in ACTIVE_CHEXPERT_INDICES:
                raw_18_probs[name] = round(float(probs[i]), 4)
            else:
                raw_18_probs[name] = None

        inactive_slots = [RAW_CHECKPOINT_TARGETS[i] for i in INACTIVE_MASKED_INDICES]

        return {
            "target_probabilities": chexpert_5_probs,
            "op_calibrated_probabilities": op_calibrated_5_probs,
            "active_11_probabilities": active_11_probs,
            "raw_18_probabilities": raw_18_probs,
            "inactive_untrained_slots": inactive_slots,
            "model_metadata": {
                "backbone": "DenseNet-121",
                "checkpoint": str(self.weights_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "origin": "TorchXRayVision v1.5.5 (Cohen et al., MIDL 2022)",
                "training_dataset": "Stanford CheXpert Dataset (single-dataset 'chex' weights)",
                "license": "Apache License 2.0",
                "input_resolution": "224x224 grayscale (1 channel, square center-cropped)",
                "input_normalization": "[-1024.0, 1024.0]",
                "active_slot_count": len(ACTIVE_CHEXPERT_INDICES),
                "masked_untrained_slot_count": len(INACTIVE_MASKED_INDICES),
                "label_equivalence_note": (
                    "MIMIC-CXR 'Pleural Effusion' is mapped from TorchXRayVision slot 7 ('Effusion'), "
                    "which was trained directly on Stanford CheXpert 'Pleural Effusion' labels."
                ),
                "evaluation_status": "PRETRAINED_BASELINE_VERIFIED",
            },
        }


class PretrainedNihDenseNet121:
    """
    Standardized, reproducible 14-label NIH ChestX-ray14 DenseNet-121 adapter.

    Supports two audited checkpoint profiles:
    - 'nih14' (default): TorchXRayVision 'densenet121-res224-nih' ('models/nih-densenet121-nih14.pt').
      Trained on all 14 NIH ChestX-ray14 classes (slots 0..13 active; slots 14..17 masked).
      Note: Upstream TorchXRayVision trained 'nih' weights on its internal patient split (seed=0)
      of NIH ChestX-ray14, which may partially overlap NIH's official test_list.txt; documented
      explicitly in model_metadata and benchmark reports.
    - 'chex_external': TorchXRayVision 'densenet121-res224-chex' ('models/chex-densenet121-chexpert.pt').
      External out-of-domain baseline with 0% NIH training overlap; active on the 7 shared slots
      (Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Pneumonia, Pneumothorax) and
      explicitly masks the 7 untrained NIH-only slots as None (never emitting constant 0.5).

    Supports streaming batch preprocessing and batch inference (`preprocess_batch`, `predict_batch`)
    so full 112,120-image evaluations run with bounded constant RAM.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        checkpoint_profile: str = "nih14",
        device: Optional[str] = None,
        verify_sha256: bool = True,
        center_crop: bool = True,
    ):
        if checkpoint_profile not in {"nih14", "chex_external"}:
            raise ValueError(
                f"Unsupported checkpoint_profile='{checkpoint_profile}'. Must be 'nih14' or 'chex_external'."
            )
        self.checkpoint_profile = checkpoint_profile
        self.verify_sha256 = verify_sha256
        self.center_crop = center_crop

        if weights_path is None:
            self.weights_path = (
                Path("models/nih-densenet121-nih14.pt")
                if checkpoint_profile == "nih14"
                else Path("models/chex-densenet121-chexpert.pt")
            )
        else:
            self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            hint_url = (
                NIH_DENSENET121_RELEASE_URL
                if checkpoint_profile == "nih14"
                else "https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
            )
            raise FileNotFoundError(
                f"Verified pretrained checkpoint not found at: {self.weights_path}. "
                f"Download it first from: {hint_url}"
            )

        self.checkpoint_sha256 = PretrainedCheXpertDenseNet121._compute_sha256(self.weights_path)
        expected_sha = (
            EXPECTED_NIH_DENSENET121_SHA256
            if checkpoint_profile == "nih14"
            else EXPECTED_CHEX_DENSENET121_SHA256
        )
        if self.verify_sha256 and self.checkpoint_sha256 != expected_sha:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {self.weights_path} (profile={checkpoint_profile})! "
                f"Expected {expected_sha}, got {self.checkpoint_sha256}."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_profile == "nih14":
            self.active_slot_indices = list(ACTIVE_NIH_14_INDICES)
            self.masked_slot_indices = list(INACTIVE_NIH_4_INDICES)
            self.op_threshs = list(NIH_OP_THRESHS)
            self.active_nih_labels = list(NIH_14_LABELS)
            self.masked_nih_labels: List[str] = []
        else:
            self.active_slot_indices = list(ACTIVE_CHEXPERT_INDICES)
            self.masked_slot_indices = list(INACTIVE_MASKED_INDICES)
            self.op_threshs = list(CHEX_OP_THRESHS)
            self.active_nih_labels = [
                lbl for lbl in NIH_14_LABELS if NIH_14_MAPPING[lbl] in SHARED_CHEX_NIH_7_INDICES
            ]
            self.masked_nih_labels = [
                lbl for lbl in NIH_14_LABELS if NIH_14_MAPPING[lbl] in UNTRAINED_CHEX_ON_NIH_7_INDICES
            ]

        logger.info(
            "Loading pretrained NIH-14 DenseNet-121 (%s) on %s (SHA-256: %s...)",
            self.checkpoint_profile,
            self.device,
            self.checkpoint_sha256[:12],
        )
        self.model = self._load_verified_model()
        self.model.to(self.device)
        self.model.eval()

        self.target_mapping = dict(NIH_14_MAPPING)
        self.target_names = list(NIH_14_LABELS)

    def _load_verified_model(self) -> nn.Module:
        try:
            import torchxrayvision as xrv
            with torch.serialization.safe_globals([xrv.models.DenseNet]):
                model = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        except Exception:
            model = torch.load(self.weights_path, map_location=self.device, weights_only=False)

        for mod in model.modules():
            if not hasattr(mod, "_non_persistent_buffers_set"):
                mod._non_persistent_buffers_set = set()

        checkpoint_targets = getattr(model, "targets", None)
        if checkpoint_targets != RAW_CHECKPOINT_TARGETS:
            raise ValueError(
                f"Checkpoint targets mismatch! Expected {RAW_CHECKPOINT_TARGETS[:5]}..., got {checkpoint_targets}"
            )

        model.pathologies = (
            list(NIH_ACTIVE_SLOT_LABELS)
            if self.checkpoint_profile == "nih14"
            else list(CHEX_ACTIVE_SLOT_LABELS)
        )

        with torch.no_grad():
            weight_norms = model.classifier.weight.detach().cpu().norm(dim=1).numpy()
            for idx in self.active_slot_indices:
                if weight_norms[idx] <= 0.0:
                    raise ValueError(
                        f"Active slot {idx} ({RAW_CHECKPOINT_TARGETS[idx]}) has zero weight norm!"
                    )
            for idx in self.masked_slot_indices:
                if weight_norms[idx] > 1e-6:
                    raise ValueError(
                        f"Expected masked slot {idx} ({RAW_CHECKPOINT_TARGETS[idx]}) to have zero weights, "
                        f"got norm={weight_norms[idx]}"
                    )

        return model

    def _preprocess_single_cpu(self, image_input: Union[bytes, str, Path, Image.Image]) -> torch.Tensor:
        """Preprocesses one image on CPU to shape (1, 224, 224) float32 in [-1024.0, 1024.0]."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as raw_img:
                img = raw_img.convert("L")
        elif isinstance(image_input, (bytes, bytearray)):
            with Image.open(io.BytesIO(image_input)) as raw_img:
                img = raw_img.convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_arr = np.array(img, dtype=np.float32)
        if img_arr.ndim != 2:
            raise ValueError(f"Expected 2D grayscale array after conversion, got shape {img_arr.shape}")

        norm_arr = (2.0 * (img_arr / 255.0) - 1.0) * 1024.0
        if self.center_crop:
            h, w = norm_arr.shape
            crop_size = min(h, w)
            start_x = w // 2 - (crop_size // 2)
            start_y = h // 2 - (crop_size // 2)
            norm_arr = norm_arr[start_y : start_y + crop_size, start_x : start_x + crop_size]

        tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0)
        if tensor.shape[2] != 224 or tensor.shape[3] != 224:
            tensor = F.interpolate(
                tensor,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return tensor.squeeze(0)  # (1, 224, 224)

    def preprocess_image(self, image_input: Union[bytes, str, Path, Image.Image]) -> torch.Tensor:
        """Returns preprocessed tensor of shape (1, 1, 224, 224) on self.device."""
        return self._preprocess_single_cpu(image_input).unsqueeze(0).to(self.device)

    def preprocess_batch(
        self, image_inputs: Sequence[Union[bytes, str, Path, Image.Image]]
    ) -> torch.Tensor:
        """
        Preprocesses a mini-batch of images into a contiguous tensor of shape (B, 1, 224, 224)
        on self.device without retaining raw image buffers in memory.
        """
        if len(image_inputs) == 0:
            raise ValueError("image_inputs batch cannot be empty")
        tensors = [self._preprocess_single_cpu(item) for item in image_inputs]
        return torch.stack(tensors, dim=0).to(self.device)

    def get_source_op_thresh(self, label_name: str) -> Optional[float]:
        """Returns published TorchXRayVision operating point threshold for an NIH-14 label (metadata only)."""
        if label_name not in self.target_mapping:
            raise KeyError(f"Unknown NIH-14 label: {label_name}")
        idx = self.target_mapping[label_name]
        if idx not in self.active_slot_indices:
            return None
        val = self.op_threshs[idx]
        return None if np.isnan(val) else round(float(val), 6)

    def get_slot_audit_table(self) -> List[Dict[str, Any]]:
        """Returns 18-slot classifier head verification table for the loaded checkpoint."""
        with torch.no_grad():
            w_norms = self.model.classifier.weight.detach().cpu().norm(dim=1).numpy()
            biases = self.model.classifier.bias.detach().cpu().numpy()

        inv_nih = {v: k for k, v in NIH_14_MAPPING.items()}
        table = []
        for i, raw_name in enumerate(RAW_CHECKPOINT_TARGETS):
            is_active = i in self.active_slot_indices
            nih_label = inv_nih.get(i)
            table.append({
                "slot_index": i,
                "raw_target_name": raw_name,
                "nih_14_label": nih_label,
                "is_in_nih_14": nih_label is not None,
                "is_trained_in_checkpoint": is_active,
                "is_evaluatable_on_nih_14": bool(is_active and nih_label is not None),
                "weight_l2_norm": round(float(w_norms[i]), 4),
                "bias": round(float(biases[i]), 6),
                "op_thresh": None if np.isnan(self.op_threshs[i]) else round(float(self.op_threshs[i]), 6),
            })
        return table

    def get_model_metadata(self) -> Dict[str, Any]:
        if self.checkpoint_profile == "nih14":
            training_dataset = "NIH ChestX-ray14 (TorchXRayVision 'densenet121-res224-nih' single-dataset weights)"
            overlap_disclosure = (
                "Upstream TorchXRayVision trained 'densenet121-res224-nih' on its internal patient-grouped "
                "split (seed=0) of NIH ChestX-ray14, which may partially overlap NIH's official test_list.txt. "
                "For zero-overlap out-of-domain evaluation on the 7 shared classes, run --checkpoint-profile chex_external."
            )
        else:
            training_dataset = "Stanford CheXpert Dataset (TorchXRayVision 'densenet121-res224-chex' external weights)"
            overlap_disclosure = (
                "Zero NIH ChestX-ray14 training overlap (100% external out-of-domain baseline). "
                "Evaluates the 7 shared active classes only; 7 NIH-only classes are masked as None."
            )
        return {
            "backbone": "DenseNet-121",
            "checkpoint_profile": self.checkpoint_profile,
            "checkpoint": str(self.weights_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "origin": "TorchXRayVision v1.5.5 (Cohen et al., MIDL 2022)",
            "training_dataset": training_dataset,
            "training_overlap_disclosure": overlap_disclosure,
            "license": "Apache License 2.0",
            "input_resolution": "224x224 grayscale (1 channel, square center-cropped)",
            "input_normalization": "[-1024.0, 1024.0]",
            "nih_14_active_labels": list(self.active_nih_labels),
            "nih_14_masked_labels": list(self.masked_nih_labels),
            "active_slot_count": len(self.active_slot_indices),
            "masked_untrained_slot_count": len(self.masked_slot_indices),
            "label_governance_note": (
                "NIH ChestX-ray14 labels are NLP-mined binary {0, 1} labels from radiological reports "
                "(Wang et al., CVPR 2017) and are NOT equivalent to CheXpert labels or radiologist consensus ground truth."
            ),
        }

    def predict_batch(
        self, image_inputs: Sequence[Union[bytes, str, Path, Image.Image]]
    ) -> List[Dict[str, Any]]:
        """
        Runs batched forward pass on `image_inputs` and returns per-image NIH-14 probability dictionaries.
        """
        batch_tensor = self.preprocess_batch(image_inputs)
        with torch.no_grad():
            features = self.model.features2(batch_tensor)
            logits = self.model.classifier(features)
            probs_batch = torch.sigmoid(logits).cpu().numpy()

        if np.isnan(probs_batch).any() or np.isinf(probs_batch).any():
            raise ValueError("Model produced NaN or Inf probabilities in batch inference")
        probs_batch = np.clip(probs_batch, 0.0, 1.0)

        results: List[Dict[str, Any]] = []
        for row_idx in range(probs_batch.shape[0]):
            probs = probs_batch[row_idx]
            nih_14_probs: Dict[str, Optional[float]] = {}
            op_calibrated_probs: Dict[str, Optional[float]] = {}

            for label in NIH_14_LABELS:
                slot_idx = self.target_mapping[label]
                if slot_idx in self.active_slot_indices:
                    nih_14_probs[label] = round(float(probs[slot_idx]), 6)
                    op_calibrated_probs[label] = PretrainedCheXpertDenseNet121._apply_op_norm(
                        float(probs[slot_idx]), self.op_threshs[slot_idx]
                    )
                else:
                    nih_14_probs[label] = None
                    op_calibrated_probs[label] = None

            raw_18_probs: Dict[str, Optional[float]] = {
                name: (round(float(probs[i]), 6) if i in self.active_slot_indices else None)
                for i, name in enumerate(RAW_CHECKPOINT_TARGETS)
            }

            results.append({
                "nih_14_probabilities": nih_14_probs,
                "target_probabilities": nih_14_probs,
                "op_calibrated_probabilities": op_calibrated_probs,
                "raw_18_probabilities": raw_18_probs,
            })

        return results

    def extract_features(self, image_input: Union[bytes, str, Path, Image.Image]) -> torch.Tensor:
        """
        Extracts the 1024-dimensional pooled visual representation F_v from DenseNet-121
        prior to the 18-slot classification head. Shape: (1, 1024).
        """
        tensor = self.preprocess_image(image_input)
        with torch.no_grad():
            features = self.model.features2(tensor)
        return features

    def predict(self, image_input: Union[bytes, str, Path, Image.Image]) -> Dict[str, Any]:
        """Single-image convenience wrapper around `predict_batch`."""
        res = self.predict_batch([image_input])[0]
        res["inactive_untrained_slots"] = [RAW_CHECKPOINT_TARGETS[i] for i in self.masked_slot_indices]
        res["model_metadata"] = self.get_model_metadata()
        return res
