"""
MedFusion-CXR Reproducible Baseline Evaluation Module.

Provides decoupled, reproducible statistical evaluation for medical imaging and multimodal models:
1. Validated input schema with strict study-level and patient-level integrity checks.
2. Explicit CheXpert uncertainty policy handling (U-Ignore, U-Zeros, U-Ones).
3. Primary metrics: AUROC, AUPRC, F1 (threshold selected on validation split only), Brier score, and ECE.
4. Patient-clustered bootstrap confidence intervals (grouped by subject_id).
5. Strict multimodal linkage and synthetic data provenance guardrails.
6. Machine-readable JSON report serialization.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

EVALUATION_SCRIPT_VERSION = "1.4.0"
AUPRC_EXACT_DEFINITION = (
    "Average Precision (AP) computed via step-wise rectangular precision-recall integration: "
    "AP = sum_n (R_n - R_{n-1}) * P_n (sklearn.metrics.average_precision_score; "
    "no linear/trapezoidal interpolation)."
)

# Canonical 14 binary pathology classes in NIH ChestX-ray14 (Wang et al., CVPR 2017).
# Note: 'No Finding' in Data_Entry_2017.csv is represented as all 14 labels == 0, NOT a 15th disease class.
NIH_LABELS: List[str] = [
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


def compute_file_sha256(filepath: Optional[Union[str, Path]]) -> Optional[str]:
    """Computes SHA-256 hex digest of a file on disk, or None if unavailable."""
    if filepath is None:
        return None
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_package_version(dist_name: str, module_name: Optional[str] = None) -> Optional[str]:
    """Retrieves installed package version without fabricating values if absent."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        if module_name:
            try:
                mod = __import__(module_name)
                return getattr(mod, "__version__", None)
            except Exception:
                return None
        return None


def _detect_git_revision(repo_dir: Optional[Path] = None) -> Dict[str, Optional[Union[str, bool]]]:
    """Queries git commit SHA and working-tree status if available; returns None fields otherwise."""
    cwd = str(repo_dir) if repo_dir else None
    commit_sha: Optional[str] = None
    is_dirty: Optional[bool] = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            commit_sha = res.stdout.strip()
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if status_res.returncode == 0:
                is_dirty = bool(status_res.stdout.strip())
    except Exception:
        commit_sha = None
        is_dirty = None
    return {"git_commit": commit_sha, "git_working_tree_dirty": is_dirty}


def collect_reproducibility_metadata(
    *,
    random_seed: int,
    uncertainty_policy: str,
    execution_mode: str = "standalone_evaluator",
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_sha256: Optional[str] = None,
    input_manifest_path: Optional[Union[str, Path]] = None,
    input_manifest_sha256: Optional[str] = None,
    bootstrap_config: Optional[Dict[str, Any]] = None,
    threshold_provenance: Optional[Dict[str, Any]] = None,
    target_label_definitions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collects verifiable runtime, environment, cryptographic, and methodological metadata.
    Never fabricates unavailable fields (uses None explicitly).
    """
    resolved_ckpt_sha = checkpoint_sha256 or compute_file_sha256(checkpoint_path)
    resolved_manifest_sha = input_manifest_sha256 or compute_file_sha256(input_manifest_path)
    git_info = _detect_git_revision()

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
        "evaluation_script_version": EVALUATION_SCRIPT_VERSION,
        "git_commit": git_info["git_commit"],
        "git_working_tree_dirty": git_info["git_working_tree_dirty"],
        "random_seed": random_seed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": resolved_ckpt_sha,
        "input_manifest_path": str(input_manifest_path) if input_manifest_path else None,
        "input_manifest_sha256": resolved_manifest_sha,
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": _safe_package_version("torch", "torch"),
            "torchxrayvision": _safe_package_version("torchxrayvision", "torchxrayvision"),
            "pillow": _safe_package_version("Pillow", "PIL"),
            "numpy": _safe_package_version("numpy", "numpy"),
            "pandas": _safe_package_version("pandas", "pandas"),
            "scikit_learn": _safe_package_version("scikit-learn", "sklearn"),
        },
        "uncertainty_policy": uncertainty_policy,
        "target_label_definitions": target_label_definitions,
        "threshold_provenance": threshold_provenance,
        "bootstrap_configuration": bootstrap_config,
    }


class UncertaintyPolicy(str, Enum):
    """CheXpert label uncertainty (-1) resolution policy (not applicable to strict binary datasets like NIH)."""
    U_IGNORE = "u_ignore"  # Drop all samples with label == -1
    U_ZEROS = "u_zeros"    # Map -1 to 0 (assume negative)
    U_ONES = "u_ones"      # Map -1 to 1 (assume positive)


class LabelDomainPolicy(str, Enum):
    """Ground-truth label domain enforcement policy."""
    CHEXPERT_UNCERTAINTY = "chexpert_uncertainty"  # Permits {0, 1, -1, NaN} and applies UncertaintyPolicy
    BINARY_STRICT = "binary_strict"                # Strictly {0, 1, NaN}; rejects -1 uncertainty values (NIH ChestX-ray14)


class ModalityType(str, Enum):
    CXR_ONLY = "cxr_only"
    CLINICAL_ONLY = "clinical_only"
    MULTIMODAL = "multimodal"


class PatientLeakageError(ValueError):
    """Raised when the same patient (subject_id) appears across distinct evaluation splits."""
    pass


class DuplicateRecordError(ValueError):
    """Raised when duplicate study_id records exist within the same split."""
    pass


class UnverifiedLinkageError(ValueError):
    """Raised when multimodal evaluation is requested on unverified or synthetic image-clinical pairings."""
    pass


class SyntheticDataQuarantineError(ValueError):
    """Raised when synthetic test fixtures are detected in clinical benchmark evaluation."""
    pass


class InvalidLabelError(ValueError):
    """Raised when ground truth labels contain values outside {0, 1, -1, NaN}."""
    pass


class MixedLabelSourceError(ValueError):
    """Raised when evaluation data contains multiple inconsistent label sources."""
    pass


class InsufficientValidationDiversityError(ValueError):
    """Raised when validation split lacks both positive (1) and negative (0) samples required for threshold selection."""
    pass


@dataclass
class EvaluationRecord:
    """Standardized prediction record representing one evaluation unit (image_id / study_id)."""
    subject_id: Optional[str] = None
    study_id: Optional[str] = None
    y_true: Union[int, float] = 0
    split: str = "test"  # 'train', 'val' / 'validate', 'test'
    p_cxr: Optional[float] = None
    p_clinical: Optional[float] = None
    p_multimodal: Optional[float] = None
    label_source: str = "NIH_CHESTXRAY14_NLP_2017"
    linkage_verified: bool = False
    is_synthetic: bool = False
    patient_id: Optional[str] = None
    image_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.subject_id is None and self.patient_id is not None:
            self.subject_id = self.patient_id
        if self.patient_id is None and self.subject_id is not None:
            self.patient_id = self.subject_id
        if self.study_id is None and self.image_id is not None:
            self.study_id = self.image_id
        if self.image_id is None and self.study_id is not None:
            self.image_id = self.study_id

    def validate(self, label_domain_policy: LabelDomainPolicy = LabelDomainPolicy.CHEXPERT_UNCERTAINTY) -> None:
        if self.subject_id is None or pd.isna(self.subject_id) or not str(self.subject_id).strip():
            raise ValueError("subject_id cannot be null or empty (patient_id/subject_id required)")
        if self.study_id is None or pd.isna(self.study_id) or not str(self.study_id).strip():
            raise ValueError("study_id cannot be null or empty (image_id/study_id required)")
        
        valid_splits = {"train", "val", "validate", "test"}
        if self.split is None or pd.isna(self.split) or str(self.split).lower() not in valid_splits:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of {valid_splits}")

        # Validate label domain
        if self.y_true is not None and not pd.isna(self.y_true):
            if label_domain_policy == LabelDomainPolicy.BINARY_STRICT:
                if self.y_true not in {0, 1, 0.0, 1.0}:
                    raise InvalidLabelError(
                        f"Invalid y_true={self.y_true} under BINARY_STRICT policy. "
                        "Permitted values are strictly {0, 1} or NaN (-1 uncertainty is forbidden)."
                    )
            else:
                if self.y_true not in {0, 1, -1, 0.0, 1.0, -1.0}:
                    raise InvalidLabelError(f"Invalid y_true={self.y_true}. Permitted values are {0, 1, -1} or NaN.")

        for mod_name, prob in [
            ("p_cxr", self.p_cxr),
            ("p_clinical", self.p_clinical),
            ("p_multimodal", self.p_multimodal),
        ]:
            if prob is not None:
                if pd.isna(prob):
                    raise ValueError(f"Probability {mod_name} cannot be NaN")
                if not (0.0 <= float(prob) <= 1.0):
                    raise ValueError(f"Probability {mod_name}={prob} is out of range [0.0, 1.0]")


@dataclass
class CalibrationBin:
    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_error: float


@dataclass
class MetricSummary:
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    f1: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    brier_score: Optional[float] = None
    ece: Optional[float] = None
    threshold: float = 0.5
    sample_count: int = 0
    pos_count: int = 0
    neg_count: int = 0
    auroc_ci: Optional[Tuple[float, float]] = None
    auprc_ci: Optional[Tuple[float, float]] = None
    f1_ci: Optional[Tuple[float, float]] = None
    sensitivity_ci: Optional[Tuple[float, float]] = None
    specificity_ci: Optional[Tuple[float, float]] = None
    brier_ci: Optional[Tuple[float, float]] = None
    ece_ci: Optional[Tuple[float, float]] = None
    calibration_bins: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ExpectedCalibrationErrorCalculator:
    """
    Computes Expected Calibration Error (ECE) using standard equal-width binning.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """

    @staticmethod
    def compute_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_bins: int = 10,
    ) -> Tuple[float, List[CalibrationBin]]:
        if len(y_true) == 0:
            return 0.0, []

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        total_samples = len(y_true)
        bins_data: List[CalibrationBin] = []

        for i in range(num_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            # In the final bin, include right edge <= 1.0
            if i == num_bins - 1:
                in_bin = (y_prob >= lower) & (y_prob <= upper)
            else:
                in_bin = (y_prob >= lower) & (y_prob < upper)

            bin_count = int(np.sum(in_bin))
            if bin_count > 0:
                bin_acc = float(np.mean(y_true[in_bin]))
                bin_conf = float(np.mean(y_prob[in_bin]))
                cal_err = abs(bin_acc - bin_conf)
                ece += (bin_count / total_samples) * cal_err
            else:
                bin_acc = 0.0
                bin_conf = (lower + upper) / 2.0
                cal_err = 0.0

            bins_data.append(
                CalibrationBin(
                    bin_index=i,
                    bin_lower=round(float(lower), 4),
                    bin_upper=round(float(upper), 4),
                    sample_count=bin_count,
                    mean_confidence=round(bin_conf, 4),
                    empirical_accuracy=round(bin_acc, 4),
                    calibration_error=round(cal_err, 4),
                )
            )

        return round(float(ece), 4), bins_data


class BenchmarkFusionEvaluator:
    """
    Independent baseline evaluation engine for MedFusion-CXR.
    Evaluates CXR-only, Clinical-only, and Multimodal prediction tables.
    """

    def __init__(
        self,
        uncertainty_policy: UncertaintyPolicy = UncertaintyPolicy.U_IGNORE,
        bootstrap_iterations: int = 1000,
        random_seed: int = 42,
        num_calibration_bins: int = 10,
        benchmark_version: str = "1.0.0",
        strict_patient_split: bool = True,
        label_domain_policy: LabelDomainPolicy = LabelDomainPolicy.CHEXPERT_UNCERTAINTY,
    ):
        self.uncertainty_policy = uncertainty_policy
        self.bootstrap_iterations = bootstrap_iterations
        self.random_seed = random_seed
        self.num_calibration_bins = num_calibration_bins
        self.benchmark_version = benchmark_version
        self.strict_patient_split = strict_patient_split
        self.label_domain_policy = label_domain_policy

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates schema, checks for duplicates, patient overlap across splits,
        invalid probabilities, and null/invalid identifiers.
        Supports both NIH schema (patient_id, image_id) and legacy schema (subject_id, study_id).
        """
        has_patient_id = "patient_id" in df.columns
        has_subject_id = "subject_id" in df.columns
        has_image_id = "image_id" in df.columns
        has_study_id = "study_id" in df.columns

        missing_base = {"y_true", "split"} - set(df.columns)
        missing_patient = not (has_patient_id or has_subject_id)
        missing_record = not (has_image_id or has_study_id)
        if missing_base or missing_patient or missing_record:
            missing_desc = set(missing_base)
            if missing_patient:
                missing_desc.add("patient_id/subject_id")
            if missing_record:
                missing_desc.add("image_id/study_id")
            raise ValueError(f"Input DataFrame missing required columns: {missing_desc}")

        df = df.copy()
        patient_col = "patient_id" if has_patient_id else "subject_id"
        record_col = "image_id" if has_image_id else "study_id"

        # Check nulls BEFORE converting to string (prevents "nan" bypass)
        for col_to_check in ([c for c in ("patient_id", "subject_id") if c in df.columns]):
            if df[col_to_check].isna().any() or (
                df[col_to_check].astype(str).str.strip().str.lower().isin(["", "nan", "none"])
            ).any():
                raise ValueError(f"Found null, empty, or NaN {col_to_check}")

        for col_to_check in ([c for c in ("image_id", "study_id") if c in df.columns]):
            if df[col_to_check].isna().any() or (
                df[col_to_check].astype(str).str.strip().str.lower().isin(["", "nan", "none"])
            ).any():
                raise ValueError(f"Found null, empty, or NaN {col_to_check}")

        if has_patient_id and has_subject_id:
            if (df["patient_id"].astype(str) != df["subject_id"].astype(str)).any():
                raise ValueError("Mismatched patient_id and subject_id values in input DataFrame")
        elif has_patient_id:
            df["subject_id"] = df["patient_id"]
        else:
            df["patient_id"] = df["subject_id"]

        if has_image_id and has_study_id:
            if (df["image_id"].astype(str) != df["study_id"].astype(str)).any():
                raise ValueError("Mismatched image_id and study_id values in input DataFrame")
        elif has_image_id:
            df["study_id"] = df["image_id"]
        else:
            df["image_id"] = df["study_id"]

        df.attrs["resampling_unit"] = patient_col
        df.attrs["record_unit"] = record_col

        if df["split"].isna().any():
            raise ValueError("Found null or missing values in 'split' column")

        df["split"] = df["split"].astype(str).str.lower().replace({"val": "validate"})
        valid_splits = {"train", "validate", "test"}
        invalid_splits = set(df["split"].unique()) - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid split values found: {invalid_splits}. Must be one of {valid_splits}")

        # 1. Check duplicate image_id / study_id within the same split
        duplicate_studies = df[df.duplicated(subset=["split", "study_id"], keep=False)]
        if not duplicate_studies.empty:
            dup_ids = duplicate_studies["study_id"].unique()[:5].tolist()
            raise DuplicateRecordError(
                f"Found {len(duplicate_studies)} duplicate records with same {record_col} within split: {dup_ids}"
            )

        # 2. Check patient overlap across splits (Patient Leakage Guard)
        train_subjects = set(df[df["split"] == "train"]["subject_id"].astype(str).unique())
        val_subjects = set(df[df["split"] == "validate"]["subject_id"].astype(str).unique())
        test_subjects = set(df[df["split"] == "test"]["subject_id"].astype(str).unique())

        train_test_overlap = train_subjects.intersection(test_subjects)
        val_test_overlap = val_subjects.intersection(test_subjects)
        train_val_overlap = train_subjects.intersection(val_subjects)

        if train_test_overlap:
            raise PatientLeakageError(
                f"CRITICAL PATIENT LEAKAGE: {len(train_test_overlap)} patients ({patient_col}) overlap between train and test splits! "
                f"Sample overlapping IDs: {list(train_test_overlap)[:3]}"
            )
        if val_test_overlap:
            raise PatientLeakageError(
                f"CRITICAL PATIENT LEAKAGE: {len(val_test_overlap)} patients ({patient_col}) overlap between validate and test splits! "
                f"Sample overlapping IDs: {list(val_test_overlap)[:3]}"
            )
        if train_val_overlap:
            if self.strict_patient_split:
                raise PatientLeakageError(
                    f"CRITICAL PATIENT LEAKAGE: {len(train_val_overlap)} patients ({patient_col}) overlap between train and validate splits! "
                    f"Sample overlapping IDs: {list(train_val_overlap)[:3]}"
                )
            else:
                logger.warning(
                    "Patient overlap between train and validate splits detected: %d patients.",
                    len(train_val_overlap),
                )

        # Validate probabilities if columns exist
        for col in ["p_cxr", "p_clinical", "p_multimodal"]:
            if col in df.columns:
                valid_probs = df[col].dropna()
                if not valid_probs.empty:
                    if (valid_probs < 0.0).any() or (valid_probs > 1.0).any():
                        raise ValueError(f"Column '{col}' has probabilities outside [0.0, 1.0]")

        return df

    def apply_uncertainty_policy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validates y_true domain under configured LabelDomainPolicy:
        - BINARY_STRICT (NIH ChestX-ray14): strictly {0, 1, NaN}; explicitly rejects -1 uncertainty values.
        - CHEXPERT_UNCERTAINTY: permits {0, 1, -1, NaN} and applies UncertaintyPolicy.
        """
        df_attrs = dict(df.attrs)
        df = df.copy()
        df.attrs.update(df_attrs)

        non_null_labels = df["y_true"].dropna()
        if self.label_domain_policy == LabelDomainPolicy.BINARY_STRICT:
            invalid_mask = ~non_null_labels.isin([0, 1, 0.0, 1.0])
            if invalid_mask.any():
                invalid_values = non_null_labels[invalid_mask].unique()[:5].tolist()
                raise InvalidLabelError(
                    f"Strict binary label domain violated: {invalid_values}. "
                    "NIH ChestX-ray14 uses binary {0, 1} labels only; CheXpert -1 uncertainty values are forbidden."
                )
        else:
            invalid_mask = ~non_null_labels.isin([0, 1, -1, 0.0, 1.0, -1.0])
            if invalid_mask.any():
                invalid_values = non_null_labels[invalid_mask].unique()[:5].tolist()
                raise InvalidLabelError(
                    f"Invalid ground truth labels detected: {invalid_values}. "
                    "Permitted values are strictly 0, 1, -1, or NaN."
                )

        stats = {
            "initial_samples": len(df),
            "positive_count": int((df["y_true"] == 1).sum()),
            "negative_count": int((df["y_true"] == 0).sum()),
            "uncertain_count": int((df["y_true"] == -1).sum()),
            "missing_count": int(df["y_true"].isna().sum()),
        }

        # Drop missing NaN labels
        df = df.dropna(subset=["y_true"])

        if self.label_domain_policy == LabelDomainPolicy.CHEXPERT_UNCERTAINTY:
            if self.uncertainty_policy == UncertaintyPolicy.U_IGNORE:
                df = df[df["y_true"] != -1]
            elif self.uncertainty_policy == UncertaintyPolicy.U_ZEROS:
                df["y_true"] = df["y_true"].replace(-1, 0)
            elif self.uncertainty_policy == UncertaintyPolicy.U_ONES:
                df["y_true"] = df["y_true"].replace(-1, 1)

        df["y_true"] = df["y_true"].astype(int)
        df.attrs.update(df_attrs)
        stats["evaluated_samples"] = len(df)
        return df, stats

    def find_optimal_threshold(
        self,
        y_val: np.ndarray,
        p_val: np.ndarray,
        metric: str = "f1",
        strict: bool = False,
    ) -> float:
        """
        Finds optimal decision threshold on VALIDATION set ONLY.
        Maximizes F1-score or Youden's J statistic over a deterministic 91-point grid [0.05, 0.95].
        Never inspects test labels.
        """
        if metric not in {"f1", "youden"}:
            raise ValueError(f"Unsupported threshold selection metric '{metric}'. Must be 'f1' or 'youden'.")

        if len(y_val) == 0 or len(np.unique(y_val)) < 2:
            if strict:
                pos_n = int(np.sum(y_val == 1)) if len(y_val) > 0 else 0
                neg_n = int(np.sum(y_val == 0)) if len(y_val) > 0 else 0
                raise InsufficientValidationDiversityError(
                    f"Validation split lacks class diversity for threshold optimization: "
                    f"total={len(y_val)}, pos={pos_n}, neg={neg_n}. Both class 0 and class 1 are required."
                )
            return 0.5

        thresholds = np.linspace(0.05, 0.95, 91)
        best_threshold = 0.5
        best_score = -1.0

        for t in thresholds:
            y_pred = (p_val >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_val, y_pred, zero_division=0)
            else:  # metric == "youden"
                tp = np.sum((y_pred == 1) & (y_val == 1))
                fn = np.sum((y_pred == 0) & (y_val == 1))
                fp = np.sum((y_pred == 1) & (y_val == 0))
                tn = np.sum((y_pred == 0) & (y_val == 0))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                score = tpr - fpr

            if score > best_score:
                best_score = score
                best_threshold = float(t)

        return round(best_threshold, 4)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> MetricSummary:
        """
        Computes AUROC, AUPRC (Average Precision), F1, Brier score, and ECE for single evaluation split.
        Handles single-class targets gracefully while logging explicit evaluation limits.
        """
        summary = MetricSummary()
        summary.sample_count = len(y_true)
        summary.threshold = threshold
        summary.pos_count = int(np.sum(y_true == 1))
        summary.neg_count = int(np.sum(y_true == 0))

        if len(y_true) == 0:
            summary.warnings.append("Empty dataset provided; cannot compute metrics.")
            return summary

        # Check single-class target edge cases
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            summary.warnings.append(
                f"LIMITATION: Single-class target detected (only class {unique_classes[0]} present). "
                "AUROC and AUPRC (Average Precision) are mathematically undefined and set to None. "
                "Brier score and ECE remain valid."
            )
            y_pred = (y_prob >= threshold).astype(int)
            summary.f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
            if summary.pos_count > 0:
                tp_sc = int(np.sum((y_pred == 1) & (y_true == 1)))
                summary.sensitivity = round(float(tp_sc / summary.pos_count), 4)
            if summary.neg_count > 0:
                tn_sc = int(np.sum((y_pred == 0) & (y_true == 0)))
                summary.specificity = round(float(tn_sc / summary.neg_count), 4)
            summary.brier_score = round(float(np.mean((y_prob - y_true) ** 2)), 4)
            ece, cal_bins = ExpectedCalibrationErrorCalculator.compute_ece(
                y_true, y_prob, self.num_calibration_bins
            )
            summary.ece = ece
            summary.calibration_bins = [asdict(b) for b in cal_bins]
            return summary

        # AUROC
        try:
            summary.auroc = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception as e:
            summary.warnings.append(f"AUROC calculation error: {e}")

        # AUPRC: Exact Average Precision (step-wise precision-recall integral, non-trapezoidal)
        try:
            summary.auprc = round(float(average_precision_score(y_true, y_prob)), 4)
        except Exception as e:
            summary.warnings.append(f"AUPRC (Average Precision) calculation error: {e}")

        # F1, Sensitivity, and Specificity at frozen validation threshold
        y_pred = (y_prob >= threshold).astype(int)
        summary.f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        summary.sensitivity = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else None
        summary.specificity = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else None

        # Brier Score
        summary.brier_score = round(float(np.mean((y_prob - y_true) ** 2)), 4)

        # ECE & Calibration Bins
        ece, cal_bins = ExpectedCalibrationErrorCalculator.compute_ece(
            y_true, y_prob, self.num_calibration_bins
        )
        summary.ece = ece
        summary.calibration_bins = [asdict(b) for b in cal_bins]

        return summary

    def compute_clustered_bootstrap_ci(
        self,
        df_split: pd.DataFrame,
        prob_col: str,
        threshold: float,
        alpha: float = 0.05,
        min_valid_iterations: int = 2,
        resampling_unit: Optional[str] = None,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
        """
        Computes (1 - alpha) Confidence Intervals using patient-clustered bootstrap resampling.
        - Resamples strictly at the patient (patient_id / subject_id) level with replacement.
        - Skips single-class resamples safely without computing undefined AUROC/AUPRC.
        - Records requested, successful, and skipped iterations plus failure_rate.
        - Computes percentile confidence intervals exclusively from valid dual-class iterations.
        Returns: (ci_results, bootstrap_diagnostics)
        """
        resolved_unit = resampling_unit or df_split.attrs.get("resampling_unit", "subject_id")
        subjects = (
            np.array(sorted(df_split["subject_id"].astype(str).unique()))
            if len(df_split) > 0
            else np.array([], dtype=str)
        )
        n_subjects = int(len(subjects))
        boot_warnings: List[str] = []

        diagnostics: Dict[str, Any] = {
            "resampling_unit": resolved_unit,
            "n_unique_subjects": n_subjects,
            "n_unique_patients": n_subjects,
            "requested_iterations": int(self.bootstrap_iterations),
            "successful_iterations": 0,
            "skipped_iterations": 0,
            "single_class_skipped_iterations": 0,
            "metric_error_skipped_iterations": 0,
            "failure_rate": 0.0,
            "confidence_level": round(float(1.0 - alpha), 4),
            "alpha": float(alpha),
            "ci_method": "percentile_from_valid_iterations_only",
            "min_valid_iterations_for_ci": int(min_valid_iterations),
            "auprc_definition": AUPRC_EXACT_DEFINITION,
            "warnings": boot_warnings,
        }

        if self.bootstrap_iterations <= 0:
            return {}, diagnostics

        if n_subjects == 0 or len(df_split) == 0:
            diagnostics["skipped_iterations"] = int(self.bootstrap_iterations)
            diagnostics["failure_rate"] = 1.0
            boot_warnings.append("Bootstrap skipped: evaluation split is empty (0 subjects).")
            return {}, diagnostics

        rng = np.random.RandomState(self.random_seed)
        boot_metrics: Dict[str, List[float]] = {
            "auroc": [],
            "auprc": [],
            "f1": [],
            "sensitivity": [],
            "specificity": [],
            "brier": [],
            "ece": [],
        }

        # Pre-group dataframe by string subject_id in deterministic order
        df_grouped = df_split.copy()
        df_grouped["_subject_key"] = df_grouped["subject_id"].astype(str)
        subject_groups = {sub: group for sub, group in df_grouped.groupby("_subject_key", sort=True)}

        for _ in range(self.bootstrap_iterations):
            sampled_subs = rng.choice(subjects, size=n_subjects, replace=True)
            resampled_dfs = [subject_groups[s] for s in sampled_subs]
            boot_df = pd.concat(resampled_dfs, ignore_index=True)

            y_b = boot_df["y_true"].to_numpy()
            p_b = boot_df[prob_col].to_numpy()

            # Single-class guard: AUROC and AUPRC are undefined when only 1 class is sampled
            if len(np.unique(y_b)) < 2:
                diagnostics["single_class_skipped_iterations"] += 1
                continue

            try:
                auroc_val = float(roc_auc_score(y_b, p_b))
                auprc_val = float(average_precision_score(y_b, p_b))
                y_pred = (p_b >= threshold).astype(int)
                f1_val = float(f1_score(y_b, y_pred, zero_division=0))
                tp_b = int(np.sum((y_pred == 1) & (y_b == 1)))
                fn_b = int(np.sum((y_pred == 0) & (y_b == 1)))
                tn_b = int(np.sum((y_pred == 0) & (y_b == 0)))
                fp_b = int(np.sum((y_pred == 1) & (y_b == 0)))
                sens_val = float(tp_b / (tp_b + fn_b))
                spec_val = float(tn_b / (tn_b + fp_b))
                brier_val = float(np.mean((p_b - y_b) ** 2))
                ece_val, _ = ExpectedCalibrationErrorCalculator.compute_ece(
                    y_b, p_b, num_bins=self.num_calibration_bins
                )
            except Exception as exc:
                diagnostics["metric_error_skipped_iterations"] += 1
                logger.warning("Bootstrap iteration skipped due to metric error: %s", exc)
                continue

            boot_metrics["auroc"].append(auroc_val)
            boot_metrics["auprc"].append(auprc_val)
            boot_metrics["f1"].append(f1_val)
            boot_metrics["sensitivity"].append(sens_val)
            boot_metrics["specificity"].append(spec_val)
            boot_metrics["brier"].append(brier_val)
            boot_metrics["ece"].append(ece_val)

        successful = len(boot_metrics["auroc"])
        skipped = (
            diagnostics["single_class_skipped_iterations"]
            + diagnostics["metric_error_skipped_iterations"]
        )
        diagnostics["successful_iterations"] = int(successful)
        diagnostics["skipped_iterations"] = int(skipped)
        diagnostics["failure_rate"] = round(
            float(skipped / max(1, self.bootstrap_iterations)), 4
        )

        if skipped > 0:
            boot_warnings.append(
                f"{skipped}/{self.bootstrap_iterations} bootstrap iterations skipped "
                f"(single_class={diagnostics['single_class_skipped_iterations']}, "
                f"metric_error={diagnostics['metric_error_skipped_iterations']}; "
                f"failure_rate={diagnostics['failure_rate']:.4f})."
            )

        lower_pct = 100.0 * (alpha / 2.0)
        upper_pct = 100.0 * (1.0 - alpha / 2.0)

        ci_results: Dict[str, Tuple[float, float]] = {}
        if successful >= min_valid_iterations:
            for metric_name, values in boot_metrics.items():
                if len(values) >= min_valid_iterations:
                    low = np.percentile(values, lower_pct)
                    high = np.percentile(values, upper_pct)
                    ci_results[metric_name] = (round(float(low), 4), round(float(high), 4))
        else:
            boot_warnings.append(
                f"Insufficient valid bootstrap iterations ({successful} < {min_valid_iterations}) "
                "to form percentile confidence intervals; CIs set to None."
            )

        return ci_results, diagnostics

    def evaluate(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        modality: ModalityType = ModalityType.CXR_ONLY,
        strict_provenance: bool = True,
        source_op_thresh: Optional[float] = None,
        threshold_metric: str = "f1",
        require_validation_diversity: Optional[bool] = None,
        target_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs full baseline evaluation protocol.
        1. Validates dataset integrity and checks split leakage (all splits).
        2. Validates label domain and applies CheXpert uncertainty policy.
        3. Enforces single consistent label_source.
        4. Quarantines synthetic data (blocks clinical claims in strict mode).
        5. Enforces multimodal linkage verification.
        6. Selects optimal decision threshold strictly on validation set (fails explicitly if validation
           lacks class diversity when require_validation_diversity is True). Never uses source_op_thresh
           or test labels to select the benchmark decision threshold.
        7. Computes test set metrics using the frozen validation threshold.
        8. Computes subject_id-clustered bootstrap 95% CIs from valid iterations only.
        9. Returns comprehensive JSON-serializable report with full threshold and reproducibility provenance.
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        enforce_val_diversity = (
            strict_provenance
            if require_validation_diversity is None
            else bool(require_validation_diversity)
        )

        input_has_patient_id = "patient_id" in df.columns and "subject_id" not in df.columns
        resampling_unit = "patient_id" if "patient_id" in df.columns else "subject_id"
        if "subject_id" in df.columns and "patient_id" not in df.columns:
            resampling_unit = "subject_id"
        elif input_has_patient_id:
            resampling_unit = "patient_id"

        # Step 1: Schema and patient-level integrity validation
        df = self.validate_dataset(df)

        # Step 2: Validate label_source consistency
        if "label_source" in df.columns:
            sources = df["label_source"].dropna().unique()
            if len(sources) > 1:
                raise MixedLabelSourceError(
                    f"Mixed label sources detected: {list(sources)}. "
                    "Evaluation dataset must have exactly one consistent label definition per benchmark run."
                )
            label_source = str(sources[0]) if len(sources) == 1 else "UNKNOWN"
        else:
            label_source = "UNKNOWN"

        # Step 3: Apply CheXpert uncertainty policy
        df, uncertainty_stats = self.apply_uncertainty_policy(df)

        # Modality probability column mapping
        prob_cols = {
            ModalityType.CXR_ONLY: "p_cxr",
            ModalityType.CLINICAL_ONLY: "p_clinical",
            ModalityType.MULTIMODAL: "p_multimodal",
        }
        prob_col = prob_cols[modality]

        if prob_col not in df.columns:
            raise ValueError(f"Required probability column '{prob_col}' not found for modality '{modality}'")

        # Step 4: Synthetic Quarantine & Modality Guardrails
        is_software_test = False
        detected_synthetic = 0
        excluded_synthetic = 0
        unverified_linkage_samples = 0

        if "is_synthetic" in df.columns:
            synthetic_mask = df["is_synthetic"].fillna(False).astype(bool)
            detected_synthetic = int(synthetic_mask.sum())
            if detected_synthetic > 0:
                is_software_test = True
                if strict_provenance:
                    raise SyntheticDataQuarantineError(
                        f"Refusing clinical evaluation: {detected_synthetic} synthetic samples detected in {modality.value}. "
                        "Synthetic fixtures cannot be used for clinical benchmark reporting. "
                        "Set strict_provenance=False ONLY for software test execution."
                    )
                else:
                    logger.warning(
                        "Quarantining %d synthetic samples from evaluation cohort.",
                        detected_synthetic,
                    )
                    df = df[~synthetic_mask]
                    excluded_synthetic = detected_synthetic

        if modality == ModalityType.MULTIMODAL:
            if "linkage_verified" in df.columns:
                unverified_mask = ~df["linkage_verified"].fillna(False).astype(bool)
                unverified_linkage_samples = int(unverified_mask.sum())
                if unverified_linkage_samples > 0:
                    if strict_provenance:
                        raise UnverifiedLinkageError(
                            f"Refusing multimodal benchmark: {unverified_linkage_samples} samples have unverified "
                            "image-clinical linkage. Cannot claim multimodal fusion without verified encounter linkage."
                        )
                    else:
                        logger.warning(
                            "Filtering out %d unverified multimodal linkage samples.",
                            unverified_linkage_samples,
                        )
                        df = df[~unverified_mask]
            else:
                if strict_provenance:
                    raise UnverifiedLinkageError(
                        "linkage_verified column missing. Cannot evaluate multimodal fusion without verified linkage flag."
                    )

        # Filter rows having valid probabilities for the target modality
        df_mod = df.dropna(subset=[prob_col]).copy()
        missing_probs_dropped = len(df) - len(df_mod)

        # Step 5: Split datasets
        val_df = df_mod[df_mod["split"] == "validate"]
        test_df = df_mod[df_mod["split"] == "test"]
        train_df = df_mod[df_mod["split"] == "train"]

        warnings_list: List[str] = []

        val_sample_count = int(len(val_df))
        val_pos_count = int((val_df["y_true"] == 1).sum()) if val_sample_count > 0 else 0
        val_neg_count = int((val_df["y_true"] == 0).sum()) if val_sample_count > 0 else 0
        val_has_diversity = (val_sample_count > 0) and (len(np.unique(val_df["y_true"])) >= 2)

        # Step 6: Select decision threshold exclusively on validation set
        optimization_method = "grid_search_linspace_0.05_to_0.95_91_points_on_validation_split"
        if val_has_diversity:
            validation_optimal_threshold: Optional[float] = self.find_optimal_threshold(
                val_df["y_true"].to_numpy(),
                val_df[prob_col].to_numpy(),
                metric=threshold_metric,
                strict=True,
            )
            threshold = float(validation_optimal_threshold)
            threshold_source = f"validation_optimized_{threshold_metric}"
        else:
            if enforce_val_diversity:
                target_label_msg = f" for target '{target_name}'" if target_name else ""
                raise InsufficientValidationDiversityError(
                    f"Insufficient validation class diversity{target_label_msg}: "
                    f"validation_sample_count={val_sample_count} (pos={val_pos_count}, neg={val_neg_count}). "
                    "Both positive (1) and negative (0) validation samples are required to select "
                    "validation_optimal_threshold in benchmark evaluation."
                )
            validation_optimal_threshold = None
            threshold = 0.5
            threshold_source = "default_fallback_0.5_non_benchmark"
            warnings_list.append(
                f"Validation set has < 2 classes or is empty (n={val_sample_count}, pos={val_pos_count}, neg={val_neg_count}). "
                "validation_optimal_threshold is None; non-benchmark fallback threshold 0.5 used."
            )

        clean_source_op_thresh = (
            None
            if source_op_thresh is None or pd.isna(source_op_thresh)
            else round(float(source_op_thresh), 6)
        )

        # Step 7: Compute evaluation metrics using frozen validation threshold
        val_metrics = self.compute_metrics(
            val_df["y_true"].to_numpy(),
            val_df[prob_col].to_numpy(),
            threshold=threshold,
        )

        test_metrics = self.compute_metrics(
            test_df["y_true"].to_numpy(),
            test_df[prob_col].to_numpy(),
            threshold=threshold,
        )

        # Step 8: Grouped Bootstrap Confidence Intervals on Test Split
        ci_results, boot_diag = self.compute_clustered_bootstrap_ci(
            test_df,
            prob_col=prob_col,
            threshold=threshold,
            resampling_unit=resampling_unit,
        )
        test_metrics.auroc_ci = ci_results.get("auroc")
        test_metrics.auprc_ci = ci_results.get("auprc")
        test_metrics.f1_ci = ci_results.get("f1")
        test_metrics.sensitivity_ci = ci_results.get("sensitivity")
        test_metrics.specificity_ci = ci_results.get("specificity")
        test_metrics.brier_ci = ci_results.get("brier")
        test_metrics.ece_ci = ci_results.get("ece")

        effective_uncertainty_policy = (
            "binary_strict_not_applicable"
            if self.label_domain_policy == LabelDomainPolicy.BINARY_STRICT
            else self.uncertainty_policy.value
        )

        threshold_provenance: Dict[str, Any] = {
            "source_op_thresh": clean_source_op_thresh,
            "source_op_thresh_origin": (
                "TorchXRayVision published operating point (reference metadata only; never used for test_metrics)"
                if clean_source_op_thresh is not None
                else None
            ),
            "source_op_thresh_used_for_test_metrics": False,
            "validation_optimal_threshold": validation_optimal_threshold,
            "selected_threshold": threshold,
            "applied_test_threshold": threshold,
            "strategy": threshold_source,
            "selection_metric": threshold_metric,
            "optimization_method": optimization_method,
            "validation_sample_count": val_sample_count,
            "validation_pos_count": val_pos_count,
            "validation_neg_count": val_neg_count,
            "validation_class_diversity_satisfied": val_has_diversity,
            "used_test_labels_for_threshold": False,
            "note": (
                "Decision threshold selected strictly on validation split; frozen and applied to test split. "
                "TorchXRayVision source_op_thresh is recorded strictly for reference provenance."
            ),
        }

        bootstrap_config: Dict[str, Any] = {
            "resampling_unit": resampling_unit,
            "requested_iterations": self.bootstrap_iterations,
            "confidence_level": 0.95,
            "alpha": 0.05,
            "random_seed": self.random_seed,
            "ci_method": "percentile_from_valid_iterations_only",
        }

        metric_definitions: Dict[str, str] = {
            "auroc": (
                "Area Under the Receiver Operating Characteristic curve (sklearn.metrics.roc_auc_score); "
                "undefined (None) if < 2 classes."
            ),
            "auprc": AUPRC_EXACT_DEFINITION,
            "f1": (
                "Binary F1-score evaluated on test split using frozen validation_optimal_threshold "
                "(sklearn.metrics.f1_score, zero_division=0)."
            ),
            "sensitivity": (
                "True Positive Rate TP / (TP + FN) evaluated on test split using frozen validation_optimal_threshold."
            ),
            "specificity": (
                "True Negative Rate TN / (TN + FP) evaluated on test split using frozen validation_optimal_threshold."
            ),
            "brier_score": "Mean squared error between predicted probability and binary label: mean((p - y)^2).",
            "ece": f"Expected Calibration Error across {self.num_calibration_bins} equal-width bins in [0, 1].",
        }

        test_metrics_dict = asdict(test_metrics)
        val_metrics_dict = asdict(val_metrics)
        eval_ts = datetime.now(timezone.utc).isoformat()

        # Step 9: Assemble comprehensive report
        report: Dict[str, Any] = {
            "source_op_thresh": clean_source_op_thresh,
            "validation_optimal_threshold": validation_optimal_threshold,
            "benchmark_metadata": {
                "benchmark_name": "MedFusion-CXR Baseline Evaluation",
                "version": self.benchmark_version,
                "evaluation_script_version": EVALUATION_SCRIPT_VERSION,
                "modality": modality.value,
                "target_name": target_name,
                "evaluation_timestamp_utc": eval_ts,
                "is_software_test": is_software_test,
                "label_domain_policy": self.label_domain_policy.value,
                "uncertainty_policy": effective_uncertainty_policy,
                "label_source": label_source,
                "metric_definitions": metric_definitions,
                "bootstrap_configuration": bootstrap_config,
                "bootstrap_diagnostics": boot_diag,
                "random_seed": self.random_seed,
            },
            "reproducibility_metadata": collect_reproducibility_metadata(
                random_seed=self.random_seed,
                uncertainty_policy=effective_uncertainty_policy,
                execution_mode="software_test" if is_software_test else "strict_evaluation",
                bootstrap_config=bootstrap_config,
                threshold_provenance=threshold_provenance,
            ),
            "cohort_summary": {
                "total_records_ingested": uncertainty_stats["initial_samples"],
                "records_evaluated": len(df_mod),
                "train_count": len(train_df),
                "validate_count": val_sample_count,
                "validate_pos_count": val_pos_count,
                "validate_neg_count": val_neg_count,
                "test_count": len(test_df),
                "uncertain_labels_handled": uncertainty_stats["uncertain_count"],
                "missing_labels_dropped": uncertainty_stats["missing_count"],
                "missing_probabilities_dropped": missing_probs_dropped,
                "detected_synthetic_samples": detected_synthetic,
                "excluded_synthetic_samples": excluded_synthetic,
                "unverified_linkage_samples": unverified_linkage_samples,
            },
            "threshold_selection": threshold_provenance,
            "threshold_provenance": threshold_provenance,
            "metric_definitions": metric_definitions,
            "validation_results": val_metrics_dict,
            "validation_metrics": val_metrics_dict,
            "test_results": test_metrics_dict,
            "test_metrics": test_metrics_dict,
            "warnings": warnings_list + val_metrics.warnings + test_metrics.warnings + boot_diag["warnings"],
        }

        return report

    @staticmethod
    def export_report_to_json(report: Dict[str, Any], filepath: str) -> None:
        """Serializes benchmark report to a pretty-printed JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

