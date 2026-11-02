"""
Comprehensive Test Suite for MedFusion-CXR Baseline Evaluation Module.

Tests cover:
1. Perfect and deliberately incorrect predictions.
2. Imbalanced labels and single-class subsets (graceful handling of undefined metrics).
3. Missing labels, uncertain labels (-1) under U-Ignore, U-Zeros, U-Ones.
4. NaN and out-of-range probabilities.
5. Duplicate study IDs and patient overlap across splits (Patient Leakage Guard).
6. Validation-only decision threshold selection.
7. Reproducible subject-clustered bootstrap confidence intervals.
8. Modality separation, unverified linkage rejection, and synthetic data quarantine.
9. JSON report serialization and schema audit.
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark_fusion import (
    AUPRC_EXACT_DEFINITION,
    BenchmarkFusionEvaluator,
    EvaluationRecord,
    UncertaintyPolicy,
    ModalityType,
    PatientLeakageError,
    DuplicateRecordError,
    UnverifiedLinkageError,
    SyntheticDataQuarantineError,
    InvalidLabelError,
    MixedLabelSourceError,
    InsufficientValidationDiversityError,
    ExpectedCalibrationErrorCalculator,
)


def _generate_synthetic_eval_data(
    n_patients: int = 20,
    studies_per_patient: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Helper to generate deterministic, leak-free evaluation predictions."""
    rng = np.random.RandomState(seed)
    records = []

    # Partition patients strictly across splits
    n_train_pts = int(n_patients * 0.5)
    n_val_pts = int(n_patients * 0.25)
    
    patient_ids = [f"P_{i:04d}" for i in range(n_patients)]
    train_pts = set(patient_ids[:n_train_pts])
    val_pts = set(patient_ids[n_train_pts : n_train_pts + n_val_pts])
    test_pts = set(patient_ids[n_train_pts + n_val_pts :])

    study_idx = 0
    for pid in patient_ids:
        if pid in train_pts:
            split = "train"
        elif pid in val_pts:
            split = "validate"
        else:
            split = "test"

        for _ in range(studies_per_patient):
            study_idx += 1
            y_true = rng.choice([0, 1], p=[0.7, 0.3])
            
            # Simulated well-calibrated CXR probability
            if y_true == 1:
                p_cxr = float(np.clip(rng.beta(7, 2), 0.0, 1.0))
            else:
                p_cxr = float(np.clip(rng.beta(2, 7), 0.0, 1.0))

            records.append({
                "subject_id": pid,
                "study_id": f"S_{study_idx:05d}",
                "y_true": int(y_true),
                "p_cxr": round(p_cxr, 4),
                "p_clinical": round(float(np.clip(p_cxr + rng.normal(0, 0.05), 0.0, 1.0)), 4),
                "p_multimodal": round(float(np.clip(p_cxr + rng.normal(0, 0.02), 0.0, 1.0)), 4),
                "split": split,
                "label_source": "MIMIC_CXR_JPG_CHEXPERT_V2.1.0",
                "linkage_verified": True,
                "is_synthetic": False,
            })

    return pd.DataFrame(records)


def test_evaluation_record_schema_validation():
    """Verifies dataclass input validation catches invalid splits and out-of-range probabilities."""
    # Valid record
    rec = EvaluationRecord(
        subject_id="1001",
        study_id="5001",
        y_true=1,
        split="test",
        p_cxr=0.85,
    )
    rec.validate()

    # Empty subject_id
    with pytest.raises(ValueError, match="subject_id cannot be null or empty"):
        EvaluationRecord(subject_id="", study_id="5001", y_true=1, split="test").validate()

    # Invalid split
    with pytest.raises(ValueError, match="Invalid split"):
        EvaluationRecord(subject_id="1001", study_id="5001", y_true=1, split="holdout_fake").validate()

    # Invalid label
    with pytest.raises(InvalidLabelError, match="Invalid y_true"):
        EvaluationRecord(subject_id="1001", study_id="5001", y_true=5, split="test").validate()

    # Probability out of range
    with pytest.raises(ValueError, match="out of range"):
        EvaluationRecord(subject_id="1001", study_id="5001", y_true=1, split="test", p_cxr=1.5).validate()

    # NaN probability
    with pytest.raises(ValueError, match="cannot be NaN"):
        EvaluationRecord(subject_id="1001", study_id="5001", y_true=1, split="test", p_cxr=float("nan")).validate()


def test_perfect_vs_inverted_predictions():
    """Verifies metric behavior on perfect vs deliberately inverted predictions."""
    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=0)
    
    # 1. Perfect predictions
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    p_perfect = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    perf_metrics = evaluator.compute_metrics(y_true, p_perfect, threshold=0.5)

    assert perf_metrics.auroc == 1.0
    assert perf_metrics.auprc == 1.0
    assert perf_metrics.f1 == 1.0
    assert perf_metrics.brier_score == 0.0
    assert perf_metrics.ece == 0.0

    # 2. Deliberately inverted predictions
    p_inverted = 1.0 - p_perfect
    inv_metrics = evaluator.compute_metrics(y_true, p_inverted, threshold=0.5)

    assert inv_metrics.auroc == 0.0
    assert inv_metrics.f1 == 0.0
    assert inv_metrics.brier_score == 1.0


def test_single_class_edge_cases():
    """AUROC and AUPRC must be None and not crash when only 1 class is present."""
    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=0)

    # All zeros
    y_zeros = np.zeros(20, dtype=int)
    p_zeros = np.linspace(0.1, 0.9, 20)
    m_zeros = evaluator.compute_metrics(y_zeros, p_zeros, threshold=0.5)

    assert m_zeros.auroc is None
    assert m_zeros.auprc is None
    assert any("Single-class target detected" in w for w in m_zeros.warnings)
    assert m_zeros.brier_score is not None

    # All ones
    y_ones = np.ones(20, dtype=int)
    p_ones = np.linspace(0.1, 0.9, 20)
    m_ones = evaluator.compute_metrics(y_ones, p_ones, threshold=0.5)

    assert m_ones.auroc is None
    assert m_ones.auprc is None
    assert any("Single-class target detected" in w for w in m_ones.warnings)


def test_expected_calibration_error_bins():
    """Verifies equal-width binning for ECE computation."""
    # Predict uniform 0.5 confidence, while half are 1 and half are 0
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p_prob = np.full(10, 0.5)

    ece, cal_bins = ExpectedCalibrationErrorCalculator.compute_ece(y_true, p_prob, num_bins=10)
    
    assert ece == 0.0
    assert len(cal_bins) == 10
    
    bin_5 = cal_bins[5]
    assert bin_5.sample_count == 10
    assert bin_5.empirical_accuracy == 0.5
    assert bin_5.mean_confidence == 0.5
    assert bin_5.calibration_error == 0.0


def test_chexpert_uncertainty_policies():
    """Tests U-Ignore, U-Zeros, and U-Ones policies on CheXpert label -1."""
    raw_data = pd.DataFrame([
        {"subject_id": "P1", "study_id": "S1", "y_true": 1, "split": "test", "p_cxr": 0.8},
        {"subject_id": "P2", "study_id": "S2", "y_true": 0, "split": "test", "p_cxr": 0.2},
        {"subject_id": "P3", "study_id": "S3", "y_true": -1, "split": "test", "p_cxr": 0.6},
        {"subject_id": "P4", "study_id": "S4", "y_true": None, "split": "test", "p_cxr": 0.5},
    ])

    # 1. U-Ignore: drops -1 and NaN
    ev_ignore = BenchmarkFusionEvaluator(uncertainty_policy=UncertaintyPolicy.U_IGNORE)
    df_ignore, stats_ignore = ev_ignore.apply_uncertainty_policy(raw_data)
    assert len(df_ignore) == 2
    assert set(df_ignore["y_true"].unique()) == {0, 1}
    assert stats_ignore["uncertain_count"] == 1
    assert stats_ignore["missing_count"] == 1

    # 2. U-Zeros: maps -1 to 0
    ev_zeros = BenchmarkFusionEvaluator(uncertainty_policy=UncertaintyPolicy.U_ZEROS)
    df_zeros, _ = ev_zeros.apply_uncertainty_policy(raw_data)
    assert len(df_zeros) == 3
    s3_row = df_zeros[df_zeros["study_id"] == "S3"].iloc[0]
    assert s3_row["y_true"] == 0

    # 3. U-Ones: maps -1 to 1
    ev_ones = BenchmarkFusionEvaluator(uncertainty_policy=UncertaintyPolicy.U_ONES)
    df_ones, _ = ev_ones.apply_uncertainty_policy(raw_data)
    assert len(df_ones) == 3
    s3_row_ones = df_ones[df_ones["study_id"] == "S3"].iloc[0]
    assert s3_row_ones["y_true"] == 1


def test_invalid_label_domain_rejection():
    """Rejects labels outside {0, 1, -1, NaN}."""
    invalid_data = pd.DataFrame([
        {"subject_id": "P1", "study_id": "S1", "y_true": 2, "split": "test", "p_cxr": 0.8},
    ])
    evaluator = BenchmarkFusionEvaluator()
    with pytest.raises(InvalidLabelError, match="Invalid ground truth labels detected"):
        evaluator.apply_uncertainty_policy(invalid_data)


def test_patient_leakage_detection():
    """Detects and rejects patient overlap between all splits (train/val/test)."""
    # Train-test overlap
    leaky_data1 = pd.DataFrame([
        {"subject_id": "P_LEAK_01", "study_id": "S_01", "y_true": 1, "split": "train", "p_cxr": 0.8},
        {"subject_id": "P_LEAK_01", "study_id": "S_02", "y_true": 0, "split": "test", "p_cxr": 0.3},
        {"subject_id": "P_CLEAN_02", "study_id": "S_03", "y_true": 1, "split": "validate", "p_cxr": 0.7},
    ])
    evaluator = BenchmarkFusionEvaluator(strict_patient_split=True)
    with pytest.raises(PatientLeakageError, match="train and test splits"):
        evaluator.validate_dataset(leaky_data1)

    # Train-val overlap (strict mode)
    leaky_data2 = pd.DataFrame([
        {"subject_id": "P_LEAK_02", "study_id": "S_01", "y_true": 1, "split": "train", "p_cxr": 0.8},
        {"subject_id": "P_LEAK_02", "study_id": "S_02", "y_true": 0, "split": "validate", "p_cxr": 0.3},
        {"subject_id": "P_CLEAN_03", "study_id": "S_03", "y_true": 1, "split": "test", "p_cxr": 0.7},
    ])
    with pytest.raises(PatientLeakageError, match="train and validate splits"):
        evaluator.validate_dataset(leaky_data2)


def test_duplicate_study_id_detection():
    """Detects and rejects duplicate study_id records within the same split."""
    duplicate_data = pd.DataFrame([
        {"subject_id": "P1", "study_id": "S_DUP", "y_true": 1, "split": "test", "p_cxr": 0.8},
        {"subject_id": "P2", "study_id": "S_DUP", "y_true": 0, "split": "test", "p_cxr": 0.2},
    ])
    evaluator = BenchmarkFusionEvaluator()
    with pytest.raises(DuplicateRecordError, match="duplicate records with same study_id"):
        evaluator.validate_dataset(duplicate_data)


def test_null_identifiers_detection():
    """Detects null, empty string, or 'nan' string in identifiers."""
    null_sub = pd.DataFrame([
        {"subject_id": np.nan, "study_id": "S1", "y_true": 1, "split": "test", "p_cxr": 0.8},
    ])
    evaluator = BenchmarkFusionEvaluator()
    with pytest.raises(ValueError, match="Found null, empty, or NaN subject_id"):
        evaluator.validate_dataset(null_sub)

    null_split = pd.DataFrame([
        {"subject_id": "P1", "study_id": "S1", "y_true": 1, "split": None, "p_cxr": 0.8},
    ])
    with pytest.raises(ValueError, match="Found null or missing values in 'split' column"):
        evaluator.validate_dataset(null_split)


def test_mixed_label_source_detection():
    """Refuses benchmark if multiple inconsistent label_source values are provided."""
    mixed_data = pd.DataFrame([
        {"subject_id": "P1", "study_id": "S1", "y_true": 1, "split": "train", "p_cxr": 0.8, "label_source": "CHEXPERT_NLP"},
        {"subject_id": "P2", "study_id": "S2", "y_true": 0, "split": "validate", "p_cxr": 0.2, "label_source": "CHEXPERT_NLP"},
        {"subject_id": "P3", "study_id": "S3", "y_true": 1, "split": "test", "p_cxr": 0.7, "label_source": "EXPERT_ADJUDICATED"},
    ])
    evaluator = BenchmarkFusionEvaluator()
    with pytest.raises(MixedLabelSourceError, match="Mixed label sources detected"):
        evaluator.evaluate(mixed_data, modality=ModalityType.CXR_ONLY, strict_provenance=False)


def test_validation_only_threshold_selection():
    """Ensures optimal threshold is tuned on validation set ONLY, never test set."""
    evaluator = BenchmarkFusionEvaluator()

    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_val = np.array([0.1, 0.2, 0.6, 0.65, 0.75, 0.8, 0.85, 0.9])
    
    threshold = evaluator.find_optimal_threshold(y_val, p_val, metric="f1")
    assert 0.65 <= threshold <= 0.75

    df = _generate_synthetic_eval_data(n_patients=30, seed=123)
    report = evaluator.evaluate(df, modality=ModalityType.CXR_ONLY, strict_provenance=False)

    assert "threshold_selection" in report
    assert report["threshold_selection"]["strategy"] == "validation_optimized_f1"
    assert report["test_results"]["threshold"] == report["threshold_selection"]["selected_threshold"]


def test_patient_clustered_bootstrap_reproducibility():
    """Subject-clustered bootstrap must be deterministic given fixed random seed."""
    df = _generate_synthetic_eval_data(n_patients=40, studies_per_patient=3, seed=42)
    test_df = df[df["split"] == "test"].copy()

    ev1 = BenchmarkFusionEvaluator(bootstrap_iterations=50, random_seed=999)
    ci1, diag1 = ev1.compute_clustered_bootstrap_ci(test_df, prob_col="p_cxr", threshold=0.5)

    ev2 = BenchmarkFusionEvaluator(bootstrap_iterations=50, random_seed=999)
    ci2, diag2 = ev2.compute_clustered_bootstrap_ci(test_df, prob_col="p_cxr", threshold=0.5)

    assert ci1["auroc"] == ci2["auroc"]
    assert ci1["f1"] == ci2["f1"]
    assert ci1["brier"] == ci2["brier"]
    assert ci1["auroc"][0] <= ci1["auroc"][1]
    assert diag1["successful_iterations"] > 0
    assert diag1["failure_rate"] >= 0.0


def test_synthetic_data_quarantine_in_cxr_and_multimodal():
    """Synthetic test fixtures cannot be used for clinical benchmark claims in strict mode."""
    df = _generate_synthetic_eval_data(n_patients=20, seed=42)
    df["is_synthetic"] = True
    df["linkage_verified"] = True

    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=10)
    # Strict mode refuses synthetic data for clinical evaluation
    with pytest.raises(SyntheticDataQuarantineError, match="synthetic samples detected"):
        evaluator.evaluate(df, modality=ModalityType.CXR_ONLY, strict_provenance=True)

    with pytest.raises(SyntheticDataQuarantineError, match="synthetic samples detected"):
        evaluator.evaluate(df, modality=ModalityType.MULTIMODAL, strict_provenance=True)

    # Software test mode allows quarantining with explicit flag
    report = evaluator.evaluate(df, modality=ModalityType.CXR_ONLY, strict_provenance=False)
    assert report["benchmark_metadata"]["is_software_test"] is True
    assert report["cohort_summary"]["excluded_synthetic_samples"] == len(df)


def test_multimodal_unverified_linkage_rejection():
    """Multimodal evaluation must refuse unverified linkage in strict mode."""
    df = _generate_synthetic_eval_data(n_patients=20, seed=42)
    df["linkage_verified"] = False
    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=10)

    with pytest.raises(UnverifiedLinkageError, match="Refusing multimodal benchmark"):
        evaluator.evaluate(df, modality=ModalityType.MULTIMODAL, strict_provenance=True)


def test_json_report_serialization(tmp_path):
    """Verifies that full benchmark report serializes to valid, comprehensive JSON."""
    df = _generate_synthetic_eval_data(n_patients=30, seed=777)
    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=30, random_seed=777)

    report = evaluator.evaluate(df, modality=ModalityType.CXR_ONLY, strict_provenance=False)
    json_path = tmp_path / "baseline_cxr_evaluation_report.json"
    
    evaluator.export_report_to_json(report, str(json_path))
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["benchmark_metadata"]["modality"] == "cxr_only"
    assert loaded["benchmark_metadata"]["benchmark_name"] == "MedFusion-CXR Baseline Evaluation"
    assert "validation_results" in loaded
    assert "test_results" in loaded
    assert "threshold_selection" in loaded
    assert loaded["test_results"]["auroc"] is not None
    assert loaded["test_results"]["ece"] is not None
    assert loaded["test_results"]["brier_score"] is not None


def test_test_labels_never_influence_threshold_selection():
    """
    Verifies Requirement 1 & 6:
    Modifying or completely inverting test split labels and test probabilities MUST NEVER
    change validation_optimal_threshold.
    """
    df_a = _generate_synthetic_eval_data(n_patients=32, studies_per_patient=2, seed=2026)
    df_b = df_a.copy()

    # Invert every test label and radically shift test probabilities in df_b
    test_mask = df_b["split"] == "test"
    df_b.loc[test_mask, "y_true"] = 1 - df_b.loc[test_mask, "y_true"]
    df_b.loc[test_mask, "p_cxr"] = 1.0 - df_b.loc[test_mask, "p_cxr"]

    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=20, random_seed=42)
    rep_a = evaluator.evaluate(
        df_a, modality=ModalityType.CXR_ONLY, strict_provenance=True, source_op_thresh=0.198897
    )
    rep_b = evaluator.evaluate(
        df_b, modality=ModalityType.CXR_ONLY, strict_provenance=True, source_op_thresh=0.198897
    )

    assert rep_a["validation_optimal_threshold"] is not None
    assert rep_a["validation_optimal_threshold"] == rep_b["validation_optimal_threshold"]
    assert rep_a["threshold_provenance"]["used_test_labels_for_threshold"] is False
    assert rep_b["threshold_provenance"]["used_test_labels_for_threshold"] is False
    assert rep_a["test_metrics"]["threshold"] == rep_a["validation_optimal_threshold"]
    assert rep_b["test_metrics"]["threshold"] == rep_b["validation_optimal_threshold"]


def test_source_op_thresh_vs_validation_threshold_and_diversity_guard():
    """
    Verifies Requirement 1 & 6:
    1. source_op_thresh (from TorchXRayVision) and validation_optimal_threshold are recorded
       distinctly and never confused; test_metrics uses validation_optimal_threshold.
    2. Benchmark evaluation fails explicitly with InsufficientValidationDiversityError when
       validation contains < 2 classes or 0 samples.
    """
    records = [
        # Train
        {"subject_id": "T1", "study_id": "ST1", "y_true": 0, "p_cxr": 0.10, "split": "train", "label_source": "CHEX_NLP", "is_synthetic": False},
        {"subject_id": "T2", "study_id": "ST2", "y_true": 1, "p_cxr": 0.90, "split": "train", "label_source": "CHEX_NLP", "is_synthetic": False},
        # Validation: separable specifically around 0.72..0.78 (very far from source_op_thresh=0.057106)
        {"subject_id": "V1", "study_id": "SV1", "y_true": 0, "p_cxr": 0.60, "split": "validate", "label_source": "CHEX_NLP", "is_synthetic": False},
        {"subject_id": "V2", "study_id": "SV2", "y_true": 0, "p_cxr": 0.68, "split": "validate", "label_source": "CHEX_NLP", "is_synthetic": False},
        {"subject_id": "V3", "study_id": "SV3", "y_true": 1, "p_cxr": 0.80, "split": "validate", "label_source": "CHEX_NLP", "is_synthetic": False},
        {"subject_id": "V4", "study_id": "SV4", "y_true": 1, "p_cxr": 0.88, "split": "validate", "label_source": "CHEX_NLP", "is_synthetic": False},
        # Test: sample at 0.30 would be positive under source_op_thresh=0.057106, but MUST be negative under val threshold >= 0.69
        {"subject_id": "E1", "study_id": "SE1", "y_true": 0, "p_cxr": 0.30, "split": "test", "label_source": "CHEX_NLP", "is_synthetic": False},
        {"subject_id": "E2", "study_id": "SE2", "y_true": 1, "p_cxr": 0.85, "split": "test", "label_source": "CHEX_NLP", "is_synthetic": False},
    ]
    df = pd.DataFrame(records)
    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=10, random_seed=42)

    rep = evaluator.evaluate(
        df,
        modality=ModalityType.CXR_ONLY,
        strict_provenance=True,
        source_op_thresh=0.057106,
        threshold_metric="f1",
        target_name="Consolidation",
    )

    assert rep["source_op_thresh"] == pytest.approx(0.057106, abs=1e-6)
    assert rep["validation_optimal_threshold"] >= 0.69
    assert rep["source_op_thresh"] != rep["validation_optimal_threshold"]
    assert rep["threshold_provenance"]["source_op_thresh_used_for_test_metrics"] is False
    assert rep["threshold_provenance"]["selection_metric"] == "f1"
    assert "grid_search" in rep["threshold_provenance"]["optimization_method"]
    assert rep["threshold_provenance"]["validation_sample_count"] == 4
    assert rep["test_metrics"]["threshold"] == rep["validation_optimal_threshold"]
    # Because test sample E1 (p=0.30, y=0) is below validation threshold (>=0.69) and E2 (p=0.85, y=1) is above, F1 == 1.0!
    # Had source_op_thresh (0.057106) been used on test, E1 would be false positive and F1 would be 0.6667!
    assert rep["test_metrics"]["f1"] == 1.0

    # Now make validation single-class (all zeros) -> must fail explicitly in strict mode
    df_single_val = df.copy()
    df_single_val.loc[df_single_val["split"] == "validate", "y_true"] = 0
    with pytest.raises(InsufficientValidationDiversityError, match="Insufficient validation class diversity"):
        evaluator.evaluate(
            df_single_val,
            modality=ModalityType.CXR_ONLY,
            strict_provenance=True,
            target_name="Consolidation",
        )


def test_bootstrap_single_class_iterations_and_valid_only_ci():
    """
    Verifies Requirement 3 & 6:
    - Resamples at subject_id level.
    - Safely skips single-class bootstrap iterations without computing AUROC.
    - Records requested, successful, and skipped iterations plus failure_rate.
    - Computes 95% CIs exclusively from valid dual-class iterations.
    - Records exact Average Precision definition for AUPRC.
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    # 5 negative patients and 1 positive patient -> (5/6)^6 ~= 33.5% of resamples are all-negative single-class
    test_df = pd.DataFrame([
        {"subject_id": "SUB_01", "study_id": "S1", "y_true": 0, "p_cxr": 0.12, "split": "test"},
        {"subject_id": "SUB_01", "study_id": "S2", "y_true": 0, "p_cxr": 0.18, "split": "test"},
        {"subject_id": "SUB_02", "study_id": "S3", "y_true": 0, "p_cxr": 0.25, "split": "test"},
        {"subject_id": "SUB_03", "study_id": "S4", "y_true": 0, "p_cxr": 0.31, "split": "test"},
        {"subject_id": "SUB_04", "study_id": "S5", "y_true": 0, "p_cxr": 0.40, "split": "test"},
        {"subject_id": "SUB_05", "study_id": "S6", "y_true": 0, "p_cxr": 0.22, "split": "test"},
        {"subject_id": "SUB_06", "study_id": "S7", "y_true": 1, "p_cxr": 0.88, "split": "test"},
    ])

    n_iter = 80
    seed = 42
    thresh = 0.5
    ev = BenchmarkFusionEvaluator(bootstrap_iterations=n_iter, random_seed=seed)
    ci_results, diag = ev.compute_clustered_bootstrap_ci(test_df, prob_col="p_cxr", threshold=thresh)

    assert diag["resampling_unit"] == "subject_id"
    assert diag["requested_iterations"] == n_iter
    assert diag["single_class_skipped_iterations"] > 0
    assert diag["skipped_iterations"] == diag["single_class_skipped_iterations"]
    assert diag["successful_iterations"] + diag["skipped_iterations"] == n_iter
    assert diag["failure_rate"] == round(diag["skipped_iterations"] / n_iter, 4)
    assert diag["auprc_definition"] == AUPRC_EXACT_DEFINITION
    assert "Average Precision" in diag["auprc_definition"]

    # Independently replay exact valid iterations to verify CI comes ONLY from valid dual-class draws
    subjects = np.array(sorted(test_df["subject_id"].unique()))
    groups = {s: g for s, g in test_df.groupby("subject_id", sort=True)}
    rng = np.random.RandomState(seed)

    valid_aurocs = []
    valid_auprcs = []
    valid_f1s = []
    for _ in range(n_iter):
        drawn = rng.choice(subjects, size=len(subjects), replace=True)
        b_df = pd.concat([groups[s] for s in drawn], ignore_index=True)
        yb = b_df["y_true"].to_numpy()
        pb = b_df["p_cxr"].to_numpy()
        if len(np.unique(yb)) < 2:
            continue
        valid_aurocs.append(float(roc_auc_score(yb, pb)))
        valid_auprcs.append(float(average_precision_score(yb, pb)))
        valid_f1s.append(float(f1_score(yb, (pb >= thresh).astype(int), zero_division=0)))

    assert len(valid_aurocs) == diag["successful_iterations"]
    expected_auroc_ci = (
        round(float(np.percentile(valid_aurocs, 2.5)), 4),
        round(float(np.percentile(valid_aurocs, 97.5)), 4),
    )
    expected_auprc_ci = (
        round(float(np.percentile(valid_auprcs, 2.5)), 4),
        round(float(np.percentile(valid_auprcs, 97.5)), 4),
    )
    expected_f1_ci = (
        round(float(np.percentile(valid_f1s, 2.5)), 4),
        round(float(np.percentile(valid_f1s, 97.5)), 4),
    )

    assert ci_results["auroc"] == expected_auroc_ci
    assert ci_results["auprc"] == expected_auprc_ci
    assert ci_results["f1"] == expected_f1_ci


def test_sensitivity_and_specificity_with_clustered_bootstrap_ci():
    """
    Verifies Giai doan 6B & Section 10 Q4:
    1. compute_metrics accurately calculates sensitivity = TP/(TP+FN) and specificity = TN/(TN+FP)
       at the frozen validation_optimal_threshold.
    2. compute_clustered_bootstrap_ci produces deterministic 95% CIs for sensitivity and specificity
       exclusively from valid dual-class subject_id resamples.
    3. evaluate() populates sensitivity, specificity, sensitivity_ci, specificity_ci, and metric_definitions.
    """
    evaluator = BenchmarkFusionEvaluator(bootstrap_iterations=60, random_seed=42)

    # Direct check: 4 positives (3 >= 0.5 -> TP=3, FN=1 => sens=0.75)
    #               4 negatives (2 <  0.5 -> TN=2, FP=2 => spec=0.50)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.6, 0.2, 0.1, 0.3, 0.7, 0.85])
    m = evaluator.compute_metrics(y_true, y_prob, threshold=0.5)
    assert m.sensitivity == pytest.approx(0.75, abs=1e-4)
    assert m.specificity == pytest.approx(0.50, abs=1e-4)

    # Full evaluate() check with clustered bootstrap CIs
    df = _generate_synthetic_eval_data(n_patients=36, studies_per_patient=2, seed=2026)
    report = evaluator.evaluate(df, modality=ModalityType.CXR_ONLY, strict_provenance=True)

    tm = report["test_metrics"]
    assert tm["sensitivity"] is not None and 0.0 <= tm["sensitivity"] <= 1.0
    assert tm["specificity"] is not None and 0.0 <= tm["specificity"] <= 1.0
    assert tm["sensitivity_ci"] is not None
    assert tm["specificity_ci"] is not None
    assert tm["sensitivity_ci"][0] <= tm["sensitivity_ci"][1]
    assert tm["specificity_ci"][0] <= tm["specificity_ci"][1]
    assert "sensitivity" in report["metric_definitions"]
    assert "specificity" in report["metric_definitions"]
    assert report["threshold_provenance"]["used_test_labels_for_threshold"] is False



