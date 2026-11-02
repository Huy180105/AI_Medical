"""
Unit Tests for NIH-14 DenseNet-121 Adapter, Streaming Batch Inference, and Strict Binary Benchmark Evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.run_nih_cxr_inference import run_nih_pipeline_evaluation
from src.evaluation.benchmark_fusion import (
    BenchmarkFusionEvaluator,
    InvalidLabelError,
    LabelDomainPolicy,
    ModalityType,
    NIH_LABELS,
    SyntheticDataQuarantineError,
)
from src.vision.pretrained_cxr import (
    EXPECTED_NIH_DENSENET121_SHA256,
    NIH_14_LABELS,
    PretrainedNihDenseNet121,
)


def _save_synthetic_png(path: Path, seed: int) -> None:
    rng = np.random.RandomState(seed)
    arr = rng.randint(15, 240, size=(64, 64), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path, format="PNG")


def test_pretrained_nih_densenet121_slot_audit_and_batch_consistency():
    """
    Verifies `models/nih-densenet121-nih14.pt` SHA-256, 14 active NIH slots (||w||_2 > 0),
    4 masked non-NIH slots (||w||_2 == 0), and numerical parity between single and batch inference.
    """
    adapter = PretrainedNihDenseNet121(checkpoint_profile="nih14")
    assert adapter.checkpoint_sha256 == EXPECTED_NIH_DENSENET121_SHA256
    assert adapter.active_nih_labels == NIH_14_LABELS
    assert adapter.masked_nih_labels == []

    slot_table = adapter.get_slot_audit_table()
    assert len(slot_table) == 18
    for row in slot_table[:14]:
        assert row["is_trained_in_checkpoint"] is True
        assert row["weight_l2_norm"] > 0.0
    for row in slot_table[14:]:
        assert row["is_trained_in_checkpoint"] is False
        assert row["weight_l2_norm"] == 0.0

    # Batch vs single inference parity
    img_a = Image.fromarray(np.full((80, 64), 90, dtype=np.uint8), mode="L")
    img_b = Image.fromarray(np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64), mode="L")

    batch_tensor = adapter.preprocess_batch([img_a, img_b])
    assert tuple(batch_tensor.shape) == (2, 1, 224, 224)
    assert float(batch_tensor.min()) >= -1024.01
    assert float(batch_tensor.max()) <= 1024.01

    single_a = adapter.predict(img_a)["nih_14_probabilities"]
    batch_ab = adapter.predict_batch([img_a, img_b])
    for lbl in NIH_14_LABELS:
        assert abs(single_a[lbl] - batch_ab[0]["nih_14_probabilities"][lbl]) < 1e-5


def test_chex_external_profile_masks_7_untrained_nih_labels():
    """Verifies `chex_external` profile evaluates the 7 shared classes and masks the 7 untrained NIH classes as None."""
    ext_adapter = PretrainedNihDenseNet121(checkpoint_profile="chex_external")
    assert len(ext_adapter.active_nih_labels) == 7
    assert len(ext_adapter.masked_nih_labels) == 7

    img = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L")
    preds = ext_adapter.predict(img)["nih_14_probabilities"]
    for lbl in ext_adapter.active_nih_labels:
        assert isinstance(preds[lbl], float) and 0.0 <= preds[lbl] <= 1.0
    for lbl in ext_adapter.masked_nih_labels:
        assert preds[lbl] is None  # Never constant 0.5!


def test_benchmark_evaluator_nih_schema_and_binary_strict_policy():
    """
    Verifies `BenchmarkFusionEvaluator` accepts `(patient_id, image_id)` directly,
    clusters bootstrap by `patient_id`, and strictly rejects `-1` uncertainty labels under `BINARY_STRICT`.
    """
    evaluator = BenchmarkFusionEvaluator(
        bootstrap_iterations=50,
        random_seed=42,
        label_domain_policy=LabelDomainPolicy.BINARY_STRICT,
    )

    records = [
        {"patient_id": "101", "image_id": "101_000.png", "split": "validate", "y_true": 1, "p_cxr": 0.88, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False},
        {"patient_id": "102", "image_id": "102_000.png", "split": "validate", "y_true": 0, "p_cxr": 0.12, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False},
        {"patient_id": "201", "image_id": "201_000.png", "split": "test", "y_true": 1, "p_cxr": 0.91, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False},
        {"patient_id": "201", "image_id": "201_001.png", "split": "test", "y_true": 1, "p_cxr": 0.84, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False},
        {"patient_id": "202", "image_id": "202_000.png", "split": "test", "y_true": 0, "p_cxr": 0.09, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False},
        {"patient_id": "203", "image_id": "203_000.png", "split": "test", "y_true": 0, "p_cxr": 0.15, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False},
    ]
    report = evaluator.evaluate(
        records,
        modality=ModalityType.CXR_ONLY,
        strict_provenance=True,
        source_op_thresh=0.039118,
        target_name="Atelectasis",
    )
    assert report["benchmark_metadata"]["bootstrap_diagnostics"]["resampling_unit"] == "patient_id"
    assert report["benchmark_metadata"]["bootstrap_diagnostics"]["n_unique_patients"] == 3
    assert report["threshold_provenance"]["source_op_thresh_used_for_test_metrics"] is False
    assert report["test_metrics"]["auroc"] == 1.0

    # Injecting CheXpert -1 uncertainty label must raise InvalidLabelError under BINARY_STRICT
    bad_records = list(records) + [
        {"patient_id": "204", "image_id": "204_000.png", "split": "test", "y_true": -1, "p_cxr": 0.4, "label_source": "NIH_CHESTXRAY14_NLP_2017", "is_synthetic": False}
    ]
    with pytest.raises(InvalidLabelError, match="Strict binary label domain violated"):
        evaluator.evaluate(bad_records, modality=ModalityType.CXR_ONLY, strict_provenance=True)


def test_streaming_nih_pipeline_end_to_end(tmp_path: Path):
    """
    Tests end-to-end streaming inference (`batch_size=3`) + 14-label evaluation,
    verifying MISSING/CORRUPT row accounting and synthetic quarantine enforcement.
    """
    img_dir = tmp_path / "raw" / "images_001" / "images"
    rows: List[Dict[str, object]] = []

    # Create 4 validate patients and 4 test patients (plus 1 MISSING row and 1 CORRUPT row)
    for idx, (pid, split) in enumerate(
        [("1", "validate"), ("2", "validate"), ("3", "validate"), ("4", "validate"),
         ("5", "test"), ("6", "test"), ("7", "test"), ("8", "test")]
    ):
        img_name = f"0000000{pid}_000.png"
        img_path = img_dir / img_name
        _save_synthetic_png(img_path, seed=100 + idx)

        rec: Dict[str, object] = {
            "image_id": img_name,
            "patient_id": pid,
            "split": split,
            "image_path": img_path.as_posix(),
            "shard_id": "images_001",
            "image_status": "VALID",
            "view_position": "PA",
            "patient_gender": "M" if idx % 2 == 0 else "F",
            "age_group": "40-59",
            "label_source": "NIH_CHESTXRAY14_NLP_2017",
            "is_synthetic": True,
        }
        # Alternate positive/negative across all 14 NIH labels so both splits have class diversity
        is_pos = int(idx % 2 == 0)
        for lbl in NIH_LABELS:
            rec[lbl] = is_pos
        rows.append(rec)

    # Append 1 MISSING and 1 CORRUPT row to verify explicit accounting without crash
    missing_rec = dict(rows[0], image_id="00000098_000.png", patient_id="98", split="test", image_path="", image_status="MISSING")
    corrupt_path = img_dir / "00000099_000.png"
    corrupt_path.write_bytes(b"corrupt")
    corrupt_rec = dict(rows[0], image_id="00000099_000.png", patient_id="99", split="test", image_path=corrupt_path.as_posix(), image_status="CORRUPT")
    rows.extend([missing_rec, corrupt_rec])

    manifest_csv = tmp_path / "nih_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)

    preds_csv = tmp_path / "reports" / "nih_predictions.csv"
    report_json = tmp_path / "reports" / "nih_benchmark_report.json"

    # Benchmark mode MUST quarantine synthetic fixtures
    with pytest.raises(SyntheticDataQuarantineError):
        run_nih_pipeline_evaluation(
            manifest_path=manifest_csv,
            predictions_out=preds_csv,
            report_out=report_json,
            mode="benchmark",
            batch_size=3,
            bootstrap_iterations=20,
        )

    # Software test mode streams mini-batches of 3 and evaluates all 14 NIH labels
    report = run_nih_pipeline_evaluation(
        manifest_path=manifest_csv,
        predictions_out=preds_csv,
        report_out=report_json,
        mode="software_test",
        batch_size=3,
        bootstrap_iterations=25,
        seed=42,
    )

    assert preds_csv.exists() and report_json.exists()
    preds_df = pd.read_csv(preds_csv)
    assert len(preds_df) == 8
    assert report["cohort_accounting"]["manifest_missing_image_rows_skipped"] == 1
    assert report["cohort_accounting"]["manifest_corrupt_image_rows_skipped"] == 1
    assert report["macro_summary"]["evaluated_label_count"] == 14
    assert report["dataset_governance"]["physionet_or_mimic_used"] is False
    assert len(json.loads(report_json.read_text(encoding="utf-8"))["per_pathology_results"]) == 14
