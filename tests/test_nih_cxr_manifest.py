"""
Unit Tests for NIH ChestX-ray14 Manifest Builder, Label Encoding, Integrity Audit, and Kaggle Ingestion Guards.
"""

from __future__ import annotations

import ast
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.build_nih_cxr_manifest import (
    build_nih_manifest,
    encode_finding_labels,
    inspect_png_integrity,
    parse_nih_patient_age,
)
from scripts.download_kaggle_nih import (
    InsufficientDiskSpaceError,
    SecurityConfigurationError,
    check_disk_space,
    required_free_gb_for_plan,
    safe_extract_zip,
    stage_from_local_directory,
)
from src.evaluation.benchmark_fusion import InvalidLabelError, NIH_LABELS


def _make_valid_png_bytes(seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(32, 32), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_nih_14_label_encoding_and_no_finding_rule():
    """Verifies exact 14 binary NIH labels, 'No Finding' -> 14 zeros, and rejection of -1 / unknown labels."""
    assert len(NIH_LABELS) == 14
    assert "No Finding" not in NIH_LABELS

    # 'No Finding' must produce 14 zeros and no_finding == 1
    zero_map, no_finding, cnt = encode_finding_labels("No Finding")
    assert no_finding == 1
    assert cnt == 0
    assert all(v == 0 for v in zero_map.values())
    assert set(zero_map.keys()) == set(NIH_LABELS)

    # Multi-label pipe-separated string
    multi_map, no_finding_m, cnt_m = encode_finding_labels("Cardiomegaly|Effusion|Pleural_Thickening")
    assert no_finding_m == 0
    assert cnt_m == 3
    assert multi_map["Cardiomegaly"] == 1
    assert multi_map["Effusion"] == 1
    assert multi_map["Pleural_Thickening"] == 1
    assert sum(multi_map.values()) == 3

    # Contradictory 'No Finding' + pathology must fail
    with pytest.raises(InvalidLabelError, match="Contradictory"):
        encode_finding_labels("No Finding|Pneumonia")

    # CheXpert '-1' uncertainty or non-NIH labels must fail strictly
    with pytest.raises(InvalidLabelError, match="Forbidden CheXpert uncertainty"):
        encode_finding_labels("-1")

    with pytest.raises(InvalidLabelError, match="Unknown pathology label"):
        encode_finding_labels("Enlarged Cardiomediastinum")


def test_nih_patient_age_parsing_and_typo_preservation():
    """Verifies Kaggle integer ages, legacy Y/M/D ages, and NIH >120 typo ages are preserved with anomaly flags."""
    yrs, grp, anom = parse_nih_patient_age(58)
    assert yrs == 58.0 and grp == "40-59" and anom == 0

    yrs_m, grp_m, anom_m = parse_nih_patient_age("006M")
    assert yrs_m == 0.5 and grp_m == "0-17" and anom_m == 0

    # Upstream NIH Data_Entry_2017.csv contains known typo ages like 411; must flag, never crash
    yrs_bad, grp_bad, anom_bad = parse_nih_patient_age(411)
    assert yrs_bad == 411.0
    assert grp_bad == "ANOMALY_OUT_OF_RANGE"
    assert anom_bad == 1


def test_build_nih_manifest_retains_valid_missing_and_corrupt_rows(tmp_path: Path):
    """
    Verifies manifest construction with Kaggle's exact unquoted-bracket CSV header,
    retaining VALID, MISSING, and CORRUPT images without silently dropping any row.
    """
    nih_root = tmp_path / "kaggle_nih"
    meta_dir = nih_root / "metadata"
    raw_dir = nih_root / "raw"
    manifests_dir = nih_root / "manifests"
    meta_dir.mkdir(parents=True)
    (raw_dir / "images_001" / "images").mkdir(parents=True)
    (raw_dir / "images_002" / "images").mkdir(parents=True)

    # Simulate exact Kaggle Data_Entry_2017.csv header with bracket commas + trailing comma
    csv_lines = [
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],",
        "00000001_000.png,Cardiomegaly|Effusion,0,1,58,M,PA,2682,2749,0.143,0.143,",
        "00000001_001.png,Cardiomegaly,1,1,59,M,AP,2500,2500,0.143,0.143,",
        "00000002_000.png,No Finding,0,2,34,F,PA,2048,2500,0.168,0.168,",
        "00000003_000.png,Hernia,0,3,81,F,PA,2021,2021,0.194,0.194,",
        "00000004_000.png,Pneumothorax|Mass,0,4,412,M,AP,3000,3000,0.139,0.139,",
        "00000005_000.png,Atelectasis,0,5,45,F,PA,2020,2020,0.150,0.150,",
    ]
    (meta_dir / "Data_Entry_2017.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    bbox_lines = [
        "Image Index,Finding Label,Bbox [x,y,w,h],,,",
        "00000001_000.png,Cardiomegaly,100,120,200,210,,,",
    ]
    (meta_dir / "BBox_List_2017.csv").write_text("\n".join(bbox_lines) + "\n", encoding="utf-8")

    # Official train_val_list.txt (patients 1, 2, 3, 4) and test_list.txt (patient 5)
    (meta_dir / "train_val_list.txt").write_text(
        "00000001_000.png\n00000001_001.png\n00000002_000.png\n00000003_000.png\n00000004_000.png\n",
        encoding="utf-8",
    )
    (meta_dir / "test_list.txt").write_text("00000005_000.png\n", encoding="utf-8")

    # Create VALID PNGs for patients 1 & 2, CORRUPT file for patient 3, and leave patients 4 & 5 MISSING
    (raw_dir / "images_001" / "images" / "00000001_000.png").write_bytes(_make_valid_png_bytes(1))
    (raw_dir / "images_001" / "images" / "00000001_001.png").write_bytes(_make_valid_png_bytes(2))
    (raw_dir / "images_002" / "images" / "00000002_000.png").write_bytes(_make_valid_png_bytes(3))
    (raw_dir / "images_002" / "images" / "00000003_000.png").write_bytes(b"corrupt_not_a_png")

    audit_out = tmp_path / "reports" / "nih_dataset_audit.json"
    manifest_df, audit = build_nih_manifest(
        nih_root=nih_root,
        audit_report_path=audit_out,
        val_fraction=0.25,
        seed=42,
        verify_pixel_decode=True,
    )

    assert len(manifest_df) == 6
    assert audit["record_accounting"]["rows_silently_dropped"] == 0
    assert audit["image_integrity_audit"]["VALID"] == 3
    assert audit["image_integrity_audit"]["CORRUPT"] == 1
    assert audit["image_integrity_audit"]["MISSING"] == 2
    assert audit["demographic_and_acquisition_distribution"]["age_anomaly_count"] == 1
    assert audit["record_accounting"]["bbox_annotated_images"] == 1

    # Verify patient 1's two images stayed in the same split
    p1_splits = manifest_df.loc[manifest_df["patient_id"].astype(str) == "1", "split"].unique()
    assert len(p1_splits) == 1
    assert manifest_df.loc[manifest_df["image_id"] == "00000005_000.png", "split"].iloc[0] == "test"

    # Strict full-count check must reject partial 6-row metadata when enabled
    with pytest.raises(ValueError, match="112120"):
        build_nih_manifest(nih_root=nih_root, audit_report_path=audit_out, strict_full_count=True)


def test_disk_space_guard_and_safe_nested_shard_extraction(tmp_path: Path):
    """Verifies disk-space preflight blocks small partitions (like A:\\ 30.8 GB) and blocks ZipSlip."""
    # Simulate A:\ with 30.76 GB free when full mode requires 60 GB
    fake_usage = type("Usage", (), {"total": int(406 * 1024**3), "used": int(375.24 * 1024**3), "free": int(30.76 * 1024**3)})()
    req_full = required_free_gb_for_plan("full", list(range(1, 13)), cleanup_zips=True)
    assert req_full >= 60.0

    with patch("scripts.download_kaggle_nih.shutil.disk_usage", return_value=fake_usage):
        with pytest.raises(InsufficientDiskSpaceError, match="available=30.76 GB"):
            check_disk_space(tmp_path, required_gb=req_full)

        # Incremental 2-shard batch mode (~11 GB required) succeeds safely on 30.76 GB drive!
        req_batch = required_free_gb_for_plan("shards", [1, 2], cleanup_zips=True)
        res = check_disk_space(tmp_path, required_gb=req_batch)
        assert res["preflight_passed"] is True

    # Test nested shard ZIP staging + ZipSlip protection
    staging_src = tmp_path / "local_dl"
    staging_src.mkdir()
    (staging_src / "Data_Entry_2017.csv").write_text("Image Index,Finding Labels\n", encoding="utf-8")

    shard_zip = staging_src / "images_001.zip"
    with zipfile.ZipFile(shard_zip, "w") as zf:
        zf.writestr("images/00000001_000.png", _make_valid_png_bytes(7))

    staged = stage_from_local_directory(
        source_dir=staging_src,
        nih_root=tmp_path / "nih_dest",
        shards=[1],
        cleanup_zips=False,
    )
    assert staged["extracted_shards"][0]["png_count"] == 1
    extracted_png = tmp_path / "nih_dest" / "raw" / "images_001" / "images" / "00000001_000.png"
    assert inspect_png_integrity(extracted_png, verify_pixel_decode=True)[0] == "VALID"

    # Malicious ZipSlip archive must be blocked
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../escaped.txt", "pwned")
    with pytest.raises(SecurityConfigurationError, match="Unsafe ZIP path traversal"):
        safe_extract_zip(evil_zip, tmp_path / "unpack_safe")


def test_official_nih_pipeline_has_zero_mimic_or_physionet_imports():
    """Verifies none of the official NIH CXR scripts import MIMIC or PhysioNet modules."""
    nih_scripts = [
        Path("scripts/download_kaggle_nih.py"),
        Path("scripts/split_nih_by_patient.py"),
        Path("scripts/build_nih_cxr_manifest.py"),
        Path("scripts/run_nih_cxr_inference.py"),
        Path("scripts/validate_nih_acquisition.py"),
        Path("scripts/validate_nih_batch_002_004.py"),
        Path("scripts/validate_nih_batch_005_008.py"),
        Path("scripts/validate_nih_batch_009_012.py"),
    ]
    forbidden_tokens = ("mimic", "physionet")

    for script_path in nih_scripts:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(tok in alias.name.lower() for tok in forbidden_tokens), (
                        f"Forbidden import '{alias.name}' in {script_path}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not any(tok in node.module.lower() for tok in forbidden_tokens), (
                    f"Forbidden from-import '{node.module}' in {script_path}"
                )
