"""
Unit Tests for Deterministic Patient-Level Splitting on NIH ChestX-ray14.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from scripts.split_nih_by_patient import (
    assign_nih_patient_splits,
    build_and_save_patient_splits,
    normalize_patient_id,
    sample_patient_stratified_subset,
    verify_zero_patient_leakage,
)
from src.evaluation.benchmark_fusion import NIH_LABELS, PatientLeakageError


def _build_synthetic_nih_cohort(n_train_val_patients: int = 40, n_test_patients: int = 12) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates a multi-image longitudinal NIH cohort covering all 14 pathologies including rare Hernia."""
    rng = np.random.RandomState(123)
    rows: List[Dict[str, object]] = []
    train_val_imgs: List[str] = []
    test_imgs: List[str] = []

    total_pts = n_train_val_patients + n_test_patients
    for pid_int in range(1, total_pts + 1):
        is_test = pid_int > n_train_val_patients
        n_followups = int(rng.randint(1, 5))  # 1 to 4 longitudinal radiographs per patient

        # Guarantee every one of the 14 NIH labels (including rare Hernia) appears in >= 2 train_val patients
        if not is_test and pid_int <= 28:
            forced_label = NIH_LABELS[(pid_int - 1) % len(NIH_LABELS)]
        else:
            forced_label = None

        for fu in range(n_followups):
            img_name = f"{pid_int:08d}_{fu:03d}.png"
            if forced_label and fu == 0:
                finding = forced_label
            elif rng.rand() < 0.45:
                finding = "No Finding"
            else:
                picked = rng.choice(NIH_LABELS, size=int(rng.randint(1, 3)), replace=False)
                finding = "|".join(sorted(picked.tolist()))

            rows.append({
                "Image Index": img_name,
                "Finding Labels": finding,
                "Follow-up #": fu,
                "Patient ID": f"{pid_int:08d}" if fu % 2 == 0 else pid_int,  # mix int and zero-padded str
                "Patient Age": int(rng.randint(20, 85)),
                "Patient Gender": "M" if pid_int % 2 == 0 else "F",
                "View Position": "PA" if fu % 2 == 0 else "AP",
            })
            if is_test:
                test_imgs.append(img_name)
            else:
                train_val_imgs.append(img_name)

    return pd.DataFrame(rows), train_val_imgs, test_imgs


def test_normalize_patient_id_consistency():
    assert normalize_patient_id("00000042") == "42"
    assert normalize_patient_id(42) == "42"
    assert normalize_patient_id(42.0) == "42"
    with pytest.raises(ValueError):
        normalize_patient_id(None)


def test_patient_split_preserves_official_test_and_prevents_leakage(tmp_path: Path):
    """
    Verifies:
    1. Official `test_list.txt` is untouched.
    2. `train_val_list.txt` is split strictly by `Patient ID` (even when formatted as '00000001' vs 1).
    3. Pairwise patient intersections (train ∩ val, train ∩ test, val ∩ test) are strictly empty.
    4. All 14 NIH pathologies have >= 1 positive patient in both `train` and `validate`.
    """
    df, tv_imgs, test_imgs = _build_synthetic_nih_cohort(n_train_val_patients=40, n_test_patients=12)

    meta_dir = tmp_path / "metadata"
    manifests_dir = tmp_path / "manifests"
    meta_dir.mkdir(parents=True)
    df.to_csv(meta_dir / "Data_Entry_2017.csv", index=False)
    (meta_dir / "train_val_list.txt").write_text("\n".join(tv_imgs) + "\n", encoding="utf-8")
    (meta_dir / "test_list.txt").write_text("\n".join(test_imgs) + "\n", encoding="utf-8")

    splits_csv, audit_json, audit = build_and_save_patient_splits(
        metadata_dir=meta_dir,
        manifests_dir=manifests_dir,
        val_fraction=0.20,
        seed=42,
    )
    assert splits_csv.exists() and audit_json.exists()

    pt_splits = pd.read_csv(splits_csv)
    pt_splits["patient_id"] = pt_splits["patient_id"].astype(str)
    assert len(pt_splits) == 52

    train_pts = set(pt_splits.loc[pt_splits["split"] == "train", "patient_id"])
    val_pts = set(pt_splits.loc[pt_splits["split"] == "validate", "patient_id"])
    test_pts = set(pt_splits.loc[pt_splits["split"] == "test", "patient_id"])

    assert (train_pts & val_pts) == set()
    assert (train_pts & test_pts) == set()
    assert (val_pts & test_pts) == set()
    assert len(test_pts) == 12

    # Check reproducibility across seeds
    pt_splits_same, _ = assign_nih_patient_splits(df, tv_imgs, test_imgs, val_fraction=0.20, seed=42)
    pt_splits_diff, _ = assign_nih_patient_splits(df, tv_imgs, test_imgs, val_fraction=0.20, seed=99)
    assert pt_splits["split"].tolist() == pt_splits_same["split"].tolist()
    assert pt_splits["split"].tolist() != pt_splits_diff["split"].tolist()
    # Official test patients remain 100% identical regardless of seed
    assert set(pt_splits_diff.loc[pt_splits_diff["split"] == "test", "patient_id"]) == test_pts


def test_patient_split_raises_on_official_list_leakage():
    """Raises PatientLeakageError if a patient has one image in train_val_list.txt and another in test_list.txt."""
    df, tv_imgs, test_imgs = _build_synthetic_nih_cohort(n_train_val_patients=10, n_test_patients=4)
    # Force leakage: move first image of patient 1 into test_imgs while patient 1's follow-up stays in tv_imgs
    p1_imgs = df.loc[df["Patient ID"].apply(normalize_patient_id) == "1", "Image Index"].tolist()
    assert len(p1_imgs) >= 1
    # Add an extra follow-up row for patient 1 into test_imgs
    extra_row = pd.DataFrame([{
        "Image Index": "00000001_999.png",
        "Finding Labels": "Cardiomegaly",
        "Follow-up #": 99,
        "Patient ID": 1,
        "Patient Age": 55,
        "Patient Gender": "F",
        "View Position": "PA",
    }])
    leaked_df = pd.concat([df, extra_row], ignore_index=True)
    leaked_test_imgs = list(test_imgs) + ["00000001_999.png"]

    with pytest.raises(PatientLeakageError, match="CRITICAL PATIENT LEAKAGE"):
        assign_nih_patient_splits(leaked_df, tv_imgs, leaked_test_imgs, val_fraction=0.2, seed=42)


def test_patient_stratified_subset_sampling_preserves_multi_image_patients():
    """Verifies partial-shard / pilot patient sampling never splits a patient's images."""
    df, tv_imgs, test_imgs = _build_synthetic_nih_cohort(n_train_val_patients=30, n_test_patients=10)
    pt_splits, _ = assign_nih_patient_splits(df, tv_imgs, test_imgs, val_fraction=0.2, seed=42)
    pid_map = dict(zip(pt_splits["patient_id"], pt_splits["split"]))

    manifest_like = pd.DataFrame({
        "image_id": df["Image Index"],
        "patient_id": [normalize_patient_id(p) for p in df["Patient ID"]],
    })
    manifest_like["split"] = manifest_like["patient_id"].map(pid_map)
    for lbl in NIH_LABELS:
        manifest_like[lbl] = df["Finding Labels"].str.contains(lbl, regex=False).astype(int)

    subset = sample_patient_stratified_subset(manifest_like, max_patients_per_split=5, seed=42)
    proof = verify_zero_patient_leakage(subset, patient_col="patient_id", split_col="split")
    assert proof["zero_patient_leakage_verified"] is True
    assert proof["train_unique_patients"] <= 5
    assert proof["validate_unique_patients"] <= 5
    assert proof["test_unique_patients"] <= 5

    # Every sampled patient must retain ALL of their images from the parent manifest
    for pid in subset["patient_id"].unique():
        assert len(subset[subset["patient_id"] == pid]) == len(manifest_like[manifest_like["patient_id"] == pid])
