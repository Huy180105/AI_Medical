"""
Deterministic Patient-Level Split Builder for NIH ChestX-ray14.

Enforces three non-negotiable methodological invariants:
1. Official `test_list.txt` from NIH ChestX-ray14 is preserved untouched as `split = "test"`.
2. `train_val_list.txt` is partitioned into `train` and `validate` STRICTLY by `patient_id`
   (`Patient ID`) using a fixed random seed and rare-pathology-first multi-label stratification.
3. Zero patient leakage across splits:
   (train_patients ∩ validate_patients) = (train_patients ∩ test_patients)
   = (validate_patients ∩ test_patients) = ∅,
   and every image belonging to a patient is assigned to that patient's single split.

Zero dependency on PhysioNet, MIMIC-CXR-JPG, or MIMIC-IV-ED.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_fusion import NIH_LABELS, PatientLeakageError, compute_file_sha256
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_patient_id(raw_pid: Any) -> str:
    """
    Canonicalizes NIH `Patient ID` values so integer `1`, float `1.0`, and zero-padded `"00000001"`
    all normalize deterministically to `"1"` (or stripped string if non-numeric).
    """
    if raw_pid is None or pd.isna(raw_pid):
        raise ValueError("Patient ID cannot be null or NaN")
    text = str(raw_pid).strip()
    if not text or text.lower() in {"nan", "none"}:
        raise ValueError(f"Invalid empty Patient ID: {raw_pid!r}")
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


def _read_image_list_file(filepath: Path) -> List[str]:
    if not filepath.exists():
        raise FileNotFoundError(f"Official NIH split list file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"Official NIH split list file is empty: {filepath}")
    return items


def _parse_finding_tokens(raw_finding: Any) -> List[str]:
    if raw_finding is None or pd.isna(raw_finding):
        raise ValueError("Finding Labels cannot be null")
    tokens = [tok.strip() for tok in str(raw_finding).split("|") if tok.strip()]
    if not tokens:
        raise ValueError(f"Empty Finding Labels entry: {raw_finding!r}")
    return tokens


def verify_zero_patient_leakage(df: pd.DataFrame, patient_col: str = "patient_id", split_col: str = "split") -> Dict[str, Any]:
    """
    Mathematically verifies pairwise disjointness of `patient_id` sets across train, validate, and test,
    and asserts every `patient_id` maps to at most 1 unique split.
    Raises `PatientLeakageError` immediately on any violation.
    """
    train_pts: Set[str] = set(df.loc[df[split_col] == "train", patient_col].astype(str).unique())
    val_pts: Set[str] = set(df.loc[df[split_col] == "validate", patient_col].astype(str).unique())
    test_pts: Set[str] = set(df.loc[df[split_col] == "test", patient_col].astype(str).unique())

    tr_va = train_pts & val_pts
    tr_te = train_pts & test_pts
    va_te = val_pts & test_pts

    if tr_va or tr_te or va_te:
        raise PatientLeakageError(
            "CRITICAL NIH PATIENT LEAKAGE DETECTED: "
            f"|train ∩ validate|={len(tr_va)}, |train ∩ test|={len(tr_te)}, |validate ∩ test|={len(va_te)}."
        )

    splits_per_patient = df.groupby(df[patient_col].astype(str))[split_col].nunique()
    leaking_patients = splits_per_patient[splits_per_patient > 1]
    if not leaking_patients.empty:
        raise PatientLeakageError(
            f"Found {len(leaking_patients)} patients spanning multiple splits: "
            f"{leaking_patients.index[:5].tolist()}"
        )

    return {
        "zero_patient_leakage_verified": True,
        "train_unique_patients": len(train_pts),
        "validate_unique_patients": len(val_pts),
        "test_unique_patients": len(test_pts),
        "train_validate_overlap": len(tr_va),
        "train_test_overlap": len(tr_te),
        "validate_test_overlap": len(va_te),
        "max_splits_per_patient": int(splits_per_patient.max()) if len(splits_per_patient) > 0 else 0,
    }


def assign_nih_patient_splits(
    data_entry_df: pd.DataFrame,
    train_val_images: Sequence[str],
    test_images: Sequence[str],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assigns deterministic `train`, `validate`, and `test` splits strictly at the `patient_id` level.

    1. Validates zero image overlap and zero patient overlap between official `train_val_list.txt`
       and `test_list.txt`.
    2. Keeps all patients in `test_list.txt` fixed in `test`.
    3. Partitions `train_val` patients into `train` and `validate` using `seed` and rare-pathology-first
       patient stratification so all 14 NIH classes retain positive patients in both splits.
    """
    if not (0.01 <= val_fraction <= 0.50):
        raise ValueError(f"val_fraction={val_fraction} must be within [0.01, 0.50]")

    required_cols = {"Image Index", "Patient ID", "Finding Labels"}
    missing_cols = required_cols - set(data_entry_df.columns)
    if missing_cols:
        raise ValueError(f"Data_Entry_2017.csv missing required columns: {missing_cols}")

    train_val_set = set(train_val_images)
    test_set = set(test_images)
    img_overlap = train_val_set & test_set
    if img_overlap:
        raise PatientLeakageError(
            f"Official train_val_list.txt and test_list.txt share {len(img_overlap)} image IDs: "
            f"{sorted(list(img_overlap))[:5]}"
        )

    work = pd.DataFrame({
        "image_id": data_entry_df["Image Index"].astype(str).str.strip(),
        "patient_id": [normalize_patient_id(v) for v in data_entry_df["Patient ID"]],
        "finding_labels": data_entry_df["Finding Labels"].astype(str).str.strip(),
    })

    if work["image_id"].duplicated().any():
        dups = work.loc[work["image_id"].duplicated(), "image_id"].unique()[:5].tolist()
        raise ValueError(f"Duplicate Image Index rows in Data_Entry_2017.csv: {dups}")

    in_train_val = work["image_id"].isin(train_val_set)
    in_test = work["image_id"].isin(test_set)
    unlisted = work[~(in_train_val | in_test)]
    if not unlisted.empty:
        raise ValueError(
            f"{len(unlisted)} images in Data_Entry_2017.csv are absent from both train_val_list.txt "
            f"and test_list.txt (sample: {unlisted['image_id'].head(5).tolist()})."
        )

    train_val_patients = set(work.loc[in_train_val, "patient_id"].unique())
    test_patients = set(work.loc[in_test, "patient_id"].unique())
    official_pt_overlap = train_val_patients & test_patients
    if official_pt_overlap:
        raise PatientLeakageError(
            f"CRITICAL PATIENT LEAKAGE between train_val_list.txt and test_list.txt: "
            f"{len(official_pt_overlap)} patients appear in both! Sample: {sorted(list(official_pt_overlap))[:5]}"
        )

    # Encode 14 binary disease flags on train_val rows for rare-label-first patient stratification
    tv_rows = work[in_train_val].copy()
    valid_nih_set = set(NIH_LABELS)
    for lbl in NIH_LABELS:
        tv_rows[lbl] = 0

    for idx, raw_lbls in tv_rows["finding_labels"].items():
        for tok in _parse_finding_tokens(raw_lbls):
            if tok == "No Finding":
                continue
            if tok not in valid_nih_set:
                raise ValueError(f"Unknown pathology token '{tok}' in Finding Labels")
            tv_rows.at[idx, tok] = 1

    # Aggregate to 1 row per patient_id in deterministic sorted patient order
    pt_label_max = tv_rows.groupby("patient_id", sort=True)[NIH_LABELS].max()
    pt_img_counts = tv_rows.groupby("patient_id", sort=True)["image_id"].count()

    # Order NIH labels from rarest to most common patient prevalence so rare classes (e.g. Hernia)
    # get stratified cleanly without losing all validation positives
    label_patient_totals = pt_label_max.sum(axis=0).to_dict()
    rarity_ordered_labels = sorted(NIH_LABELS, key=lambda k: (label_patient_totals[k], k))

    patient_strata: Dict[str, List[str]] = {}
    patient_primary_stratum: Dict[str, str] = {}
    for pid in sorted(pt_label_max.index, key=lambda x: (int(x) if x.isdigit() else float("inf"), x)):
        row = pt_label_max.loc[pid]
        chosen = "No_Finding"
        for rare_lbl in rarity_ordered_labels:
            if int(row[rare_lbl]) == 1:
                chosen = rare_lbl
                break
        patient_primary_stratum[pid] = chosen
        patient_strata.setdefault(chosen, []).append(pid)

    rng = np.random.RandomState(seed)
    val_patient_set: Set[str] = set()
    train_patient_set: Set[str] = set()

    for stratum_name in sorted(patient_strata.keys()):
        pids = list(patient_strata[stratum_name])
        rng.shuffle(pids)
        n_total = len(pids)
        if n_total == 1:
            # Keep single-patient stratum in train
            n_val = 0
        else:
            n_val = max(1, int(round(n_total * val_fraction)))
            n_val = min(n_val, n_total - 1)
        val_patient_set.update(pids[:n_val])
        train_patient_set.update(pids[n_val:])

    # Build patient_splits table covering all patients (train, validate, test)
    all_pt_counts = work.groupby("patient_id", sort=True)["image_id"].count().to_dict()

    # Also compute positive_label_count per patient across full dataset
    all_label_flags = pd.DataFrame(0, index=work.index, columns=NIH_LABELS)
    for idx, raw_lbls in work["finding_labels"].items():
        for tok in _parse_finding_tokens(raw_lbls):
            if tok in valid_nih_set:
                all_label_flags.at[idx, tok] = 1
    all_label_flags["patient_id"] = work["patient_id"]
    all_pt_pos_label_cnt = all_label_flags.groupby("patient_id")[NIH_LABELS].max().sum(axis=1).to_dict()

    patient_records: List[Dict[str, Any]] = []
    all_unique_pids = sorted(
        work["patient_id"].unique(),
        key=lambda x: (int(x) if x.isdigit() else float("inf"), x),
    )
    for pid in all_unique_pids:
        if pid in train_patient_set:
            sp = "train"
            stratum = patient_primary_stratum[pid]
        elif pid in val_patient_set:
            sp = "validate"
            stratum = patient_primary_stratum[pid]
        elif pid in test_patients:
            sp = "test"
            stratum = "official_test_list"
        else:
            raise RuntimeError(f"Unassigned patient_id={pid}")

        patient_records.append({
            "patient_id": pid,
            "split": sp,
            "image_count": int(all_pt_counts[pid]),
            "positive_pathology_count": int(all_pt_pos_label_cnt[pid]),
            "primary_stratum": stratum,
        })

    patient_splits_df = pd.DataFrame(patient_records)

    # Map splits back onto image-level work table and run hard leakage assertion
    pid_to_split = dict(zip(patient_splits_df["patient_id"], patient_splits_df["split"]))
    work["split"] = work["patient_id"].map(pid_to_split)
    leakage_proof = verify_zero_patient_leakage(work, patient_col="patient_id", split_col="split")

    # Verify official test_list exact match
    assigned_test_images = set(work.loc[work["split"] == "test", "image_id"])
    if assigned_test_images != test_set.intersection(set(work["image_id"])):
        raise RuntimeError("Official test_list.txt image membership mismatch after patient split assignment!")

    split_image_counts = work["split"].value_counts().to_dict()
    audit: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(seed),
        "val_fraction_of_train_val": float(val_fraction),
        "official_test_list_preserved": True,
        "total_images": int(len(work)),
        "total_unique_patients": int(len(patient_splits_df)),
        "images_by_split": {
            "train": int(split_image_counts.get("train", 0)),
            "validate": int(split_image_counts.get("validate", 0)),
            "test": int(split_image_counts.get("test", 0)),
        },
        "leakage_verification": leakage_proof,
        "patient_ids_by_split": {
            "train": patient_splits_df.loc[patient_splits_df["split"] == "train", "patient_id"].tolist(),
            "validate": patient_splits_df.loc[patient_splits_df["split"] == "validate", "patient_id"].tolist(),
            "test": patient_splits_df.loc[patient_splits_df["split"] == "test", "patient_id"].tolist(),
        },
    }
    return patient_splits_df, audit


def sample_patient_stratified_subset(
    manifest_df: pd.DataFrame,
    max_patients_per_split: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Samples up to `max_patients_per_split` patients from each split (`train`, `validate`, `test`)
    STRICTLY at the `patient_id` level (retaining all images of each sampled patient),
    prioritizing rare-pathology coverage so pilot subsets preserve multi-label diversity
    without ever leaking a patient across splits.
    """
    if max_patients_per_split <= 0:
        raise ValueError("max_patients_per_split must be >= 1")

    verify_zero_patient_leakage(manifest_df, patient_col="patient_id", split_col="split")
    rng = np.random.RandomState(seed)
    present_labels = [lbl for lbl in NIH_LABELS if lbl in manifest_df.columns]

    kept_frames: List[pd.DataFrame] = []
    for sp in ("train", "validate", "test"):
        sp_df = manifest_df[manifest_df["split"] == sp]
        if sp_df.empty:
            continue
        unique_pids = sorted(
            sp_df["patient_id"].astype(str).unique(),
            key=lambda x: (int(x) if x.isdigit() else float("inf"), x),
        )
        if len(unique_pids) <= max_patients_per_split:
            kept_frames.append(sp_df)
            continue

        selected_pids: List[str] = []
        selected_set: Set[str] = set()

        # First guarantee >=1 positive patient per present pathology if available
        if present_labels:
            pt_lbl = sp_df.groupby(sp_df["patient_id"].astype(str))[present_labels].max()
            for lbl in sorted(present_labels, key=lambda c: int(pt_lbl[c].sum())):
                if len(selected_pids) >= max_patients_per_split:
                    break
                pos_pids = [p for p in pt_lbl.index[pt_lbl[lbl] == 1] if p not in selected_set]
                if pos_pids:
                    pick = str(rng.choice(pos_pids))
                    selected_pids.append(pick)
                    selected_set.add(pick)

        remaining = [p for p in unique_pids if p not in selected_set]
        rng.shuffle(remaining)
        needed = max_patients_per_split - len(selected_pids)
        if needed > 0:
            selected_pids.extend(remaining[:needed])

        sub = sp_df[sp_df["patient_id"].astype(str).isin(set(selected_pids))].copy()
        kept_frames.append(sub)

    out_df = pd.concat(kept_frames, ignore_index=True) if kept_frames else manifest_df.iloc[0:0].copy()
    verify_zero_patient_leakage(out_df, patient_col="patient_id", split_col="split")
    return out_df


def build_and_save_patient_splits(
    metadata_dir: Path = Path("data/kaggle_nih/metadata"),
    manifests_dir: Path = Path("data/kaggle_nih/manifests"),
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """CLI entrypoint helper: reads NIH metadata files and writes `patient_splits.csv` + audit JSON."""
    data_entry_path = Path(metadata_dir) / "Data_Entry_2017.csv"
    train_val_path = Path(metadata_dir) / "train_val_list.txt"
    test_path = Path(metadata_dir) / "test_list.txt"

    if not data_entry_path.exists():
        raise FileNotFoundError(f"Missing NIH metadata CSV: {data_entry_path}")

    data_entry_df = pd.read_csv(data_entry_path)
    train_val_imgs = _read_image_list_file(train_val_path)
    test_imgs = _read_image_list_file(test_path)

    patient_splits_df, audit = assign_nih_patient_splits(
        data_entry_df=data_entry_df,
        train_val_images=train_val_imgs,
        test_images=test_imgs,
        val_fraction=val_fraction,
        seed=seed,
    )
    audit["input_sha256"] = {
        "Data_Entry_2017.csv": compute_file_sha256(data_entry_path),
        "train_val_list.txt": compute_file_sha256(train_val_path),
        "test_list.txt": compute_file_sha256(test_path),
    }

    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_csv_path = manifests_dir / "patient_splits.csv"
    audit_json_path = manifests_dir / "patient_split_audit.json"

    patient_splits_df.to_csv(splits_csv_path, index=False)
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved patient-grouped NIH splits to %s (train=%d pts, validate=%d pts, test=%d pts).",
        splits_csv_path,
        audit["leakage_verification"]["train_unique_patients"],
        audit["leakage_verification"]["validate_unique_patients"],
        audit["leakage_verification"]["test_unique_patients"],
    )
    return splits_csv_path, audit_json_path, audit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split NIH ChestX-ray14 train_val_list.txt into train/validate strictly by Patient ID."
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/kaggle_nih/metadata"),
        help="Directory containing Data_Entry_2017.csv, train_val_list.txt, and test_list.txt.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/kaggle_nih/manifests"),
        help="Output directory for patient_splits.csv and patient_split_audit.json.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of train_val patients assigned to validate split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for patient split (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    build_and_save_patient_splits(
        metadata_dir=args.metadata_dir,
        manifests_dir=args.manifests_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
