"""
NIH ChestX-ray14 Canonical Multi-Label Manifest Builder & Dataset Auditor.

Reads official Kaggle NIH ChestX-ray14 metadata (`Data_Entry_2017.csv`, optional `BBox_List_2017.csv`,
`train_val_list.txt`, `test_list.txt`) and physical image shards (`data/kaggle_nih/raw/images_001..012`),
producing:
1. `data/kaggle_nih/manifests/nih_manifest.csv`
2. `data/kaggle_nih/manifests/data_audit.json`
3. `reports/nih_dataset_audit.json`

Methodological & Clinical Data Governance Rules:
- Exact 14 binary {0, 1} pathology columns (`NIH_LABELS`).
- `"No Finding"` encodes all 14 disease columns as `0` (`no_finding = 1`) — NEVER a 15th disease class.
- Rejects CheXpert `-1` uncertainty values and unknown label names.
- Never silently drops missing or corrupted radiographs: retains every row from `Data_Entry_2017.csv`
  with explicit `image_status` in `{"VALID", "MISSING", "CORRUPT"}`.
- Audits sample and patient counts by pathology, split, gender, age group, view position, and shard.
- Zero dependency on PhysioNet, MIMIC-CXR-JPG, or MIMIC-IV-ED.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_nih_by_patient import (
    _read_image_list_file,
    assign_nih_patient_splits,
    normalize_patient_id,
    verify_zero_patient_leakage,
)
from src.evaluation.benchmark_fusion import InvalidLabelError, NIH_LABELS, compute_file_sha256
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_FULL_NIH_IMAGE_COUNT = 112_120
EXPECTED_FULL_NIH_PATIENT_COUNT = 30_805
NIH_LABEL_SET = set(NIH_LABELS)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_finding_labels(raw_finding: Any) -> Tuple[Dict[str, int], int, int]:
    """
    Parses pipe-delimited NIH `Finding Labels` string into:
    - 14 binary {0, 1} pathology flags (`NIH_LABELS`)
    - `no_finding` flag (1 iff all 14 diseases are 0)
    - `disease_count` (number of positive NIH pathologies in [0..14])

    Strictly rejects unknown labels, empty labels, or CheXpert `-1` uncertainty tokens.
    """
    if raw_finding is None or pd.isna(raw_finding):
        raise InvalidLabelError("Finding Labels cannot be null or NaN")

    text = str(raw_finding).strip()
    if not text:
        raise InvalidLabelError("Finding Labels cannot be empty")

    tokens = [tok.strip() for tok in text.split("|") if tok.strip()]
    if not tokens:
        raise InvalidLabelError(f"Malformed Finding Labels string: {raw_finding!r}")

    binary_map = {lbl: 0 for lbl in NIH_LABELS}
    saw_no_finding = False

    for tok in tokens:
        if tok in {"-1", "-1.0", "Uncertain"}:
            raise InvalidLabelError(
                f"Forbidden CheXpert uncertainty token '{tok}' in NIH Finding Labels: {raw_finding!r}"
            )
        if tok == "No Finding":
            saw_no_finding = True
            continue
        if tok not in NIH_LABEL_SET:
            raise InvalidLabelError(
                f"Unknown pathology label '{tok}' in Finding Labels ({raw_finding!r}). "
                f"Valid NIH-14 labels: {NIH_LABELS}"
            )
        binary_map[tok] = 1

    disease_count = sum(binary_map.values())
    if saw_no_finding and disease_count > 0:
        raise InvalidLabelError(
            f"Contradictory Finding Labels '{raw_finding}': 'No Finding' cannot co-occur with positive diseases."
        )

    no_finding = 1 if disease_count == 0 else 0
    return binary_map, no_finding, disease_count


def parse_nih_patient_age(raw_age: Any) -> Tuple[Optional[float], str, int]:
    """
    Parses NIH `Patient Age` supporting both Kaggle integer ages (e.g. `58`) and legacy NIH unit strings
    (`058Y`, `009M`, `014D`), while flagging known upstream NIH `Data_Entry_2017.csv` typo ages (> 120 years)
    without dropping the record.

    Returns: `(age_years, age_group, age_anomaly_flag)`
    """
    if raw_age is None or pd.isna(raw_age):
        return None, "UNKNOWN", 1

    s = str(raw_age).strip().upper()
    if not s:
        return None, "UNKNOWN", 1

    age_years: Optional[float] = None
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*([YMD])?$", s)
    if m:
        val = float(m.group(1))
        unit = m.group(2) or "Y"
        if unit == "Y":
            age_years = val
        elif unit == "M":
            age_years = round(val / 12.0, 3)
        elif unit == "D":
            age_years = round(val / 365.25, 4)
    else:
        return None, "UNKNOWN", 1

    if age_years < 0.0 or age_years > 120.0:
        return round(age_years, 2), "ANOMALY_OUT_OF_RANGE", 1

    if age_years < 18.0:
        bucket = "0-17"
    elif age_years < 40.0:
        bucket = "18-39"
    elif age_years < 60.0:
        bucket = "40-59"
    elif age_years < 80.0:
        bucket = "60-79"
    else:
        bucket = "80+"
    return round(age_years, 2), bucket, 0


def index_raw_nih_images(raw_dir: Path) -> Dict[str, Tuple[Path, str]]:
    """
    Indexes physical `.png` files under `raw_dir` (e.g. `raw/images_001/images/00000001_000.png`).
    Returns mapping: `image_filename -> (resolved_path, shard_id)`.
    Raises ValueError if duplicate filenames exist across distinct shards.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return {}

    index: Dict[str, Tuple[Path, str]] = {}
    shard_re = re.compile(r"(images_\d{3})", re.IGNORECASE)

    for png_path in sorted(raw_dir.rglob("*.png")):
        if not png_path.is_file():
            continue
        fname = png_path.name
        rel_parts = png_path.relative_to(raw_dir).parts
        shard_id = "unsharded"
        for part in rel_parts:
            m = shard_re.match(part)
            if m:
                shard_id = m.group(1).lower()
                break
        if fname in index and index[fname][0].resolve() != png_path.resolve():
            raise ValueError(
                f"Duplicate image filename '{fname}' found in multiple locations: "
                f"{index[fname][0]} and {png_path}"
            )
        index[fname] = (png_path, shard_id)

    return index


def inspect_png_integrity(filepath: Path, verify_pixel_decode: bool = False) -> Tuple[str, str]:
    """
    Audits physical PNG file without deleting or mutating it.
    Returns `(image_status, status_detail)` where `image_status` in `{"VALID", "CORRUPT"}`.
    """
    try:
        size = filepath.stat().st_size
        if size == 0:
            return "CORRUPT", "zero_byte_file"

        with open(filepath, "rb") as f:
            sig = f.read(8)
        if sig != PNG_SIGNATURE:
            return "CORRUPT", "invalid_png_signature"

        with Image.open(filepath) as img:
            img.verify()

        if verify_pixel_decode:
            with Image.open(filepath) as img2:
                img2.load()

        return "VALID", "ok"
    except Exception as exc:
        return "CORRUPT", f"decode_error:{type(exc).__name__}"


def _extract_optional_numeric_col(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def _load_bbox_annotations(bbox_csv_path: Optional[Path]) -> Dict[str, List[str]]:
    """Loads optional `BBox_List_2017.csv` into `image_id -> sorted list of localized labels`."""
    if bbox_csv_path is None or not Path(bbox_csv_path).exists():
        return {}
    bbox_df = pd.read_csv(bbox_csv_path)
    img_col = "Image Index" if "Image Index" in bbox_df.columns else bbox_df.columns[0]
    lbl_col = (
        "Finding Label"
        if "Finding Label" in bbox_df.columns
        else ("Finding Labels" if "Finding Labels" in bbox_df.columns else bbox_df.columns[1])
    )
    mapping: Dict[str, set] = {}
    for _, row in bbox_df.iterrows():
        img_id = str(row[img_col]).strip()
        lbl = str(row[lbl_col]).strip()
        if img_id and lbl and lbl.lower() != "nan":
            mapping.setdefault(img_id, set()).add(lbl)
    return {k: sorted(list(v)) for k, v in mapping.items()}


def build_nih_manifest(
    nih_root: Path = Path("data/kaggle_nih"),
    metadata_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    manifests_dir: Optional[Path] = None,
    audit_report_path: Path = Path("reports/nih_dataset_audit.json"),
    val_fraction: float = 0.15,
    seed: int = 42,
    verify_pixel_decode: bool = False,
    strict_full_count: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs `nih_manifest.csv` and comprehensive dataset audit JSONs (`manifests/data_audit.json`
    and `reports/nih_dataset_audit.json`).
    """
    nih_root = Path(nih_root)
    metadata_dir = Path(metadata_dir) if metadata_dir else (nih_root / "metadata")
    raw_dir = Path(raw_dir) if raw_dir else (nih_root / "raw")
    manifests_dir = Path(manifests_dir) if manifests_dir else (nih_root / "manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    data_entry_path = metadata_dir / "Data_Entry_2017.csv"
    if not data_entry_path.exists():
        raise FileNotFoundError(
            f"Data_Entry_2017.csv not found at {data_entry_path}. "
            "Run `python scripts/download_kaggle_nih.py --mode metadata_only` first."
        )

    raw_df = pd.read_csv(data_entry_path)
    # Drop trailing empty column ('Unnamed: 11') present in Kaggle's official Data_Entry_2017.csv
    unnamed_cols = [c for c in raw_df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        raw_df = raw_df.drop(columns=unnamed_cols)

    if strict_full_count and len(raw_df) != EXPECTED_FULL_NIH_IMAGE_COUNT:
        raise ValueError(
            f"Strict full-dataset count failed: expected {EXPECTED_FULL_NIH_IMAGE_COUNT} rows in "
            f"Data_Entry_2017.csv, got {len(raw_df)}."
        )

    gender_col = "Patient Gender" if "Patient Gender" in raw_df.columns else "Patient Sex"
    required_cols = {"Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", gender_col, "View Position"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Data_Entry_2017.csv is missing required columns: {missing}")

    # Load or generate patient-level splits
    patient_splits_path = manifests_dir / "patient_splits.csv"
    train_val_path = metadata_dir / "train_val_list.txt"
    test_path = metadata_dir / "test_list.txt"

    if patient_splits_path.exists():
        pt_splits_df = pd.read_csv(patient_splits_path)
        pt_splits_df["patient_id"] = [normalize_patient_id(p) for p in pt_splits_df["patient_id"]]
    else:
        train_val_imgs = _read_image_list_file(train_val_path)
        test_imgs = _read_image_list_file(test_path)
        pt_splits_df, split_audit = assign_nih_patient_splits(
            data_entry_df=raw_df,
            train_val_images=train_val_imgs,
            test_images=test_imgs,
            val_fraction=val_fraction,
            seed=seed,
        )
        pt_splits_df.to_csv(patient_splits_path, index=False)
        with open(manifests_dir / "patient_split_audit.json", "w", encoding="utf-8") as f:
            json.dump(split_audit, f, indent=2, ensure_ascii=False)

    pid_to_split = dict(zip(pt_splits_df["patient_id"], pt_splits_df["split"]))

    # Optional bounding-box metadata
    bbox_path = metadata_dir / "BBox_List_2017.csv"
    bbox_map = _load_bbox_annotations(bbox_path if bbox_path.exists() else None)

    # Index raw PNGs across downloaded shards
    png_index = index_raw_nih_images(raw_dir)

    orig_w = _extract_optional_numeric_col(raw_df, ["OriginalImage[Width", "OriginalImageWidth", "orig_width"])
    orig_h = _extract_optional_numeric_col(raw_df, ["Height]", "OriginalImageHeight", "orig_height"])
    spac_x = _extract_optional_numeric_col(raw_df, ["OriginalImagePixelSpacing[x", "PixelSpacingX", "pixel_spacing_x"])
    spac_y = _extract_optional_numeric_col(raw_df, ["y]", "PixelSpacingY", "pixel_spacing_y"])

    records: List[Dict[str, Any]] = []
    for row_idx, row in raw_df.iterrows():
        image_id = str(row["Image Index"]).strip()
        if not image_id or image_id.lower() == "nan":
            raise ValueError(f"Empty Image Index at row {row_idx}")

        patient_id = normalize_patient_id(row["Patient ID"])
        if patient_id not in pid_to_split:
            raise ValueError(
                f"Patient ID '{patient_id}' (image '{image_id}') not found in patient_splits.csv! "
                "Rebuild splits via `scripts/split_nih_by_patient.py`."
            )
        split = pid_to_split[patient_id]

        binary_labels, no_finding, disease_count = encode_finding_labels(row["Finding Labels"])
        age_years, age_group, age_anomaly = parse_nih_patient_age(row["Patient Age"])

        gender_raw = str(row[gender_col]).strip().upper() if pd.notna(row[gender_col]) else "UNKNOWN"
        gender = gender_raw if gender_raw in {"M", "F"} else "UNKNOWN"
        view_pos = str(row["View Position"]).strip().upper() if pd.notna(row["View Position"]) else "UNKNOWN"
        follow_up = int(row["Follow-up #"]) if pd.notna(row["Follow-up #"]) else 0

        if image_id in png_index:
            phys_path, shard_id = png_index[image_id]
            img_status, status_detail = inspect_png_integrity(phys_path, verify_pixel_decode=verify_pixel_decode)
            try:
                rel_img_path = phys_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                rel_img_path = phys_path.resolve().as_posix()
        else:
            img_status = "MISSING"
            status_detail = "shard_not_downloaded_or_file_absent"
            rel_img_path = ""
            shard_id = ""

        bbox_lbls = bbox_map.get(image_id, [])

        rec: Dict[str, Any] = {
            "image_id": image_id,
            "patient_id": patient_id,
            "split": split,
            "image_path": rel_img_path,
            "shard_id": shard_id,
            "image_status": img_status,
            "status_detail": status_detail,
            "finding_labels_raw": str(row["Finding Labels"]).strip(),
            "no_finding": no_finding,
            "disease_count": disease_count,
            "follow_up_num": follow_up,
            "patient_age_raw": row["Patient Age"],
            "patient_age_years": age_years,
            "age_group": age_group,
            "age_anomaly_flag": age_anomaly,
            "patient_gender": gender,
            "view_position": view_pos,
            "orig_width": None if pd.isna(orig_w.iloc[row_idx]) else int(orig_w.iloc[row_idx]),
            "orig_height": None if pd.isna(orig_h.iloc[row_idx]) else int(orig_h.iloc[row_idx]),
            "pixel_spacing_x": None if pd.isna(spac_x.iloc[row_idx]) else round(float(spac_x.iloc[row_idx]), 6),
            "pixel_spacing_y": None if pd.isna(spac_y.iloc[row_idx]) else round(float(spac_y.iloc[row_idx]), 6),
            "has_bbox": 1 if bbox_lbls else 0,
            "bbox_labels": "|".join(bbox_lbls),
            "label_source": "NIH_CHESTXRAY14_NLP_2017",
            "is_synthetic": False,
        }
        for lbl in NIH_LABELS:
            rec[lbl] = int(binary_labels[lbl])
        records.append(rec)

    manifest_df = pd.DataFrame(records)
    if manifest_df["image_id"].duplicated().any():
        dups = manifest_df.loc[manifest_df["image_id"].duplicated(), "image_id"].unique()[:5].tolist()
        raise ValueError(f"Duplicate image_id detected in constructed NIH manifest: {dups}")

    leakage_audit = verify_zero_patient_leakage(manifest_df, patient_col="patient_id", split_col="split")

    manifest_csv_path = manifests_dir / "nih_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)

    # Compute comprehensive audit statistics
    per_label_stats: Dict[str, Any] = {}
    total_n = max(1, len(manifest_df))
    for lbl in NIH_LABELS:
        pos_total = int(manifest_df[lbl].sum())
        pos_pts = int(manifest_df.loc[manifest_df[lbl] == 1, "patient_id"].nunique())
        by_split = {
            sp: int(manifest_df.loc[manifest_df["split"] == sp, lbl].sum())
            for sp in ("train", "validate", "test")
        }
        valid_on_disk_pos = int(
            manifest_df.loc[(manifest_df["image_status"] == "VALID") & (manifest_df[lbl] == 1), lbl].sum()
        )
        per_label_stats[lbl] = {
            "positive_images": pos_total,
            "negative_images": int(len(manifest_df) - pos_total),
            "prevalence_pct": round(100.0 * pos_total / total_n, 4),
            "positive_unique_patients": pos_pts,
            "positive_images_by_split": by_split,
            "valid_on_disk_positive_images": valid_on_disk_pos,
        }

    status_counts = manifest_df["image_status"].value_counts().to_dict()
    split_counts = manifest_df["split"].value_counts().to_dict()
    gender_counts = manifest_df["patient_gender"].value_counts().to_dict()
    age_group_counts = manifest_df["age_group"].value_counts().to_dict()
    view_counts = manifest_df["view_position"].value_counts().to_dict()
    shard_counts = (
        manifest_df.loc[manifest_df["shard_id"] != "", "shard_id"].value_counts().sort_index().to_dict()
    )

    audit_report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "NIH ChestX-ray14",
        "source_platform": "Kaggle (nih-chest-xrays/data)",
        "originating_institution": "NIH Clinical Center",
        "primary_citation": (
            "Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray "
            "Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax "
            "Diseases. IEEE CVPR 2017 (arXiv:1705.02315)."
        ),
        "physionet_or_mimic_used": False,
        "label_ontology": {
            "num_disease_classes": len(NIH_LABELS),
            "nih_14_labels": list(NIH_LABELS),
            "no_finding_encoding": "All 14 disease columns == 0 (never encoded as a 15th disease target)",
            "label_domain": "strict_binary_0_1",
            "chexpert_uncertainty_used": False,
            "nlp_label_limitation": (
                "Labels were text-mined via NLP from radiological reports (~>90% NLP precision reported by "
                "Wang et al., CVPR 2017; Oakden-Rayner 2020 documented label noise relative to visual re-reads). "
                "Do NOT treat NIH NLP labels as equivalent to CheXpert labels or radiologist biopsy/consensus ground truth."
            ),
        },
        "record_accounting": {
            "total_metadata_rows": int(len(manifest_df)),
            "expected_full_nih_rows": EXPECTED_FULL_NIH_IMAGE_COUNT,
            "is_complete_112120_metadata": bool(len(manifest_df) == EXPECTED_FULL_NIH_IMAGE_COUNT),
            "total_unique_patients": int(manifest_df["patient_id"].nunique()),
            "expected_full_nih_patients": EXPECTED_FULL_NIH_PATIENT_COUNT,
            "no_finding_images": int(manifest_df["no_finding"].sum()),
            "any_pathology_images": int((manifest_df["disease_count"] > 0).sum()),
            "multi_label_2plus_images": int((manifest_df["disease_count"] >= 2).sum()),
            "bbox_annotated_images": int(manifest_df["has_bbox"].sum()),
            "rows_silently_dropped": 0,
        },
        "image_integrity_audit": {
            "VALID": int(status_counts.get("VALID", 0)),
            "MISSING": int(status_counts.get("MISSING", 0)),
            "CORRUPT": int(status_counts.get("CORRUPT", 0)),
            "verify_pixel_decode_enabled": bool(verify_pixel_decode),
            "downloaded_shard_counts": {k: int(v) for k, v in shard_counts.items()},
        },
        "split_distribution": {
            "images": {
                "train": int(split_counts.get("train", 0)),
                "validate": int(split_counts.get("validate", 0)),
                "test": int(split_counts.get("test", 0)),
            },
            "leakage_verification": leakage_audit,
        },
        "demographic_and_acquisition_distribution": {
            "patient_gender": {str(k): int(v) for k, v in gender_counts.items()},
            "age_group": {str(k): int(v) for k, v in age_group_counts.items()},
            "age_anomaly_count": int(manifest_df["age_anomaly_flag"].sum()),
            "view_position": {str(k): int(v) for k, v in view_counts.items()},
        },
        "label_distribution": per_label_stats,
        "artifact_hashes_sha256": {
            "Data_Entry_2017.csv": compute_file_sha256(data_entry_path),
            "BBox_List_2017.csv": compute_file_sha256(bbox_path) if bbox_path.exists() else None,
            "train_val_list.txt": compute_file_sha256(train_val_path) if train_val_path.exists() else None,
            "test_list.txt": compute_file_sha256(test_path) if test_path.exists() else None,
            "patient_splits.csv": compute_file_sha256(patient_splits_path),
            "nih_manifest.csv": compute_file_sha256(manifest_csv_path),
        },
    }

    local_audit_path = manifests_dir / "data_audit.json"
    with open(local_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    audit_report_path = Path(audit_report_path)
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    logger.info(
        "Built NIH manifest (%d rows: VALID=%d, MISSING=%d, CORRUPT=%d) -> %s",
        len(manifest_df),
        audit_report["image_integrity_audit"]["VALID"],
        audit_report["image_integrity_audit"]["MISSING"],
        audit_report["image_integrity_audit"]["CORRUPT"],
        manifest_csv_path,
    )
    return manifest_df, audit_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical 14-label NIH ChestX-ray14 manifest and dataset audit JSON."
    )
    parser.add_argument(
        "--nih-root",
        type=Path,
        default=Path("data/kaggle_nih"),
        help="Root directory of NIH ChestX-ray14 workspace (default: data/kaggle_nih).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Validation patient fraction if patient_splits.csv must be generated (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient split generation (default: 42).",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Perform full pixel decode (`img.load()`) in addition to PNG signature & `img.verify()`.",
    )
    parser.add_argument(
        "--strict-full-count",
        action="store_true",
        help="Require exact 112,120 rows in Data_Entry_2017.csv.",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("reports/nih_dataset_audit.json"),
        help="Output path for dataset audit report JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    build_nih_manifest(
        nih_root=args.nih_root,
        audit_report_path=args.audit_out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        verify_pixel_decode=args.verify_images,
        strict_full_count=args.strict_full_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
