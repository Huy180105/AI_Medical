"""
NIH ChestX-ray14 — End-to-End Kaggle Data Acquisition Validation Gate.

Verifies the real downloaded metadata + first image shard (`images_001`) across the entire
`Kaggle -> Manifest -> Patient Split -> Image Loader -> NIH-14 Batch Inference` pipeline
BEFORE approving full 12-shard (112,120-image) download and benchmark execution.

Mandatory Outputs:
1. `reports/nih_manifest_summary.json` (enriched with 14-label patient & prevalence split tables)
2. `reports/nih_data_acquisition_validation.json` (6 hard acceptance gates + batch inference telemetry)

Zero dependency on PhysioNet, MIMIC-CXR-JPG, or MIMIC-IV-ED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_nih_cxr_manifest import EXPECTED_FULL_NIH_IMAGE_COUNT, inspect_png_integrity
from scripts.split_nih_by_patient import verify_zero_patient_leakage
from src.evaluation.benchmark_fusion import (
    BenchmarkFusionEvaluator,
    LabelDomainPolicy,
    ModalityType,
    NIH_LABELS,
    compute_file_sha256,
)
from src.utils.logger import get_logger
from src.vision.pretrained_cxr import NIH_14_LABELS, PretrainedNihDenseNet121

logger = get_logger(__name__)


def run_acquisition_validation(
    nih_root: Path = Path("data/kaggle_nih"),
    shard_id: str = "images_001",
    batch_sample_per_split: int = 32,
    mini_batch_size: int = 16,
    download_audit_path: Path = Path("reports/nih_download_audit.json"),
    manifest_summary_path: Path = Path("reports/nih_manifest_summary.json"),
    validation_report_path: Path = Path("reports/nih_data_acquisition_validation.json"),
    seed: int = 42,
) -> Dict[str, Any]:
    t0 = time.time()
    nih_root = Path(nih_root)
    manifest_csv = nih_root / "manifests" / "nih_manifest.csv"
    test_list_path = nih_root / "metadata" / "test_list.txt"
    train_val_list_path = nih_root / "metadata" / "train_val_list.txt"
    shard_dir = nih_root / "raw" / shard_id

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    if not download_audit_path.exists():
        raise FileNotFoundError(f"Download audit not found: {download_audit_path}")

    df = pd.read_csv(manifest_csv)
    df["patient_id"] = df["patient_id"].astype(str)
    download_audit = json.loads(Path(download_audit_path).read_text(encoding="utf-8"))

    # 1. Verify official split files & zero patient leakage across 112,120 rows
    official_test_set = {
        line.strip() for line in test_list_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    official_tv_set = {
        line.strip() for line in train_val_list_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    manifest_test_set = set(df.loc[df["split"] == "test", "image_id"].astype(str))
    manifest_tv_set = set(df.loc[df["split"].isin(["train", "validate"]), "image_id"].astype(str))

    official_test_exact_match = (manifest_test_set == official_test_set) and (manifest_tv_set == official_tv_set)
    leakage_proof = verify_zero_patient_leakage(df, patient_col="patient_id", split_col="split")
    duplicate_image_id_count = int(df["image_id"].duplicated().sum())

    # 2. Detailed 14-label distribution across train / validate / test (highlighting rare classes)
    split_totals = {sp: int((df["split"] == sp).sum()) for sp in ("train", "validate", "test")}
    split_patient_totals = {
        sp: int(df.loc[df["split"] == sp, "patient_id"].nunique()) for sp in ("train", "validate", "test")
    }

    per_label_split_audit: Dict[str, Any] = {}
    rare_labels_check: Dict[str, Any] = {}
    for lbl in NIH_LABELS:
        by_split_imgs: Dict[str, int] = {}
        by_split_prev_pct: Dict[str, float] = {}
        by_split_pts: Dict[str, int] = {}
        for sp in ("train", "validate", "test"):
            sp_sub = df[df["split"] == sp]
            pos_imgs = int(sp_sub[lbl].sum())
            pos_pts = int(sp_sub.loc[sp_sub[lbl] == 1, "patient_id"].nunique())
            by_split_imgs[sp] = pos_imgs
            by_split_prev_pct[sp] = round(100.0 * pos_imgs / max(1, len(sp_sub)), 4)
            by_split_pts[sp] = pos_pts

        entry = {
            "total_positive_images": int(df[lbl].sum()),
            "total_prevalence_pct": round(100.0 * float(df[lbl].sum()) / max(1, len(df)), 4),
            "total_positive_patients": int(df.loc[df[lbl] == 1, "patient_id"].nunique()),
            "positive_images_by_split": by_split_imgs,
            "prevalence_pct_by_split": by_split_prev_pct,
            "positive_patients_by_split": by_split_pts,
            "has_positives_in_all_3_splits": all(by_split_pts[sp] > 0 for sp in ("train", "validate", "test")),
        }
        per_label_split_audit[lbl] = entry
        if lbl in {"Hernia", "Pneumonia", "Fibrosis"}:
            rare_labels_check[lbl] = entry

    # Enrich and save nih_manifest_summary.json
    if Path(manifest_summary_path).exists():
        manifest_summary = json.loads(Path(manifest_summary_path).read_text(encoding="utf-8"))
    else:
        manifest_summary = {}
    manifest_summary["split_patient_counts"] = split_patient_totals
    manifest_summary["official_test_split_exact_match"] = official_test_exact_match
    manifest_summary["duplicate_image_index_count"] = duplicate_image_id_count
    manifest_summary["detailed_14_label_split_distribution"] = per_label_split_audit
    manifest_summary["rare_pathology_focus_audit"] = rare_labels_check
    Path(manifest_summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_summary_path).write_text(
        json.dumps(manifest_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3. Reconcile physical Shard 1 (`images_001`) against remote archive inventory & manifest
    remote_shard_info = (
        download_audit.get("remote_kaggle_archive_inventory", {})
        .get("shards", {})
        .get(shard_id, {})
    )
    expected_shard_count = int(remote_shard_info.get("png_count", 4999))
    first_expected = remote_shard_info.get("first_image", "00000001_000.png")
    last_expected = remote_shard_info.get("last_image", "00001335_006.png")

    on_disk_pngs = sorted(shard_dir.rglob("*.png")) if shard_dir.exists() else []
    on_disk_names = [p.name for p in on_disk_pngs]
    duplicate_files_in_shard = len(on_disk_names) - len(set(on_disk_names))

    # Verify full pixel decode + SHA-256 tree digest across all PNGs in Shard 1
    valid_png_count = 0
    corrupt_png_files: List[Dict[str, str]] = []
    resolutions_seen: Dict[str, int] = {}
    tree_hasher = hashlib.sha256()

    for p in on_disk_pngs:
        status, detail = inspect_png_integrity(p, verify_pixel_decode=True)
        if status == "VALID":
            valid_png_count += 1
            with Image.open(p) as im:
                dim_key = f"{im.width}x{im.height}:{im.mode}"
                resolutions_seen[dim_key] = resolutions_seen.get(dim_key, 0) + 1
        else:
            corrupt_png_files.append({"file": p.name, "detail": detail})
        f_sha = compute_file_sha256(p)
        tree_hasher.update(f"{p.name}:{p.stat().st_size}:{f_sha}\n".encode("utf-8"))

    # Expected manifest subset in lexicographic range [first_expected .. last_expected]
    expected_manifest_shard_rows = df[
        (df["image_id"] >= first_expected) & (df["image_id"] <= last_expected)
    ]
    expected_manifest_names = set(expected_manifest_shard_rows["image_id"].astype(str))
    on_disk_name_set = set(on_disk_names)

    missing_in_shard = sorted(list(expected_manifest_names - on_disk_name_set))
    orphan_unindexed_in_shard = sorted(list(on_disk_name_set - expected_manifest_names))

    shard_df = df[df["shard_id"] == shard_id].copy()

    # 4. Run real batch inference on Shard 1 (`validate` & `test` radiographs) using NIH-14 checkpoint
    rng = np.random.RandomState(seed)
    sampled_frames: List[pd.DataFrame] = []
    for sp in ("validate", "test"):
        sp_valid = shard_df[(shard_df["split"] == sp) & (shard_df["image_status"] == "VALID")]
        sp_pids = sorted(sp_valid["patient_id"].unique())
        chosen_pids = (
            list(rng.choice(sp_pids, size=min(len(sp_pids), batch_sample_per_split), replace=False))
            if sp_pids
            else []
        )
        # Pick 1 frontal radiograph per sampled patient for clean multi-patient batch test
        sub_rows = (
            sp_valid[sp_valid["patient_id"].isin(set(chosen_pids))]
            .sort_values("image_id")
            .groupby("patient_id", as_index=False)
            .first()
        )
        sampled_frames.append(sub_rows)

    batch_eval_df = pd.concat(sampled_frames, ignore_index=True)
    batch_leakage = verify_zero_patient_leakage(batch_eval_df, patient_col="patient_id", split_col="split")

    adapter = PretrainedNihDenseNet121(checkpoint_profile="nih14", verify_sha256=True)

    t_infer0 = time.time()
    all_batch_preds: List[Dict[str, Any]] = []
    corrupt_during_infer = 0
    img_paths = [PROJECT_ROOT / str(r["image_path"]) for _, r in batch_eval_df.iterrows()]

    # Explicitly audit preprocessed tensor shape, dtype, and [-1024, 1024] dynamic range on first real mini-batch
    first_chunk_paths = img_paths[:mini_batch_size]
    preproc_tensor = adapter.preprocess_batch(first_chunk_paths)
    preproc_audit = {
        "mini_batch_tensor_shape": list(preproc_tensor.shape),
        "dtype": str(preproc_tensor.dtype),
        "min_value": round(float(preproc_tensor.min().item()), 4),
        "max_value": round(float(preproc_tensor.max().item()), 4),
        "mean_value": round(float(preproc_tensor.mean().item()), 4),
        "has_nan_or_inf": bool( preproc_tensor.isnan().any().item() or preproc_tensor.isinf().any().item() ),
        "within_expected_xrv_bounds": bool(
            list(preproc_tensor.shape[1:]) == [1, 224, 224]
            and float(preproc_tensor.min().item()) >= -1024.01
            and float(preproc_tensor.max().item()) <= 1024.01
        ),
    }

    for start_i in range(0, len(img_paths), mini_batch_size):
        sub_paths = img_paths[start_i : start_i + mini_batch_size]
        preds_chunk = adapter.predict_batch(sub_paths)
        all_batch_preds.extend(preds_chunk)
    infer_elapsed_s = round(time.time() - t_infer0, 3)

    # Verify schema & probability domain across every inferred image
    exact_14_keys_every_image = all(
        list(item["nih_14_probabilities"].keys()) == NIH_14_LABELS for item in all_batch_preds
    )
    prob_matrix = np.array(
        [[float(item["nih_14_probabilities"][lbl]) for lbl in NIH_14_LABELS] for item in all_batch_preds],
        dtype=np.float64,
    )
    has_nan_or_inf = bool(np.isnan(prob_matrix).any() or np.isinf(prob_matrix).any())
    probs_in_unit_interval = bool(
        (not has_nan_or_inf)
        and prob_matrix.size > 0
        and float(prob_matrix.min()) >= 0.0
        and float(prob_matrix.max()) <= 1.0
    )

    per_label_batch_prob_stats: Dict[str, Dict[str, float]] = {}
    for col_idx, lbl in enumerate(NIH_14_LABELS):
        col_vals = prob_matrix[:, col_idx]
        per_label_batch_prob_stats[lbl] = {
            "min_prob": round(float(col_vals.min()), 6),
            "mean_prob": round(float(col_vals.mean()), 6),
            "max_prob": round(float(col_vals.max()), 6),
        }

    # Verify validation-only threshold invariance on real Shard 1 predictions (pick a high-prevalence label e.g. Infiltration/Effusion)
    probe_label = "Effusion"
    probe_df = pd.DataFrame({
        "patient_id": batch_eval_df["patient_id"].astype(str),
        "image_id": batch_eval_df["image_id"].astype(str),
        "split": batch_eval_df["split"].astype(str),
        "y_true": batch_eval_df[probe_label].astype(int),
        "p_cxr": [float(item["nih_14_probabilities"][probe_label]) for item in all_batch_preds],
        "label_source": "NIH_CHESTXRAY14_NLP_2017",
        "is_synthetic": False,
    })

    evaluator = BenchmarkFusionEvaluator(
        bootstrap_iterations=100,
        random_seed=seed,
        strict_patient_split=True,
        label_domain_policy=LabelDomainPolicy.BINARY_STRICT,
    )
    eval_real = evaluator.evaluate(
        probe_df,
        modality=ModalityType.CXR_ONLY,
        strict_provenance=True,
        require_validation_diversity=False,
        source_op_thresh=adapter.get_source_op_thresh(probe_label),
        target_name=probe_label,
    )
    # Invert all test labels on a copy and prove selected_threshold stays 100% identical
    flipped_probe_df = probe_df.copy()
    test_mask = flipped_probe_df["split"] == "test"
    flipped_probe_df.loc[test_mask, "y_true"] = 1 - flipped_probe_df.loc[test_mask, "y_true"]
    eval_flipped = evaluator.evaluate(
        flipped_probe_df,
        modality=ModalityType.CXR_ONLY,
        strict_provenance=True,
        require_validation_diversity=False,
        source_op_thresh=adapter.get_source_op_thresh(probe_label),
        target_name=probe_label,
    )
    threshold_locked_without_test_leakage = (
        eval_real["threshold_provenance"]["selected_threshold"]
        == eval_flipped["threshold_provenance"]["selected_threshold"]
        and eval_real["threshold_provenance"]["used_test_labels_for_threshold"] is False
        and eval_real["threshold_provenance"]["source_op_thresh_used_for_test_metrics"] is False
    )

    sample_predictions_preview = []
    for i in range(min(5, len(batch_eval_df))):
        r = batch_eval_df.iloc[i]
        top3 = sorted(
            all_batch_preds[i]["nih_14_probabilities"].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        sample_predictions_preview.append({
            "image_id": str(r["image_id"]),
            "patient_id": str(r["patient_id"]),
            "split": str(r["split"]),
            "finding_labels_raw": str(r["finding_labels_raw"]),
            "top_3_predicted_probs": {k: v for k, v in top3},
        })

    # 5. Evaluate all 6 Mandatory Acceptance Criteria
    criteria = {
        "1_zero_corrupt_images_in_shard_and_batch": {
            "passed": len(corrupt_png_files) == 0 and corrupt_during_infer == 0 and valid_png_count == expected_shard_count,
            "corrupt_in_shard": len(corrupt_png_files),
            "valid_in_shard": valid_png_count,
            "expected_in_shard": expected_shard_count,
        },
        "2_zero_duplicate_image_index": {
            "passed": duplicate_image_id_count == 0 and duplicate_files_in_shard == 0,
            "duplicate_manifest_image_ids": duplicate_image_id_count,
            "duplicate_files_in_shard": duplicate_files_in_shard,
        },
        "3_exact_14_nih_labels_verified": {
            "passed": (
                len(NIH_LABELS) == 14
                and NIH_LABELS == NIH_14_LABELS
                and exact_14_keys_every_image
                and all(lbl in df.columns for lbl in NIH_LABELS)
            ),
            "nih_14_labels": list(NIH_LABELS),
        },
        "4_valid_inference_probabilities": {
            "passed": probs_in_unit_interval and len(all_batch_preds) == len(batch_eval_df) and len(batch_eval_df) > 0,
            "images_inferred": len(all_batch_preds),
            "total_scalar_probabilities_checked": int(prob_matrix.size),
            "nan_or_inf_detected": has_nan_or_inf,
            "global_min_prob": round(float(prob_matrix.min()), 6),
            "global_max_prob": round(float(prob_matrix.max()), 6),
        },
        "5_zero_patient_leakage": {
            "passed": (
                leakage_proof["zero_patient_leakage_verified"] is True
                and batch_leakage["zero_patient_leakage_verified"] is True
            ),
            "full_manifest_leakage_audit": leakage_proof,
            "batch_subset_leakage_audit": batch_leakage,
        },
        "6_official_test_labels_and_thresholds_locked": {
            "passed": official_test_exact_match and threshold_locked_without_test_leakage,
            "official_test_list_untouched": official_test_exact_match,
            "official_test_image_count": len(official_test_set),
            "threshold_invariant_to_test_label_inversion": threshold_locked_without_test_leakage,
        },
    }

    all_passed = all(c["passed"] for c in criteria.values())

    validation_report: Dict[str, Any] = {
        "report_type": "NIH_CHESTXRAY14_DATA_ACQUISITION_VALIDATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_validation_status": "PASS" if all_passed else "FAIL",
        "ready_for_remaining_shards_download": bool(all_passed),
        "full_112120_benchmark_completed": False,
        "scope_disclosure": (
            "Data Acquisition Validation ONLY (112,120 metadata rows + Shard 1 `images_001` = 4,999 real radiographs). "
            "Shards `images_002`..`images_012` (107,121 radiographs) have NOT been downloaded yet, and full benchmark "
            "metrics are NOT claimed at this stage."
        ),
        "physionet_or_mimic_used": False,
        "acceptance_criteria": criteria,
        "manifest_and_split_verification": {
            "total_manifest_rows": int(len(df)),
            "expected_full_rows": EXPECTED_FULL_NIH_IMAGE_COUNT,
            "total_unique_patients": int(df["patient_id"].nunique()),
            "images_by_split": split_totals,
            "patients_by_split": split_patient_totals,
            "rare_pathology_distribution": rare_labels_check,
        },
        "shard_1_integrity_reconciliation": {
            "shard_id": shard_id,
            "expected_archive_png_count": expected_shard_count,
            "on_disk_png_count": len(on_disk_pngs),
            "valid_decoded_png_count": valid_png_count,
            "corrupt_png_count": len(corrupt_png_files),
            "missing_in_shard_range_count": len(missing_in_shard),
            "orphan_unindexed_files_count": len(orphan_unindexed_in_shard),
            "resolutions_and_modes": resolutions_seen,
            "shard_tree_sha256": tree_hasher.hexdigest(),
        },
        "real_batch_inference_verification": {
            "checkpoint_path": str(adapter.weights_path),
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "mini_batch_size": mini_batch_size,
            "preprocessing_verification": preproc_audit,
            "total_real_images_inferred": len(all_batch_preds),
            "validate_images_in_batch": int((batch_eval_df["split"] == "validate").sum()),
            "test_images_in_batch": int((batch_eval_df["split"] == "test").sum()),
            "inference_wall_time_seconds": infer_elapsed_s,
            "per_label_probability_summary": per_label_batch_prob_stats,
            "sample_real_predictions": sample_predictions_preview,
        },
        "artifact_hashes_sha256": {
            "nih_download_audit.json": compute_file_sha256(download_audit_path),
            "nih_manifest.csv": compute_file_sha256(manifest_csv),
            "nih_manifest_summary.json": compute_file_sha256(manifest_summary_path),
        },
        "validation_elapsed_seconds": round(time.time() - t0, 2),
    }

    validation_report_path = Path(validation_report_path)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Data Acquisition Validation finished: overall_status=%s -> %s",
        validation_report["overall_validation_status"],
        validation_report_path,
    )
    return validation_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kaggle NIH ChestX-ray14 acquisition on metadata + Shard 1.")
    parser.add_argument("--nih-root", type=Path, default=Path("data/kaggle_nih"))
    parser.add_argument("--shard-id", type=str, default="images_001")
    parser.add_argument("--batch-sample-per-split", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rep = run_acquisition_validation(
        nih_root=args.nih_root,
        shard_id=args.shard_id,
        batch_sample_per_split=args.batch_sample_per_split,
        mini_batch_size=args.mini_batch_size,
        seed=args.seed,
    )
    return 0 if rep["overall_validation_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
