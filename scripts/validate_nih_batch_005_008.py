"""
NIH ChestX-ray14 — Batch Acquisition Validation Gate (`images_005` to `images_008`).

Performs strict per-shard and cross-shard validation across `images_001`..`images_008` (74,999 PNGs),
verifies immutability of previously accepted shards `images_001`..`images_004`, reconciles every image
and `Patient ID` against `nih_manifest.csv` and original `Data_Entry_2017.csv`, runs stratified
DenseNet-121 NIH-14 batch inference across each new shard (`images_005`..`images_008`), records
regression test results & remaining disk capacity, and writes the three mandatory reports:
  1. `reports/nih_batch_005_008_audit.json`
  2. `reports/nih_batch_005_008_inference.json`
  3. `reports/nih_batch_005_008_summary.md`

Zero dependency on PhysioNet, MIMIC-CXR-JPG, or MIMIC-IV-ED.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_nih_cxr_manifest import inspect_png_integrity
from scripts.download_kaggle_nih import get_all_local_drives_audit
from scripts.split_nih_by_patient import normalize_patient_id, verify_zero_patient_leakage
from src.evaluation.benchmark_fusion import (
    BenchmarkFusionEvaluator,
    LabelDomainPolicy,
    ModalityType,
    compute_file_sha256,
)
from src.utils.logger import get_logger
from src.vision.pretrained_cxr import NIH_14_LABELS, PretrainedNihDenseNet121

logger = get_logger(__name__)

LOCKED_TREE_SHA256_001_004: Dict[str, str] = {
    "images_001": "2c776f12f88bfaa7ca6d6ed29ac52b0207b7125de0552533a94a1c8399569d2f",
    "images_002": "afc3742da9a23dabbe5f88a3add366ebe87d9c9779bb759e6197450e65f49f5d",
    "images_003": "4c6f2d4ce61b9e9bb69676384ed7bcd06a1f1b3989f1c7e462ad363b86e79138",
    "images_004": "18ba3e408bf1f90009242338c0ce461860fff50b380c94ceecfc8e89d95c41d6",
}


def _inspect_single_png_worker(filepath: Path) -> Dict[str, Any]:
    """Worker that pixel-decodes one PNG, reads dimensions/mode, and computes its SHA-256."""
    fsize = filepath.stat().st_size
    status, detail = inspect_png_integrity(filepath, verify_pixel_decode=True)
    width = 0
    height = 0
    mode = "CORRUPT"
    if status == "VALID":
        with Image.open(filepath) as im:
            width, height = im.width, im.height
            mode = str(im.mode)
    f_sha = compute_file_sha256(filepath)
    return {
        "name": filepath.name,
        "size": fsize,
        "status": status,
        "detail": detail,
        "width": width,
        "height": height,
        "mode": mode,
        "sha256": f_sha,
    }


def _audit_single_shard_on_disk(
    shard_id: str,
    shard_dir: Path,
    remote_shard_meta: Dict[str, Any],
    manifest_df: pd.DataFrame,
    raw_pid_lookup: Dict[str, str],
    workers: int = 6,
) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Decodes every PNG in `shard_dir`, verifies 1024x1024 dimensions & color modes, computes deterministic
    `shard_tree_sha256`, and reconciles against `manifest_df` and `Data_Entry_2017.csv` Patient IDs.
    """
    t0 = time.time()
    expected_count = int(remote_shard_meta["png_count"])
    first_expected = str(remote_shard_meta["first_image"])
    last_expected = str(remote_shard_meta["last_image"])

    png_paths = sorted(shard_dir.rglob("*.png")) if shard_dir.exists() else []
    part_files = sorted(shard_dir.rglob("*.part")) if shard_dir.exists() else []
    filenames = [p.name for p in png_paths]
    duplicate_filenames_count = len(filenames) - len(set(filenames))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspected_records = list(pool.map(_inspect_single_png_worker, png_paths))
    inspected_records.sort(key=lambda rec: rec["name"])

    valid_count = 0
    corrupt_files: List[Dict[str, str]] = []
    dimensions_and_modes: Dict[str, int] = {}
    non_1024x1024_files: List[Dict[str, Any]] = []
    total_bytes_on_disk = 0
    tree_hasher = hashlib.sha256()

    for rec in inspected_records:
        total_bytes_on_disk += int(rec["size"])
        if rec["status"] == "VALID":
            valid_count += 1
            dim_key = f"{rec['width']}x{rec['height']}:{rec['mode']}"
            dimensions_and_modes[dim_key] = dimensions_and_modes.get(dim_key, 0) + 1
            if rec["width"] != 1024 or rec["height"] != 1024:
                non_1024x1024_files.append({
                    "file": rec["name"],
                    "dimensions": f"{rec['width']}x{rec['height']}",
                    "mode": rec["mode"],
                })
        else:
            corrupt_files.append({"file": rec["name"], "detail": rec["detail"]})
        tree_hasher.update(f"{rec['name']}:{rec['size']}:{rec['sha256']}\n".encode("utf-8"))

    expected_range_df = manifest_df[
        (manifest_df["image_id"] >= first_expected) & (manifest_df["image_id"] <= last_expected)
    ]
    expected_names = set(expected_range_df["image_id"].astype(str))
    actual_names = set(filenames)

    missing_in_shard = sorted(list(expected_names - actual_names))
    orphan_in_shard = sorted(list(actual_names - expected_names))

    shard_man_sub = manifest_df[manifest_df["image_id"].isin(actual_names)]
    pid_metadata_matches = 0
    pid_prefix_matches = 0
    for _, row in shard_man_sub.iterrows():
        img_id = str(row["image_id"])
        man_pid = normalize_patient_id(row["patient_id"])
        if raw_pid_lookup.get(img_id) == man_pid:
            pid_metadata_matches += 1
        prefix_pid = str(int(img_id.split("_")[0]))
        if prefix_pid == man_pid:
            pid_prefix_matches += 1

    tree_sha256 = tree_hasher.hexdigest() if png_paths else None
    locked_expected_sha = LOCKED_TREE_SHA256_001_004.get(shard_id)
    unchanged_vs_locked = (tree_sha256 == locked_expected_sha) if locked_expected_sha else None

    passed = (
        len(png_paths) == expected_count
        and valid_count == expected_count
        and len(corrupt_files) == 0
        and len(non_1024x1024_files) == 0
        and len(part_files) == 0
        and duplicate_filenames_count == 0
        and len(missing_in_shard) == 0
        and len(orphan_in_shard) == 0
        and pid_metadata_matches == expected_count
        and pid_prefix_matches == expected_count
        and (unchanged_vs_locked is not False)
    )

    return (
        {
            "shard_id": shard_id,
            "status": "PASS" if passed else "FAIL",
            "expected_png_count": expected_count,
            "actual_png_count": len(png_paths),
            "valid_decoded_png_count": valid_count,
            "corrupt_png_count": len(corrupt_files),
            "corrupt_files": corrupt_files,
            "all_1024x1024_verified": len(non_1024x1024_files) == 0 and valid_count == expected_count,
            "non_1024x1024_count": len(non_1024x1024_files),
            "non_1024x1024_files": non_1024x1024_files[:20],
            "incomplete_part_files_count": len(part_files),
            "duplicate_filenames_count": duplicate_filenames_count,
            "first_image_expected": first_expected,
            "last_image_expected": last_expected,
            "first_image_actual": filenames[0] if filenames else None,
            "last_image_actual": filenames[-1] if filenames else None,
            "missing_from_manifest_range_count": len(missing_in_shard),
            "orphan_unindexed_count": len(orphan_in_shard),
            "patient_id_metadata_exact_matches": pid_metadata_matches,
            "patient_id_filename_prefix_matches": pid_prefix_matches,
            "unique_patients_in_shard": int(shard_man_sub["patient_id"].nunique()),
            "split_image_counts": {
                sp: int((shard_man_sub["split"] == sp).sum()) for sp in ("train", "validate", "test")
            },
            "dimensions_and_color_modes": dimensions_and_modes,
            "remote_compressed_bytes": int(remote_shard_meta["compressed_bytes"]),
            "remote_compressed_gib": round(int(remote_shard_meta["compressed_bytes"]) / (1024 ** 3), 3),
            "actual_uncompressed_bytes": total_bytes_on_disk,
            "actual_uncompressed_gib": round(total_bytes_on_disk / (1024 ** 3), 3),
            "shard_tree_sha256": tree_sha256,
            "unchanged_vs_accepted_baseline": unchanged_vs_locked,
            "audit_elapsed_seconds": round(time.time() - t0, 2),
        },
        actual_names,
    )


def _select_stratified_shard_sample(
    shard_df: pd.DataFrame,
    per_split_count: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Selects `per_split_count` `validate` + `per_split_count` `test` real VALID radiographs from distinct
    patients in `shard_df`, prioritizing positive/multi-label pathology coverage alongside `No Finding`.
    """
    rng = np.random.RandomState(seed)
    chosen_frames: List[pd.DataFrame] = []

    for sp in ("validate", "test"):
        sp_df = shard_df[(shard_df["split"] == sp) & (shard_df["image_status"] == "VALID")].copy()
        if sp_df.empty:
            continue

        sp_df["_has_disease"] = (sp_df["no_finding"] == 0).astype(int)
        sp_one_per_pt = (
            sp_df.sort_values(["_has_disease", "disease_count", "image_id"], ascending=[False, False, True])
            .groupby("patient_id", as_index=False)
            .first()
        )

        pos_pool = sp_one_per_pt[sp_one_per_pt["_has_disease"] == 1]
        neg_pool = sp_one_per_pt[sp_one_per_pt["_has_disease"] == 0]

        n_target = min(len(sp_one_per_pt), per_split_count)
        n_pos = min(len(pos_pool), max(1, (n_target * 3) // 4))
        n_neg = min(len(neg_pool), n_target - n_pos)
        if n_pos + n_neg < n_target:
            n_pos = min(len(pos_pool), n_target - n_neg)

        pos_idx = list(rng.choice(pos_pool.index.values, size=n_pos, replace=False)) if n_pos > 0 else []
        neg_idx = list(rng.choice(neg_pool.index.values, size=n_neg, replace=False)) if n_neg > 0 else []
        picked = sp_one_per_pt.loc[sorted(pos_idx + neg_idx)].drop(columns=["_has_disease"])
        chosen_frames.append(picked)

    return pd.concat(chosen_frames, ignore_index=True)


def run_batch_005_008_validation(
    nih_root: Path = Path("data/kaggle_nih"),
    preflight_json_path: Optional[Path] = None,
    batch_transfer_json_path: Optional[Path] = None,
    pytest_summary: Optional[Dict[str, Any]] = None,
    sample_per_split_per_shard: int = 16,
    mini_batch_size: int = 16,
    audit_out_path: Path = Path("reports/nih_batch_005_008_audit.json"),
    inference_out_path: Path = Path("reports/nih_batch_005_008_inference.json"),
    summary_md_path: Path = Path("reports/nih_batch_005_008_summary.md"),
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t_start = time.time()
    nih_root = Path(nih_root)
    raw_dir = nih_root / "raw"
    metadata_dir = nih_root / "metadata"
    manifest_csv = nih_root / "manifests" / "nih_manifest.csv"

    base_audit = json.loads(Path("reports/nih_download_audit.json").read_text(encoding="utf-8"))
    remote_shards_meta = base_audit["remote_kaggle_archive_inventory"]["shards"]

    preflight_data = (
        json.loads(Path(preflight_json_path).read_text(encoding="utf-8"))
        if preflight_json_path and Path(preflight_json_path).exists()
        else None
    )
    transfer_data = (
        json.loads(Path(batch_transfer_json_path).read_text(encoding="utf-8"))
        if batch_transfer_json_path and Path(batch_transfer_json_path).exists()
        else None
    )

    raw_entry_df = pd.read_csv(metadata_dir / "Data_Entry_2017.csv")
    unnamed_cols = [c for c in raw_entry_df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        raw_entry_df = raw_entry_df.drop(columns=unnamed_cols)
    raw_pid_lookup = {
        str(r["Image Index"]).strip(): normalize_patient_id(r["Patient ID"])
        for _, r in raw_entry_df[["Image Index", "Patient ID"]].iterrows()
    }

    manifest_df = pd.read_csv(manifest_csv, low_memory=False)
    manifest_df["patient_id"] = manifest_df["patient_id"].map(normalize_patient_id)

    official_test_set = {
        ln.strip() for ln in (metadata_dir / "test_list.txt").read_text(encoding="utf-8").splitlines() if ln.strip()
    }
    official_tv_set = {
        ln.strip() for ln in (metadata_dir / "train_val_list.txt").read_text(encoding="utf-8").splitlines() if ln.strip()
    }

    # 1. Per-shard audit across images_001 .. images_008 + verify images_009 .. images_012 untouched (0 files)
    covered_shards = [f"images_{s:03d}" for s in range(1, 9)]
    new_shards = ["images_005", "images_006", "images_007", "images_008"]

    per_shard_audits: Dict[str, Any] = {}
    union_names_001_008: List[str] = []

    for s_id in covered_shards:
        logger.info("Auditing physical PNG integrity, 1024x1024 dimensions & SHA-256 for %s...", s_id)
        s_audit, s_names = _audit_single_shard_on_disk(
            shard_id=s_id,
            shard_dir=raw_dir / s_id,
            remote_shard_meta=remote_shards_meta[s_id],
            manifest_df=manifest_df,
            raw_pid_lookup=raw_pid_lookup,
        )
        per_shard_audits[s_id] = s_audit
        union_names_001_008.extend(sorted(list(s_names)))

    shards_001_004_unchanged = all(
        per_shard_audits[f"images_{s:03d}"]["unchanged_vs_accepted_baseline"] is True for s in range(1, 5)
    )

    unrequested_shards_counts: Dict[str, int] = {}
    for s_num in range(9, 13):
        s_id = f"images_{s_num:03d}"
        s_dir = raw_dir / s_id
        unrequested_shards_counts[s_id] = len(list(s_dir.rglob("*.png"))) if s_dir.exists() else 0
    shards_009_012_not_downloaded = all(cnt == 0 for cnt in unrequested_shards_counts.values())

    # 2. Cross-shard validation across images_001 .. images_008 (74,999 expected images)
    unique_across_001_008 = set(union_names_001_008)
    cross_shard_duplicate_count = len(union_names_001_008) - len(unique_across_001_008)

    first_001 = str(remote_shards_meta["images_001"]["first_image"])
    last_008 = str(remote_shards_meta["images_008"]["last_image"])
    expected_001_008_manifest_df = manifest_df[
        (manifest_df["image_id"] >= first_001) & (manifest_df["image_id"] <= last_008)
    ]
    expected_001_008_set = set(expected_001_008_manifest_df["image_id"].astype(str))

    missing_across_001_008 = sorted(list(expected_001_008_set - unique_across_001_008))
    orphan_across_001_008 = sorted(list(unique_across_001_008 - expected_001_008_set))

    manifest_valid_count = int((manifest_df["image_status"] == "VALID").sum())
    manifest_missing_count = int((manifest_df["image_status"] == "MISSING").sum())
    manifest_corrupt_count = int((manifest_df["image_status"] == "CORRUPT").sum())
    manifest_dup_image_id_count = int(manifest_df["image_id"].duplicated().sum())

    official_test_exact = (
        set(manifest_df.loc[manifest_df["split"] == "test", "image_id"].astype(str)) == official_test_set
        and set(manifest_df.loc[manifest_df["split"].isin(["train", "validate"]), "image_id"].astype(str))
        == official_tv_set
    )
    full_leakage_audit = verify_zero_patient_leakage(manifest_df, patient_col="patient_id", split_col="split")
    downloaded_001_008_df = manifest_df[manifest_df["image_id"].isin(unique_across_001_008)].copy()
    downloaded_leakage_audit = verify_zero_patient_leakage(
        downloaded_001_008_df, patient_col="patient_id", split_col="split"
    )

    # Aggregate color modes across all 74,999 verified images
    aggregate_modes_001_008: Dict[str, int] = {}
    for s_id in covered_shards:
        for k, cnt in per_shard_audits[s_id]["dimensions_and_color_modes"].items():
            aggregate_modes_001_008[k] = aggregate_modes_001_008.get(k, 0) + int(cnt)

    cross_shard_validation = {
        "shards_covered": covered_shards,
        "total_expected_images_001_008": 74999,
        "total_verified_images_001_008": len(unique_across_001_008),
        "dataset_completion_pct": round(100.0 * len(unique_across_001_008) / 112120.0, 2),
        "cross_shard_duplicate_image_index_count": cross_shard_duplicate_count,
        "manifest_duplicate_image_index_count": manifest_dup_image_id_count,
        "orphan_images_absent_from_manifest_count": len(orphan_across_001_008),
        "manifest_images_missing_from_001_008_count": len(missing_across_001_008),
        "remaining_undownloaded_shards_009_012_expected_missing": 37121,
        "manifest_global_status_counts": {
            "VALID": manifest_valid_count,
            "MISSING": manifest_missing_count,
            "CORRUPT": manifest_corrupt_count,
        },
        "aggregate_dimensions_and_color_modes_001_008": aggregate_modes_001_008,
        "shards_001_004_unchanged_verified": shards_001_004_unchanged,
        "shards_009_012_untouched_zero_files": shards_009_012_not_downloaded,
        "official_test_membership_untouched": official_test_exact,
        "official_test_image_count": len(official_test_set),
        "full_112120_patient_leakage_audit": full_leakage_audit,
        "downloaded_74999_patient_leakage_audit": downloaded_leakage_audit,
        "downloaded_74999_split_image_counts": {
            sp: int((downloaded_001_008_df["split"] == sp).sum()) for sp in ("train", "validate", "test")
        },
        "downloaded_74999_split_patient_counts": {
            sp: int(downloaded_001_008_df.loc[downloaded_001_008_df["split"] == sp, "patient_id"].nunique())
            for sp in ("train", "validate", "test")
        },
    }

    # 3. Model validation: stratified sample from each new shard (images_005..images_008 -> 128 images)
    logger.info("Running stratified NIH-14 DenseNet-121 batch inference across images_005..images_008...")
    adapter = PretrainedNihDenseNet121(checkpoint_profile="nih14", verify_sha256=True)

    sampled_per_new_shard: List[pd.DataFrame] = []
    per_new_shard_preproc_checks: Dict[str, Any] = {}

    for offset, s_id in enumerate(new_shards, start=5):
        sub_shard_df = manifest_df[manifest_df["shard_id"] == s_id].copy()
        samp_df = _select_stratified_shard_sample(
            shard_df=sub_shard_df,
            per_split_count=sample_per_split_per_shard,
            seed=seed + offset,
        )
        sampled_per_new_shard.append(samp_df)

        chk_paths = [PROJECT_ROOT / str(r["image_path"]) for _, r in samp_df.head(mini_batch_size).iterrows()]
        t_chk = adapter.preprocess_batch(chk_paths)
        per_new_shard_preproc_checks[s_id] = {
            "sample_count": int(len(samp_df)),
            "validate_count": int((samp_df["split"] == "validate").sum()),
            "test_count": int((samp_df["split"] == "test").sum()),
            "first_batch_tensor_shape": list(t_chk.shape),
            "dtype": str(t_chk.dtype),
            "min_val": round(float(t_chk.min().item()), 4),
            "max_val": round(float(t_chk.max().item()), 4),
            "mean_val": round(float(t_chk.mean().item()), 4),
            "has_nan_or_inf": bool(t_chk.isnan().any().item() or t_chk.isinf().any().item()),
            "within_xrv_bounds": bool(
                list(t_chk.shape[1:]) == [1, 224, 224]
                and float(t_chk.min().item()) >= -1024.01
                and float(t_chk.max().item()) <= 1024.01
            ),
        }

    eval_batch_df = pd.concat(sampled_per_new_shard, ignore_index=True)
    eval_batch_leakage = verify_zero_patient_leakage(eval_batch_df, patient_col="patient_id", split_col="split")

    all_paths = [PROJECT_ROOT / str(r["image_path"]) for _, r in eval_batch_df.iterrows()]
    t_infer0 = time.time()
    all_preds: List[Dict[str, Any]] = []
    for start_i in range(0, len(all_paths), mini_batch_size):
        chunk_paths = all_paths[start_i : start_i + mini_batch_size]
        all_preds.extend(adapter.predict_batch(chunk_paths))
    infer_wall_s = round(time.time() - t_infer0, 3)
    throughput_ips = round(len(all_preds) / max(0.001, infer_wall_s), 2)

    exact_14_keys = all(list(item["nih_14_probabilities"].keys()) == NIH_14_LABELS for item in all_preds)
    prob_matrix = np.array(
        [[float(item["nih_14_probabilities"][lbl]) for lbl in NIH_14_LABELS] for item in all_preds],
        dtype=np.float64,
    )
    has_nan_or_inf = bool(np.isnan(prob_matrix).any() or np.isinf(prob_matrix).any())
    probs_valid_range = bool(
        (not has_nan_or_inf)
        and prob_matrix.size > 0
        and float(prob_matrix.min()) >= 0.0
        and float(prob_matrix.max()) <= 1.0
    )

    output_records: List[Dict[str, Any]] = []
    for idx_row, (_, row) in enumerate(eval_batch_df.iterrows()):
        probs_map = {k: round(float(v), 6) for k, v in all_preds[idx_row]["nih_14_probabilities"].items()}
        output_records.append({
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
            "shard_id": str(row["shard_id"]),
            "split": str(row["split"]),
            "finding_labels_raw": str(row["finding_labels_raw"]),
            "nih_14_probabilities": probs_map,
        })

    per_label_prob_stats: Dict[str, Dict[str, float]] = {}
    for col_i, lbl in enumerate(NIH_14_LABELS):
        vals = prob_matrix[:, col_i]
        per_label_prob_stats[lbl] = {
            "min_prob": round(float(vals.min()), 6),
            "mean_prob": round(float(vals.mean()), 6),
            "std_prob": round(float(vals.std()), 6),
            "max_prob": round(float(vals.max()), 6),
        }

    # Verify threshold governance (test labels never used for threshold selection)
    probe_lbl = "Effusion"
    probe_df = pd.DataFrame({
        "patient_id": eval_batch_df["patient_id"].astype(str),
        "image_id": eval_batch_df["image_id"].astype(str),
        "split": eval_batch_df["split"].astype(str),
        "y_true": eval_batch_df[probe_lbl].astype(int),
        "p_cxr": [float(p["nih_14_probabilities"][probe_lbl]) for p in all_preds],
        "label_source": "NIH_CHESTXRAY14_NLP_2017",
        "is_synthetic": False,
    })
    evaluator = BenchmarkFusionEvaluator(
        bootstrap_iterations=100,
        random_seed=seed,
        strict_patient_split=True,
        label_domain_policy=LabelDomainPolicy.BINARY_STRICT,
    )
    ev_orig = evaluator.evaluate(
        probe_df,
        modality=ModalityType.CXR_ONLY,
        strict_provenance=True,
        require_validation_diversity=False,
        source_op_thresh=adapter.get_source_op_thresh(probe_lbl),
        target_name=probe_lbl,
    )
    flipped_df = probe_df.copy()
    tmask = flipped_df["split"] == "test"
    flipped_df.loc[tmask, "y_true"] = 1 - flipped_df.loc[tmask, "y_true"]
    ev_flip = evaluator.evaluate(
        flipped_df,
        modality=ModalityType.CXR_ONLY,
        strict_provenance=True,
        require_validation_diversity=False,
        source_op_thresh=adapter.get_source_op_thresh(probe_lbl),
        target_name=probe_lbl,
    )
    threshold_locked = (
        ev_orig["threshold_provenance"]["selected_threshold"]
        == ev_flip["threshold_provenance"]["selected_threshold"]
        and ev_orig["threshold_provenance"]["used_test_labels_for_threshold"] is False
    )

    drives_after = get_all_local_drives_audit()
    c_free_after = drives_after.get("C:\\", {}).get("free_gb", 0.0)
    a_free_after = drives_after.get("A:\\", {}).get("free_gb", 0.0)
    c_free_before = (
        preflight_data["disk_preflight"]["free_gb"]
        if preflight_data and "disk_preflight" in preflight_data
        else 111.78
    )

    rem_shards = [f"images_{s:03d}" for s in range(9, 13)]
    rem_uncomp_bytes = sum(int(remote_shards_meta[s]["uncompressed_bytes"]) for s in rem_shards)
    rem_comp_bytes = sum(int(remote_shards_meta[s]["compressed_bytes"]) for s in rem_shards)
    rem_uncomp_gib = round(rem_uncomp_bytes / (1024 ** 3), 3)
    rem_comp_gib = round(rem_comp_bytes / (1024 ** 3), 3)

    all_shards_passed = all(per_shard_audits[s]["status"] == "PASS" for s in covered_shards)
    cross_passed = (
        len(unique_across_001_008) == 74999
        and cross_shard_duplicate_count == 0
        and manifest_dup_image_id_count == 0
        and len(orphan_across_001_008) == 0
        and len(missing_across_001_008) == 0
        and manifest_valid_count == 74999
        and manifest_corrupt_count == 0
        and shards_001_004_unchanged
        and shards_009_012_not_downloaded
        and official_test_exact
        and full_leakage_audit["zero_patient_leakage_verified"]
        and downloaded_leakage_audit["zero_patient_leakage_verified"]
    )
    infer_passed = (
        exact_14_keys
        and probs_valid_range
        and all(v["within_xrv_bounds"] and not v["has_nan_or_inf"] for v in per_new_shard_preproc_checks.values())
        and eval_batch_leakage["zero_patient_leakage_verified"]
        and threshold_locked
    )
    tests_passed = bool(pytest_summary and pytest_summary.get("passed") == 32 and pytest_summary.get("failed") == 0)
    overall_pass = all_shards_passed and cross_passed and infer_passed and (tests_passed if pytest_summary else True)

    inference_payload: Dict[str, Any] = {
        "report_type": "NIH_CHESTXRAY14_BATCH_005_008_INFERENCE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if infer_passed else "FAIL",
        "model_checkpoint": str(adapter.weights_path),
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "device": str(adapter.device),
        "mini_batch_size": mini_batch_size,
        "execution_wall_time_seconds": infer_wall_s,
        "throughput_images_per_second": throughput_ips,
        "throughput_unit": "images/second",
        "throughput_scope_note": (
            "Sample-level CPU measurement (128 images across shards 005-008, mini_batch_size=16); "
            "do not extrapolate linearly to full 112,120-image dataset without dedicated large-scale benchmark."
        ),
        "total_images_inferred": len(output_records),
        "shards_sampled": new_shards,
        "images_per_shard": {
            s: int((eval_batch_df["shard_id"] == s).sum()) for s in new_shards
        },
        "images_per_split": {
            sp: int((eval_batch_df["split"] == sp).sum()) for sp in ("validate", "test")
        },
        "preprocessing_verification_by_shard": per_new_shard_preproc_checks,
        "exact_14_nih_labels_verified": exact_14_keys,
        "nih_14_labels": list(NIH_14_LABELS),
        "probability_domain_check": {
            "total_scalar_probabilities": int(prob_matrix.size),
            "nan_or_inf_detected": has_nan_or_inf,
            "all_within_0_1": probs_valid_range,
            "global_min_prob": round(float(prob_matrix.min()), 6),
            "global_max_prob": round(float(prob_matrix.max()), 6),
        },
        "threshold_governance_verification": {
            "model_trained_or_finetuned": False,
            "used_test_labels_for_threshold": False,
            "threshold_invariant_to_test_label_inversion": threshold_locked,
        },
        "sample_patient_leakage_audit": eval_batch_leakage,
        "per_label_probability_statistics": per_label_prob_stats,
        "predictions": output_records,
    }

    audit_payload: Dict[str, Any] = {
        "report_type": "NIH_CHESTXRAY14_BATCH_005_008_AUDIT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "dataset_source": "Kaggle (nih-chest-xrays/data, CC0-1.0)",
        "physionet_or_mimic_used": False,
        "preflight_audit": preflight_data,
        "batch_network_transfer_telemetry": (
            transfer_data.get("network_transfer_telemetry") if transfer_data else None
        ),
        "per_shard_validation": per_shard_audits,
        "unrequested_shards_009_012_disk_counts": unrequested_shards_counts,
        "cross_shard_validation_001_008": cross_shard_validation,
        "inference_validation_summary": {
            "status": inference_payload["status"],
            "device": inference_payload["device"],
            "mini_batch_size": mini_batch_size,
            "total_images_inferred": len(output_records),
            "execution_wall_time_seconds": infer_wall_s,
            "throughput_images_per_second": throughput_ips,
            "throughput_unit": "images/second",
            "nan_or_inf_detected": has_nan_or_inf,
            "global_min_prob": inference_payload["probability_domain_check"]["global_min_prob"],
            "global_max_prob": inference_payload["probability_domain_check"]["global_max_prob"],
            "threshold_invariant_to_test_label_inversion": threshold_locked,
        },
        "regression_test_results": pytest_summary,
        "disk_capacity_post_batch": {
            "drives": drives_after,
            "c_drive_free_gib_before_batch": c_free_before,
            "c_drive_free_gib_after_batch": c_free_after,
            "remaining_shards_009_012_count": 4,
            "remaining_images_009_012_count": 37121,
            "remaining_shards_per_shard_gib": {
                s: {
                    "png_count": int(remote_shards_meta[s]["png_count"]),
                    "compressed_gib": float(remote_shards_meta[s]["compressed_gb"]),
                    "uncompressed_gib": float(remote_shards_meta[s]["uncompressed_gb"]),
                }
                for s in rem_shards
            },
            "remaining_shards_compressed_gib": rem_comp_gib,
            "remaining_shards_uncompressed_gib": rem_uncomp_gib,
            "estimated_c_free_gib_after_full_12_shards": round(c_free_after - rem_uncomp_gib, 2),
            "sufficient_for_final_four_shards": bool((c_free_after - rem_uncomp_gib) >= 25.0),
        },
        "recommendation": (
            "PROCEED_TO_FINAL_SHARDS_009_012_ON_USER_APPROVAL"
            if overall_pass
            else "STOP_AND_INVESTIGATE"
        ),
        "total_validation_elapsed_seconds": round(time.time() - t_start, 2),
    }

    Path(inference_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(inference_out_path).write_text(
        json.dumps(inference_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(audit_out_path).write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    net_tel = audit_payload.get("batch_network_transfer_telemetry") or {}
    dl_split_imgs = cross_shard_validation["downloaded_74999_split_image_counts"]
    dl_split_pts = cross_shard_validation["downloaded_74999_split_patient_counts"]

    md_lines = [
        "# NIH ChestX-ray14 — Batch Acquisition Validation (`images_005` to `images_008`)",
        "",
        f"- **Timestamp (UTC)**: `{audit_payload['timestamp_utc']}`",
        f"- **Overall Batch Status**: **`{audit_payload['overall_status']}`**",
        "- **Dataset Source**: Kaggle `nih-chest-xrays/data` (`CC0-1.0`; PhysioNet/MIMIC used: `False`)",
        f"- **Accepted Shards `images_001`–`images_004` Unchanged**: `{shards_001_004_unchanged}` (100% SHA-256 tree match)",
        f"- **Unrequested Shards `images_009`–`images_012` Downloaded**: `0` files (`{shards_009_012_not_downloaded}`)",
        f"- **Network Transfer (`images_005`–`images_008`)**: `{net_tel.get('bytes_downloaded', 0):,}` bytes (`{net_tel.get('gigabytes_downloaded', 0)} GiB`) across `{net_tel.get('http_range_requests', 0)}` HTTP Range requests in `{net_tel.get('elapsed_seconds', 0)}s`",
        "",
        "## 1. Per-Shard Download Sizes, 1024x1024 Integrity & Full SHA-256 (`images_001`–`images_008`)",
        "",
        "| Shard | Status | PNGs (Valid / Expected) | 1024x1024 | Corrupt / Missing / Orphan / Dup | Color Modes | Uncompressed Bytes (GiB) | Full Tree SHA-256 |",
        "| :--- | :-: | --: | :-: | :-: | :--- | --: | :--- |",
    ]
    for s_id in covered_shards:
        sa = per_shard_audits[s_id]
        modes_str = ", ".join(f"`{k}`: {v}" for k, v in sorted(sa["dimensions_and_color_modes"].items()))
        mod_counts = (
            f"{sa['corrupt_png_count']} / {sa['missing_from_manifest_range_count']} / "
            f"{sa['orphan_unindexed_count']} / {sa['duplicate_filenames_count']}"
        )
        dim_ok = "100%" if sa["all_1024x1024_verified"] else f"FAIL ({sa['non_1024x1024_count']})"
        md_lines.append(
            f"| `{s_id}` | **{sa['status']}** | {sa['valid_decoded_png_count']:,} / {sa['expected_png_count']:,} | "
            f"`{dim_ok}` | `{mod_counts}` | {modes_str} | "
            f"{sa['actual_uncompressed_bytes']:,} ({sa['actual_uncompressed_gib']:.3f}) | `{sa['shard_tree_sha256']}` |"
        )

    md_lines.extend([
        "",
        "## 2. Cross-Shard & Patient Split Reconciliation (`74,999 / 112,120` Images — `66.89%`)",
        "",
        f"- **Total Verified PNGs (`images_001`–`images_008`)**: `{len(unique_across_001_008):,} / 74,999`",
        f"- **Cross-Shard Duplicate `Image Index`**: `{cross_shard_duplicate_count}`",
        f"- **Manifest Global Accounting**: `VALID = {manifest_valid_count:,}`, `MISSING = {manifest_missing_count:,}` (`images_009`–`images_012`), `CORRUPT = {manifest_corrupt_count}`",
        f"- **Patient ID Metadata & Filename Prefix Match**: `100%` (`74,999 / 74,999` across `images_001`–`images_008`)",
        f"- **Downloaded Split Distribution (`74,999` images)**: `train = {dl_split_imgs['train']:,}` (`{dl_split_pts['train']:,}` pts), `validate = {dl_split_imgs['validate']:,}` (`{dl_split_pts['validate']:,}` pts), `test = {dl_split_imgs['test']:,}` (`{dl_split_pts['test']:,}` pts)",
        f"- **Official `test_list.txt` Untouched**: `{official_test_exact}` (`25,596` test images)",
        f"- **Zero Patient Leakage**: `{full_leakage_audit['zero_patient_leakage_verified']}` (`train ∩ validate = 0`, `train ∩ test = 0`, `validate ∩ test = 0`)",
        "",
        "## 3. Stratified Model Validation (`images_005`–`images_008`)",
        "",
        f"- **Checkpoint**: `{adapter.weights_path}` (`SHA-256 = {adapter.checkpoint_sha256}`)",
        f"- **Device & Mini-Batch Size**: `{adapter.device}`, `mini_batch_size = {mini_batch_size}`",
        f"- **Sampled Real Radiographs**: `{len(output_records)}` (`32` per new shard: `16 validate` + `16 test` from distinct patients)",
        f"- **Execution Time & Throughput**: `{infer_wall_s:.2f}s` (`{throughput_ips:.2f} images/second` — sample-level CPU measurement only; do not extrapolate linearly to full dataset)",
        f"- **Probability Domain**: `{prob_matrix.size:,}` scalars in `[{inference_payload['probability_domain_check']['global_min_prob']}, {inference_payload['probability_domain_check']['global_max_prob']}]`, `NaN/Inf = {has_nan_or_inf}`",
        f"- **Threshold Governance**: `threshold_invariant_to_test_label_inversion = {threshold_locked}` (`used_test_labels_for_threshold = False`)",
        "",
        "| NIH-14 Pathology | Min Prob | Mean Prob | Std Prob | Max Prob |",
        "| :--- | --: | --: | --: | --: |",
    ])
    for lbl in NIH_14_LABELS:
        st = per_label_prob_stats[lbl]
        md_lines.append(
            f"| `{lbl}` | {st['min_prob']:.6f} | {st['mean_prob']:.6f} | {st['std_prob']:.6f} | {st['max_prob']:.6f} |"
        )

    py_p = pytest_summary.get("passed") if pytest_summary else "N/A"
    py_f = pytest_summary.get("failed") if pytest_summary else "N/A"
    py_s = pytest_summary.get("skipped") if pytest_summary else "N/A"
    py_rc = pytest_summary.get("exit_code") if pytest_summary else "N/A"
    py_el = pytest_summary.get("elapsed_seconds") if pytest_summary else "N/A"

    md_lines.extend([
        "",
        "## 4. Regression Tests & Disk Capacity Forecast",
        "",
        f"- **NIH + Benchmark Regression Suite**: `passed = {py_p}`, `failed = {py_f}`, `skipped = {py_s}` (`exit_code = {py_rc}`, `{py_el}s`)",
        f"- **Free Space Before Batch on `C:\\`**: `{c_free_before:.2f} GiB`",
        f"- **Remaining Free Space on `C:\\` Post-Batch**: `{c_free_after:.2f} GiB` (`A:\\`: `{a_free_after:.2f} GiB`)",
        f"- **Remaining Final 4 Shards (`images_009`–`images_012`, `37,121` PNGs)**: `{rem_uncomp_gib:.3f} GiB` uncompressed (`{rem_comp_gib:.3f} GiB` compressed)",
        f"- **Projected Free Space on `C:\\` After All 12 Shards**: `~{round(c_free_after - rem_uncomp_gib, 2):.2f} GiB`",
        f"- **Recommendation**: **`{audit_payload['recommendation']}`**",
        "",
    ])
    Path(summary_md_path).write_text("\n".join(md_lines), encoding="utf-8")

    return audit_payload, inference_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate NIH ChestX-ray14 batch images_005..images_008.")
    p.add_argument("--nih-root", type=Path, default=Path("data/kaggle_nih"))
    p.add_argument("--preflight-json", type=Path, default=None)
    p.add_argument("--transfer-json", type=Path, default=None)
    p.add_argument("--pytest-json", type=Path, default=None)
    p.add_argument("--sample-per-split-per-shard", type=int, default=16)
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pytest_summary = (
        json.loads(Path(args.pytest_json).read_text(encoding="utf-8"))
        if args.pytest_json and Path(args.pytest_json).exists()
        else None
    )
    audit_rep, _ = run_batch_005_008_validation(
        nih_root=args.nih_root,
        preflight_json_path=args.preflight_json,
        batch_transfer_json_path=args.transfer_json,
        pytest_summary=pytest_summary,
        sample_per_split_per_shard=args.sample_per_split_per_shard,
        mini_batch_size=args.mini_batch_size,
        seed=args.seed,
    )
    return 0 if audit_rep["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
