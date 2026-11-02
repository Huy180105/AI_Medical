"""
Streaming Multi-Label Inference & Reproducible Statistical Benchmark for NIH ChestX-ray14.

Features:
1. Streaming mini-batch inference (`--batch-size`, default 32) via `PretrainedNihDenseNet121`:
   never loads all 112,120 radiographs into RAM and streams row-level predictions directly to CSV.
2. Strict binary {0, 1} label governance (`LabelDomainPolicy.BINARY_STRICT`) across all 14 `NIH_LABELS`
   (explicitly rejects CheXpert `-1` uncertainty values).
3. Validation-only threshold selection (`validation_optimal_threshold` on `validate` split; frozen and
   applied to `test` split; TorchXRayVision `source_op_thresh` recorded as metadata only).
4. Patient-clustered bootstrap 95% confidence intervals grouped strictly by `patient_id`.
5. Outputs:
   - `reports/nih_predictions.csv`
   - `reports/nih_benchmark_report.json`

Zero dependency on PhysioNet, MIMIC-CXR-JPG, or MIMIC-IV-ED.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_nih_by_patient import (
    sample_patient_stratified_subset,
    verify_zero_patient_leakage,
)
from src.evaluation.benchmark_fusion import (
    AUPRC_EXACT_DEFINITION,
    BenchmarkFusionEvaluator,
    InsufficientValidationDiversityError,
    LabelDomainPolicy,
    ModalityType,
    NIH_LABELS,
    SyntheticDataQuarantineError,
    collect_reproducibility_metadata,
    compute_file_sha256,
)
from src.utils.logger import get_logger
from src.vision.pretrained_cxr import PretrainedNihDenseNet121

logger = get_logger(__name__)


def _resolve_manifest_image_path(raw_path_str: str, project_root: Path = PROJECT_ROOT) -> Path:
    p = Path(raw_path_str)
    if p.is_absolute():
        return p
    return project_root / p


def stream_nih_batch_inference(
    eval_manifest_df: pd.DataFrame,
    adapter: PretrainedNihDenseNet121,
    predictions_csv_path: Path,
    batch_size: int = 32,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, int]:
    """
    Streams mini-batches of `VALID` radiographs through `PretrainedNihDenseNet121.predict_batch()`
    and writes predictions incrementally to `predictions_csv_path` with O(batch_size) RAM usage.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    predictions_csv_path = Path(predictions_csv_path)
    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)

    meta_cols = [
        "image_id",
        "patient_id",
        "split",
        "shard_id",
        "view_position",
        "patient_gender",
        "age_group",
        "label_source",
        "is_synthetic",
    ]
    y_cols = [f"y_true_{lbl}" for lbl in NIH_LABELS]
    p_cols = [f"p_cxr_{lbl}" for lbl in NIH_LABELS]
    fieldnames = meta_cols + y_cols + p_cols

    total_rows = len(eval_manifest_df)
    processed_rows = 0
    runtime_corrupt_skipped = 0

    with open(predictions_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for start_idx in range(0, total_rows, batch_size):
            chunk = eval_manifest_df.iloc[start_idx : start_idx + batch_size]
            valid_chunk_rows: List[pd.Series] = []
            valid_img_paths: List[Path] = []

            for _, row in chunk.iterrows():
                img_path = _resolve_manifest_image_path(str(row["image_path"]), project_root=project_root)
                if not img_path.exists():
                    runtime_corrupt_skipped += 1
                    continue
                valid_chunk_rows.append(row)
                valid_img_paths.append(img_path)

            if not valid_img_paths:
                continue

            try:
                batch_preds = adapter.predict_batch(valid_img_paths)
            except Exception as batch_exc:
                # Isolate any unexpectedly corrupted image at decode time without aborting healthy rows
                logger.warning("Batch decode warning (%s); isolating per-image in current mini-batch.", batch_exc)
                batch_preds = []
                filtered_rows: List[pd.Series] = []
                for r_item, p_item in zip(valid_chunk_rows, valid_img_paths):
                    try:
                        pred_one = adapter.predict_batch([p_item])[0]
                        filtered_rows.append(r_item)
                        batch_preds.append(pred_one)
                    except Exception as single_exc:
                        logger.warning("Skipping unreadable image %s: %s", p_item, single_exc)
                        runtime_corrupt_skipped += 1
                valid_chunk_rows = filtered_rows

            for row, pred in zip(valid_chunk_rows, batch_preds):
                out_row: Dict[str, Any] = {
                    "image_id": str(row["image_id"]),
                    "patient_id": str(row["patient_id"]),
                    "split": str(row["split"]),
                    "shard_id": str(row.get("shard_id", "")),
                    "view_position": str(row.get("view_position", "")),
                    "patient_gender": str(row.get("patient_gender", "")),
                    "age_group": str(row.get("age_group", "")),
                    "label_source": str(row.get("label_source", "NIH_CHESTXRAY14_NLP_2017")),
                    "is_synthetic": bool(row.get("is_synthetic", False)),
                }
                probs_map = pred["nih_14_probabilities"]
                for lbl in NIH_LABELS:
                    out_row[f"y_true_{lbl}"] = int(row[lbl])
                    out_row[f"p_cxr_{lbl}"] = probs_map[lbl]
                writer.writerow(out_row)
                processed_rows += 1

    return {
        "requested_valid_rows": int(total_rows),
        "inferred_rows_written": int(processed_rows),
        "runtime_unreadable_skipped": int(runtime_corrupt_skipped),
    }


def _compute_macro_summary(per_label_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Computes unweighted macro-average across evaluatable NIH-14 pathologies."""
    metric_keys = ["auroc", "auprc", "f1", "sensitivity", "specificity", "brier_score", "ece"]
    macro: Dict[str, Any] = {}
    evaluatable_labels: List[str] = []
    skipped_labels: Dict[str, str] = {}

    for lbl, rep in per_label_reports.items():
        if rep.get("status") != "EVALUATED":
            skipped_labels[lbl] = str(rep.get("reason", "not_evaluated"))
            continue
        evaluatable_labels.append(lbl)

    for mk in metric_keys:
        vals = [
            rep["test_metrics"][mk]
            for lbl, rep in per_label_reports.items()
            if rep.get("status") == "EVALUATED" and rep["test_metrics"].get(mk) is not None
        ]
        macro[f"macro_{mk}"] = round(float(np.mean(vals)), 4) if vals else None
        macro[f"{mk}_valid_label_count"] = len(vals)

    macro["evaluated_label_count"] = len(evaluatable_labels)
    macro["evaluated_labels"] = evaluatable_labels
    macro["non_evaluated_labels"] = skipped_labels
    return macro


def run_nih_pipeline_evaluation(
    manifest_path: Path = Path("data/kaggle_nih/manifests/nih_manifest.csv"),
    predictions_out: Path = Path("reports/nih_predictions.csv"),
    report_out: Path = Path("reports/nih_benchmark_report.json"),
    checkpoint_profile: str = "nih14",
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
    mode: str = "benchmark",
    batch_size: int = 32,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
    threshold_metric: str = "f1",
    splits_to_infer: Sequence[str] = ("validate", "test"),
    shard_filter: Optional[Sequence[str]] = None,
    max_patients_per_split: Optional[int] = None,
    reuse_predictions: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """
    End-to-end streaming inference and 14-label patient-clustered statistical benchmark on NIH ChestX-ray14.
    """
    if mode not in {"benchmark", "real_pilot", "software_test"}:
        raise ValueError(f"Unsupported mode='{mode}'. Must be 'benchmark', 'real_pilot', or 'software_test'.")

    if mode == "benchmark" and (max_patients_per_split is not None or shard_filter is not None):
        raise ValueError(
            "Subsampling (`--max-patients-per-split` or `--shards`) is forbidden in `--mode benchmark`. "
            "Use `--mode real_pilot` for partial-shard or batch pilot runs."
        )

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"NIH manifest not found at {manifest_path}. Run `python scripts/build_nih_cxr_manifest.py` first."
        )

    manifest_df = pd.read_csv(manifest_path, low_memory=False)
    manifest_df["patient_id"] = manifest_df["patient_id"].astype(str)

    # Enforce synthetic quarantine on real modes
    if mode in {"benchmark", "real_pilot"} and "is_synthetic" in manifest_df.columns:
        syn_cnt = int(manifest_df["is_synthetic"].fillna(False).astype(bool).sum())
        if syn_cnt > 0:
            raise SyntheticDataQuarantineError(
                f"Refusing `{mode}` run: {syn_cnt} synthetic records detected in {manifest_path}."
            )

    # Verify zero patient leakage across the entire manifest upfront
    leakage_proof = verify_zero_patient_leakage(manifest_df, patient_col="patient_id", split_col="split")

    total_manifest_rows = len(manifest_df)
    status_series = manifest_df["image_status"].astype(str) if "image_status" in manifest_df.columns else pd.Series(["VALID"] * len(manifest_df))
    missing_rows = int((status_series == "MISSING").sum())
    corrupt_rows = int((status_series == "CORRUPT").sum())

    valid_df = manifest_df[status_series == "VALID"].copy()
    if shard_filter:
        norm_shards = {s.strip().lower() for s in shard_filter}
        valid_df = valid_df[valid_df["shard_id"].astype(str).str.lower().isin(norm_shards)].copy()

    if max_patients_per_split is not None:
        valid_df = sample_patient_stratified_subset(
            valid_df,
            max_patients_per_split=max_patients_per_split,
            seed=seed,
        )

    infer_df = valid_df[valid_df["split"].isin(set(splits_to_infer))].copy()
    if infer_df.empty:
        raise ValueError(
            f"Zero VALID images available for splits {list(splits_to_infer)} in {manifest_path} "
            f"(MISSING={missing_rows}, CORRUPT={corrupt_rows}). Download image shards first via `scripts/download_kaggle_nih.py`."
        )

    adapter = PretrainedNihDenseNet121(
        weights_path=weights_path,
        checkpoint_profile=checkpoint_profile,
        device=device,
        verify_sha256=True,
    )

    predictions_out = Path(predictions_out)
    if reuse_predictions and predictions_out.exists():
        logger.info("Reusing existing predictions CSV at %s", predictions_out)
        stream_stats = {
            "requested_valid_rows": int(len(infer_df)),
            "inferred_rows_written": int(len(pd.read_csv(predictions_out))),
            "runtime_unreadable_skipped": 0,
            "reused_existing_predictions": True,
        }
    else:
        stream_stats = stream_nih_batch_inference(
            eval_manifest_df=infer_df,
            adapter=adapter,
            predictions_csv_path=predictions_out,
            batch_size=batch_size,
            project_root=project_root,
        )
        stream_stats["reused_existing_predictions"] = False

    preds_df = pd.read_csv(predictions_out, low_memory=False)
    preds_df["patient_id"] = preds_df["patient_id"].astype(str)

    strict_provenance = mode != "software_test"
    require_val_diversity = mode == "benchmark"

    evaluator = BenchmarkFusionEvaluator(
        bootstrap_iterations=bootstrap_iterations,
        random_seed=seed,
        strict_patient_split=True,
        label_domain_policy=LabelDomainPolicy.BINARY_STRICT,
    )

    per_label_reports: Dict[str, Dict[str, Any]] = {}
    for lbl in NIH_LABELS:
        p_col = f"p_cxr_{lbl}"
        y_col = f"y_true_{lbl}"

        if lbl in adapter.masked_nih_labels:
            per_label_reports[lbl] = {
                "status": "MASKED_UNTRAINED_IN_CHECKPOINT",
                "reason": (
                    f"Label '{lbl}' is masked (untrained zero weights) in checkpoint profile "
                    f"'{checkpoint_profile}' and excluded from evaluation rather than emitting constant 0.5."
                ),
            }
            continue

        label_eval_df = pd.DataFrame({
            "patient_id": preds_df["patient_id"].astype(str),
            "image_id": preds_df["image_id"].astype(str),
            "split": preds_df["split"].astype(str),
            "y_true": preds_df[y_col],
            "p_cxr": preds_df[p_col],
            "label_source": preds_df["label_source"],
            "is_synthetic": preds_df["is_synthetic"].astype(bool),
        })

        try:
            rep = evaluator.evaluate(
                label_eval_df,
                modality=ModalityType.CXR_ONLY,
                strict_provenance=strict_provenance,
                source_op_thresh=adapter.get_source_op_thresh(lbl),
                threshold_metric=threshold_metric,
                require_validation_diversity=require_val_diversity,
                target_name=lbl,
            )
            rep["status"] = "EVALUATED"
            per_label_reports[lbl] = rep
        except InsufficientValidationDiversityError as div_err:
            if mode == "benchmark":
                raise
            per_label_reports[lbl] = {
                "status": "SKIPPED_INSUFFICIENT_PILOT_VALIDATION_POSITIVES",
                "reason": str(div_err),
            }

    macro_summary = _compute_macro_summary(per_label_reports)
    model_meta = adapter.get_model_metadata()

    bootstrap_cfg = {
        "resampling_unit": "patient_id",
        "requested_iterations": int(bootstrap_iterations),
        "confidence_level": 0.95,
        "alpha": 0.05,
        "random_seed": int(seed),
        "ci_method": "percentile_from_valid_iterations_only",
    }

    repro_meta = collect_reproducibility_metadata(
        random_seed=seed,
        uncertainty_policy="binary_strict_not_applicable",
        execution_mode=mode,
        checkpoint_path=adapter.weights_path,
        checkpoint_sha256=adapter.checkpoint_sha256,
        input_manifest_path=manifest_path,
        bootstrap_config=bootstrap_cfg,
        threshold_provenance={
            "selection_split": "validate",
            "selection_metric": threshold_metric,
            "source_op_thresh_used_for_test_metrics": False,
            "used_test_labels_for_threshold": False,
        },
        target_label_definitions={
            "dataset": "NIH ChestX-ray14",
            "labels": list(NIH_LABELS),
            "no_finding_rule": "All 14 disease columns == 0 (not a 15th disease class)",
            "label_domain_policy": LabelDomainPolicy.BINARY_STRICT.value,
        },
    )
    repro_meta["predictions_csv_path"] = str(predictions_out)
    repro_meta["predictions_csv_sha256"] = compute_file_sha256(predictions_out)

    final_report: Dict[str, Any] = {
        "report_schema_version": "2.0.0-nih14",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": mode,
        "dataset_governance": {
            "dataset_name": "NIH ChestX-ray14",
            "kaggle_source": "https://www.kaggle.com/datasets/nih-chest-xrays/data",
            "originating_institution": "NIH Clinical Center",
            "primary_citation": (
                "Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray "
                "Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax "
                "Diseases. IEEE CVPR 2017 (arXiv:1705.02315)."
            ),
            "physionet_or_mimic_used": False,
            "scientific_limitations": [
                "NIH ChestX-ray14 labels are NLP-extracted binary {0, 1} tags from clinical radiology reports and contain documented NLP extraction noise; they are NOT radiologist consensus ground truth.",
                "NIH ChestX-ray14 ontology is NOT equivalent to Stanford CheXpert; CheXpert -1 uncertainty policies are strictly excluded.",
                model_meta["training_overlap_disclosure"],
            ],
        },
        "model_metadata": model_meta,
        "slot_verification_table": adapter.get_slot_audit_table(),
        "reproducibility_metadata": repro_meta,
        "cohort_accounting": {
            "manifest_total_rows": int(total_manifest_rows),
            "manifest_missing_image_rows_skipped": missing_rows,
            "manifest_corrupt_image_rows_skipped": corrupt_rows,
            "shard_filter": list(shard_filter) if shard_filter else None,
            "max_patients_per_split": max_patients_per_split,
            "streaming_batch_size": int(batch_size),
            "streaming_inference_stats": stream_stats,
            "evaluated_validate_images": int((preds_df["split"] == "validate").sum()),
            "evaluated_validate_patients": int(preds_df.loc[preds_df["split"] == "validate", "patient_id"].nunique()),
            "evaluated_test_images": int((preds_df["split"] == "test").sum()),
            "evaluated_test_patients": int(preds_df.loc[preds_df["split"] == "test", "patient_id"].nunique()),
            "leakage_verification": leakage_proof,
        },
        "metric_definitions": {
            "auroc": "Area Under the ROC Curve (sklearn.metrics.roc_auc_score).",
            "auprc": AUPRC_EXACT_DEFINITION,
            "f1": "Binary F1-score on test split using frozen validation_optimal_threshold.",
            "sensitivity": "TP / (TP + FN) on test split using frozen validation_optimal_threshold.",
            "specificity": "TN / (TN + FP) on test split using frozen validation_optimal_threshold.",
            "brier_score": "Mean squared probability error mean((p - y)^2).",
            "ece": "10-bin Expected Calibration Error on [0, 1].",
            "confidence_intervals": "95% percentile CIs from valid dual-class patient_id-clustered bootstrap resamples.",
        },
        "macro_summary": macro_summary,
        "per_pathology_results": per_label_reports,
    }

    report_out = Path(report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(
        "Completed NIH-14 evaluation (%s): macro_auroc=%s, macro_auprc=%s -> %s",
        mode,
        macro_summary.get("macro_auroc"),
        macro_summary.get("macro_auprc"),
        report_out,
    )
    return final_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run streaming mini-batch CXR inference and 14-label patient-clustered evaluation on NIH ChestX-ray14."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/kaggle_nih/manifests/nih_manifest.csv"),
        help="Path to canonical NIH manifest CSV.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("reports/nih_predictions.csv"),
        help="Output CSV path for streamed per-image predictions.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/nih_benchmark_report.json"),
        help="Output JSON path for benchmark report.",
    )
    parser.add_argument(
        "--checkpoint-profile",
        choices=["nih14", "chex_external"],
        default="nih14",
        help="Checkpoint profile: 'nih14' (all 14 NIH labels) or 'chex_external' (zero-overlap 7 shared labels).",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional custom path to DenseNet-121 .pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cpu' or 'cuda'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "real_pilot", "software_test"],
        default="benchmark",
        help="Execution mode: 'benchmark' (full valid validate+test), 'real_pilot' (batch/shard subset), or 'software_test'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for streaming inference (default: 32).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of patient_id-clustered bootstrap resamples (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "youden"],
        default="f1",
        help="Validation-only threshold optimization criterion (default: f1).",
    )
    parser.add_argument(
        "--shards",
        type=str,
        nargs="+",
        default=None,
        help="Optional shard filter (e.g. images_001 images_002) for `--mode real_pilot`.",
    )
    parser.add_argument(
        "--max-patients-per-split",
        type=int,
        default=None,
        help="Optional patient-stratified cap per split for `--mode real_pilot`.",
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Skip model forward pass and re-evaluate existing `--predictions-out` CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_nih_pipeline_evaluation(
        manifest_path=args.manifest,
        predictions_out=args.predictions_out,
        report_out=args.report_out,
        checkpoint_profile=args.checkpoint_profile,
        weights_path=args.weights_path,
        device=args.device,
        mode=args.mode,
        batch_size=args.batch_size,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        threshold_metric=args.threshold_metric,
        shard_filter=args.shards,
        max_patients_per_split=args.max_patients_per_split,
        reuse_predictions=args.reuse_predictions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
