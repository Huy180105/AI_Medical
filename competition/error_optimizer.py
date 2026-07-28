import csv
import time
from pathlib import Path
from typing import Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

ANALYSIS_DIR = Path("data/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


class ErrorDiagnosticProfiler:
    """
    Quantitative Error Diagnostic Profiler categorizing pipeline failures into
    6 specialized CSV reports under data/analysis/.
    """

    def __init__(self, output_dir: Path = ANALYSIS_DIR) -> None:
        self.output_dir = output_dir
        self.boundary_errors: list[dict[str, Any]] = []
        self.assertion_errors: list[dict[str, Any]] = []
        self.relation_errors: list[dict[str, Any]] = []
        self.ranking_errors: list[dict[str, Any]] = []
        self.icd_errors: list[dict[str, Any]] = []
        self.rxnorm_errors: list[dict[str, Any]] = []
        self.latency_logs: list[dict[str, Any]] = []

    def profile_prediction(self, ground_truth: dict[str, Any], prediction: dict[str, Any], latency_ms: float = 0.0) -> None:
        """
        Compares ground truth vs prediction record and categorizes errors.
        """
        doc_id = ground_truth.get("document_id", "doc_unknown")

        # 1. Latency log
        self.latency_logs.append({
            "document_id": doc_id,
            "latency_ms": round(latency_ms, 2)
        })

        # 2. Boundary & Entity errors
        gt_entities = {e.get("text", "").lower(): e for e in ground_truth.get("entities", [])}
        pred_entities = {e.get("text", "").lower(): e for e in prediction.get("entities", [])}

        for text, gt_e in gt_entities.items():
            if text not in pred_entities:
                self.boundary_errors.append({
                    "document_id": doc_id,
                    "entity_text": text,
                    "type": gt_e.get("type", "UNKNOWN"),
                    "error_type": "MISSING_ENTITY"
                })
            else:
                pred_e = pred_entities[text]
                if gt_e.get("start") != pred_e.get("start") or gt_e.get("end") != pred_e.get("end"):
                    self.boundary_errors.append({
                        "document_id": doc_id,
                        "entity_text": text,
                        "expected_span": f"{gt_e.get('start')}:{gt_e.get('end')}",
                        "predicted_span": f"{pred_e.get('start')}:{pred_e.get('end')}",
                        "error_type": "SPAN_MISALIGNMENT"
                    })

        # 3. Assertion errors
        gt_assertions = {a.get("text", "").lower(): a for a in ground_truth.get("assertions", [])}
        pred_assertions = {a.get("text", "").lower(): a for a in prediction.get("assertions", [])}

        for text, gt_a in gt_assertions.items():
            if text in pred_assertions:
                p_a = pred_assertions[text]
                if gt_a.get("is_negated") != p_a.get("is_negated"):
                    self.assertion_errors.append({
                        "document_id": doc_id,
                        "entity_text": text,
                        "expected_negated": gt_a.get("is_negated"),
                        "predicted_negated": p_a.get("is_negated"),
                        "error_type": "NEGATION_MISMATCH"
                    })

        # 4. Relation errors
        gt_relations = {(r.get("subject"), r.get("object")): r for r in ground_truth.get("relations", [])}
        pred_relations = {(r.get("subject"), r.get("object")): r for r in prediction.get("relations", [])}

        for pair, gt_r in gt_relations.items():
            if pair not in pred_relations:
                self.relation_errors.append({
                    "document_id": doc_id,
                    "subject": pair[0],
                    "object": pair[1],
                    "expected_relation": gt_r.get("relation_type"),
                    "error_type": "MISSING_RELATION"
                })

        # 5. Candidate & ICD-10/RxNorm errors
        gt_icd = {c.get("entity_text", "").lower(): c for c in ground_truth.get("icd10_codes", [])}
        pred_icd = {c.get("entity_text", "").lower(): c for c in prediction.get("icd10_codes", [])}

        for text, gt_c in gt_icd.items():
            if text in pred_icd:
                p_c = pred_icd[text]
                if gt_c.get("code") != p_c.get("code"):
                    self.icd_errors.append({
                        "document_id": doc_id,
                        "entity_text": text,
                        "expected_code": gt_c.get("code"),
                        "predicted_code": p_c.get("code"),
                        "error_type": "ICD10_MAPPING_MISMATCH"
                    })

    def export_all_error_csvs(self) -> dict[str, str]:
        """
        Exports all error logs into 6 structured CSV reports under data/analysis/.
        """
        reports = {}

        exports = [
            ("boundary_errors.csv", self.boundary_errors, ["document_id", "entity_text", "type", "error_type", "expected_span", "predicted_span"]),
            ("assertion_errors.csv", self.assertion_errors, ["document_id", "entity_text", "expected_negated", "predicted_negated", "error_type"]),
            ("relation_errors.csv", self.relation_errors, ["document_id", "subject", "object", "expected_relation", "error_type"]),
            ("ranking_errors.csv", self.ranking_errors, ["document_id", "entity_text", "expected_code", "predicted_code", "error_type"]),
            ("icd_errors.csv", self.icd_errors, ["document_id", "entity_text", "expected_code", "predicted_code", "error_type"]),
            ("latency.csv", self.latency_logs, ["document_id", "latency_ms"])
        ]

        for filename, data, fieldnames in exports:
            file_path = self.output_dir / filename
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            reports[filename] = str(file_path)

        logger.info("Successfully exported %d error diagnostic reports to '%s'.", len(reports), self.output_dir)
        return reports
