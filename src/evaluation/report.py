import json
from typing import Any


class EvaluationReportGenerator:
    """
    Formats evaluation run metrics into clean Markdown and JSON reports.
    """

    @staticmethod
    def generate_markdown_report(benchmark_data: dict[str, Any]) -> str:
        """
        Generates a formatted Markdown evaluation report.
        """
        score = benchmark_data.get("overall_score_percentage", 0.0)
        throughput = benchmark_data.get("throughput", {})
        ner = benchmark_data.get("ner_evaluation", {})
        cdss = benchmark_data.get("cdss_evaluation", {})
        wearable = benchmark_data.get("wearable_evaluation", {})

        md = f"""# Clinical Intelligence Platform — Benchmark Report

## 1. Executive Summary
- **Platform Overall Evaluation Score**: **{score}%**
- **Evaluation Throughput**: `{throughput.get('throughput_samples_per_sec', 0.0)} samples/sec`
- **Average Latency**: `{throughput.get('avg_latency_ms', 0.0)} ms`

---

## 2. Clinical NER Extraction Performance
- **Precision**: `{ner.get('precision', 0.0)}`
- **Recall**: `{ner.get('recall', 0.0)}`
- **F1 Score**: `{ner.get('f1', 0.0)}`

---

## 3. Clinical Decision Support System (CDSS)
- **Clinical Accuracy**: `{cdss.get('accuracy', 0.0)}`
- **Sensitivity**: `{cdss.get('sensitivity', 0.0)}`
- **Specificity**: `{cdss.get('specificity', 0.0)}`
- **Safety Compliance Rate**: `{cdss.get('safety_compliance_rate', 1.0) * 100.0}%`

---

## 4. Wearable AI & Sensor Telemetry
- **Anomaly Detection Sensitivity**: `{wearable.get('sensitivity', 0.0)}`
- **Anomaly Detection Specificity**: `{wearable.get('specificity', 0.0)}`
- **AUROC Score**: `{wearable.get('auroc', 0.5)}`

---
*Generated automatically by EvaluationReportGenerator.*
"""
        return md

    @staticmethod
    def generate_json_report(benchmark_data: dict[str, Any]) -> str:
        """
        Returns pretty-printed JSON evaluation payload.
        """
        return json.dumps(benchmark_data, indent=2, ensure_ascii=False)
