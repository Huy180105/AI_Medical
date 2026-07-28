import csv
import html
import json
from pathlib import Path
from typing import Any


class BenchmarkReportExporter:
    def export_json(self, report: dict[str, Any], path: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return output

    def export_csv(self, report: dict[str, Any], path: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self._flatten(report)
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        return output

    def export_html(self, report: dict[str, Any], path: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            f"<tr><td>{html.escape(row['metric'])}</td><td>{html.escape(str(row['value']))}</td></tr>"
            for row in self._flatten(report)
        )
        output.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Benchmark Report</title>"
            "<style>body{font-family:Arial,sans-serif;margin:32px}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #ddd;padding:8px}th{background:#f6f6f6;text-align:left}</style>"
            "</head><body><h1>Medical AI Agent Benchmark Report</h1>"
            f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
            "</body></html>",
            encoding="utf-8",
        )
        return output

    def _flatten(self, payload: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, value in payload.items():
            metric = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                rows.extend(self._flatten(value, metric))
            elif isinstance(value, list):
                rows.append({"metric": metric, "value": json.dumps(value, ensure_ascii=True)})
            else:
                rows.append({"metric": metric, "value": value})
        return rows
