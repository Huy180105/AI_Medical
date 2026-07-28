import csv
from pathlib import Path
from typing import Any

ERROR_DIR = Path("data/evaluation")
ERROR_DIR.mkdir(parents=True, exist_ok=True)


class ErrorDiagnosticAnalyzer:
    """
    Analyzes misclassified evaluation instances and logs detailed diagnostic reports to errors.csv.
    """

    ERROR_CATEGORIES = [
        "NER_BOUNDARY",
        "NEGATION_MISMATCH",
        "ASSERTION_LEAK",
        "CANDIDATE_CONFUSION"
    ]

    def __init__(self, output_csv_path: Path = ERROR_DIR / "errors.csv") -> None:
        self.output_csv_path = output_csv_path
        self.error_logs: list[dict[str, Any]] = []

    def log_error(
        self,
        query_text: str,
        category: str,
        expected: str,
        predicted: str,
        confidence: float = 0.0
    ) -> dict[str, Any]:
        """
        Logs a single misclassification instance.
        """
        if category not in self.ERROR_CATEGORIES:
            category = "CANDIDATE_CONFUSION"

        entry = {
            "query_text": query_text,
            "category": category,
            "expected": expected,
            "predicted": predicted,
            "confidence": round(confidence, 4)
        }
        self.error_logs.append(entry)
        return entry

    def export_csv_report(self) -> str:
        """
        Writes all logged errors to CSV file and returns file path string.
        """
        file_path = str(self.output_csv_path)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["query_text", "category", "expected", "predicted", "confidence"]
            )
            writer.writeheader()
            for err in self.error_logs:
                writer.writerow(err)
        return file_path

    def summarize_errors(self) -> dict[str, Any]:
        """
        Summarizes error counts by category.
        """
        summary = {cat: 0 for cat in self.ERROR_CATEGORIES}
        for err in self.error_logs:
            cat = err["category"]
            summary[cat] = summary.get(cat, 0) + 1

        return {
            "total_errors": len(self.error_logs),
            "category_counts": summary
        }
