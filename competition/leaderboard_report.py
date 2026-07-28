from typing import Any


class LeaderboardReportGenerator:
    """
    Formats empirical encoder evaluation results into a quantitative Markdown leaderboard table.
    """

    @staticmethod
    def generate_leaderboard_markdown(benchmark_summary: dict[str, Any]) -> str:
        """
        Generates a formatted Markdown leaderboard table comparing model backbones.
        """
        md_lines = [
            "# Viettel AI Race — Quantitative Empirical Leaderboard",
            "",
            "| Rank | Encoder Backbone | Model ID | Top-1 Accuracy | Top-5 Accuracy | Recall@5 | MRR | Latency (ms) |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
        ]

        # Sort models descending by Top-1 Accuracy then MRR
        sorted_models = sorted(
            benchmark_summary.items(),
            key=lambda x: (x[1].get("top1_accuracy", 0.0), x[1].get("mrr", 0.0)),
            reverse=True
        )

        for rank, (name, metrics) in enumerate(sorted_models, 1):
            top1 = metrics.get("top1_accuracy", 0.0)
            top5 = metrics.get("top5_accuracy", 0.0)
            rec5 = metrics.get("recall_at_5", 0.0)
            mrr = metrics.get("mrr", 0.0)
            lat = metrics.get("avg_latency_ms", 0.0)
            
            md_lines.append(
                f"| {rank} | **{name}** | `models/{name.lower()}` | {top1:.4f} | {top5:.4f} | {rec5:.4f} | {mrr:.4f} | {lat:.2f} ms |"
            )

        md_lines.append("")
        md_lines.append("*All runs empirically logged into MLflow Registry.*")
        return "\n".join(md_lines)
