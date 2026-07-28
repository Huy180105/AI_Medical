import time
from typing import Any


class EvaluationLeaderboardManager:
    """
    Stores and ranks platform model benchmark evaluation runs.
    """

    def __init__(self) -> None:
        self.leaderboard: list[dict[str, Any]] = []

    def record_run(self, model_name: str, score_percentage: float, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Records a new evaluation run into the leaderboard.
        """
        run_entry = {
            "rank": 1,
            "model_name": model_name,
            "overall_score_percentage": score_percentage,
            "metrics": metrics,
            "timestamp": time.time()
        }
        self.leaderboard.append(run_entry)
        self._recalculate_ranks()
        return run_entry

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """
        Returns the sorted leaderboard table (highest score first).
        """
        return self.leaderboard

    def _recalculate_ranks(self) -> None:
        """Sorts runs descending by overall score percentage and assigns ranks."""
        self.leaderboard.sort(key=lambda x: x["overall_score_percentage"], reverse=True)
        for idx, entry in enumerate(self.leaderboard):
            entry["rank"] = idx + 1
