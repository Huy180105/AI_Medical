from typing import Any


class IdentityReranker:
    def rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)
