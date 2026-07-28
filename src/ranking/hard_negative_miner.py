from typing import Any
from src.ranking.candidate_models import EntityCandidate
from src.ranking.vector_store import ClinicalVectorStore


class HardNegativeMiner:
    """
    Mines confusing, high-similarity Hard Negatives (e.g. K21.0 GERD with esophagitis vs K21.9 GERD unspecified)
    and computes margin ranking loss to calibrate CrossEncoder decision boundaries.
    """

    def __init__(self, margin_gamma: float = 0.2) -> None:
        self.vector_store = ClinicalVectorStore()
        self.margin_gamma = margin_gamma

    def mine_hard_negatives(self, query: str, expected_code: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieves top candidate matches and filters out the true positive to generate hard negatives.
        """
        all_candidates = self.vector_store.search_candidates(query, top_k=top_k + 5)
        
        hard_negatives = []
        for cand in all_candidates:
            if cand.code != expected_code:
                hard_negatives.append({
                    "code": cand.code,
                    "display": cand.display,
                    "similarity_score": cand.score
                })
            if len(hard_negatives) >= top_k:
                break

        return hard_negatives

    def calculate_margin_ranking_loss(self, pos_score: float, neg_score: float) -> float:
        """
        Calculates Triple Margin Ranking Loss: L = max(0, gamma - pos_score + neg_score).
        """
        loss = max(0.0, self.margin_gamma - pos_score + neg_score)
        return round(loss, 4)
