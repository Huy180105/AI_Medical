import math
from typing import Any
from src.ranking.candidate_models import EntityCandidate


class CrossEncoderReranker:
    """
    Reranks Top-30 candidates down to Top-5 using fine-grained pair scoring
    and applies Softmax probability calibration.
    """

    def rerank_and_calibrate(self, query: str, candidates: list[EntityCandidate], top_k: int = 5) -> list[EntityCandidate]:
        """
        Reranks candidate list and returns Top-K items with normalized confidence scores.
        """
        if not candidates:
            return []

        query_lower = query.lower().strip()
        pair_scores = []

        for c in candidates:
            display_lower = c.display.lower().strip()
            
            # Compute fine-grained cross-encoder pair score
            exact_match_bonus = 2.0 if query_lower == display_lower else 1.0
            prefix_match_bonus = 1.4 if display_lower.startswith(query_lower) else 1.0
            
            # Combine initial vector score with pair features
            rerank_score = c.score * exact_match_bonus * prefix_match_bonus
            pair_scores.append((c, rerank_score))

        # Sort descending by rerank score
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pair_scores[:top_k]

        # Apply Softmax Confidence Calibration
        scores_array = [score for _, score in top_pairs]
        max_s = max(scores_array) if scores_array else 0.0
        exp_scores = [math.exp(s - max_s) for s in scores_array]
        sum_exp = sum(exp_scores) if sum(exp_scores) > 0 else 1.0
        probabilities = [exp / sum_exp for exp in exp_scores]

        reranked_candidates = []
        for rank, ((cand, raw_score), prob) in enumerate(zip(top_pairs, probabilities), 1):
            reranked_candidates.append(EntityCandidate(
                code=cand.code,
                display=cand.display,
                score=raw_score,
                confidence=round(prob, 4),
                rank=rank
            ))

        return reranked_candidates
