from typing import Any
from src.ranking.candidate_models import CandidateRankingResult
from src.ranking.vector_store import ClinicalVectorStore
from src.ranking.cross_encoder_reranker import CrossEncoderReranker


class CandidateRetrievalRanker:
    """
    Production-grade 2-stage Candidate Retrieval & Ranking Pipeline for ICD-10 and RxNorm normalization.
    Stage 1: FAISS ANN Vector Retrieval (Top 30 candidates).
    Stage 2: Cross-Encoder Pair Reranker & Softmax Calibrator (Top 5 candidates).
    Features LRU query caching for sub-millisecond repeated queries.
    """

    def __init__(self, cache_size: int = 1000) -> None:
        self.vector_store = ClinicalVectorStore()
        self.reranker = CrossEncoderReranker()
        self.cache_size = cache_size
        self._cache: dict[str, CandidateRankingResult] = {}

    def rank_entity(self, entity_text: str, entity_type: str = "ALL", top_k: int = 5) -> CandidateRankingResult:
        """
        Retrieves and ranks normalization candidates for a single medical entity.
        """
        cache_key = f"{entity_text.lower()}:{entity_type}:{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Stage 1: Vector Store Retrieval (Top 30)
        retrieved_30 = self.vector_store.search_candidates(entity_text, entity_type=entity_type, top_k=30)

        # Stage 2: Cross-Encoder Reranking & Calibration (Top K)
        reranked_top_k = self.reranker.rerank_and_calibrate(entity_text, retrieved_30, top_k=top_k)

        result = CandidateRankingResult(
            entity_text=entity_text,
            entity_type=entity_type,
            top_candidates=reranked_top_k
        )

        # Update Cache
        if len(self._cache) >= self.cache_size:
            # Simple eviction
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[cache_key] = result

        return result

    def rank_batch(self, entities: list[dict[str, Any]], top_k: int = 5) -> list[CandidateRankingResult]:
        """
        Processes a batch of extracted entities and returns Top-K candidate results for each.
        """
        results = []
        for ent in entities:
            text = ent.get("text", "")
            ent_type = ent.get("type", "ALL")
            res = self.rank_entity(text, entity_type=ent_type, top_k=top_k)
            results.append(res)
        return results
