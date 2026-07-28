import time
import pytest
from src.ranking.vector_store import ClinicalVectorStore
from src.ranking.cross_encoder_reranker import CrossEncoderReranker
from src.ranking.candidate_ranker import CandidateRetrievalRanker
from src.ranking.benchmark_ranking import RankingBenchmarkSuite


def test_vector_store_retrieval():
    vs = ClinicalVectorStore()
    cands = vs.search_candidates("viêm phổi", entity_type="DISEASE", top_k=30)
    
    assert len(cands) >= 1
    codes = [c.code for c in cands]
    assert "J18.9" in codes or "J18.0" in codes


def test_cross_encoder_reranker_and_softmax():
    vs = ClinicalVectorStore()
    cands = vs.search_candidates("Paracetamol 500mg", entity_type="MEDICINE", top_k=30)
    
    reranker = CrossEncoderReranker()
    top5 = reranker.rerank_and_calibrate("Paracetamol 500mg", cands, top_k=5)
    
    assert len(top5) <= 5
    assert top5[0].rank == 1
    # Check Softmax probability sum ~ 1.0
    conf_sum = sum(c.confidence for c in top5)
    assert abs(conf_sum - 1.0) < 1e-3


def test_candidate_retrieval_ranker_caching_and_batch():
    ranker = CandidateRetrievalRanker()
    
    # 1. Single query
    res1 = ranker.rank_entity("trào ngược dạ dày", entity_type="DISEASE", top_k=5)
    assert res1.entity_text == "trào ngược dạ dày"
    assert len(res1.top_candidates) >= 1
    assert res1.top_candidates[0].code in ("K21.9", "K21.0")

    # 2. LRU Cache hit test
    t0 = time.perf_counter()
    res2 = ranker.rank_entity("trào ngược dạ dày", entity_type="DISEASE", top_k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 1.0  # Zero-latency cache hit
    assert res1.top_candidates[0].code == res2.top_candidates[0].code

    # 3. Batch processing
    batch = [
        {"text": "ho kéo dài", "type": "SYMPTOM"},
        {"text": "Ibuprofen", "type": "MEDICINE"}
    ]
    batch_res = ranker.rank_batch(batch)
    assert len(batch_res) == 2


def test_ranking_benchmark_suite():
    suite = RankingBenchmarkSuite()
    test_ds = [
        {"query": "viêm phổi thùy", "type": "DISEASE", "expected_code": "J18.9"},
        {"query": "Paracetamol", "type": "MEDICINE", "expected_code": "RxNorm:161"},
        {"query": "sốt cao", "type": "SYMPTOM", "expected_code": "R50.9"}
    ]
    
    metrics = suite.evaluate_benchmark(test_ds)
    assert "top1_accuracy" in metrics
    assert "top5_accuracy" in metrics
    assert "mrr" in metrics
    assert metrics["top1_accuracy"] > 0.0
    assert metrics["top5_accuracy"] >= metrics["top1_accuracy"]
