from pathlib import Path
import pytest
from src.ranking.biomedical_encoder import BiomedicalEncoderBenchmark
from src.ranking.hard_negative_miner import HardNegativeMiner
from src.evaluation.error_analyzer import ErrorDiagnosticAnalyzer


def test_biomedical_encoder_benchmark():
    bm = BiomedicalEncoderBenchmark()
    ds = [
        {"query": "viêm phổi thùy", "type": "DISEASE", "expected_code": "J18.9"},
        {"query": "Paracetamol", "type": "MEDICINE", "expected_code": "RxNorm:161"}
    ]
    
    results = bm.benchmark_all_encoders(ds)
    assert len(results) == 4
    for r in results:
        assert "encoder_name" in r
        assert "top1_accuracy" in r


def test_hard_negative_miner():
    miner = HardNegativeMiner(margin_gamma=0.2)
    negs = miner.mine_hard_negatives("trào ngược dạ dày", expected_code="K21.9", top_k=3)
    
    assert len(negs) >= 1
    for n in negs:
        assert n["code"] != "K21.9"

    # Loss calculation
    loss = miner.calculate_margin_ranking_loss(pos_score=0.9, neg_score=0.8)
    # 0.2 - 0.9 + 0.8 = 0.1
    assert loss == 0.1


def test_error_diagnostic_analyzer():
    analyzer = ErrorDiagnosticAnalyzer()
    analyzer.log_error("ho đờm", "NER_BOUNDARY", expected="ho đờm", predicted="ho", confidence=0.7)
    analyzer.log_error("không sốt", "NEGATION_MISMATCH", expected="Negated", predicted="Positive", confidence=0.6)
    
    summary = analyzer.summarize_errors()
    assert summary["total_errors"] == 2
    assert summary["category_counts"]["NER_BOUNDARY"] == 1
    
    csv_path = analyzer.export_csv_report()
    assert Path(csv_path).exists()
    
    # Cleanup test CSV
    Path(csv_path).unlink(missing_ok=True)
