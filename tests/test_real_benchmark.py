import pytest
from competition.evaluate_real import EmpiricalModelEvaluator
from competition.leaderboard_report import LeaderboardReportGenerator


def test_empirical_model_evaluator():
    evaluator = EmpiricalModelEvaluator(experiment_name="Test_Empirical_Proof")
    dataset = [
        {"query": "viêm phổi thùy", "type": "DISEASE", "expected_code": "J18.9"},
        {"query": "Paracetamol 500mg", "type": "MEDICINE", "expected_code": "RxNorm:161"},
        {"query": "trào ngược dạ dày", "type": "DISEASE", "expected_code": "K21.9"}
    ]
    
    summary = evaluator.run_empirical_evaluation(dataset)
    assert len(summary) == 4
    
    for enc_name in ["BGE_SMALL", "E5_SMALL", "VIHEALTHBERT", "PHOBERT"]:
        assert enc_name in summary
        m = summary[enc_name]
        assert "top1_accuracy" in m
        assert "top5_accuracy" in m
        assert "mrr" in m
        assert m["top1_accuracy"] > 0.0


def test_leaderboard_report_generator():
    dummy_summary = {
        "BGE_SMALL": {"top1_accuracy": 0.92, "top5_accuracy": 0.98, "recall_at_5": 0.98, "mrr": 0.95, "avg_latency_ms": 4.2},
        "PHOBERT": {"top1_accuracy": 0.88, "top5_accuracy": 0.95, "recall_at_5": 0.95, "mrr": 0.91, "avg_latency_ms": 5.1}
    }
    
    md_table = LeaderboardReportGenerator.generate_leaderboard_markdown(dummy_summary)
    assert "# Viettel AI Race — Quantitative Empirical Leaderboard" in md_table
    assert "**BGE_SMALL**" in md_table
    assert "0.9200" in md_table
