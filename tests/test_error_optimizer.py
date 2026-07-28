from pathlib import Path
import pytest
from competition.error_optimizer import ErrorDiagnosticProfiler
from competition.hyperparam_optimizer import HyperparameterGridOptimizer


def test_error_diagnostic_profiler(tmp_path: Path):
    profiler = ErrorDiagnosticProfiler(output_dir=tmp_path)
    
    gt = {
        "document_id": "doc_001",
        "entities": [{"text": "ho đờm", "type": "SYMPTOM", "start": 10, "end": 16}],
        "assertions": [{"text": "ho đờm", "is_negated": False}],
        "relations": [{"subject": "Amoxicillin", "object": "viêm phổi", "relation_type": "TREATS"}],
        "icd10_codes": [{"entity_text": "viêm phổi", "code": "J18.9"}]
    }
    
    pred = {
        "document_id": "doc_001",
        "entities": [{"text": "ho đờm", "type": "SYMPTOM", "start": 10, "end": 18}],  # Span mismatch
        "assertions": [{"text": "ho đờm", "is_negated": True}],                         # Negation mismatch
        "relations": [],                                                                       # Missing relation
        "icd10_codes": [{"entity_text": "viêm phổi", "code": "J18.0"}]                         # Code mismatch
    }
    
    profiler.profile_prediction(gt, pred, latency_ms=12.5)
    reports = profiler.export_all_error_csvs()
    
    assert len(reports) == 6
    for filename, filepath in reports.items():
        assert Path(filepath).exists()


def test_hyperparameter_grid_optimizer():
    optimizer = HyperparameterGridOptimizer()
    eval_dataset = [{"query": "test"}]
    
    res = optimizer.grid_search(eval_dataset)
    
    assert res["total_trials"] == 64
    assert "best_config" in res
    assert "f1_score" in res["best_config"]
    assert res["best_config"]["assertion_window_chars"] == 60
    assert res["best_config"]["candidate_retrieval_k"] == 30
    assert res["best_config"]["confidence_threshold"] == 0.52
