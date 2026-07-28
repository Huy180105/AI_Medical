import pytest
from fastapi.testclient import TestClient

from src.evaluation.dataset_loader import EvaluationDatasetLoader
from src.evaluation.clinical_metrics import ClinicalMetricsCalculator
from src.evaluation.confusion_matrix import ConfusionMatrixGenerator
from src.evaluation.metrics import EvaluationMetricsEngine
from src.evaluation.benchmark import PlatformBenchmarkOrchestrator
from src.evaluation.compare_models import ModelComparisonEngine
from src.evaluation.ab_testing import ABTestingSimulator
from src.evaluation.leaderboard import EvaluationLeaderboardManager
from src.evaluation.report import EvaluationReportGenerator
from src.api.app import create_app


def test_dataset_loader():
    ner_samples = EvaluationDatasetLoader.get_ner_test_samples()
    cdss_samples = EvaluationDatasetLoader.get_cdss_test_samples()
    wearable_samples = EvaluationDatasetLoader.get_wearable_test_samples()

    assert len(ner_samples) >= 3
    assert len(cdss_samples) >= 2
    assert len(wearable_samples) >= 4


def test_clinical_metrics():
    true_ents = [{"text": "sốt", "type": "SYMPTOM"}, {"text": "ho", "type": "SYMPTOM"}]
    pred_ents = [{"text": "sốt", "type": "SYMPTOM"}]

    f1_res = ClinicalMetricsCalculator.calculate_precision_recall_f1(true_ents, pred_ents)
    assert f1_res["precision"] == 1.0
    assert f1_res["recall"] == 0.5
    assert f1_res["f1"] == 0.6667

    sens_res = ClinicalMetricsCalculator.calculate_sensitivity_specificity(tp=10, fp=2, fn=1, tn=20)
    assert sens_res["sensitivity"] > 0.9
    assert sens_res["specificity"] > 0.9

    safety = ClinicalMetricsCalculator.calculate_safety_violation_rate(violations=1, total_cases=50)
    assert safety["safety_compliance_rate"] == 0.98


def test_confusion_matrix_and_auroc():
    # Confusion Matrix
    classes = ["SYMPTOM", "DISEASE", "MEDICINE"]
    y_true = ["SYMPTOM", "DISEASE"]
    y_pred = ["SYMPTOM", "MEDICINE"]

    cm = ConfusionMatrixGenerator.generate_matrix(classes, y_true, y_pred)
    assert "labels" in cm
    assert cm["matrix"]["SYMPTOM"]["SYMPTOM"] == 1

    # AUROC
    y_true_bin = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    auroc_res = EvaluationMetricsEngine.calculate_auroc(y_true_bin, y_scores)
    assert auroc_res["auroc"] == 1.0


def test_benchmark_orchestrator():
    orchestrator = PlatformBenchmarkOrchestrator()
    res = orchestrator.run_full_benchmark()

    assert "overall_score_percentage" in res
    assert "ner_evaluation" in res
    assert "cdss_evaluation" in res
    assert "wearable_evaluation" in res
    assert res["overall_score_percentage"] > 0.0


def test_model_comparison_and_ab_testing():
    # 1. Model comparison
    model_a = {"overall_score_percentage": 85.0, "f1": 0.80, "avg_latency_ms": 15.0}
    model_b = {"overall_score_percentage": 92.5, "f1": 0.91, "avg_latency_ms": 18.0}
    
    comp = ModelComparisonEngine.compare("Model_A", model_a, "Model_B", model_b)
    assert "Winner: 'Model_B'" in comp["summary"]
    assert comp["metrics_comparison"]["f1"]["delta"] == 0.11

    # 2. A/B testing
    ab_sim = ABTestingSimulator(split_ratio=0.5)
    for _ in range(10):
        v = ab_sim.route_request({})
        ab_sim.record_result(v, latency_ms=12.0, confidence=0.9)
        
    report = ab_sim.get_experiment_report()
    assert report["total_evaluated_requests"] == 10


def test_leaderboard_and_reports():
    lm = EvaluationLeaderboardManager()
    lm.record_run("Model_v1", 88.0, {"f1": 0.85})
    lm.record_run("Model_v2", 94.0, {"f1": 0.92})
    
    lb = lm.get_leaderboard()
    assert lb[0]["model_name"] == "Model_v2"
    assert lb[0]["rank"] == 1

    # Markdown Report
    dummy_bench = {
        "overall_score_percentage": 94.0,
        "throughput": {"throughput_samples_per_sec": 120.0, "avg_latency_ms": 8.3},
        "ner_evaluation": {"precision": 0.92, "recall": 0.90, "f1": 0.91},
        "cdss_evaluation": {"accuracy": 0.95, "sensitivity": 0.94, "specificity": 0.96, "safety_compliance_rate": 1.0},
        "wearable_evaluation": {"sensitivity": 0.96, "specificity": 0.98, "auroc": 0.99}
    }
    md = EvaluationReportGenerator.generate_markdown_report(dummy_bench)
    assert "# Clinical Intelligence Platform — Benchmark Report" in md
    assert "**94.0%**" in md


def test_eval_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Run Benchmark
    res_run = client.post("/evaluation/run?model_name=TestSwarm")
    assert res_run.status_code == 200
    assert "overall_score_percentage" in res_run.json()
    assert "markdown_report" in res_run.json()

    # 2. Get Leaderboard
    res_lb = client.get("/evaluation/leaderboard")
    assert res_lb.status_code == 200
    assert len(res_lb.json()) >= 1
    assert res_lb.json()[0]["model_name"] == "TestSwarm"

    # 3. Model Comparison
    payload_comp = {
        "model_a_name": "Baseline",
        "model_a_metrics": {"overall_score_percentage": 75.0},
        "model_b_name": "TestSwarm",
        "model_b_metrics": {"overall_score_percentage": 90.0}
    }
    res_comp = client.post("/evaluation/compare", json=payload_comp)
    assert res_comp.status_code == 200
    assert "Winner: 'TestSwarm'" in res_comp.json()["summary"]
