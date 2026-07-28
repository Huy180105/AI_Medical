from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.evaluation.benchmark import PlatformBenchmarkOrchestrator
from src.evaluation.leaderboard import EvaluationLeaderboardManager
from src.evaluation.compare_models import ModelComparisonEngine
from src.evaluation.report import EvaluationReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["AI Evaluation & Benchmark"])

# Global Leaderboard Singleton Manager
_leaderboard_manager = EvaluationLeaderboardManager()


class ModelCompareRequest(BaseModel):
    model_a_name: str = Field(default="PhoBERT Baseline")
    model_a_metrics: dict[str, Any] = Field(default_factory=dict)
    model_b_name: str = Field(default="Clinical Multi-Agent Swarm")
    model_b_metrics: dict[str, Any] = Field(default_factory=dict)


@router.post("/run")
async def run_benchmark_evaluation(fastapi_req: Request, model_name: str = "Clinical Swarm v1.0") -> dict[str, Any]:
    """
    Triggers an end-to-end evaluation benchmark across NER, CDSS, and Wearables, returning metrics and Markdown report.
    """
    try:
        predictor = getattr(fastapi_req.app.state, "predictor", None)
        decision_engine = getattr(fastapi_req.app.state, "decision_engine", None)
        
        orchestrator = PlatformBenchmarkOrchestrator(ner_predictor=predictor, decision_engine=decision_engine)
        results = orchestrator.run_full_benchmark()
        
        # Generate Markdown Report
        md_report = EvaluationReportGenerator.generate_markdown_report(results)
        results["markdown_report"] = md_report
        
        # Record in Leaderboard
        _leaderboard_manager.record_run(model_name, results["overall_score_percentage"], results)
        
        return results
    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


@router.get("/leaderboard")
async def get_evaluation_leaderboard() -> list[dict[str, Any]]:
    """
    Returns the sorted evaluation leaderboard table.
    """
    return _leaderboard_manager.get_leaderboard()


@router.post("/compare")
async def compare_models(request: ModelCompareRequest) -> dict[str, Any]:
    """
    Compares two model variants side-by-side and returns percentage deltas.
    """
    try:
        return ModelComparisonEngine.compare(
            request.model_a_name,
            request.model_a_metrics,
            request.model_b_name,
            request.model_b_metrics
        )
    except Exception as exc:
        logger.error("Model comparison failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")
