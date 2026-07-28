import uuid
from pathlib import Path
from typing import Any
import networkx as nx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from src.explainability.engine import ExplainableAIEngine
from src.explainability.visualizer import ExplanationVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/xai", tags=["Explainable AI (XAI)"])

# Local directory to store static XAI reports and diagrams
XAI_DIR = Path("data/xai")
XAI_DIR.mkdir(parents=True, exist_ok=True)


class ExplainRequest(BaseModel):
    """Payload format containing sensor data, anomalies, risk, and CDSS decisions."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    telemetry: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: str = Field(default="Low")
    cdss_alert: dict[str, Any] = Field(default_factory=dict)


def _generate_and_save_visualizations(explanation_data: dict[str, Any], session_id: str) -> None:
    """Helper to reconstruct DiGraph and write PNG/HTML visualizations."""
    try:
        # Reconstruct NetworkX DiGraph
        graph_data = explanation_data["evidence_graph"]
        g = nx.DiGraph()
        
        for node in graph_data["nodes"]:
            g.add_node(
                node["id"],
                label=node["label"],
                type=node["type"],
                value=node["value"]
            )
            
        for edge in graph_data["edges"]:
            g.add_edge(
                edge["from"],
                edge["to"],
                label=edge["label"]
            )

        html_path = XAI_DIR / f"{session_id}.html"
        png_path = XAI_DIR / f"{session_id}.png"

        # Export visualizations
        ExplanationVisualizer.export_html(g, str(html_path))
        ExplanationVisualizer.export_png(g, str(png_path))
        
        logger.info("Saved XAI visualizations for session '%s'.", session_id)
    except Exception as exc:
        logger.error("Failed to generate XAI visual files for '%s': %s", session_id, exc)


@router.post("/explain")
async def explain_decision(request: ExplainRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Formulates a structured explainability report from CDSS anomalies and telemetry.
    Saves visual diagrams asynchronously.
    """
    try:
        # Generate structured explanation
        explanation = ExplainableAIEngine.generate_explanation(request.model_dump())
        
        # Trigger PNG/HTML generation as a background task to keep latency low
        background_tasks.add_task(
            _generate_and_save_visualizations,
            explanation_data=explanation,
            session_id=request.session_id
        )

        # Append reference URLs
        explanation["visualization_html"] = f"/xai/visualize/{request.session_id}"
        explanation["visualization_png"] = f"/xai/image/{request.session_id}"
        explanation["session_id"] = request.session_id
        
        return explanation
    except Exception as exc:
        logger.error("Error generating clinical explanation: %s", exc)
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {exc}")


@router.get("/visualize/{session_id}", response_class=HTMLResponse)
async def get_visualize_html(session_id: str) -> HTMLResponse:
    """
    Serves the interactive Vis.js HTML graph dashboard for the specified session.
    """
    html_path = XAI_DIR / f"{session_id}.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Explanation dashboard not found for this session.")
        
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    return HTMLResponse(content=html_content)


@router.get("/image/{session_id}", response_class=FileResponse)
async def get_visualize_image(session_id: str) -> FileResponse:
    """
    Serves the static Matplotlib PNG explanation diagram for the specified session.
    """
    png_path = XAI_DIR / f"{session_id}.png"
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="Explanation image diagram not found for this session.")
        
    return FileResponse(path=png_path, media_type="image/png")
