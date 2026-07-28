import os
from pathlib import Path
import pytest
import networkx as nx
from fastapi.testclient import TestClient

from src.explainability.engine import ExplainableAIEngine
from src.explainability.visualizer import ExplanationVisualizer
from src.api.app import create_app

# Temporary directory for visual reports
TEMP_XAI_DIR = Path("data/xai")


@pytest.fixture
def mock_payload():
    return {
        "session_id": "test_session_123",
        "telemetry": {
            "heart_rate": 128.0,
            "spo2": 89.0,
            "skin_temp": 38.6
        },
        "features": {
            "hr_mean": 126.0,
            "spo2_mean": 89.5,
            "hrv_rmssd_ms": 150.0,
            "dominant_frequency": 2.1,
            "temp_trend": "Rising"
        },
        "anomalies": [
            {"type": "Hypoxia", "severity": "High", "description": "low oxygen", "timestamp": 1234.5},
            {"type": "Tachycardia", "severity": "Medium", "description": "high HR", "timestamp": 1234.5}
        ],
        "risk_level": "High",
        "cdss_alert": {
            "evidence": {
                "red_flags": ["khó thở"],
                "contraindications": [
                    {"medication": "Ibuprofen", "disease": "Kidney disease", "reason": "Renal risk"}
                ],
                "guideline_sources": [{"title": "Renal Guideline", "code": "RG-1"}]
            },
            "recommendations": {
                "recommended_labs": ["Renal function test", "CBC"],
                "recommended_medication_categories": [
                    {"category": "NSAIDs", "status": "Contraindicated", "note": "Avoid"}
                ],
                "referral_suggestion": "Immediate consult",
                "lifestyle_advice": ["Rest"]
            },
            "confidence": 0.85
        }
    }


def test_xai_engine_logic(mock_payload):
    explanation = ExplainableAIEngine.generate_explanation(mock_payload)
    
    assert "summary" in explanation
    assert explanation["confidence"] == 0.85
    
    # Verify reasoning path
    path = explanation["reasoning_path"]
    assert len(path) == 6
    assert path[0]["title"] == "Raw Telemetry Ingestion"
    assert "Heart Rate: 128.0" in path[0]["value"]
    assert path[4]["title"] == "Risk Stratification Outcome"
    assert "Risk Level: High" in path[4]["value"]
    
    # Verify evidence graph
    graph_data = explanation["evidence_graph"]
    assert len(graph_data["nodes"]) > 0
    assert len(graph_data["edges"]) > 0
    
    # Assert nodes exist
    nodes = {n["id"]: n for n in graph_data["nodes"]}
    assert "telemetry_hr" in nodes
    assert "telemetry_spo2" in nodes
    assert "risk_node" in nodes
    assert "anomaly_0" in nodes  # Hypoxia
    assert "anomaly_1" in nodes  # Tachycardia


def test_xai_visualizer(mock_payload):
    explanation = ExplainableAIEngine.generate_explanation(mock_payload)
    graph_data = explanation["evidence_graph"]
    
    # Reconstruct DiGraph
    g = nx.DiGraph()
    for node in graph_data["nodes"]:
        g.add_node(node["id"], label=node["label"], type=node["type"], value=node["value"])
    for edge in graph_data["edges"]:
        g.add_edge(edge["from"], edge["to"], label=edge["label"])
        
    html_path = TEMP_XAI_DIR / "test_session_123.html"
    png_path = TEMP_XAI_DIR / "test_session_123.png"
    
    # Cleanup previous if any
    html_path.unlink(missing_ok=True)
    png_path.unlink(missing_ok=True)
    
    # Export visual representations
    ExplanationVisualizer.export_html(g, str(html_path))
    ExplanationVisualizer.export_png(g, str(png_path))
    
    assert html_path.exists()
    assert html_path.stat().st_size > 0
    assert png_path.exists()
    assert png_path.stat().st_size > 0
    
    # Clean up test output
    html_path.unlink(missing_ok=True)
    png_path.unlink(missing_ok=True)


def test_xai_api_endpoints(mock_payload):
    app = create_app()
    client = TestClient(app)
    
    # Make request
    response = client.post("/xai/explain", json=mock_payload)
    assert response.status_code == 200
    res_data = response.json()
    
    assert res_data["session_id"] == "test_session_123"
    assert "visualization_html" in res_data
    assert "visualization_png" in res_data
    
    # Wait briefly or query endpoints directly (endpoints look for file, which generated in background)
    # Since background tasks complete synchronously in TestClient:
    html_response = client.get(f"/xai/visualize/test_session_123")
    assert html_response.status_code == 200
    assert "vis-network" in html_response.text
    
    png_response = client.get(f"/xai/image/test_session_123")
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    
    # Cleanup generated files
    (TEMP_XAI_DIR / "test_session_123.html").unlink(missing_ok=True)
    (TEMP_XAI_DIR / "test_session_123.png").unlink(missing_ok=True)
