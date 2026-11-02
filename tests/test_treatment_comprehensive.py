import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api.app import create_app


def create_sample_cxr_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


def test_predict_comprehensive_standard(client):
    data = {
        "age": "58",
        "gender": "Male",
        "heart_rate": "92.0",
        "sys_bp": "148.0",
        "dia_bp": "94.0",
        "spo2": "93.5",
        "stress_score": "8",
        "sleep_hours": "4.5",
        "physical_activity_hrs": "1.0",
        "chronic_conditions": "2",
        "symptoms_severity": "4",
        "enable_fhe": "false",
    }
    resp = client.post("/api/v1/treatment/predict-comprehensive", data=data)
    assert resp.status_code == 200, resp.text
    res = resp.json()
    assert "treatment_needed" in res
    assert "treatment_probability_pct" in res
    assert "champion_model" in res
    assert res.get("fhe_metadata") is None
    assert res.get("multimodal_xray") is None


def test_predict_comprehensive_with_fhe(client):
    data = {
        "age": "60",
        "gender": "Female",
        "heart_rate": "85.0",
        "sys_bp": "135.0",
        "dia_bp": "88.0",
        "spo2": "96.0",
        "stress_score": "5",
        "sleep_hours": "6.0",
        "physical_activity_hrs": "2.0",
        "chronic_conditions": "1",
        "symptoms_severity": "3",
        "enable_fhe": "true",
    }
    resp = client.post("/api/v1/treatment/predict-comprehensive", data=data)
    assert resp.status_code == 200, resp.text
    res = resp.json()
    assert res.get("fhe_metadata") is not None
    fhe = res["fhe_metadata"]
    assert fhe["enabled"] is True
    assert "CKKS" in fhe["scheme"]
    assert "predicted_risk_level" in fhe
    assert "ciphertext_size_bytes" in fhe
    assert res.get("multimodal_xray") is None


def test_predict_comprehensive_all_inclusive(client):
    img_bytes = create_sample_cxr_image_bytes()
    data = {
        "age": "65",
        "gender": "Male",
        "heart_rate": "100.0",
        "sys_bp": "155.0",
        "dia_bp": "95.0",
        "spo2": "91.0",
        "stress_score": "9",
        "sleep_hours": "3.5",
        "physical_activity_hrs": "0.5",
        "chronic_conditions": "3",
        "symptoms_severity": "5",
        "enable_fhe": "true",
    }
    files = {
        "xray_file": ("chest_sample.png", img_bytes, "image/png")
    }
    resp = client.post("/api/v1/treatment/predict-comprehensive", data=data, files=files)
    assert resp.status_code == 200, resp.text
    res = resp.json()
    
    # 1. Standard prediction
    assert res["treatment_needed"] is True
    assert "CRITICAL" in res["risk_level"] or "HIGH" in res["risk_level"]
    assert len(res["predicted_diseases"]) > 0

    # 2. FHE Metadata
    assert res.get("fhe_metadata") is not None
    assert res["fhe_metadata"]["enabled"] is True

    # 3. Multimodal CXR Analysis
    assert res.get("multimodal_xray") is not None
    mx = res["multimodal_xray"]
    assert "multimodal_category" in mx
    assert "xray_finding" in mx
    assert "concordance_score" in mx
    assert "clinical_summary" in mx
