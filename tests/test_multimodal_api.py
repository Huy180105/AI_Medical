import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_multimodal_consult_json_endpoint(client):
    # Use one of the CXR images in input/images/
    cxr_sample = os.path.join("input", "images", "cxr_37887480.png")
    payload = {
        "clinical_text": "Bệnh nhân sốt cao rét run, đau ngực và khó thở 2 ngày nay.",
        "patient_id": "TEST_PATIENT_01",
        "image_path": cxr_sample if os.path.exists(cxr_sample) else None,
        "vitals": {
            "heart_rate": 110.0,
            "respiratory_rate": 24.0,
            "systolic_bp": 92.0,
            "spo2": 91.0
        },
        "labs": {
            "wbc": 16.5,
            "crp": 52.0
        },
        "save_output": True
    }

    response = client.post("/api/v1/multimodal/consult-json", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "consultation_id" in data
    assert "clinical_decision" in data
    decision = data["clinical_decision"]
    assert len(decision["primary_diagnosis"]) > 0
    assert len(decision["differential_diagnoses"]) > 0
    assert len(decision["actionable_recommendations"]) > 0
    assert len(decision["evidence_chain"]) > 0
    assert len(decision["differential_diagnoses"]) > 0
    assert len(decision["actionable_recommendations"]) > 0

    # Verify that file was saved to output/
    if "output_file" in data:
        assert os.path.exists(data["output_file"])


def test_multimodal_consult_multipart_with_image_upload(client, tmp_path):
    # Create a small dummy CXR image
    img = Image.new("L", (224, 224), color=100)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    form_data = {
        "clinical_text": "Bệnh nhân khó thở cấp tính, tiền sử bệnh tim.",
        "patient_id": "TEST_PATIENT_CARDIO",
        "heart_rate": 98.0,
        "respiratory_rate": 20.0,
        "systolic_bp": 130.0,
        "spo2": 95.0,
        "save_output": True
    }

    files = {
        "image": ("test_cxr.png", img_bytes, "image/png")
    }

    response = client.post("/api/v1/multimodal/consult", data=form_data, files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "consultation_id" in data
    decision = data["clinical_decision"]
    assert decision["modalities_used"]["cxr_imaging"] is True
    assert decision["modalities_used"]["symptoms_history"] is True
    assert decision["modalities_used"]["vital_signs"] is True
