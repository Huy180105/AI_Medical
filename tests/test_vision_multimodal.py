import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.vision.xray_classifier import ChestXRayClassifier
from src.vision.multimodal_fusion import MultimodalClinicalFusion
from src.api.app import create_app


def create_sample_cxr_image_bytes() -> bytes:
    # Create a 224x224 grayscale-like chest radiograph simulation
    img = Image.new("RGB", (224, 224), color=(60, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_xray_classifier_untrained_guard():
    """Classifier must refuse clinical initialization without weights unless allow_untrained=True."""
    from src.vision.xray_classifier import UntrainedModelError
    with pytest.raises(UntrainedModelError, match="A trained checkpoint is required"):
        ChestXRayClassifier(device="cpu", weights_path=None, allow_untrained=False)


def test_xray_classifier():
    classifier = ChestXRayClassifier(device="cpu", allow_untrained=True)
    img_bytes = create_sample_cxr_image_bytes()

    res = classifier.predict(img_bytes)
    assert "primary_finding" in res
    assert "confidence" in res
    assert "pathology_probabilities" in res
    assert len(res["pathology_probabilities"]) == 6
    assert res["model_training_status"] == "UNTRAINED_SOFTWARE_TEST_ONLY"

    cam = classifier.generate_heatmap_cam(img_bytes)
    assert "grid_resolution" in cam
    assert "hotspots" in cam


def test_multimodal_fusion():
    classifier = ChestXRayClassifier(device="cpu", allow_untrained=True)
    fusion = MultimodalClinicalFusion(xray_classifier=classifier)
    img_bytes = create_sample_cxr_image_bytes()

    text = "Bệnh nhân sốt cao 39 độ kèm ho có đờm và khó thở nhẹ."
    entities = [
        {"text": "sốt cao", "type": "SYMPTOM"},
        {"text": "ho có đờm", "type": "SYMPTOM"},
        {"text": "khó thở nhẹ", "type": "SYMPTOM"}
    ]

    res = fusion.analyze_multimodal(img_bytes, text, entities)
    assert "multimodal_category" in res
    assert "overall_confidence" in res
    assert "concordance_score" in res
    assert "clinical_summary" in res
    assert len(res["recommended_actions"]) > 0


def test_vision_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # Health
    h_res = client.get("/vision/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"

    # X-ray inference
    img_bytes = create_sample_cxr_image_bytes()
    x_res = client.post(
        "/vision/xray/predict",
        files={"file": ("test_cxr.png", img_bytes, "image/png")}
    )
    assert x_res.status_code == 200
    data = x_res.json()
    assert data["status"] == "success"
    assert "prediction" in data

    # Multimodal inference
    m_res = client.post(
        "/vision/multimodal/predict",
        files={"file": ("test_cxr.png", img_bytes, "image/png")},
        data={"clinical_text": "Bệnh nhân sốt cao và ho đờm."}
    )
    assert m_res.status_code == 200
    m_data = m_res.json()
    assert m_data["status"] == "success"
    assert "result" in m_data
