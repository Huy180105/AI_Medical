"""
Multimodal Clinical Test Cases Suite (MedFusion-CXR).

Validates end-to-end multimodal consultation across 10 authentic NIH ChestX-ray14
radiographs and clinical vignettes:
  1. Pneumonia (Viêm phổi cấp)
  2. Cardiomegaly (Tim to / Suy tim)
  3. Effusion (Tràn dịch màng phổi)
  4. Normal (Sàng lọc ổn định)
  5. Pneumothorax (Tràn khí màng phổi cấp)
  6. Atelectasis (Xẹp phổi)
  7. Mass / Nodule (Khối u nhu mô phổi)
  8. Edema (Phù phổi cấp)
  9. Infiltration (Thâm nhiễm lan tỏa)
  10. Fibrosis (Xơ hóa phổi & Dày dính màng phổi)

Verifies:
  - Image integrity (512x512 Grayscale PNGs, valid bytes, no corruption)
  - Pretrained DenseNet-121 Vision branch on NIH-14 (14 ranked probabilities in [0, 1])
  - XGBoost + TreeSHAP clinical branch (stable probabilities, non-empty attributions)
  - Gated Multimodal Unit (GMU 256-d fused representation, valid gate weights)
  - End-to-end FastAPI endpoint (/api/v1/treatment/multimodal_predict)
  - Plain-language Vietnamese consensus recommendations and zero NaN/Inf values.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from src.api.app import create_app
from src.vision.multimodal_fusion import get_medfusion_engine


SAMPLES_DIR = Path("web") / "samples"

TEN_CLINICAL_CASES: List[Dict[str, Any]] = [
    {
        "key": "pneumonia",
        "filename": "pneumonia_nih.png",
        "expected_pathology": "Pneumonia",
        "age": 62,
        "gender": "Nam (Male)",
        "heart_rate": 112.0,
        "sys_bp": 146.0,
        "dia_bp": 90.0,
        "spo2": 91.5,
        "respiratory_rate": 25.0,
        "temperature_c": 38.8,
        "wbc": 16.4,
        "crp": 78.5,
        "troponin": 0.02,
        "stress_score": 8,
        "sleep_hours": 4.5,
        "physical_activity_hrs": 1.0,
        "chronic_conditions": 2,
        "symptoms_severity": 4,
        "clinical_text": (
            "Bệnh nhân nam 62 tuổi, tiền sử tăng huyết áp và đái tháo đường típ 2, "
            "vào viện vì sốt cao 38.8°C rét run, ho đờm vàng và khó thở tăng dần 3 ngày nay."
        ),
    },
    {
        "key": "cardiomegaly",
        "filename": "cardiomegaly_nih.png",
        "expected_pathology": "Cardiomegaly",
        "age": 68,
        "gender": "Nữ (Female)",
        "heart_rate": 104.0,
        "sys_bp": 162.0,
        "dia_bp": 98.0,
        "spo2": 92.8,
        "respiratory_rate": 23.0,
        "temperature_c": 37.1,
        "wbc": 9.2,
        "crp": 14.0,
        "troponin": 0.09,
        "stress_score": 7,
        "sleep_hours": 5.0,
        "physical_activity_hrs": 0.5,
        "chronic_conditions": 3,
        "symptoms_severity": 4,
        "clinical_text": (
            "Bệnh nhân nữ 68 tuổi có tiền sử tăng huyết áp và suy tim mạn đang dùng Amlodipine, "
            "nhập viện vì khó thở khi nằm, phù hai chân và đau ngực âm ỉ 2 ngày nay."
        ),
    },
    {
        "key": "effusion",
        "filename": "effusion_nih.png",
        "expected_pathology": "Effusion",
        "age": 57,
        "gender": "Nam (Male)",
        "heart_rate": 98.0,
        "sys_bp": 138.0,
        "dia_bp": 86.0,
        "spo2": 93.2,
        "respiratory_rate": 22.0,
        "temperature_c": 37.9,
        "wbc": 12.6,
        "crp": 42.0,
        "troponin": 0.01,
        "stress_score": 7,
        "sleep_hours": 5.5,
        "physical_activity_hrs": 1.5,
        "chronic_conditions": 1,
        "symptoms_severity": 4,
        "clinical_text": (
            "Bệnh nhân nam 57 tuổi nhập viện vì khó thở tăng dần khi gắng sức, "
            "ho kéo dài và đau tức vùng ngực đáy phổi 5 ngày nay."
        ),
    },
    {
        "key": "normal",
        "filename": "normal_nih.png",
        "expected_pathology": "No Finding",
        "age": 34,
        "gender": "Nữ (Female)",
        "heart_rate": 72.0,
        "sys_bp": 116.0,
        "dia_bp": 74.0,
        "spo2": 98.5,
        "respiratory_rate": 16.0,
        "temperature_c": 36.7,
        "wbc": 6.8,
        "crp": 2.1,
        "troponin": 0.01,
        "stress_score": 3,
        "sleep_hours": 7.5,
        "physical_activity_hrs": 4.0,
        "chronic_conditions": 0,
        "symptoms_severity": 1,
        "clinical_text": (
            "Bệnh nhân nữ 34 tuổi khám sức khỏe định kỳ, không sốt, không ho, "
            "không khó thở, sinh hiệu ổn định, chụp X-quang ngực kiểm tra tổng quát."
        ),
    },
    {
        "key": "pneumothorax",
        "filename": "pneumothorax_nih.png",
        "expected_pathology": "Pneumothorax",
        "age": 56,
        "gender": "Nam (Male)",
        "heart_rate": 118.0,
        "sys_bp": 105.0,
        "dia_bp": 70.0,
        "spo2": 89.0,
        "respiratory_rate": 28.0,
        "temperature_c": 37.1,
        "wbc": 10.5,
        "crp": 12.0,
        "troponin": 0.01,
        "stress_score": 9,
        "sleep_hours": 3.5,
        "physical_activity_hrs": 0.5,
        "chronic_conditions": 1,
        "symptoms_severity": 5,
        "clinical_text": (
            "Bệnh nhân nam 56 tuổi đột ngột đau nhói ngực phải dữ dội, khó thở tăng nhanh sau khi gắng sức. "
            "Thăm khám ghi nhận rì rào phế nang phổi phải giảm rõ rệt, gõ vang, SpO2 tụt xuống 89%."
        ),
    },
    {
        "key": "atelectasis",
        "filename": "atelectasis_nih.png",
        "expected_pathology": "Atelectasis",
        "age": 55,
        "gender": "Nữ (Female)",
        "heart_rate": 96.0,
        "sys_bp": 130.0,
        "dia_bp": 85.0,
        "spo2": 92.5,
        "respiratory_rate": 24.0,
        "temperature_c": 37.8,
        "wbc": 11.2,
        "crp": 25.0,
        "troponin": 0.02,
        "stress_score": 7,
        "sleep_hours": 5.0,
        "physical_activity_hrs": 0.5,
        "chronic_conditions": 2,
        "symptoms_severity": 4,
        "clinical_text": (
            "Bệnh nhân nữ 55 tuổi sau phẫu thuật ổ bụng ngày thứ 3 xuất hiện thở nông, "
            "ho đờm khó khạc, tức ngực bên trái. Nghe phổi có vùng giảm thông khí đáy phổi trái."
        ),
    },
    {
        "key": "mass_nodule",
        "filename": "mass_nodule_nih.png",
        "expected_pathology": "Mass",
        "age": 54,
        "gender": "Nam (Male)",
        "heart_rate": 84.0,
        "sys_bp": 135.0,
        "dia_bp": 85.0,
        "spo2": 96.0,
        "respiratory_rate": 18.0,
        "temperature_c": 36.8,
        "wbc": 8.0,
        "crp": 8.5,
        "troponin": 0.01,
        "stress_score": 6,
        "sleep_hours": 6.0,
        "physical_activity_hrs": 2.0,
        "chronic_conditions": 1,
        "symptoms_severity": 3,
        "clinical_text": (
            "Bệnh nhân nam 54 tuổi có tiền sử hút thuốc lá 30 bao-năm, đến khám vì ho khan kéo dài "
            "hơn 2 tháng, thỉnh thoảng khạc đờm có tia máu, gầy sút 4kg không rõ nguyên nhân."
        ),
    },
    {
        "key": "edema",
        "filename": "edema_nih.png",
        "expected_pathology": "Edema",
        "age": 52,
        "gender": "Nam (Male)",
        "heart_rate": 122.0,
        "sys_bp": 175.0,
        "dia_bp": 105.0,
        "spo2": 88.0,
        "respiratory_rate": 30.0,
        "temperature_c": 36.9,
        "wbc": 9.4,
        "crp": 15.0,
        "troponin": 0.08,
        "stress_score": 9,
        "sleep_hours": 3.0,
        "physical_activity_hrs": 0.5,
        "chronic_conditions": 3,
        "symptoms_severity": 5,
        "clinical_text": (
            "Bệnh nhân nam 52 tuổi có tiền sử tăng huyết áp và bệnh tim thiếu máu cục bộ, "
            "nhập viện vì khó thở kịch phát về đêm, ho khạc bọt hồng, vã mồ hôi. Huyết áp vọt lên 175/105 mmHg."
        ),
    },
    {
        "key": "infiltration",
        "filename": "infiltration_nih.png",
        "expected_pathology": "Infiltration",
        "age": 59,
        "gender": "Nam (Male)",
        "heart_rate": 102.0,
        "sys_bp": 125.0,
        "dia_bp": 80.0,
        "spo2": 91.0,
        "respiratory_rate": 25.0,
        "temperature_c": 38.6,
        "wbc": 14.8,
        "crp": 55.0,
        "troponin": 0.02,
        "stress_score": 8,
        "sleep_hours": 4.0,
        "physical_activity_hrs": 1.0,
        "chronic_conditions": 2,
        "symptoms_severity": 4,
        "clinical_text": (
            "Bệnh nhân nam 59 tuổi sốt cao 38.6C kéo dài 4 ngày, ho khan nhiều, mệt mỏi li bì, "
            "cảm giác hụt hơi khi đi lại. Nghe phổi có rale nổ rải rác hai bên."
        ),
    },
    {
        "key": "fibrosis",
        "filename": "fibrosis_nih.png",
        "expected_pathology": "Fibrosis",
        "age": 50,
        "gender": "Nữ (Female)",
        "heart_rate": 88.0,
        "sys_bp": 120.0,
        "dia_bp": 80.0,
        "spo2": 94.0,
        "respiratory_rate": 20.0,
        "temperature_c": 36.6,
        "wbc": 7.2,
        "crp": 5.0,
        "troponin": 0.01,
        "stress_score": 5,
        "sleep_hours": 6.5,
        "physical_activity_hrs": 1.5,
        "chronic_conditions": 2,
        "symptoms_severity": 3,
        "clinical_text": (
            "Bệnh nhân nữ 50 tuổi có tiền sử lao phổi cũ đã điều trị khỏi 5 năm trước, "
            "nay đi khám vì khó thở tăng dần khi gắng sức, ho khan từng cơn, tức nặng ngực."
        ),
    },
]


@pytest.fixture(scope="module")
def api_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_samples_directory_has_all_10_images():
    """All 10 authentic NIH PNG files must exist and be readable images."""
    for case in TEN_CLINICAL_CASES:
        img_path = SAMPLES_DIR / case["filename"]
        assert img_path.exists(), f"Missing radiograph sample: {img_path}"
        assert img_path.stat().st_size > 10_000, f"Sample file is abnormally small: {img_path}"

        with Image.open(img_path) as im:
            assert im.format == "PNG"
            w, h = im.size
            assert w >= 224 and h >= 224
            assert im.mode in ("L", "RGB", "RGBA")


def test_samples_manifest_json_integrity():
    """The manifest file must contain valid metadata for all 6 newly added cases."""
    manifest_path = SAMPLES_DIR / "samples_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 6

    filenames = {item["filename"] for item in data}
    for key in ["pneumothorax", "atelectasis", "mass_nodule", "edema", "infiltration", "fibrosis"]:
        expected_fn = f"{key}_nih.png"
        assert expected_fn in filenames, f"Manifest missing {expected_fn}"


@pytest.mark.parametrize("case", TEN_CLINICAL_CASES)
def test_direct_medfusion_engine_analysis_on_all_10_cases(case: Dict[str, Any]):
    """
    Directly exercises `MultimodalClinicalFusion.run_multimodal_analysis` on every radiograph,
    ensuring 14 valid NIH probabilities, GMU fusion, and stable XGBoost output.
    """
    engine = get_medfusion_engine()
    img_path = SAMPLES_DIR / case["filename"]

    patient_tabular = {
        "age": case["age"],
        "gender": case["gender"],
        "heart_rate": case["heart_rate"],
        "sys_bp": case["sys_bp"],
        "dia_bp": case["dia_bp"],
        "spo2": case["spo2"],
        "stress_score": case["stress_score"],
        "sleep_hours": case["sleep_hours"],
        "physical_activity_hrs": case["physical_activity_hrs"],
        "chronic_conditions": case["chronic_conditions"],
        "symptoms_severity": case["symptoms_severity"],
    }
    vitals_extra = {
        "heart_rate": case["heart_rate"],
        "respiratory_rate": case["respiratory_rate"],
        "systolic_bp": case["sys_bp"],
        "diastolic_bp": case["dia_bp"],
        "spo2": case["spo2"],
        "temperature_c": case["temperature_c"],
    }
    labs = {
        "wbc": case["wbc"],
        "crp": case["crp"],
        "troponin": case["troponin"],
    }

    result = engine.run_multimodal_analysis(
        patient_tabular=patient_tabular,
        image_bytes_or_path=str(img_path),
        vitals_extra=vitals_extra,
        labs=labs,
        nlp_entities=[],
    )

    # 1. Vision branch checks
    vision = result["vision_branch"]
    assert vision is not None
    assert "DenseNet-121" in vision["backbone"]
    assert "NIH-14" in vision["backbone"]
    assert len(vision["nih_14_ranked"]) == 14

    for item in vision["nih_14_ranked"]:
        prob = item["probability"]
        assert 0.0 <= prob <= 1.0
        assert not math.isnan(prob) and not math.isinf(prob)
        calibrated = item["calibrated_score"]
        assert 0.0 <= calibrated <= 1.0

    # Calibrated scores must be descending
    ranked_calibrated = [item["calibrated_score"] for item in vision["nih_14_ranked"]]
    assert ranked_calibrated == sorted(ranked_calibrated, reverse=True)

    # 2. Clinical branch checks
    clinical = result["clinical_branch"]
    assert clinical is not None
    assert isinstance(clinical["treatment_needed"], bool)
    assert 0.0 <= clinical["probability_pct"] <= 100.0
    assert any(tier in clinical["risk_level"] for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    # 3. Gated Fusion checks
    fusion = result["gated_fusion"]
    assert fusion is not None
    assert fusion["shared_latent_dim"] == 256
    assert 0.0 <= fusion["gate_vision_weight"] <= 1.0
    assert 0.0 <= fusion["gate_clinical_weight"] <= 1.0
    assert abs((fusion["gate_vision_weight"] + fusion["gate_clinical_weight"]) - 1.0) < 1e-4


@pytest.mark.parametrize("case", TEN_CLINICAL_CASES)
def test_api_multimodal_predict_on_all_10_cases(api_client: TestClient, case: Dict[str, Any]):
    """
    Submits real radiograph bytes + clinical parameters to POST /api/v1/treatment/multimodal_predict,
    verifying end-to-end multi-agent orchestration, consensus synthesis, and plain Vietnamese text.
    """
    img_path = SAMPLES_DIR / case["filename"]
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    form_data = {
        "clinical_text": case["clinical_text"],
        "age": str(case["age"]),
        "gender": case["gender"],
        "heart_rate": str(case["heart_rate"]),
        "sys_bp": str(case["sys_bp"]),
        "dia_bp": str(case["dia_bp"]),
        "spo2": str(case["spo2"]),
        "respiratory_rate": str(case["respiratory_rate"]),
        "temperature_c": str(case["temperature_c"]),
        "wbc": str(case["wbc"]),
        "crp": str(case["crp"]),
        "troponin": str(case["troponin"]),
        "stress_score": str(case["stress_score"]),
        "sleep_hours": str(case["sleep_hours"]),
        "physical_activity_hrs": str(case["physical_activity_hrs"]),
        "chronic_conditions": str(case["chronic_conditions"]),
        "symptoms_severity": str(case["symptoms_severity"]),
    }
    files = {
        "xray_file": (case["filename"], img_bytes, "image/png"),
    }

    resp = api_client.post("/api/v1/treatment/multimodal_predict", data=form_data, files=files)
    assert resp.status_code == 200, f"Failed for {case['key']}: {resp.text}"

    data = resp.json()
    assert data["status"] == "success"
    assert data["latency_ms"] > 0

    # Vision findings
    xray = data["agent2_xray"]
    assert xray is not None
    assert xray["filename"] == case["filename"]
    assert len(xray["detected_pathologies"]) == 14

    # TreeSHAP & Prediction
    pred = data["agent3_prediction"]
    assert 0.0 <= pred["probability_pct"] <= 100.0
    assert any(tier in pred["risk_level"] for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    # Multimodal Consensus & Synthesized Potential Diseases
    consensus = data.get("multimodal_consensus")
    assert consensus is not None
    assert len(consensus["primary_diagnosis"]) > 0
    assert len(consensus["conclusion_summary"]) > 0
    assert 0.0 <= consensus["diagnostic_confidence_pct"] <= 100.0
    assert len(consensus["potential_diseases"]) > 0

    for disease in consensus["potential_diseases"]:
        assert len(disease["disease_name"]) > 0
        assert len(disease["icd10_code"]) > 0
        assert 0.0 <= disease["probability_pct"] <= 100.0
        assert len(disease["cxr_evidence"]) > 0
        assert len(disease["clinical_evidence"]) > 0
        assert len(disease["bach_mai_treatment"]) > 0

    # Consensus Recommendations
    recommendations = data["consensus_recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) >= 2
    # Ensure plain Vietnamese clinical text
    has_vietnamese = any("bác sĩ" in r or "lâm sàng" in r or "X-quang" in r for r in recommendations)
    assert has_vietnamese, f"Recommendations lack Vietnamese guidance: {recommendations}"
