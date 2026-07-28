import time
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.wearable.patient_state import PatientState, HealthPacket
from src.wearable.simulator import SamsungGalaxyWatchSimulator
from src.wearable.feature_engineering import WearableFeatureEngineer
from src.wearable.timeseries_model import WearableTimeSeriesModel
from src.wearable.anomaly_detector import WearableAnomalyDetector
from src.wearable.risk_predictor import WearableRiskPredictor
from src.api.app import create_app


def test_wearable_simulator_signals():
    sim = SamsungGalaxyWatchSimulator()
    
    # 1. Normal state boundaries
    packet = sim.generate_packet(PatientState.NORMAL)
    assert 55.0 <= packet.heart_rate <= 85.0
    assert 94.0 <= packet.spo2 <= 100.0
    assert len(packet.ecg) == 100
    assert packet.activity == "Resting"
    assert packet.sleep_stage == "Wake"

    # 2. Exercise state boundaries
    packet_ex = sim.generate_packet(PatientState.EXERCISE)
    assert 110.0 <= packet_ex.heart_rate <= 160.0
    assert packet_ex.activity == "Running"
    assert packet_ex.step_count > packet.step_count

    # 3. Fever state boundaries
    packet_fv = sim.generate_packet(PatientState.FEVER)
    assert packet_fv.skin_temp >= 38.0


def test_feature_engineering():
    fe = WearableFeatureEngineer(window_size=10)
    
    # Add dummy packets
    for i in range(10):
        # Stepwise increasing HR: 70, 72, 74, ...
        p = HealthPacket(
            heart_rate=70.0 + i * 2,
            spo2=98.0,
            ecg=[0.0] * 100,
            skin_temp=36.6,
            stress=20.0,
            sleep_stage="Wake",
            step_count=1000 + i,
            calories=100.0 + i,
            activity="Resting",
            timestamp=time.time()
        )
        fe.add_packet(p)

    features = fe.extract_features()
    # Average HR: sum(70 to 88) / 10 = 79.0
    assert features["hr_mean"] == 79.0
    assert features["hr_max"] == 88.0
    assert features["hr_min"] == 70.0
    assert features["hr_trend"] == "Rising"
    # Verify HRV RMSSD calculation conversion did not fail
    assert features["hrv_rmssd_ms"] > 0.0


def test_timeseries_model():
    # 1. Autoregressive HR forecast
    hr_history = [70.0, 71.0, 72.0, 73.0, 74.0]
    forecast = WearableTimeSeriesModel.forecast_heart_rate(hr_history, seconds_ahead=5)
    assert len(forecast) == 5
    # Since recent delta is positive (+1.0), it should forecast upward or near mean
    assert forecast[0] >= 74.0

    # 2. FFT Spectral analysis
    # Generate simple sine wave ECG to test frequency peak
    t = [i / 100.0 for i in range(100)]  # 100Hz sampling
    ecg_sine = [float(1.2 * (0.8 * abs(i - 0.4) < 0.1)) for i in t]  # dummy pulse wave
    freq_feats = WearableTimeSeriesModel.compute_ecg_frequency_features(ecg_sine, sampling_rate=100.0)
    assert "dominant_frequency" in freq_feats
    assert "spectral_power_entropy" in freq_feats


def test_anomaly_detection():
    fe = WearableFeatureEngineer()
    # Mock normal features
    features = {"hrv_rmssd_ms": 25.0}

    # 1. Hypoxia detection
    p1 = HealthPacket(
        heart_rate=72.0, spo2=84.0, ecg=[0.0]*100, skin_temp=36.6, stress=20.0,
        sleep_stage="Wake", step_count=100, calories=1.0, activity="Resting", timestamp=time.time()
    )
    anoms = WearableAnomalyDetector.detect_anomalies(p1, features, PatientState.HYPOXIA)
    assert any(a["type"] == "Hypoxia" for a in anoms)
    assert any(a["severity"] == "Emergency" for a in anoms)

    # 2. Tachycardia detection
    p2 = HealthPacket(
        heart_rate=115.0, spo2=97.0, ecg=[0.0]*100, skin_temp=36.6, stress=20.0,
        sleep_stage="Wake", step_count=100, calories=1.0, activity="Resting", timestamp=time.time()
    )
    anoms_hr = WearableAnomalyDetector.detect_anomalies(p2, features, PatientState.ARRHYTHMIA)
    assert any(a["type"] == "Tachycardia" for a in anoms_hr)


def test_risk_predictor_and_cdss_trigger():
    # Setup mock CDSS decision engine
    mock_cdss = MagicMock()
    mock_cdss.make_decision.return_value = {
        "diagnosis_candidates": [{"disease": "Renal crisis", "confidence": 0.8}],
        "risk_level": "Emergency"
    }

    predictor = WearableRiskPredictor(decision_engine=mock_cdss)
    
    # Low Risk
    assert predictor.assess_risk([]) == "Low"

    # Emergency Risk (contains Hypoxia severity = Emergency)
    anoms = [
        {"type": "Hypoxia", "severity": "Emergency", "description": "low oxygen", "timestamp": time.time()},
        {"type": "Tachycardia", "severity": "Medium", "description": "high HR", "timestamp": time.time()}
    ]
    risk = predictor.assess_risk(anoms)
    assert risk == "Emergency"

    # Verify CDSS Trigger mappings
    p = HealthPacket(
        heart_rate=105.0, spo2=84.0, ecg=[0.0]*100, skin_temp=36.6, stress=20.0,
        sleep_stage="Wake", step_count=100, calories=1.0, activity="Resting", timestamp=time.time()
    )
    
    decision = predictor.trigger_clinical_decision(p, anoms, risk)
    assert decision["risk_level"] == "Emergency"
    # Check that make_decision was called with mapped entities
    called_entities = mock_cdss.make_decision.call_args[0][0]
    # Hypoxia -> 'thiếu oxy', Tachycardia -> 'tim đập nhanh'
    assert any(e["text"] == "thiếu oxy" for e in called_entities)
    assert any(e["text"] == "tim đập nhanh" for e in called_entities)


def test_wearable_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Get status
    response = client.get("/wearable/status")
    assert response.status_code == 200
    res_data = response.json()
    assert "is_running" in res_data
    assert res_data["is_running"] is False

    # 2. Start simulation
    response = client.post("/wearable/start", json={"state": "Exercise", "sampling_rate": 2.0})
    assert response.status_code == 200
    assert response.json()["status"] == "started"
    assert response.json()["current_state"] == "Exercise"
    assert response.json()["sampling_rate"] == 2.0

    # Status should be active now
    response = client.get("/wearable/status")
    assert response.json()["is_running"] is True

    # 3. Stop simulation
    response = client.post("/wearable/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"

    # Status should be inactive again
    response = client.get("/wearable/status")
    assert response.json()["is_running"] is False
