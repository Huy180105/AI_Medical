import pytest
from src.assertion.deep_assertion import DeepAssertionClassifier
from src.decision.deep_risk_network import DeepClinicalRiskEngine


def test_deep_assertion_classifier():
    classifier = DeepAssertionClassifier()

    # Negation
    t1 = "Bệnh nhân không sốt và không đau đầu."
    a1 = classifier.predict_assertion(t1, "sốt", 16, 19)
    assert a1.is_negated is True

    # Family history
    t2 = "Gia đình có bố mắc bệnh tiểu đường."
    a2 = classifier.predict_assertion(t2, "tiểu đường", 23, 33)
    assert a2.is_family is True

    # Uncertainty / hypothetical
    t3 = "Bác sĩ nghi ngờ bệnh nhân bị viêm phổi."
    a3 = classifier.predict_assertion(t3, "viêm phổi", 29, 38)
    assert a3.is_uncertain is True

    # Present (Normal condition)
    t4 = "Bệnh nhân sốt cao liên tục 3 ngày nay."
    a4 = classifier.predict_assertion(t4, "sốt cao", 10, 17)
    assert a4.is_negated is False


def test_deep_risk_network():
    engine = DeepClinicalRiskEngine()

    # Mild case
    mild_entities = [{"text": "ho nhẹ", "type": "SYMPTOM"}]
    mild_res = engine.evaluate_risk(
        entities=mild_entities,
        vitals={"heart_rate": 72.0, "blood_pressure_systolic": 118.0, "spo2": 99.0, "temperature": 36.8}
    )
    assert "risk_level" in mild_res
    assert mild_res["risk_score"] < 0.60
    assert mild_res["engine"] == "DeepNeuralResidualRiskClassifier"

    # Emergency case with red flags and abnormal vitals
    crit_entities = [
        {"text": "đau ngực dữ dội", "type": "SYMPTOM"},
        {"text": "khó thở cấp", "type": "SYMPTOM"},
        {"text": "nhồi máu cơ tim", "type": "DISEASE"}
    ]
    crit_res = engine.evaluate_risk(
        entities=crit_entities,
        red_flags=["Đau thắt ngực dữ dội", "Hạ SpO2 cấp"],
        vitals={"heart_rate": 135.0, "blood_pressure_systolic": 85.0, "spo2": 88.0, "temperature": 39.2}
    )
    assert crit_res["risk_level"] == "Emergency"
    assert crit_res["risk_score"] > 0.70
