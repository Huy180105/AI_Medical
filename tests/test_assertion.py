import pytest
from src.assertion.assertion_detector import ClinicalAssertionDetector


def test_negation_detection():
    detector = ClinicalAssertionDetector()
    
    # 1. Negated
    text1 = "Bệnh nhân hoàn toàn không sốt."
    ents1 = [{"text": "sốt", "type": "SYMPTOM"}]
    res1 = detector.detect_assertions(text1, ents1)
    assert res1[0].assertion.is_negated is True

    # 2. Non-negated
    text2 = "Bệnh nhân bị sốt cao 39 độ C."
    ents2 = [{"text": "sốt cao", "type": "SYMPTOM"}]
    res2 = detector.detect_assertions(text2, ents2)
    assert res2[0].assertion.is_negated is False


def test_family_history_detection():
    detector = ClinicalAssertionDetector()
    
    text = "Bệnh nhân bị ho nhẹ, nhưng mẹ bị đái tháo đường tuýp 2."
    ents = [
        {"text": "ho nhẹ", "type": "SYMPTOM"},
        {"text": "đái tháo đường tuýp 2", "type": "DISEASE"}
    ]
    res = detector.detect_assertions(text, ents)
    
    assert res[0].assertion.is_family is False
    assert res[1].assertion.is_family is True


def test_historical_context_detection():
    detector = ClinicalAssertionDetector()
    
    text = "Tiền sử mổ ruột thừa năm ngoái, hiện tại đau bụng cấp."
    ents = [
        {"text": "mổ ruột thừa", "type": "PROCEDURE"},
        {"text": "đau bụng cấp", "type": "SYMPTOM"}
    ]
    res = detector.detect_assertions(text, ents)
    
    assert res[0].assertion.is_historical is True
    assert res[1].assertion.is_historical is False


def test_uncertainty_detection():
    detector = ClinicalAssertionDetector()
    
    text = "Theo dõi nghi ngờ viêm phổi thùy."
    ents = [{"text": "viêm phổi thùy", "type": "DISEASE"}]
    res = detector.detect_assertions(text, ents)
    
    assert res[0].assertion.is_uncertain is True


def test_conditional_detection():
    detector = ClinicalAssertionDetector()
    
    text = "Nếu đau ngực dữ dội thì uống Paracetamol."
    ents = [
        {"text": "đau ngực dữ dội", "type": "SYMPTOM"},
        {"text": "Paracetamol", "type": "MEDICINE"}
    ]
    res = detector.detect_assertions(text, ents)
    
    assert res[0].assertion.is_conditional is True
    assert res[1].assertion.is_conditional is True


def test_scope_termination():
    detector = ClinicalAssertionDetector()
    
    # Scope terminator 'nhưng' prevents 'không' from leaking to 'đau đầu'
    text = "Bệnh nhân không sốt, nhưng đau đầu nhẹ."
    ents = [
        {"text": "sốt", "type": "SYMPTOM"},
        {"text": "đau đầu nhẹ", "type": "SYMPTOM"}
    ]
    res = detector.detect_assertions(text, ents)
    
    assert res[0].assertion.is_negated is True
    assert res[1].assertion.is_negated is False
