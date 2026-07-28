import pytest
import networkx as nx

from src.graph.ontology import NodeType, EdgeType
from src.decision.clinical_rules import ClinicalRuleEngine
from src.decision.risk_engine import ClinicalRiskEngine
from src.decision.recommendation import ClinicalRecommendationEngine
from src.decision.followup import ClinicalFollowUpEngine
from src.decision.validator import ClinicalDecisionValidator
from src.decision.decision_engine import ClinicalDecisionEngine


@pytest.fixture
def mock_graph() -> nx.MultiDiGraph:
    """Builds a basic mock graph for decision support testing."""
    graph = nx.MultiDiGraph()
    # Add nodes
    graph.add_node("Disease:acute_upper_respiratory_infection", name="Acute upper respiratory infection", type=NodeType.DISEASE.value)
    graph.add_node("Disease:kidney_disease", name="Kidney disease", type=NodeType.DISEASE.value)
    graph.add_node("Drug:paracetamol", name="Paracetamol", type=NodeType.DRUG.value)
    graph.add_node("Drug:ibuprofen", name="Ibuprofen", type=NodeType.DRUG.value)
    graph.add_node("Symptom:fever", name="Fever", type=NodeType.SYMPTOM.value)
    graph.add_node("Guideline:gl_resp", name="Respiratory infection initial guidance", type=NodeType.GUIDELINE.value, description="Check SpO2 and RR.")
    
    # Add edges
    graph.add_edge("Disease:acute_upper_respiratory_infection", "Symptom:fever", type=EdgeType.HAS_SYMPTOM.value)
    graph.add_edge("Drug:paracetamol", "Disease:acute_upper_respiratory_infection", type=EdgeType.TREATS.value)
    graph.add_edge("Drug:ibuprofen", "Disease:kidney_disease", type=EdgeType.CONTRAINDICATED_FOR.value)
    graph.add_edge("Disease:acute_upper_respiratory_infection", "Guideline:gl_resp", type=EdgeType.GUIDED_BY.value)
    
    return graph


def test_clinical_rules():
    # Test red flags
    entities = [{"text": "Bệnh nhân bị khó thở", "type": "SYMPTOM"}]
    red_flags = ClinicalRuleEngine.check_red_flags(entities)
    assert len(red_flags) == 1
    assert "khó thở" in red_flags[0]

    # Test contraindications
    meds_and_diseases = [
        {"text": "Ibuprofen", "type": "MEDICINE"},
    ]
    graph_results = [
        {
            "disease": "Kidney disease",
            "node_id": "Disease:kidney_disease",
            "confidence": 0.8,
            "evidence": {
                "contraindicated_drugs": ["Ibuprofen"]
            }
        }
    ]
    contra = ClinicalRuleEngine.check_contraindications(meds_and_diseases, graph_results)
    assert len(contra) == 1
    assert contra[0]["medication"] == "Ibuprofen"
    assert "contraindicated" in contra[0]["reason"].lower()


def test_risk_engine():
    # Low Risk
    entities = [{"text": "ho nhẹ", "type": "SYMPTOM"}]
    risk = ClinicalRiskEngine.assess_risk(entities, [], [], [])
    assert risk == "Low"

    # Medium Risk
    entities = [{"text": "sốt", "type": "SYMPTOM"}, {"text": "ho", "type": "SYMPTOM"}]
    graph_results = [{"disease": "Acute upper respiratory infection", "confidence": 0.65}]
    risk = ClinicalRiskEngine.assess_risk(entities, graph_results, [], [])
    assert risk == "Medium"

    # High Risk (due to contraindications)
    contra = [{"medication": "Ibuprofen", "disease": "Kidney disease"}]
    risk = ClinicalRiskEngine.assess_risk(entities, graph_results, [], contra)
    assert risk == "High"

    # Emergency Risk (due to red flags)
    red_flags = ["Critical sign: khó thở"]
    risk = ClinicalRiskEngine.assess_risk(entities, graph_results, red_flags, contra)
    assert risk == "Emergency"


def test_recommendation_engine():
    entities = [
        {"text": "sốt", "type": "SYMPTOM"},
        {"text": "ho", "type": "SYMPTOM"},
        {"text": "Ibuprofen", "type": "MEDICINE"}
    ]
    graph_results = [
        {
            "disease": "Kidney disease",
            "confidence": 0.8,
            "evidence": {
                "matched_labs": ["Renal function test"],
                "contraindicated_drugs": ["Ibuprofen"]
            }
        }
    ]
    contra = [{"medication": "Ibuprofen", "disease": "Kidney disease"}]
    
    recs = ClinicalRecommendationEngine.generate_recommendations(entities, graph_results, "High", contra)
    
    assert "Renal function test" in recs["recommended_labs"]
    assert "Chest X-ray" in recs["recommended_labs"]  # triggered by cough symptom
    
    meds = recs["recommended_medication_categories"]
    assert any(m["category"] == "Antipyretics and Analgesics (e.g. Paracetamol)" for m in meds)
    
    # NSAIDs should be marked as Contraindicated
    nsaid_rec = next(m for m in meds if "NSAIDs" in m["category"])
    assert nsaid_rec.get("status") == "Contraindicated"
    
    assert "specialist" in recs["referral_suggestion"].lower()
    assert "Stop taking contraindicated medications immediately." in recs["lifestyle_advice"]


def test_followup_engine():
    f_low = ClinicalFollowUpEngine.determine_followup("Low", [], [])
    assert f_low["action"] == "Observation"
    
    f_med = ClinicalFollowUpEngine.determine_followup("Medium", [], [])
    assert f_med["action"] == "Revisit"
    assert "48-72 hours" in f_med["timeframe"]

    f_high = ClinicalFollowUpEngine.determine_followup("High", [], [])
    assert f_high["action"] == "Hospitalization"
    assert "12-24 hours" in f_high["timeframe"]

    f_emerg = ClinicalFollowUpEngine.determine_followup("Emergency", [], [])
    assert f_emerg["action"] == "Emergency"
    assert f_emerg["timeframe"] == "Immediate"


def test_validator():
    valid_payload = {
        "diagnosis_candidates": [{"disease": "Flu", "confidence": 0.7, "explanation": "..."}],
        "risk_level": "Medium",
        "recommendations": {
            "recommended_labs": ["CBC"],
            "recommended_medication_categories": [],
            "referral_suggestion": "Routine primary care",
            "lifestyle_advice": []
        },
        "follow_up": {
            "action": "Observation",
            "timeframe": "2 days",
            "instructions": "..."
        },
        "evidence": {
            "supporting_paths": [],
            "red_flags": [],
            "contraindications": [],
            "guideline_sources": []
        },
        "confidence": 0.7
    }
    
    is_valid, errors = ClinicalDecisionValidator.validate(valid_payload)
    assert is_valid
    assert not errors

    # Invalid payload (out of bounds confidence)
    valid_payload["confidence"] = 1.5
    is_valid, errors = ClinicalDecisionValidator.validate(valid_payload)
    assert not is_valid
    assert len(errors) == 1
    assert "out of bounds" in errors[0]


def test_decision_engine_e2e(mock_graph):
    engine = ClinicalDecisionEngine(mock_graph)
    
    entities = [
        {"text": "sốt", "type": "SYMPTOM"},
        {"text": "ho", "type": "SYMPTOM"},
        {"text": "Ibuprofen", "type": "MEDICINE"}
    ]
    
    graph_results = [
        {
            "disease": "Kidney disease",
            "node_id": "Disease:kidney_disease",
            "confidence": 0.8,
            "path": ["[Drug:Ibuprofen] -(contraindicated_for)-> [Disease:Kidney disease]"],
            "evidence": {
                "matched_labs": [],
                "contraindicated_drugs": ["Ibuprofen"]
            }
        }
    ]

    decision = engine.make_decision(entities, graph_results)
    
    assert decision["risk_level"] == "High"
    assert len(decision["evidence"]["contraindications"]) == 1
    assert decision["confidence"] == 0.65  # 0.8 - 0.15 penalty
    assert len(decision["evidence"]["supporting_paths"]) == 1
