import pytest
from src.relation.relation_detector import ClinicalRelationDetector
from src.relation.relation_graph import ClinicalRelationGraphBuilder
from src.relation.relation_postprocess import RelationPostProcessor


def test_treats_relation():
    detector = ClinicalRelationDetector()
    text = "Bác sĩ cho bệnh nhân dùng Paracetamol để điều trị sốt cao."
    entities = [
        {"text": "Paracetamol", "type": "MEDICINE"},
        {"text": "sốt cao", "type": "SYMPTOM"}
    ]
    rels = detector.extract_relations(text, entities)
    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relation_type == "TREATS"
    assert rel.subject.text == "Paracetamol"
    assert rel.object.text == "sốt cao"


def test_caused_by_relation():
    detector = ClinicalAssertionDetector = ClinicalRelationDetector()
    text = "Bệnh nhân bị ho đờm do viêm phổi thùy."
    entities = [
        {"text": "ho đờm", "type": "SYMPTOM"},
        {"text": "viêm phổi thùy", "type": "DISEASE"}
    ]
    rels = detector.extract_relations(text, entities)
    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relation_type == "CAUSED_BY"


def test_has_test_result_relation():
    detector = ClinicalRelationDetector()
    text = "Xét nghiệm WBC cho kết quả 14.3 G/L."
    entities = [
        {"text": "WBC", "type": "TEST"},
        {"text": "14.3 G/L", "type": "VALUE"}
    ]
    rels = detector.extract_relations(text, entities)
    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relation_type == "HAS_TEST_RESULT"


def test_contraindicated_for_relation():
    detector = ClinicalRelationDetector()
    text = "Ibuprofen chống chỉ định ở bệnh nhân suy thận cấp."
    entities = [
        {"text": "Ibuprofen", "type": "MEDICINE"},
        {"text": "suy thận cấp", "type": "DISEASE"}
    ]
    rels = detector.extract_relations(text, entities)
    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relation_type == "CONTRAINDICATED_FOR"


def test_relation_graph_and_postprocessing():
    detector = ClinicalRelationDetector()
    text = "Paracetamol điều trị sốt cao."
    entities = [
        {"text": "Paracetamol", "type": "MEDICINE"},
        {"text": "sốt cao", "type": "SYMPTOM"}
    ]
    rels = detector.extract_relations(text, entities)
    
    # Deduplicate
    dedup = RelationPostProcessor.filter_and_deduplicate(rels + rels)
    assert len(dedup) == 1

    # Graph
    g = ClinicalRelationGraphBuilder.build_graph(dedup)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
