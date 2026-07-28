from src.agent.workflow import MedicalWorkflow


class FakeNER:
    def predict(self, text: str):
        return [
            {"text": "sot", "type": "SYMPTOM", "score": 0.99, "start": 0, "end": 3},
            {"text": "ho", "type": "SYMPTOM", "score": 0.98, "start": 4, "end": 6},
            {"text": "Paracetamol", "type": "MEDICINE", "score": 0.97, "start": 7, "end": 18},
        ]


class FakeRAG:
    def run(self, query: str, top_k=None):
        assert "Paracetamol" in query
        return [
            {
                "text": "Respiratory infection initial guidance",
                "score": 0.88,
                "metadata": {"source_type": "medical_guideline", "title": "Respiratory infection initial guidance"},
            }
        ]


def test_workflow_normalizes_retrieves_and_reasons():
    workflow = MedicalWorkflow(ner_predictor=FakeNER(), rag_pipeline=FakeRAG())

    result = workflow.run("sot ho Paracetamol")

    assert result["entities"][0]["text"] == "sot"
    assert result["normalized_entities"][0]["code_system"] == "ICD-10"
    assert result["normalized_entities"][2]["code_system"] == "RxNorm"
    assert result["knowledge"][0]["score"] == 0.88
    assert result["clinical_reasoning"]["possible_diseases"][0]["name"] == "Respiratory infection"
    assert result["confidence"] > 0.0
    assert set(result["processing_time"]) == {"ner_ms", "retriever_ms", "reasoner_ms", "total_ms"}
