from src.agent.medical_agent import MedicalAgent


class FakeWorkflow:
    def run(self, text: str):
        return {
            "entities": [],
            "normalized_entities": [],
            "knowledge": [],
            "clinical_reasoning": {"confidence": 0.35},
            "confidence": 0.35,
            "processing_time": {"ner_ms": 0.0, "retriever_ms": 0.0, "reasoner_ms": 0.0, "total_ms": 0.0},
            "gpu_usage": {"available": False, "device": "cpu"},
        }


class NoopConversationMemory:
    def add_turn(self, session_id, turn):
        self.session_id = session_id
        self.turn = turn


class NoopContextMemory:
    def save_context(self, session_id, context):
        self.session_id = session_id
        self.context = context


def test_agent_process_returns_plan_and_result():
    conversation_memory = NoopConversationMemory()
    context_memory = NoopContextMemory()
    agent = MedicalAgent(
        ner_predictor=None,
        workflow=FakeWorkflow(),
        conversation_memory=conversation_memory,
        context_memory=context_memory,
    )

    result = agent.process("sot ho", session_id="session-1", patient_id="patient-1")

    assert result["plan"] == ["ner", "normalize", "retrieve", "reason", "respond"]
    assert conversation_memory.session_id == "session-1"
    assert context_memory.context["patient_id"] == "patient-1"
