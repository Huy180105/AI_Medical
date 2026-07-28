from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class ResponseAgent(BaseAgent):
    """
    Finalizes the execution run of the Multi-Agent System. Packs the accumulated states
    into a structured dictionary response for downstream services or API presentation.
    """

    def __init__(self) -> None:
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "response_agent"

    def process(self, context: AgentContext) -> AgentContext:
        # Compile response dictionary
        context.response = {
            "text": context.text,
            "session_id": context.session_id,
            "patient_id": context.patient_id,
            "entities": context.entities,
            "normalized_entities": context.normalized_entities,
            "knowledge": context.knowledge,
            "clinical_decision": context.clinical_decision,
            "recommendations": context.recommendations,
            "confidence": context.confidence,
            "trace": context.trace,
            "metadata": context.metadata
        }
        return context
