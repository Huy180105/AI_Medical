from typing import Any
from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class NERAgent(BaseAgent):
    """
    Extracts clinical named entities from the user's text input using the PhoBERT NER model.
    """

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor
        self.confidence = 0.95

    @property
    def name(self) -> str:
        return "ner_agent"

    def process(self, context: AgentContext) -> AgentContext:
        if not context.text:
            return context

        entities = self.predictor.predict(context.text)
        context.entities = entities
        
        # Propagate average score of extracted entities as the agent's contribution to confidence
        if entities:
            avg_score = sum(ent.get("score", 1.0) for ent in entities) / len(entities)
            self.confidence = round(avg_score, 4)
        else:
            self.confidence = 1.0

        return context
