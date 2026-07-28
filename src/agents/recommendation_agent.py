from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class RecommendationAgent(BaseAgent):
    """
    Extracts, formats, and adjusts patient-centric guidelines, laboratory test lists,
    referrals, and lifestyle recommendations.
    """

    def __init__(self) -> None:
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "recommendation_agent"

    def process(self, context: AgentContext) -> AgentContext:
        # Extract recommendations from clinical decision block
        recs = context.clinical_decision.get("recommendations", {})
        context.recommendations = recs
        
        return context
