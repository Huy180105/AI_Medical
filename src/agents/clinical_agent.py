from typing import Any
from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class ClinicalAgent(BaseAgent):
    """
    Acts as the clinical decision Support Agent. Evaluates clinical safety rules,
    contraindications, and calculates the patient risk stratification.
    """

    def __init__(self, decision_engine: Any) -> None:
        self.decision_engine = decision_engine
        self.confidence = 0.90

    @property
    def name(self) -> str:
        return "clinical_agent"

    def process(self, context: AgentContext) -> AgentContext:
        # Run clinical decision support engine
        decision = self.decision_engine.make_decision(
            context.entities,
            context.graph_results
        )
        context.clinical_decision = decision
        
        # Propagate engine confidence
        self.confidence = decision.get("confidence", 0.90)
        
        return context
