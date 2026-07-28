from typing import Any
from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class GraphAgent(BaseAgent):
    """
    Traverses the Medical Knowledge Graph to find path explanations, candidate diseases,
    complications, and contraindications matching the clinical presentation.
    """

    def __init__(self, graph_reasoner: Any) -> None:
        self.graph_reasoner = graph_reasoner
        self.confidence = 0.92

    @property
    def name(self) -> str:
        return "graph_agent"

    def process(self, context: AgentContext) -> AgentContext:
        if not context.entities:
            return context

        # Run graph reasoning based on NER entities
        graph_findings = self.graph_reasoner.reason(context.entities)
        context.graph_results = graph_findings
        
        # Propagate top candidate confidence if available
        if graph_findings:
            top_conf = graph_findings[0].get("confidence", 0.8)
            self.confidence = round(top_conf, 4)
        else:
            self.confidence = 1.0

        return context
