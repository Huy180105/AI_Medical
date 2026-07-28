from typing import Any
from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class ReasoningAgent(BaseAgent):
    """
    Synthesizes intermediate reasoning states, cross-referencing RAG guidelines
    with graph-derived diseases and symptoms to ensure facts align.
    """

    def __init__(self) -> None:
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "reasoning_agent"

    def process(self, context: AgentContext) -> AgentContext:
        # Cross-reference RAG guidelines with graph candidate diseases
        matched_indicators = []
        
        disease_candidates = [
            cand.get("disease", "").lower()
            for cand in context.graph_results
        ]

        # For every retrieved guideline, check if it explicitly mentions any candidate disease
        for guideline in context.guidelines:
            g_text = guideline.get("text", "").lower()
            g_title = guideline.get("metadata", {}).get("title", "").lower()
            
            for disease in disease_candidates:
                if disease in g_text or disease in g_title:
                    matched_indicators.append({
                        "disease": disease,
                        "guideline": guideline.get("metadata", {}).get("title", "Clinical Guideline"),
                        "source": "rag_cross_reference"
                    })
                    
        context.metadata["reasoning_alignment"] = matched_indicators
        return context
