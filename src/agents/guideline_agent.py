from typing import Any
from src.agents.base import BaseAgent
from src.agents.context import AgentContext


class GuidelineAgent(BaseAgent):
    """
    Queries the vector database/RAG pipeline to retrieve clinical guidelines
    relevant to the patient's symptoms and history.
    """

    def __init__(self, rag_pipeline: Any | None = None) -> None:
        self.rag_pipeline = rag_pipeline
        self.confidence = 0.88

    @property
    def name(self) -> str:
        return "guideline_agent"

    def process(self, context: AgentContext) -> AgentContext:
        # Build retrieval query combining original text and normalized entity names
        normalized_names = [
            str(ent.get("name") or ent.get("original") or "")
            for ent in context.normalized_entities
        ]
        query = f"{context.text} " + " ".join(normalized_names)
        query = query.strip()

        if not query:
            return context

        if self.rag_pipeline is None:
            try:
                from src.rag.pipeline import MedicalRAGPipeline
                self.rag_pipeline = MedicalRAGPipeline()
            except Exception:
                self.rag_pipeline = False

        if self.rag_pipeline and self.rag_pipeline is not False:
            try:
                retrieved_guidelines = self.rag_pipeline.run(query)
                context.guidelines = retrieved_guidelines
                context.knowledge = retrieved_guidelines
            except Exception:
                pass

        return context
