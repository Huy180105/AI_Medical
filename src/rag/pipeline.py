from typing import Any

from src.knowledge.retriever import MedicalKnowledgeRetriever
from src.rag.reranker import IdentityReranker


class MedicalRAGPipeline:
    def __init__(
        self,
        retriever: MedicalKnowledgeRetriever | None = None,
        reranker: IdentityReranker | None = None,
    ) -> None:
        self.retriever = retriever or MedicalKnowledgeRetriever()
        self.reranker = reranker or IdentityReranker()

    def run(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        results = self.retriever.retrieve(query=query, top_k=top_k)
        return self.reranker.rerank(query=query, results=results)
