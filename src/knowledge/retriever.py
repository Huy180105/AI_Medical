from typing import Any

from src.knowledge.vector_store import FaissVectorStore
from src.utils.config import Config


class MedicalKnowledgeRetriever:
    def __init__(self, vector_store: FaissVectorStore | None = None, top_k: int | None = None) -> None:
        self.vector_store = vector_store or FaissVectorStore()
        self.top_k = top_k or Config.RETRIEVER_TOP_K

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        results = self.vector_store.search(query=query, top_k=top_k or self.top_k)
        return [
            {
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in results
        ]
