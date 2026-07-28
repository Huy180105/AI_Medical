from src.knowledge.retriever import MedicalKnowledgeRetriever
from src.knowledge.vector_store import SearchResult


class FakeVectorStore:
    def search(self, query: str, top_k: int):
        assert query == "fever cough"
        assert top_k == 2
        return [
            SearchResult(
                text="Respiratory infection guidance",
                score=0.91,
                metadata={"title": "Respiratory guidance"},
            )
        ]


def test_retriever_returns_text_score_and_metadata():
    retriever = MedicalKnowledgeRetriever(vector_store=FakeVectorStore(), top_k=2)

    results = retriever.retrieve("fever cough")

    assert results == [
        {
            "text": "Respiratory infection guidance",
            "score": 0.91,
            "metadata": {"title": "Respiratory guidance"},
        }
    ]
