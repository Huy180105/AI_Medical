import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.embeddings import Embedder, SentenceTransformerEmbedder
from src.knowledge.loader import KnowledgeDocument, KnowledgeLoader
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any]


class FaissVectorStore:
    def __init__(
        self,
        loader: KnowledgeLoader | None = None,
        embedder: Embedder | None = None,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        self.loader = loader or KnowledgeLoader()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.index_path = Path(index_path or Config.VECTOR_INDEX_PATH)
        self.metadata_path = Path(metadata_path or Config.VECTOR_METADATA_PATH)
        self._index = None
        self._documents: list[KnowledgeDocument] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.ensure_ready()
        if not query.strip() or self._index is None or not self._documents:
            return []

        query_vector = self.embedder.encode([query])
        scores, indices = self._index.search(query_vector, min(top_k, len(self._documents)))
        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            document = self._documents[int(index)]
            results.append(
                SearchResult(
                    text=document.text,
                    score=float(score),
                    metadata=document.metadata,
                )
            )
        return results

    def ensure_ready(self) -> None:
        if self._index is not None and self._documents:
            return

        documents = self.loader.load_documents()
        if self._can_load_existing(documents):
            self._load()
            return

        self.build(documents)

    def build(self, documents: list[KnowledgeDocument] | None = None) -> None:
        documents = documents if documents is not None else self.loader.load_documents()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        import faiss

        if not documents:
            self._index = faiss.IndexFlatIP(1)
            self._documents = []
            return

        embeddings = self.embedder.encode([document.text for document in documents])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
            raise ValueError("Embedding model returned invalid shape for knowledge documents.")

        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.asarray(embeddings, dtype=np.float32))

        faiss.write_index(index, os.fspath(self.index_path))
        with self.metadata_path.open("wb") as file:
            pickle.dump({"documents": documents, "document_count": len(documents)}, file)

        self._index = index
        self._documents = documents
        logger.info("Built FAISS index with %s documents at %s", len(documents), self.index_path)

    def _can_load_existing(self, documents: list[KnowledgeDocument]) -> bool:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        try:
            with self.metadata_path.open("rb") as file:
                payload = pickle.load(file)
            return int(payload.get("document_count", -1)) == len(documents)
        except Exception as exc:
            logger.warning("Unable to read vector metadata, rebuilding index: %s", exc)
            return False

    def _load(self) -> None:
        import faiss

        self._index = faiss.read_index(os.fspath(self.index_path))
        with self.metadata_path.open("rb") as file:
            payload = pickle.load(file)
        self._documents = payload["documents"]
        logger.info("Loaded FAISS index from %s", self.index_path)
