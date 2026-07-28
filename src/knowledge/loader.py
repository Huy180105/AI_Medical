import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    text: str
    metadata: dict[str, Any]


class KnowledgeLoader:
    def __init__(self, knowledge_base_dir: str | None = None) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir or Config.KNOWLEDGE_BASE_DIR)

    def load_documents(self) -> list[KnowledgeDocument]:
        if not self.knowledge_base_dir.exists():
            logger.warning("Knowledge base directory does not exist: %s", self.knowledge_base_dir)
            return []

        documents: list[KnowledgeDocument] = []
        for path in sorted(self.knowledge_base_dir.iterdir()):
            if path.name in {"metadata.pkl", "faiss.index"}:
                continue
            if path.suffix.lower() == ".csv":
                documents.extend(self._load_csv(path))
            elif path.suffix.lower() == ".json":
                documents.extend(self._load_json(path))

        logger.info("Loaded %s knowledge documents from %s", len(documents), self.knowledge_base_dir)
        return documents

    def _load_csv(self, path: Path) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row_index, row in enumerate(reader):
                doc_id = row.get("code") or row.get("id") or f"{path.stem}-{row_index}"
                title = row.get("title") or row.get("name") or doc_id
                description = row.get("description") or row.get("text") or ""
                keywords = row.get("keywords") or ""
                text = " ".join(part for part in [title, description, keywords] if part).strip()
                documents.append(
                    KnowledgeDocument(
                        id=f"{path.stem}:{doc_id}",
                        text=text,
                        metadata={
                            "source": os.fspath(path),
                            "source_type": path.stem,
                            "code": doc_id,
                            "title": title,
                            "category": row.get("category", ""),
                            "raw": dict(row),
                        },
                    )
                )
        return documents

    def _load_json(self, path: Path) -> list[KnowledgeDocument]:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        rows = payload if isinstance(payload, list) else payload.get("documents", [])
        documents: list[KnowledgeDocument] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            doc_id = str(row.get("id") or f"{path.stem}-{row_index}")
            title = str(row.get("title") or doc_id)
            body = str(row.get("text") or row.get("description") or "")
            keywords = row.get("keywords") or []
            keyword_text = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
            text = " ".join(part for part in [title, body, keyword_text] if part).strip()
            documents.append(
                KnowledgeDocument(
                    id=f"{path.stem}:{doc_id}",
                    text=text,
                    metadata={
                        "source": os.fspath(path),
                        "source_type": path.stem,
                        "code": doc_id,
                        "title": title,
                        "category": row.get("category", ""),
                        "raw": dict(row),
                    },
                )
            )
        return documents
