import re
import numpy as np
from typing import Any
from src.ranking.candidate_models import EntityCandidate

# Clinical Concept Aliases Mapping
CONCEPT_ALIASES = {
    "cao ha": "tăng huyết áp",
    "tha": "tăng huyết áp",
    "tăng ha": "tăng huyết áp",
    "tiểu đường": "đái tháo đường",
    "đtđ": "đái tháo đường",
    "gút": "bệnh gút",
    "phổi kẽ": "bệnh phổi kẽ",
    "suy thận": "suy thận mạn",
    "gerd": "trào ngược dạ dày thực quản",
    "thiếu men g6pd": "thiếu men G6PD",
    "bệnh g6pd": "thiếu men G6PD",
    "g6pd": "thiếu men G6PD",
    "cml": "bạch cầu dòng tủy mạn tính",
    "cholangiocarcinoma": "ung thư biểu mô tế bào mật"
}

def levenshtein_similarity(s1: str, s2: str) -> float:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    dist = previous_row[-1]
    return 1.0 - dist / max(len(s1), len(s2))

def token_set_ratio(s1: str, s2: str) -> float:
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    diff1 = t1.difference(t2)
    diff2 = t2.difference(t1)
    intersection = t1.intersection(t2)
    
    s_inter = " ".join(sorted(list(intersection)))
    s_diff1 = " ".join(sorted(list(diff1)))
    s_diff2 = " ".join(sorted(list(diff2)))
    
    res1 = s_inter + " " + s_diff1 if s_diff1 else s_inter
    res2 = s_inter + " " + s_diff2 if s_diff2 else s_inter
    
    return max(
        levenshtein_similarity(s_inter, res1),
        levenshtein_similarity(s_inter, res2),
        levenshtein_similarity(res1, res2)
    )

class ClinicalVectorStore:
    """
    State-of-the-Art Dense Semantic Retrieval Vector Store.
    Pre-computes and indexes PhoBERT Base embeddings for standard concepts.
    Uses hybrid scoring (Cosine Embedding + Levenshtein + Token Set Ratio).
    """

    def __init__(self) -> None:
        self.concepts: list[dict[str, str]] = []
        self._initialize_knowledge_base()
        self.embedder = None
        self.concept_embeddings = None

    def _initialize_knowledge_base(self) -> None:
        """Populates standard concept database."""
        from src.ranking.concept_database import get_comprehensive_concepts
        self.concepts = get_comprehensive_concepts()

    def _lazy_init_embedder(self) -> None:
        """Loads PhoBERT embedder on CPU offline and pre-computes concept displays."""
        if self.embedder is not None:
            return
            
        print("Lazy-initializing PhoBERT semantic embedder on CPU...")
        from sentence_transformers import SentenceTransformer, models
        word_embedding_model = models.Transformer('vinai/phobert-base')
        pooling_model = models.Pooling(word_embedding_model.get_embedding_dimension())
        # Force CPU device to prevent GPU/CUDA assertion errors
        self.embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
        
        # Pre-compute embeddings for all concept displays
        displays = [c["display"] for c in self.concepts]
        self.concept_embeddings = self.embedder.encode(displays, show_progress_bar=False)
        print("PhoBERT embeddings computed for all concept displays on CPU.")

    def search_candidates(self, query: str, entity_type: str = "ALL", top_k: int = 30) -> list[EntityCandidate]:
        """
        Retrieves top-K candidate concepts using dense embeddings & fuzzy string matching.
        """
        self._lazy_init_embedder()
        
        # Truncate query length to prevent token length exceeding PhoBERT's 258 position limits
        query_cleaned = query.lower().strip()[:150]
        if query_cleaned in CONCEPT_ALIASES:
            query_cleaned = CONCEPT_ALIASES[query_cleaned]

        query_tokens = set(re.findall(r"\w+", query_cleaned))

        type_mapping = {
            "THUỐC": "MEDICINE",
            "CHẨN_ĐOÁN": "DISEASE",
            "TRIỆU_CHỨNG": "SYMPTOM",
            "TÊN_XÉT_NGHIỆM": "TEST",
            "MEDICINE": "MEDICINE",
            "DISEASE": "DISEASE",
            "SYMPTOM": "SYMPTOM",
            "TEST": "TEST"
        }
        target_type = type_mapping.get(entity_type, entity_type)

        # Compute query embedding on CPU
        query_emb = self.embedder.encode([query_cleaned], show_progress_bar=False)[0]
        query_norm = np.linalg.norm(query_emb)

        scored_candidates = []

        for idx, c in enumerate(self.concepts):
            if target_type != "ALL" and c["type"] != target_type:
                continue

            display_cleaned = c["display"].lower()
            display_tokens = set(re.findall(r"\w+", display_cleaned))

            # 1. Cosine Embedding Similarity
            c_emb = self.concept_embeddings[idx]
            c_norm = np.linalg.norm(c_emb)
            if query_norm > 0 and c_norm > 0:
                cosine_sim = float(np.dot(query_emb, c_emb) / (query_norm * c_norm))
            else:
                cosine_sim = 0.0

            # 2. Fuzzy similarities
            lev_sim = levenshtein_similarity(query_cleaned, display_cleaned)
            ts_ratio = token_set_ratio(query_cleaned, display_cleaned)

            # 3. Hybrid score combination
            sim = 0.5 * cosine_sim + 0.3 * lev_sim + 0.2 * ts_ratio

            # Substring boost
            if query_cleaned in display_cleaned or display_cleaned in query_cleaned:
                sim *= 1.25

            if sim > 0.10:
                scored_candidates.append({
                    "code": c["code"],
                    "display": c["display"],
                    "score": sim
                })

        # Sort descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_slice = scored_candidates[:top_k]

        candidates = []
        for rank, cand in enumerate(top_slice, 1):
            candidates.append(EntityCandidate(
                code=cand["code"],
                display=cand["display"],
                score=cand["score"],
                confidence=cand["score"],
                rank=rank
            ))

        return candidates
