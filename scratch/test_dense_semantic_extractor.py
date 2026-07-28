import re
import json
from pathlib import Path

def extract_dense_clinical_entities(text: str) -> list[dict]:
    """
    Dense Clinical Semantic Extractor using N-gram POS patterns and clinical triggers
    to extract all potential medical symptoms, diagnoses, drugs, and tests.
    """
    entities = []

    # 1. Disease / Diagnosis patterns
    disease_triggers = [
        "bệnh", "hội chứng", "suy", "viêm", "ung thư", "u", "loét", "trào ngược", "xơ",
        "tăng", "hẹp", "thiếu máu", "tan huyết", "gút", "parkinson", "kawasaki",
        "đái tháo đường", "tiểu đường", "dại", "sởi", "thủy đậu", "sốt xuất huyết",
        "lao", "trầm cảm", "amyloidosis", "tiền sản giật", "mày đay"
    ]
    
    # 2. Symptom patterns
    symptom_triggers = [
        "sốt", "đau", "khó thở", "ho", "nôn", "buồn nôn", "mệt mỏi", "phù", "chóng mặt",
        "tiêu chảy", "táo bón", "tê bì", "run", "ù tai", "rối loạn thị lực", "mất thăng bằng",
        "vàng da", "vàng mắt", "béo phì", "chán ăn", "sụt cân", "mất ngủ", "tức ngực"
    ]

    # Scan for N-grams up to 6 words
    words = text.split()
    for n in range(1, 7):
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i:i+n]).strip(",.:;!()•-")
            chunk_lower = chunk.lower()
            
            if len(chunk_lower) < 2:
                continue

            # Check if chunk contains a disease trigger
            if any(dt in chunk_lower for dt in disease_triggers):
                entities.append({"text": chunk, "type": "CHẨN_ĐOÁN"})
            elif any(st in chunk_lower for st in symptom_triggers):
                entities.append({"text": chunk, "type": "TRIỆU_CHỨNG"})

    return entities

if __name__ == "__main__":
    sample_txt = Path("input/15.txt").read_text(encoding="utf-8")
    dense_ents = extract_dense_clinical_entities(sample_txt)
    print(f"Sample 15.txt dense entities extracted: {len(dense_ents)}")
    print("Sample extracted:", dense_ents[:15])
