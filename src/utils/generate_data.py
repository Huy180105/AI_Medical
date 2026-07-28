import os
import json
import random
from src.utils.segmenter import VietnameseSegmenter
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# Sample entities dictionary for Vietnamese clinical NLP
ENTITIES_POOL = {
    "SYMPTOM": [
        "sốt cao", "ho khan", "khó thở", "đau ngực", "mất vị giác", 
        "nôn mửa", "tiêu chảy", "đau đầu", "chóng mặt", "mệt mỏi", 
        "sổ mũi", "đau họng", "tức ngực", "co giật", "phù nề",
        "nổi mẩn đỏ", "khô miệng", "đau khớp", "chán ăn", "sụt cân"
    ],
    "DISEASE": [
        "cảm cúm", "viêm phổi", "sốt xuất huyết", "tiểu đường", "cao huyết áp", 
        "đau dạ dày", "suy thận", "lao phổi", "hen suyễn", "viêm phế quản", 
        "sở", "thủy đậu", "covid-19", "xơ gan", "suy tim",
        "viêm gan B", "đột quỵ", "gút", "thoái hóa khớp", "sốt siêu vi"
    ],
    "MEDICINE": [
        "Paracetamol", "Ibuprofen", "Amoxicillin", "Metformin", "Amlodipine", 
        "Insulin", "Panadol", "Decolgen", "Augmentin", "Salbutamol",
        "Aspirin", "Penicillin", "Ciprofloxacin", "Omeprazole", "Atorvastatin",
        "Loratadine", "Gliclazide", "Enalapril", "Prednisolone", "Paracetamol 500mg"
    ],
    "TEST": [
        "chụp X-quang", "xét nghiệm máu", "siêu âm bụng", "chụp CT", "chụp MRI", 
        "đo điện tâm đồ", "nội soi dạ dày", "xét nghiệm nước tiểu", "sinh thiết",
        "đo huyết áp", "xét nghiệm chức năng gan", "chụp X-quang phổi"
    ]
}

TEMPLATES = [
    # Template 1: Patient with symptoms and disease
    "Bệnh nhân có triệu chứng {SYMPTOM} và được chẩn đoán mắc {DISEASE}.",
    # Template 2: Patient diagnosed with disease, prescribed medicine
    "Bác sĩ chẩn đoán bệnh nhân bị {DISEASE} và kê đơn điều trị bằng {MEDICINE}.",
    # Template 3: Patient underwent test, showed symptoms
    "Kết quả {TEST} cho thấy bệnh nhân có dấu hiệu {SYMPTOM}.",
    # Template 4: Symptoms and recommended test
    "Bệnh nhân sốt ho kèm theo {SYMPTOM}, cần tiến hành {TEST} ngay lập tức.",
    # Template 5: Disease treated with medicine and monitored by test
    "Để điều trị {DISEASE}, bệnh nhân được chỉ định dùng {MEDICINE} và thực hiện {TEST}.",
    # Template 6: Patient taking medicine, experiencing symptom
    "Sau khi sử dụng {MEDICINE}, bệnh nhân gặp triệu chứng {SYMPTOM}.",
    # Template 7: Complex diagnostic description
    "Bệnh nhân vào viện vì {SYMPTOM}. Qua {TEST}, bác sĩ kết luận bị {DISEASE} và chỉ định uống {MEDICINE}.",
    # Template 8: Underwent multiple tests
    "Người bệnh đã được thực hiện {TEST} và chẩn đoán theo dõi {DISEASE}.",
    # Template 9: Post-treatment monitor
    "Bệnh nhân điều trị {DISEASE} bằng {MEDICINE} ghi nhận cải thiện, không còn {SYMPTOM}.",
    # Template 10: Simple symptom check
    "Khám lâm sàng phát hiện bệnh nhân bị {SYMPTOM}, nghi ngờ mắc {DISEASE}."
]

def generate_sentence_with_spans():
    """
    Generates a single sentence from templates, tracks entity spans,
    and returns (original_text, entity_spans).
    """
    template = random.choice(TEMPLATES)
    
    # Identify which placeholders are in the template
    placeholders = []
    for ent_type in ENTITIES_POOL.keys():
        placeholder = f"{{{ent_type}}}"
        if placeholder in template:
            placeholders.append(ent_type)
            
    # Select random entities for placeholders
    selected_ents = {}
    for p in placeholders:
        selected_ents[p] = random.choice(ENTITIES_POOL[p])
        
    # Build text and track character indices
    # We construct the text segment by segment to track positions accurately
    parts = []
    entity_spans = []
    
    current_idx = 0
    # A simple way is to parse the template
    temp_str = template
    while True:
        # Find next placeholder
        first_p = None
        first_pos = -1
        for p in placeholders:
            pos = temp_str.find(f"{{{p}}}")
            if pos != -1:
                if first_pos == -1 or pos < first_pos:
                    first_pos = pos
                    first_p = p
                    
        if first_p is None:
            # No more placeholders
            parts.append(temp_str)
            break
            
        # Add text before placeholder
        pre_text = temp_str[:first_pos]
        parts.append(pre_text)
        current_idx += len(pre_text)
        
        # Add entity and record span
        ent_val = selected_ents[first_p]
        start_char = current_idx
        end_char = start_char + len(ent_val)
        
        parts.append(ent_val)
        entity_spans.append({
            "start": start_char,
            "end": end_char,
            "label": first_p
        })
        
        current_idx = end_char
        
        # Crop template
        temp_str = temp_str[first_pos + len(first_p) + 2:]
        
    full_text = "".join(parts)
    return full_text, entity_spans

def align_tokens_to_entities(original_text: str, tokens: list[str], entity_spans: list[dict]) -> list[str]:
    """
    Aligns segmented tokens (with underscores) to the original character spans
    and assigns BIO labels.
    """
    labels = ["O"] * len(tokens)
    
    # Reconstruct character spans for the tokens in terms of the original text
    text_ptr = 0
    token_spans = []
    
    for token in tokens:
        clean_token = token.replace("_", "")
        start_pos = -1
        match_ptr = text_ptr
        
        while match_ptr < len(original_text):
            orig_slice = original_text[match_ptr:match_ptr + len(token)]
            orig_clean = orig_slice.replace(" ", "").replace("_", "")
            if clean_token in orig_clean or orig_clean in clean_token:
                temp_clean = ""
                idx = match_ptr
                while idx < len(original_text) and len(temp_clean) < len(clean_token):
                    char = original_text[idx]
                    if char != " " and char != "_":
                        temp_clean += char
                    idx += 1
                
                if temp_clean.lower() == clean_token.lower():
                    start_pos = match_ptr
                    end_pos = idx
                    
                    # Trim leading spaces and underscores from start_pos
                    while start_pos < end_pos and original_text[start_pos] in (" ", "_"):
                        start_pos += 1
                        
                    text_ptr = end_pos
                    token_spans.append((start_pos, end_pos))
                    break
            
            match_ptr += 1
            
        if start_pos == -1:
            token_spans.append((-1, -1))
            
    # Match token spans with entity spans to assign BIO tags
    for i, (t_start, t_end) in enumerate(token_spans):
        if t_start == -1 or t_end == -1:
            continue
            
        for ent in entity_spans:
            ent_start = ent["start"]
            ent_end = ent["end"]
            ent_label = ent["label"]
            
            # Robust overlap check
            if max(t_start, ent_start) < min(t_end, ent_end):
                is_first = True
                for prev_idx in range(i):
                    prev_start, prev_end = token_spans[prev_idx]
                    if prev_start != -1 and max(prev_start, ent_start) < min(prev_end, ent_end):
                        is_first = False
                        break
                
                if is_first:
                    labels[i] = f"B-{ent_label}"
                else:
                    labels[i] = f"I-{ent_label}"
                break
                
    return labels

def generate_dataset(num_samples: int = 500) -> list[dict]:
    """Generates dataset of specified size."""
    dataset = []
    for _ in range(num_samples):
        text, entity_spans = generate_sentence_with_spans()
        # Word segment the sentence
        tokens = VietnameseSegmenter.segment_to_words(text)
        # Align BIO tags
        labels = align_tokens_to_entities(text, tokens, entity_spans)
        
        # Verify alignment
        dataset.append({
            "text": text,
            "tokens": tokens,
            "ner_tags": labels
        })
    return dataset

def main():
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    logger.info("Generating synthetic datasets...")
    
    # Seed for reproducibility
    random.seed(42)
    
    # Generate Train and Val sets
    train_data = generate_dataset(600)
    val_data = generate_dataset(150)
    test_data = generate_dataset(100)
    
    train_path = os.path.join(Config.DATA_DIR, "train.json")
    val_path = os.path.join(Config.DATA_DIR, "val.json")
    test_path = os.path.join(Config.DATA_DIR, "test.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Generated {len(train_data)} training samples -> {train_path}")
    logger.info(f"Generated {len(val_data)} validation samples -> {val_path}")
    logger.info(f"Generated {len(test_data)} test samples -> {test_path}")

if __name__ == "__main__":
    main()
