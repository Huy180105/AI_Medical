import re
import sys

def extract_test_results(text: str, test_entities: list[dict]) -> list[dict]:
    results = []
    # Pattern to match numbers (e.g. 14,43 or 76.4), positive/negative (âm tính/dương tính), or simple clinical descriptors (tăng/giảm)
    result_pattern = re.compile(
        r"^\s*([\:\=\s]+)?(\d+[\,\.]\d+|\b(âm tính|dương tính|tăng|giảm|bình thường)\b)",
        re.IGNORECASE
    )

    for ent in test_entities:
        end = ent["end"]
        # Look ahead up to 25 characters
        lookahead = text[end:end+25]
        match = result_pattern.search(lookahead)
        if match:
            matched_text = match.group(2).strip()
            # Calculate actual offsets in text
            match_start = end + lookahead.find(matched_text)
            match_end = match_start + len(matched_text)
            
            results.append({
                "text": matched_text,
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "candidates": [],
                "assertions": [],
                "position": [match_start, match_end]
            })
    return results

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sample_text = "WBC:14,43; NEUT%:76,4; cấy máu: âm tính; marker viêm: tăng;"
    test_ents = [
        {"text": "WBC", "start": 0, "end": 3},
        {"text": "NEUT%", "start": 11, "end": 16},
        {"text": "cấy máu", "start": 23, "end": 30},
        {"text": "marker viêm", "start": 41, "end": 52}
    ]
    extracted = extract_test_results(sample_text, test_ents)
    print("Extracted test results:", extracted)
