import re
import sys

def expand_clinical_span(text: str, start: int, end: int, ent_type: str) -> tuple[str, int, int]:
    # 1. Expand left for disease prefixes ("Bệnh ", "Hội chứng ", "Hội chứng ")
    if ent_type in ["CHẨN_ĐOÁN", "DISEASE"]:
        left_text = text[:start]
        prefix_match = re.search(r"\b(Bệnh|Hội chứng)\s+$", left_text, re.IGNORECASE)
        if prefix_match:
            shift = len(prefix_match.group(0))
            start -= shift

    # 2. Expand right for clinical diagnosis modifiers (" nguyên phát", " thứ phát", " mạn tính", " cấp tính", " cộng đồng")
    if ent_type in ["CHẨN_ĐOÁN", "DISEASE"]:
        right_text = text[end:end+30]
        suffix_match = re.search(r"^\s*(nguyên phát|thứ phát|mạn tính|cấp tính|cộng đồng|tự miễn|do rượu|do thuốc)", right_text, re.IGNORECASE)
        if suffix_match:
            shift = len(suffix_match.group(0))
            end += shift

    # 3. Expand right for test/procedure organ modifiers (" nội mạc tử cung", " bụng", " tim", " dạ dày", " ngực", " não")
    if ent_type in ["TÊN_XÉT_NGHIỆM", "TEST"]:
        right_text = text[end:end+30]
        suffix_match = re.search(r"^\s*(nội mạc tử cung|buồng trứng|bụng|tim|dạ dày|ngực|não|máu|nước tiểu|tinh dịch|tử cung vòi trứng)", right_text, re.IGNORECASE)
        if suffix_match:
            shift = len(suffix_match.group(0))
            end += shift

    expanded_text = text[start:end].strip()
    return expanded_text, start, end

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sample = "Tiền sử: Tăng huyết áp nguyên phát. Bệnh phổi kẽ do sử dụng corticoid. Sinh thiết nội mạc tử cung gần đây."
    
    # Test Tăng huyết áp
    m1 = re.search(r"Tăng huyết áp", sample)
    if m1:
        t1, s1, e1 = expand_clinical_span(sample, m1.start(), m1.end(), "CHẨN_ĐOÁN")
        print(f"'Tăng huyết áp' -> '{t1}'")

    # Test phổi kẽ
    m2 = re.search(r"phổi kẽ", sample)
    if m2:
        t2, s2, e2 = expand_clinical_span(sample, m2.start(), m2.end(), "CHẨN_ĐOÁN")
        print(f"'phổi kẽ' -> '{t2}'")

    # Test Sinh thiết
    m3 = re.search(r"Sinh thiết", sample)
    if m3:
        t3, s3, e3 = expand_clinical_span(sample, m3.start(), m3.end(), "TÊN_XÉT_NGHIỆM")
        print(f"'Sinh thiết' -> '{t3}'")
