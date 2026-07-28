import json
import re
from pathlib import Path
from collections import Counter

def main():
    output_dir = Path("output")
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))
    
    total_entities = 0
    type_counts = Counter()
    entity_text_counts = Counter()
    empty_cand_counts = Counter()
    short_entity_counts = Counter()
    suspicious_terms = Counter()
    
    suspicious_keywords = [
        "gen", "đột biến", "nhiễm sắc thể", "tác nhân", "phá hủy", "chuyển hóa",
        "mong manh", "tùy thuộc", "hoạt tính", "cơ chế", "thực chất", "nguyên nhân",
        "yếu tố", "quá trình", "kết quả", "giai đoạn", "mức độ", "tình trạng", "phương pháp",
        "biện pháp", "khả năng", "tác dụng", "ảnh hưởng", "biểu hiện", "vấn đề", "trường hợp",
        "định kỳ", "bản chất", "cấu trúc", "chức năng", "enzyme", "enzym", "protein", "dna", "rna"
    ]

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        total_entities += len(data)
        for item in data:
            txt = item["text"].strip()
            etype = item["type"]
            cands = item.get("candidates", [])
            
            type_counts[etype] += 1
            entity_text_counts[(txt.lower(), etype)] += 1
            
            if not cands:
                empty_cand_counts[txt.lower()] += 1
                
            if len(txt.split()) <= 2:
                short_entity_counts[(txt.lower(), etype)] += 1
                
            txt_lower = txt.lower()
            if any(sk in txt_lower for sk in suspicious_keywords):
                suspicious_terms[(txt_lower, etype)] += 1

    report = []
    report.append("==========================================================================")
    report.append("BÁO CÁO THỐNG KÊ CHI TIẾT 100 FILE OUTPUT (AUDIT TOÀN DIỆN FALSE POSITIVES)")
    report.append("==========================================================================")
    report.append(f"1. Tổng số thực thể đang xuất ra trên 100 file: {total_entities} (Trung bình {total_entities/100:.1f} entity/file)")
    report.append("\n2. Phân bố theo Loại thực thể (Entity Types):")
    for etype, count in type_counts.most_common():
        report.append(f"   - {etype}: {count} ({count/total_entities*100:.1f}%)")
        
    report.append("\n3. Top 30 thực thể nghi ngờ sai loại / nghi ngờ False Positives:")
    for (txt, etype), count in suspicious_terms.most_common(30):
        report.append(f"   - '{txt}' [{etype}]: {count} lần")
        
    report.append("\n4. Top 30 thực thể xuất hiện nhiều nhất toàn bộ 100 file:")
    for (txt, etype), count in entity_text_counts.most_common(30):
        report.append(f"   - '{txt}' [{etype}]: {count} lần")
        
    report.append("\n5. Top 30 thực thể KHÔNG CÓ CANDIDATE (candidates == []):")
    for txt, count in empty_cand_counts.most_common(30):
        report.append(f"   - '{txt}': {count} lần")

    out_path = Path("scratch/suspicious_entities_audit.txt")
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Audit completed! Summary written to {out_path}")

if __name__ == "__main__":
    main()
