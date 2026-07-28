import json
import re
from pathlib import Path
from collections import Counter, defaultdict

def main():
    output_dir = Path("output")
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))
    
    total_files = len(json_files)
    total_entities = 0
    
    type_counts = Counter()
    type_cand_counts = Counter()
    
    text_freq = Counter()
    text_to_cand = defaultdict(list)
    text_to_type = defaultdict(list)
    
    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            entities = json.load(f)
            total_entities += len(entities)
            
            for ent in entities:
                ent_text = ent["text"]
                ent_type = ent["type"]
                cands = ent.get("candidates", [])
                
                type_counts[ent_type] += 1
                if cands:
                    type_cand_counts[ent_type] += 1
                    
                text_freq[ent_text.lower()] += 1
                text_to_type[ent_text.lower()].append(ent_type)
                if cands:
                    text_to_cand[ent_text.lower()].append(cands[0])

    lines = []
    lines.append("==========================================================================")
    lines.append("BÁO CÁO PHÂN TÍCH CHẨN ĐOÁN 5 BƯỚC TOÀN BỘ 100 FILE OUTPUT THEO MENTOR")
    lines.append("==========================================================================")
    lines.append(f"Tổng số file: {total_files}")
    lines.append(f"Tổng số thực thể: {total_entities} (Trung bình {total_entities/total_files:.1f} thực thể/file)\n")
    
    lines.append("--- BƯỚC 1: Thống kê Tỷ lệ Candidate theo Loaị Thực thể ---")
    for etype in ["THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM"]:
        t_cnt = type_counts[etype]
        c_cnt = type_cand_counts[etype]
        pct = (c_cnt / t_cnt * 100) if t_cnt > 0 else 0
        lines.append(f"  * {etype:<15}: {t_cnt:>5} thực thể | {c_cnt:>5} CÓ candidate ({pct:>5.1f}%) | {t_cnt - c_cnt:>5} KHÔNG CÓ candidate ({100 - pct:>5.1f}%)")

    lines.append("\n--- BƯỚC 2 & 3: Top 50 Thực thể Xuất hiện Nhiều nhất & Trạng thái Candidate/Synonym ---")
    lines.append(f"{'STT':<4} | {'Text thực thể':<35} | {'Tần suất':<8} | {'Loại chính':<15} | {'Mã Candidate hiện tại'}")
    lines.append("-" * 90)
    
    most_common_texts = text_freq.most_common(100)
    for idx, (txt, count) in enumerate(most_common_texts[:50], 1):
        main_type = Counter(text_to_type[txt]).most_common(1)[0][0]
        cands_used = set(text_to_cand[txt])
        cand_str = ", ".join(cands_used) if cands_used else "[TRỐNG]"
        lines.append(f"{idx:<4} | {txt:<35} | {count:<8} | {main_type:<15} | {cand_str}")

    lines.append("\n--- BƯỚC 4: Kiểm tra Lỗi Nhầm Loại Thực thể (Type Ambiguity / 2x Penalty) ---")
    ambiguous_types = 0
    for txt, types in text_to_type.items():
        type_set = set(types)
        if len(type_set) > 1 and text_freq[txt] >= 3:
            ambiguous_types += 1
            lines.append(f"  * Thực thể '{txt}' ({text_freq[txt]} lần) bị đoán nhầm giữa các loại: {dict(Counter(types))}")

    if ambiguous_types == 0:
        lines.append("  * Không phát hiện thực thể tần suất cao bị nhầm lẫn loại!")

    out_file = Path("scratch/mentor_5step_audit.txt")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print("5-step empirical audit report generated successfully!")

if __name__ == "__main__":
    main()
