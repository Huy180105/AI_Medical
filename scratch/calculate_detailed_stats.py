import json
from pathlib import Path
from collections import Counter

def main():
    output_dir = Path("output")
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))
    
    total_files = len(json_files)
    total_entities = 0
    empty_candidates_count = 0
    non_empty_candidates_count = 0
    
    type_counter = Counter()
    icd_counter = Counter()
    rxnorm_counter = Counter()
    all_candidate_counter = Counter()

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            entities = json.load(f)
            total_entities += len(entities)
            
            for ent in entities:
                ent_type = ent["type"]
                type_counter[ent_type] += 1
                
                cands = ent.get("candidates", [])
                if not cands:
                    empty_candidates_count += 1
                else:
                    non_empty_candidates_count += 1
                    for c in cands:
                        all_candidate_counter[c] += 1
                        if c.startswith("RxNorm:") or c.isdigit():
                            rxnorm_counter[c] += 1
                        else:
                            icd_counter[c] += 1

    lines = []
    lines.append("==========================================================================")
    lines.append("DỮ LIỆU THỐNG KÊ CHI TIẾT 100 FILE OUTPUT SUBMISSION (STEP 1 - STEP 4)")
    lines.append("==========================================================================")
    lines.append(f"Tổng số file output: {total_files}")
    lines.append(f"Tổng số thực thể trích xuất: {total_entities} (Trung bình {total_entities/total_files:.1f} thực thể/file)")
    lines.append("")
    lines.append("1. Phân bố loại thực thể (Entity Types):")
    for tname, tcnt in type_counter.most_common():
        lines.append(f"   - {tname:<20}: {tcnt:>5} ({tcnt/total_entities*100:5.1f}%)")
        
    lines.append("")
    lines.append("2. Thống kê Candidate mapping:")
    lines.append(f"   - Số thực thể CÓ candidate (candidates != []): {non_empty_candidates_count:>5} ({non_empty_candidates_count/total_entities*100:5.1f}%)")
    lines.append(f"   - Số thực thể KHÔNG CÓ candidate (candidates == []): {empty_candidates_count:>5} ({empty_candidates_count/total_entities*100:5.1f}%)")
    
    lines.append("")
    lines.append("3. Top 30 mã Candidate xuất hiện nhiều nhất toàn bộ 100 file:")
    for code, count in all_candidate_counter.most_common(30):
        lines.append(f"   - Code {code:<15}: {count:>4} lần")

    lines.append("")
    lines.append("==========================================================================")
    
    out_file = Path("scratch/detailed_stats_report.txt")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Stats written to {out_file}")

if __name__ == "__main__":
    main()
