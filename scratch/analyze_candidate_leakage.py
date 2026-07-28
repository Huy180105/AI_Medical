import json
from pathlib import Path
from collections import Counter

def main():
    output_dir = Path("output")
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))
    
    cand_by_type = Counter()
    total_by_type = Counter()
    has_cand_by_type = Counter()
    
    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            entities = json.load(f)
            for e in entities:
                etype = e["type"]
                total_by_type[etype] += 1
                if e["candidates"]:
                    has_cand_by_type[etype] += 1
                    for c in e["candidates"]:
                        cand_by_type[(etype, c)] += 1

    lines = []
    lines.append("==========================================================================")
    lines.append("BÁO CÁO PHÂN TÍCH CANDIDATE THEO LOẠI THỰC THỂ (ENTITY TYPE BREAKDOWN)")
    lines.append("==========================================================================")
    for etype in ["THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM"]:
        total = total_by_type[etype]
        has_c = has_cand_by_type[etype]
        pct = (has_c / total * 100) if total > 0 else 0
        lines.append(f"\n{etype}:")
        lines.append(f"  - Tổng số thực thể: {total}")
        lines.append(f"  - Số thực thể có Candidate: {has_c} ({pct:.1f}%)")
        lines.append(f"  - Top 10 Candidates của {etype}:")
        type_cands = [(c, cnt) for (t, c), cnt in cand_by_type.items() if t == etype]
        type_cands.sort(key=lambda x: x[1], reverse=True)
        for code, count in type_cands[:10]:
            lines.append(f"       * {code:<15}: {count} lần")

    out_file = Path("scratch/candidate_breakdown.txt")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print("Candidate breakdown written to scratch/candidate_breakdown.txt")

if __name__ == "__main__":
    main()
