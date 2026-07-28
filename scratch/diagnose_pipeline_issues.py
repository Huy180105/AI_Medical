import json
import re
from pathlib import Path

def main():
    input_dir = Path("input")
    output_dir = Path("output")
    
    txt_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))
    
    # 1. Check Drug Span Expansion potential
    drug_with_dosage_pattern = re.compile(r"\b[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]+\s+\d+(\.\d+)?\s*(mg|ml|g|mcg|viên|ống|chai|túi)\b", re.IGNORECASE)
    
    total_drugs = 0
    drugs_without_dosage = 0
    expandable_drugs = 0
    
    # 2. Check Boundary Noise
    noisy_boundary_count = 0
    total_entities = 0
    
    # 3. Check Section-based isHistorical potential
    historical_section_pattern = re.compile(r"(tiền sử|trước nhập viện|các bệnh mãn tính|bệnh cũ)", re.IGNORECASE)
    historical_entities_count = 0
    
    for txt_file, json_file in zip(txt_files, json_files):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        with open(json_file, "r", encoding="utf-8") as f:
            entities = json.load(f)
            
        is_hist_doc = bool(historical_section_pattern.search(text[:200]))
        
        for ent in entities:
            total_entities += 1
            ent_text = ent["text"]
            ent_type = ent["type"]
            start, end = ent["position"]
            
            # Check noise in text boundary
            if re.search(r"^[!•\-\.\s\d\:]+|[!•\-\.\s\:]+$", ent_text):
                noisy_boundary_count += 1
                
            # Check Drug dosage potential
            if ent_type == "THUỐC":
                total_drugs += 1
                # Check surrounding context in text for dosage (next 30 chars)
                after_text = text[end:end+35]
                dosage_match = re.search(r"^\s*\d+(\.\d+)?\s*(mg|ml|g|mcg|viên|gói|po|daily|bid|qid|qhs|q6h)", after_text, re.IGNORECASE)
                if dosage_match:
                    expandable_drugs += 1
                else:
                    drugs_without_dosage += 1
                    
            if is_hist_doc:
                historical_entities_count += 1

    lines = []
    lines.append("==========================================================================")
    lines.append("BÁO CÁO CHẨN ĐOÁN CHUYÊN SÂU 4 ĐIỂM NÚT THẮT THEO YÊU CẦU MENTOR")
    lines.append("==========================================================================")
    lines.append(f"1. Thống kê mở rộng Span THUỐC (Drug Span Expansion):")
    lines.append(f"   - Tổng số thực thể THUỐC hiện tại: {total_drugs}")
    lines.append(f"   - Số thực thể THUỐC có thể mở rộng kèm liều/đường dùng/tần suất: {expandable_drugs} ({expandable_drugs/total_drugs*100:.1f}%)")
    lines.append(f"   - Số thực thể THUỐC chỉ có tên hoạt chất đơn lẻ: {drugs_without_dosage} ({drugs_without_dosage/total_drugs*100:.1f}%)")
    lines.append("")
    lines.append(f"2. Thống kê Lỗi Ranh giới/Ký tự gây nhiễu (Boundary Noise):")
    lines.append(f"   - Tổng số thực thể: {total_entities}")
    lines.append(f"   - Số thực thể bị dính ký tự rác (như '!', '•', dấu câu, số thứ tự): {noisy_boundary_count} ({noisy_boundary_count/total_entities*100:.1f}%)")
    lines.append("")
    lines.append(f"3. Thống kê Tiềm năng Gán nhãn isHistorical theo Tiêu đề đoạn (Section Header):")
    lines.append(f"   - Thực thể nằm trong tài liệu/đoạn có chứa tiêu đề Tiền sử/Trước nhập viện: {historical_entities_count} ({historical_entities_count/total_entities*100:.1f}%)")
    lines.append("==========================================================================")

    out_file = Path("scratch/diagnostic_report.txt")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print("Diagnostic report generated successfully!")

if __name__ == "__main__":
    main()
