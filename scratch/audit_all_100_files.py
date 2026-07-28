import json
import re
from pathlib import Path

def main():
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Medical terms regex dictionary to check against document text
    medical_vocab = [
        # Diseases
        "thiếu men G6PD", "bệnh G6PD", "G6PD", "bệnh phổi kẽ", "phổi kẽ", "suy giảm miễn dịch",
        "viêm mô tế bào", "tổn thương vùng âm hộ", "bệnh Parkinson", "Parkinson", "bệnh Kawasaki", "Kawasaki",
        "hội chứng buồng trứng đa nang", "buồng trứng đa nang", "ung thư biểu mô tế bào mật", "cholangiocarcinoma",
        "bạch cầu dòng tủy mạn tính", "CML", "đau thắt ngực", "thiếu máu cơ tim", "nhồi máu cơ tim",
        "đột quỵ", "tai biến mạch máu não", "Tăng huyết áp", "tăng huyết áp nguyên phát", "cao huyết áp",
        "đái tháo đường", "tiểu đường", "trào ngược dạ dày thực quản", "trào ngược dạ dày", "GERD",
        "viêm phổi", "viêm phổi cộng đồng", "lao phổi", "xơ gan do rượu", "xơ gan", "suy tim",
        "viêm gan C", "viêm gan B", "bệnh gút", "gút", "hen phế quản", "hen suyễn", "viêm phế quản",
        "thiếu máu tan huyết", "tan huyết", "thiếu máu", "suy thận cấp", "suy thận mạn", "suy thận",
        "nhiễm khuẩn đường tiết niệu", "tăng lipid máu", "hẹp ống sống", "mày đay", "nổi mề đay",
        "tiền sản giật", "hội chứng thận hư", "gan nhiễm mỡ", "sỏi mật", "sỏi thận", "viêm ruột thừa",
        "bệnh dại", "thủy đậu", "bệnh sởi", "sốt xuất huyết", "trầm cảm", "thoái hóa tinh bột", "amyloidosis",
        
        # Symptoms
        "Sốt cao", "sốt", "Tim đập nhanh", "hồi hộp đánh trống ngực", "khó thở", "khó thở khi gắng sức",
        "Vàng da", "vàng da", "vàng mắt", "đau bụng", "Đau bụng", "Buồn nôn", "buồn nôn", "nôn", "đau",
        "ho", "ho khan", "ho đờm", "mệt mỏi", "phù", "phù nề", "đau đầu", "tức ngực", "đau ngực",
        "chóng mặt", "tiêu chảy", "táo bón", "run tay", "tê bì", "ù tai", "rối loạn thị lực", "mất thăng bằng",
        "chán ăn", "sụt cân", "béo phì",
        
        # Drugs
        "amlodipine", "aspirin", "metoprolol", "guaifenesin", "nystatin", "acetaminophen", "paracetamol",
        "pravastatin", "docusate", "senna", "clonazepam", "gleevec", "imatinib", "tylenol", "omeprazole",
        "allopurinol", "suboxone", "Vitamin K", "doxycycline", "bactrim", "băng phiến", "long não", "đậu tằm", "corticoid",
        
        # Tests
        "xét nghiệm máu", "công thức máu", "xét nghiệm nước tiểu", "chụp CT", "chụp MRI", "x-quang",
        "siêu âm", "siêu âm bụng", "siêu âm tim", "điện tâm đồ", "nội soi", "sinh thiết", "chức năng gan"
    ]

    missing_report = {}
    total_missing_count = 0

    for doc_id in range(1, 101):
        txt_path = input_dir / f"{doc_id}.txt"
        json_path = output_dir / f"{doc_id}.json"

        if not txt_path.exists() or not json_path.exists():
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        with open(json_path, "r", encoding="utf-8") as f:
            extracted_json = json.load(f)

        extracted_texts = {e["text"].lower() for e in extracted_json}

        doc_missing = []
        for term in medical_vocab:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                found_text = text[m.start():m.end()]
                if not any(found_text.lower() in ext for ext in extracted_texts):
                    doc_missing.append(found_text)
                    total_missing_count += 1

        if doc_missing:
            missing_report[doc_id] = list(set(doc_missing))

    report_lines = []
    report_lines.append("==========================================================================")
    report_lines.append("BÁO CÁO RÀ SOÁT TỪ CÒN SÓT TRÊN TOÀN BỘ 100 FILE INPUT (1.txt -> 100.txt)")
    report_lines.append("==========================================================================")
    report_lines.append(f"Tổng số từ y khoa bị bỏ sót trên 100 file: {total_missing_count}")
    report_lines.append(f"Số file bị thiếu thực thể: {len(missing_report)} / 100\n")

    for doc_id, missing_list in sorted(missing_report.items()):
        report_lines.append(f"File {doc_id}.txt: Bị thiếu {len(missing_list)} từ -> {missing_list}")

    out_file = Path("scratch/audit_100_files_report.txt")
    out_file.write_text("\n".join(report_lines), encoding="utf-8")
    print("Full 100-file audit finished successfully!")

if __name__ == "__main__":
    main()
