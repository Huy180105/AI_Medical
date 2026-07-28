import json
import re

# Comprehensive Normalization Dictionary
NORMALIZATION_MAP = {
    # Diseases (CHẨN_ĐOÁN)
    "cao huyết áp": ("tăng huyết áp", "I10", "CHẨN_ĐOÁN"),
    "tha": ("tăng huyết áp", "I10", "CHẨN_ĐOÁN"),
    "tiểu đường": ("đái tháo đường", "E11.9", "CHẨN_ĐOÁN"),
    "đtđ": ("đái tháo đường", "E11.9", "CHẨN_ĐOÁN"),
    "gút": ("bệnh gút", "M10.9", "CHẨN_ĐOÁN"),
    "parkinson": ("bệnh parkinson", "G20", "CHẨN_ĐOÁN"),
    "kawasaki": ("bệnh kawasaki", "M30.3", "CHẨN_ĐOÁN"),
    "dại": ("bệnh dại", "A82", "CHẨN_ĐOÁN"),
    "gerd": ("trào ngược dạ dày thực quản", "K21.9", "CHẨN_ĐOÁN"),
    "amyloidosis": ("thoái hóa tinh bột", "E85.9", "CHẨN_ĐOÁN"),
    "g6pd": ("thiếu men G6PD", "D55.0", "CHẨN_ĐOÁN"),
    "cml": ("bạch cầu dòng tủy mạn tính", "C92.1", "CHẨN_ĐOÁN"),
    "cholangiocarcinoma": ("ung thư biểu mô tế bào mật", "C22.1", "CHẨN_ĐOÁN"),
    "suy thận": ("suy thận mạn", "N18.9", "CHẨN_ĐOÁN"),
    "thận mạn": ("suy thận mạn", "N18.9", "CHẨN_ĐOÁN"),
    "thủy đậu": ("thủy đậu", "B01.9", "CHẨN_ĐOÁN"),
    "sởi": ("bệnh sởi", "B05", "CHẨN_ĐOÁN"),
    "sốt xuất huyết": ("sốt xuất huyết", "A90", "CHẨN_ĐOÁN"),
    "trầm cảm": ("trầm cảm", "F32.9", "CHẨN_ĐOÁN"),
    "viêm mô tế bào": ("viêm mô tế bào", "L03.9", "CHẨN_ĐOÁN"),
    "phổi kẽ": ("bệnh phổi kẽ", "J84.9", "CHẨN_ĐOÁN"),
    "bệnh phổi kẽ": ("bệnh phổi kẽ", "J84.9", "CHẨN_ĐOÁN"),

    # Standard Symptoms (TRIỆU_CHỨNG) with ICD-10
    "khó thở": ("khó thở", "R06.0", "TRIỆU_CHỨNG"),
    "khó thở khi gắng sức": ("khó thở khi gắng sức", "R06.02", "TRIỆU_CHỨNG"),
    "sốt": ("sốt", "R50.9", "TRIỆU_CHỨNG"),
    "sốt cao": ("Sốt cao", "R50.9", "TRIỆU_CHỨNG"),
    "đau bụng": ("đau bụng", "R10.9", "TRIỆU_CHỨNG"),
    "buồn nôn": ("Buồn nôn", "R11.0", "TRIỆU_CHỨNG"),
    "nôn": ("nôn", "R11", "TRIỆU_CHỨNG"),
    "mệt mỏi": ("mệt mỏi", "R53.83", "TRIỆU_CHỨNG"),
    "phù": ("phù", "R60.9", "TRIỆU_CHỨNG"),
    "phù nề": ("phù nề", "R60.9", "TRIỆU_CHỨNG"),
    "đau đầu": ("đau đầu", "R51", "TRIỆU_CHỨNG"),
    "đau ngực": ("đau ngực", "R07.4", "TRIỆU_CHỨNG"),
    "tức ngực": ("tức ngực", "R07.4", "TRIỆU_CHỨNG"),
    "chóng mặt": ("chóng mặt", "R42", "TRIỆU_CHỨNG"),
    "tiêu chảy": ("tiêu chảy", "R19.7", "TRIỆU_CHỨNG"),
    "vàng da": ("Vàng da", "R17", "TRIỆU_CHỨNG"),
    "vàng mắt": ("vàng mắt", "R17", "TRIỆU_CHỨNG"),
    "béo phì": ("béo phì", "E66.9", "TRIỆU_CHỨNG"),
    "tim đập nhanh": ("Tim đập nhanh", "R00.0", "TRIỆU_CHỨNG"),
    "hồi hộp đánh trống ngực": ("hồi hộp đánh trống ngực", "R00.0", "TRIỆU_CHỨNG"),
    "đánh trống ngực": ("đánh trống ngực", "R00.0", "TRIỆU_CHỨNG"),
    "tê bì": ("tê bì", "R20.2", "TRIỆU_CHỨNG"),
    "run tay": ("run tay", "R25.1", "TRIỆU_CHỨNG"),
    "ù tai": ("ù tai", "H93.1", "TRIỆU_CHỨNG"),
    "rối loạn thị lực": ("rối loạn thị lực", "H53.9", "TRIỆU_CHỨNG"),
    "mất thăng bằng": ("mất thăng bằng", "R26.81", "TRIỆU_CHỨNG")
}

print(f"Loaded {len(NORMALIZATION_MAP)} clinical normalization and ontology mappings!")
