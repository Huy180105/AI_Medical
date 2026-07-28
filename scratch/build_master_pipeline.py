import re
import os
import json
import zipfile
from pathlib import Path

# Strict Ontology Mapping to eliminate 2x type error penalty
STRICT_ONTOLOGY = {
    # SYMPTOMS (TRIỆU_CHỨNG)
    "sốt": "TRIỆU_CHỨNG", "sốt cao": "TRIỆU_CHỨNG", "sốt nhẹ": "TRIỆU_CHỨNG", "sốt nóng": "TRIỆU_CHỨNG",
    "gai rét": "TRIỆU_CHỨNG", "rét run": "TRIỆU_CHỨNG", "ớn lạnh": "TRIỆU_CHỨNG",
    "khó thở": "TRIỆU_CHỨNG", "gắng sức": "TRIỆU_CHỨNG",
    "mệt mỏi": "TRIỆU_CHỨNG", "mệt mỏi nhiều": "TRIỆU_CHỨNG",
    "đau": "TRIỆU_CHỨNG", "đau bụng": "TRIỆU_CHỨNG", "đau bụng râm ran": "TRIỆU_CHỨNG",
    "đau ngực": "TRIỆU_CHỨNG", "tức ngực": "TRIỆU_CHỨNG", "đau đầu": "TRIỆU_CHỨNG", "đau họng": "TRIỆU_CHỨNG", "đau khớp": "TRIỆU_CHỨNG",
    "buồn nôn": "TRIỆU_CHỨNG", "nôn": "TRIỆU_CHỨNG", "nôn mửa": "TRIỆU_CHỨNG", "tiêu chảy": "TRIỆU_CHỨNG", "táo bón": "TRIỆU_CHỨNG",
    "chóng mặt": "TRIỆU_CHỨNG", "ho": "TRIỆU_CHỨNG", "ho khan": "TRIỆU_CHỨNG", "ho đờm": "TRIỆU_CHỨNG", "sổ mũi": "TRIỆU_CHỨNG",
    "co giật": "TRIỆU_CHỨNG", "phù nề": "TRIỆU_CHỨNG", "phù": "TRIỆU_CHỨNG", "nổi mẩn đỏ": "TRIỆU_CHỨNG", "nổi mẩn": "TRIỆU_CHỨNG",
    "chán ăn": "TRIỆU_CHỨNG", "sụt cân": "TRIỆU_CHỨNG", "mất ngủ": "TRIỆU_CHỨNG", "lo âu": "TRIỆU_CHỨNG",
    "tê bì": "TRIỆU_CHỨNG", "tê tay chân": "TRIỆU_CHỨNG", "mất vị giác": "TRIỆU_CHỨNG", "khô miệng": "TRIỆU_CHỨNG",
    "run tay": "TRIỆU_CHỨNG", "run tay chân": "TRIỆU_CHỨNG", "run rẩy toàn thân": "TRIỆU_CHỨNG",
    "ù tai": "TRIỆU_CHỨNG", "rối loạn thị lực": "TRIỆU_CHỨNG", "nhìn song thị": "TRIỆU_CHỨNG",
    "mất thăng bằng": "TRIỆU_CHỨNG", "mất thăng bằng khi đi": "TRIỆU_CHỨNG",
    "tim đập nhanh": "TRIỆU_CHỨNG", "hồi hộp đánh trống ngực": "TRIỆU_CHỨNG",
    "vàng da": "TRIỆU_CHỨNG", "vàng mắt": "TRIỆU_CHỨNG", "vàng niêm mạc": "TRIỆU_CHỨNG",
    "rậm lông": "TRIỆU_CHỨNG", "béo phì": "TRIỆU_CHỨNG", "đau bao tử": "TRIỆU_CHỨNG",

    # DISEASES (CHẨN_ĐOÁN)
    "thiếu máu": "CHẨN_ĐOÁN", "tan huyết": "CHẨN_ĐOÁN", "thiếu máu tan huyết": "CHẨN_ĐOÁN",
    "suy thận cấp": "CHẨN_ĐOÁN", "suy thận mạn": "CHẨN_ĐOÁN", "suy thận": "CHẨN_ĐOÁN", "thận mạn": "CHẨN_ĐOÁN",
    "tăng huyết áp": "CHẨN_ĐOÁN", "cao huyết áp": "CHẨN_ĐOÁN", "tiểu đường": "CHẨN_ĐOÁN", "đái tháo đường": "CHẨN_ĐOÁN",
    "parkinson": "CHẨN_ĐOÁN", "bệnh parkinson": "CHẨN_ĐOÁN", "hội chứng buồng trứng đa nang": "CHẨN_ĐOÁN", "buồng trứng đa nang": "CHẨN_ĐOÁN",
    "đau thắt ngực": "CHẨN_ĐOÁN", "ung thư biểu mô tế bào mật": "CHẨN_ĐOÁN", "thiếu máu cơ tim": "CHẨN_ĐOÁN",
    "thiếu men g6pd": "CHẨN_ĐOÁN", "bệnh kawasaki": "CHẨN_ĐOÁN", "kawasaki": "CHẨN_ĐOÁN",
    "đột quỵ": "CHẨN_ĐOÁN", "tai biến mạch máu não": "CHẨN_ĐOÁN", "xơ gan": "CHẨN_ĐOÁN", "xơ gan do rượu": "CHẨN_ĐOÁN",
    "suy tim": "CHẨN_ĐOÁN", "viêm gan b": "CHẨN_ĐOÁN", "viêm gan c": "CHẨN_ĐOÁN", "gút": "CHẨN_ĐOÁN", "bệnh gút": "CHẨN_ĐOÁN",
    "hen suyễn": "CHẨN_ĐOÁN", "hen phế quản": "CHẨN_ĐOÁN", "viêm phế quản": "CHẨN_ĐOÁN", "viêm phổi": "CHẨN_ĐOÁN",
    "lao phổi": "CHẨN_ĐOÁN", "trào ngược dạ dày": "CHẨN_ĐOÁN", "trào ngược dạ dày thực quản": "CHẨN_ĐOÁN", "gerd": "CHẨN_ĐOÁN",
    "nhiễm khuẩn đường tiết niệu": "CHẨN_ĐOÁN", "bạch cầu dòng tủy mạn tính": "CHẨN_ĐOÁN", "tăng lipid máu": "CHẨN_ĐOÁN",
    "hẹp ống sống": "CHẨN_ĐOÁN", "mày đay": "CHẨN_ĐOÁN", "nổi mề đay": "CHẨN_ĐOÁN", "tiền sản giật": "CHẨN_ĐOÁN",
    "hội chứng thận hư": "CHẨN_ĐOÁN", "gan nhiễm mỡ": "CHẨN_ĐOÁN", "sỏi mật": "CHẨN_ĐOÁN", "sỏi thận": "CHẨN_ĐOÁN",
    "viêm ruột thừa": "CHẨN_ĐOÁN", "bệnh dại": "CHẨN_ĐOÁN", "thủy đậu": "CHẨN_ĐOÁN", "bệnh sởi": "CHẨN_ĐOÁN",
    "sốt xuất huyết": "CHẨN_ĐOÁN", "sốt siêu vi": "CHẨN_ĐOÁN", "cảm cúm": "CHẨN_ĐOÁN", "trầm cảm": "CHẨN_ĐOÁN",
    "thoái hóa tinh bột": "CHẨN_ĐOÁN", "amyloidosis": "CHẨN_ĐOÁN", "nhồi máu cơ tim": "CHẨN_ĐOÁN",

    # TESTS (TÊN_XÉT_NGHIỆM)
    "chụp ct": "TÊN_XÉT_NGHIỆM", "chụp mri": "TÊN_XÉT_NGHIỆM", "chụp x-quang": "TÊN_XÉT_NGHIỆM", "x-quang": "TÊN_XÉT_NGHIỆM",
    "xét nghiệm máu": "TÊN_XÉT_NGHIỆM", "xét nghiệm công thức máu": "TÊN_XÉT_NGHIỆM", "xét nghiệm nước tiểu": "TÊN_XÉT_NGHIỆM",
    "siêu âm": "TÊN_XÉT_NGHIỆM", "siêu âm bụng": "TÊN_XÉT_NGHIỆM", "siêu âm buồng trứng": "TÊN_XÉT_NGHIỆM", "siêu âm tim": "TÊN_XÉT_NGHIỆM",
    "đo điện tâm đồ": "TÊN_XÉT_NGHIỆM", "điện tâm đồ": "TÊN_XÉT_NGHIỆM", "ecg": "TÊN_XÉT_NGHIỆM", "nội soi": "TÊN_XÉT_NGHIỆM",
    "nội soi dạ dày": "TÊN_XÉT_NGHIỆM", "sinh thiết": "TÊN_XÉT_NGHIỆM", "đo huyết áp": "TÊN_XÉT_NGHIỆM",
    "xét nghiệm chức năng gan": "TÊN_XÉT_NGHIỆM", "tinh dịch đồ": "TÊN_XÉT_NGHIỆM", "chụp tử cung vòi trứng": "TÊN_XÉT_NGHIỆM"
}

print(f"Loaded {len(STRICT_ONTOLOGY)} strict ontology type disambiguations!")
