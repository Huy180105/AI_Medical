"""
Generates the crisp, structured, non-rambling 15-slide PowerPoint deck for MedFusion-CXR.
Strictly emphasizes:
  - DATASET (Sources, splits, zero-leakage)
  - INPUT (Dimensions, features, formats)
  - METHOD (Architectures, algorithms, math)
  - OUTPUT (Representations, probabilities, decisions)
  - PIPELINE (Step-by-step end-to-end flow)
Uses clean tables, stat callouts, and diagrams with ZERO rambling text.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PPTX_ROOT_PATH = "MedFusion_CXR_Presentation.pptx"
PPTX_REPORTS_PATH = "reports/MedFusion_CXR_Presentation.pptx"

# Colors: Dark Modern Clinical Tech
BG_COLOR = RGBColor(10, 15, 29)          # #0A0F1D (Dark Navy)
CARD_BG = RGBColor(17, 28, 53)           # #111C35 (Card Navy)
CARD_BORDER = RGBColor(30, 41, 59)       # #1E293B (Border)
CARD_DARK = RGBColor(13, 21, 39)         # #0D1527

COLOR_CYAN = RGBColor(6, 182, 212)       # #06B6D4 (Vision & Tech)
COLOR_BLUE = RGBColor(59, 130, 246)      # #3B82F6 (Primary)
COLOR_EMERALD = RGBColor(16, 185, 129)   # #10B981 (Clinical & Validated)
COLOR_AMBER = RGBColor(245, 158, 11)     # #F59E0B (Prototype)
COLOR_ROSE = RGBColor(244, 63, 94)       # #F43F5E (Alerts)
COLOR_WHITE = RGBColor(255, 255, 255)    # #FFFFFF
COLOR_SLATE = RGBColor(148, 163, 184)    # #94A3B8 (Secondary)
COLOR_MUTED = RGBColor(100, 116, 139)    # #64748B

FONT_MAIN = "Segoe UI"
FONT_CODE = "Consolas"


def set_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR


def add_header(slide, tag, title, subtitle):
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.15))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = tag.upper()
    p0.font.name = FONT_MAIN
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_CYAN
    p0.space_after = Pt(2)

    p1 = tf.add_paragraph()
    p1.text = title
    p1.font.name = FONT_MAIN
    p1.font.size = Pt(21)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_after = Pt(2)

    p2 = tf.add_paragraph()
    p2.text = subtitle
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_SLATE

    # Line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.58), Inches(11.733), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = CARD_BORDER
    line.line.color.rgb = CARD_BORDER


def add_box(slide, left, top, width, height, border_col=CARD_BORDER, bg_col=CARD_BG):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_col
    shape.line.color.rgb = border_col
    shape.line.width = Pt(1.2)
    return shape


def format_table(table, col_widths, headers, data, header_bg=RGBColor(14, 116, 144)):
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = width

    # Format header row
    for col_idx, htext in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_bg
        p = cell.text_frame.paragraphs[0]
        p.text = htext
        p.font.name = FONT_MAIN
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Format data rows
    for row_idx, row_data in enumerate(data):
        bg = CARD_BG if row_idx % 2 == 0 else CARD_DARK
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.name = FONT_MAIN
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_WHITE if col_idx == 0 or col_idx == 1 else COLOR_SLATE
            p.alignment = PP_ALIGN.LEFT if col_idx <= 1 else PP_ALIGN.CENTER
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE


def generate_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1)
    add_box(s1, Inches(1.0), Inches(1.0), Inches(11.333), Inches(5.5), border_col=COLOR_CYAN, bg_col=CARD_BG)

    tb = s1.shapes.add_textbox(Inches(1.5), Inches(1.35), Inches(10.333), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "BÁO CÁO KHOA HỌC & ĐỒ ÁN TỐT NGHIỆP CDSS"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(8)

    p = tf.add_paragraph()
    p.text = "MedFusion-CXR"
    p.font.name = FONT_MAIN
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_after = Pt(6)

    p = tf.add_paragraph()
    p.text = "Hệ thống Đa phương thức & Đa tác tử Hỗ trợ Đánh giá Lâm sàng"
    p.font.name = FONT_MAIN
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.space_after = Pt(12)

    # Structured 1-line architecture summary
    p = tf.add_paragraph()
    p.text = "SƠ ĐỒ TỔNG THỂ CỐT LÕI (END-TO-END PIPELINE):"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = "Input (CXR + Sinh hiệu)  ──>  DenseNet-121 (F_v) + XGBoost (F_c)  ──>  Gated Fusion (GMU ℝ²⁵⁶)  ──>  Output (CDSS Bạch Mai)"
    p.font.name = FONT_CODE
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_after = Pt(18)

    # 4 Pillars
    p = tf.add_paragraph()
    p.text = "• VISION BRANCH: Pretrained DenseNet-121 (NIH-14)  |  • CLINICAL BRANCH: XGBoost + TreeSHAP (11 features)\n" \
             "• FUSION LAYER: Gated Feature Fusion (GMU ℝ²⁵⁶)   |  • ORCHESTRATION: 5-Stage Multi-Agent Swarm"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 2: MOTIVATION - SOLUTION - KEY LEARNINGS (3-Pillar Crisp Layout)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2)
    add_header(s2, "TỔNG QUAN ĐỀ TÀI & GIÁ TRỊ THU HOẠCH",
               "Lý Do Chọn Đề Tài · Giải Pháp Nêu Ra · Giá Trị Học Được",
               "Khái quát ba trụ cột trọng tâm: Nhu cầu thực tế -> Kiến trúc đề xuất -> Bài học khoa học và kỹ thuật")

    col_w = Inches(3.64)
    col_h = Inches(5.15)
    top_pos = Inches(1.82)
    gap = Inches(0.40)

    pillars = [
        # Pillar 1: Lý do chọn đề tài
        {
            "title": "1. LÝ DO CHỌN ĐỀ TÀI",
            "tag": "BỐI CẢNH & NỖI ĐAU THỰC TẾ",
            "color": COLOR_ROSE,
            "sections": [
                ("Áp lực 'Thời gian vàng' Cấp cứu", [
                    "Bệnh nhân suy hô hấp diễn biến nhanh trong vài phút, cần quyết định khẩn cấp.",
                    "Dữ liệu phân mảnh giữa khoa Chẩn đoán hình ảnh và Hồi sức cấp cứu."
                ]),
                ("Bế tắc của AI Đơn phương thức", [
                    "Vision-only (Ảnh CXR): Thấy tổn thương nhưng hoàn toàn mù sinh hiệu (không biết sốt cao, tụt SpO₂, sốc).",
                    "Clinical-only (Sinh hiệu): Thấy rối loạn cấp tính nhưng thiếu bằng chứng hình thái (khó phân biệt Viêm phổi vs Suy tim)."
                ]),
                ("Khoảng trống công nghệ", [
                    "Cần một hệ thống dung hòa đồng thời cả 2 nguồn tin để hỗ trợ bác sĩ ra quyết định toàn diện."
                ])
            ]
        },
        # Pillar 2: Giải pháp nêu ra
        {
            "title": "2. GIẢI PHÁP NÊU RA",
            "tag": "HỆ THỐNG MEDFUSION-CXR",
            "color": COLOR_EMERALD,
            "sections": [
                ("Kiến trúc Đa phương thức 2 nhánh", [
                    "Nhánh Thị giác: DenseNet-121 quét 14 bệnh lý phổi NIH-14, xuất vector F_v ∈ ℝ¹⁰²⁴.",
                    "Nhánh Lâm sàng: XGBoost + TreeSHAP lượng giá nguy cơ can thiệp, xuất vector F_c ∈ ℝ³²."
                ]),
                ("Cơ chế Cổng thích ứng (GMU)", [
                    "Cổng g tự cân bằng trọng số đóng góp: 48% ảnh X-quang · 52% sinh hiệu cấp cứu.",
                    "Chế độ Fallback an toàn: Tự động chuyển 100% sang sinh hiệu khi chưa có phim X-quang."
                ]),
                ("Điều phối 5 Agent & Phác đồ Bạch Mai", [
                    "Gác cổng mốc t₀ (chống rò rỉ), PhoBERT bóc tách bệnh sử tiếng Việt, RAG phác đồ Bạch Mai + Mã ICD-10."
                ])
            ]
        },
        # Pillar 3: Qua đó học được những gì
        {
            "title": "3. QUA ĐÓ HỌC ĐƯỢC GÌ",
            "tag": "BÀI HỌC THU HOẠCH ĐỒ ÁN",
            "color": COLOR_AMBER,
            "sections": [
                ("Kỹ thuật AI Y tế Đa phương thức", [
                    "Nắm vững xử lý dữ liệu phức hợp (Ảnh xám y tế 224×224 kết hợp bảng sinh hiệu cấp cứu).",
                    "Hiểu sâu cơ chế Gated Fusion và kỹ thuật giải thích mô hình cá thể hóa bằng TreeSHAP."
                ]),
                ("Liêm chính & Phương pháp luận", [
                    "Thấu hiểu bài học kiểm soát rò rỉ dữ liệu (Zero patient leakage, temporal guardrail t₀).",
                    "Nhận thức rõ ranh giới giữa kiến trúc đề xuất và thực nghiệm trên tập dữ liệu chưa ghép cặp."
                ]),
                ("Tư duy Xây dựng Sản phẩm", [
                    "Thiết kế hệ điều phối Multi-Agent Swarm, đóng gói trọn vẹn Docker, giao diện CDSS thực tế."
                ])
            ]
        }
    ]

    for idx, p_data in enumerate(pillars):
        x = Inches(0.8) + idx * (col_w + gap)
        box = add_box(s2, x, top_pos, col_w, col_h, border_col=p_data["color"], bg_col=CARD_BG)
        tb = s2.shapes.add_textbox(x + Inches(0.15), top_pos + Inches(0.15), col_w - Inches(0.3), col_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = p_data["title"]
        p.font.name = FONT_MAIN
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = p_data["color"]
        p.space_after = Pt(2)

        p = tf.add_paragraph()
        p.text = p_data["tag"]
        p.font.name = FONT_MAIN
        p.font.size = Pt(8.2)
        p.font.bold = True
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(8)

        for sec_title, bullets in p_data["sections"]:
            p = tf.add_paragraph()
            p.text = sec_title
            p.font.name = FONT_MAIN
            p.font.size = Pt(9.5)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(2)

            for b in bullets:
                p = tf.add_paragraph()
                p.text = f"• {b}"
                p.font.name = FONT_MAIN
                p.font.size = Pt(8.5)
                p.font.color.rgb = COLOR_SLATE
                p.space_after = Pt(2)
            p.space_after = Pt(5)

    # =========================================================================
    # SLIDE 3: SYSTEM SPECIFICATIONS TABLE (Input, Method, Output)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3)
    add_header(s3, "ĐẶC TẢ KỸ THUẬT HỆ THỐNG", "Bảng Ma Trận: Input - Method - Output Toàn Hệ Thống",
               "Quy chuẩn hóa chính xác từng trường dữ liệu, phương pháp áp dụng và kích thước vector đầu ra")

    # Table layout
    t_shape = s3.shapes.add_table(5, 4, Inches(0.8), Inches(1.9), Inches(11.733), Inches(4.9))
    table = t_shape.table

    col_widths = [Inches(2.2), Inches(3.2), Inches(3.2), Inches(3.133)]
    headers = ["Thành phần", "Input cụ thể", "Phương pháp cốt lõi", "Output cụ thể"]
    data = [
        ["Vision Branch\n(Nhánh Hình ảnh)",
         "• Phim X-quang ngực (CXR)\n• Format: Grayscale 224×224\n• Chuẩn hóa: [-1024, +1024]",
         "• DenseNet-121 Backbone\n• Checkpoint: nih14 (Apache 2.0)\n• Thresholding: Validation F1-grid",
         "• Vector đặc trưng: F_v ∈ ℝ¹⁰²⁴\n• 14 xác suất bệnh lý NIH-14\n• Nhãn phát hiện nổi bật nhất"],
        ["Clinical Branch\n(Nhánh Lâm sàng)",
         "• 11 biến sinh hiệu/thể trạng\n• 8 chỉ số cấp (qSOFA, Shock Index, RR, Temp, WBC, CRP, Trop)",
         "• XGBoost (160 trees, depth 4)\n• StandardScaler fit trên Train\n• TreeSHAP (pred_contribs=True)",
         "• Vector biểu diễn: F_c ∈ ℝ³²\n• Xác suất điều trị P(treatment)\n• Vector đóng góp TreeSHAP ϕ_i"],
        ["Gated Fusion\n(Tầng Hợp nhất)",
         "• Vector F_v ∈ ℝ¹⁰²⁴ (Vision)\n• Vector F_c ∈ ℝ³² (Clinical)",
         "• Linear + LayerNorm + GELU\n• Cổng Sigmoid g = σ(W[F̃_v; F̃_c]+b)\n• F_fusion = g⊙F̃_v + (1-g)⊙F̃_c",
         "• Vector hợp nhất F_fusion ∈ ℝ²⁵⁶\n• Trọng số đóng góp (g, 1-g)\n• Clinical Fallback khi g=0"],
        ["Multi-Agent Layer\n(Lớp Điều phối)",
         "• Dữ liệu tiếp nhận ban đầu\n• Bệnh sử & Triệu chứng tự do\n• Mốc thời gian tiếp nhận t₀",
         "• 5 Agents chuyên khoa\n• PhoBERT NER Tokenizer\n• RAG Phác đồ BV Bạch Mai",
         "• Báo cáo hội chẩn đa phương thức\n• Khuyến nghị can thiệp lâm sàng\n• Phác đồ Bạch Mai + mã ICD-10"]
    ]
    format_table(table, col_widths, headers, data)

    # =========================================================================
    # SLIDE 4: DATASET & DATA ACCOUNTING TABLE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4)
    add_header(s4, "NGUỒN DỮ LIỆU & QUẢN TRỊ KHOA HỌC", "Bảng Thống Kê & Kế Toán Hai Tập Dữ Liệu Độc Lập",
               "Cam kết liêm chính khoa học: Tuyệt đối không tạo cặp giả giữa ảnh bệnh nhân A và sinh hiệu bệnh nhân B")

    t_shape = s4.shapes.add_table(7, 3, Inches(0.8), Inches(1.9), Inches(11.733), Inches(4.9))
    table = t_shape.table

    col_widths = [Inches(2.5), Inches(4.6), Inches(4.633)]
    headers = ["Tiêu chí quản trị", "Dataset 1: NIH ChestX-ray14", "Dataset 2: Clinical Tabular Dataset"]
    data = [
        ["Nguồn gốc & Đơn vị", "NIH Clinical Center (Wang et al., CVPR 2017)", "Mô phỏng sinh lý (1.500) + Bệnh án trích xuất (100)"],
        ["Tổng quy mô mẫu", "112.120 ảnh X-quang / 30.805 bệnh nhân duy nhất", "1.600 mẫu bệnh nhân dạng bảng"],
        ["Định dạng Input", "Ảnh Grayscale PNG (Kích thước gốc 1024×1024)", "11 trường số và phân loại (CSV format)"],
        ["Nhãn mục tiêu (Target)", "14 nhãn bệnh lý lồng ngực nhị phân {0, 1}", "Nhãn nhị phân treatment_needed (Tỷ lệ dương 42.25%)"],
        ["Phân chia tập (Splits)", "Train: 73.089  |  Val: 13.435  |  Test: 25.596 ảnh", "Train: 1.120 (70%)  |  Val: 240 (15%)  |  Test: 240 (15%)"],
        ["Kiểm soát rò rỉ bệnh nhân", "Patient-Level Split (Overlap bệnh nhân = 0, zero-leakage)", "Stratified Split theo nhãn mục tiêu, scaler fit trên Train"]
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(30, 41, 59))

    # =========================================================================
    # SLIDE 5: PREPROCESSING PIPELINES (Vision vs Clinical)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5)
    add_header(s5, "KỸ THUẬT TIỀN XỬ LÝ", "Pipeline Tiền Xử Lý & Chống Rò Rỉ Đa Tầng",
               "Quy trình xử lý riêng biệt cho 2 modality và 3 tầng kiểm soát rò rỉ dữ liệu")

    cw = Inches(3.64)
    ch = Inches(5.1)
    top_pos = Inches(1.85)

    p_data = [
        ("PIPELINE 1: ẢNH X-QUANG", COLOR_CYAN,
         "1. ĐẦU VÀO:\n   Ảnh CXR định dạng PNG/JPEG/DICOM.\n\n"
         "2. CHUẨN HÓA CƯỜNG ĐỘ:\n   Chuyển 1 kênh xám (Grayscale).\n   Rescale voxel về range [-1024.0, +1024.0].\n\n"
         "3. CHUẨN HÓA KÍCH THƯỚC:\n   Center-crop khung vuông 224 × 224 pixels.\n\n"
         "4. TENSOR ĐẦU RA:\n   Tensor (1, 1, 224, 224) sẵn sàng đưa vào DenseNet-121."),
        ("PIPELINE 2: LÂM SÀNG", COLOR_EMERALD,
         "1. ĐẦU VÀO:\n   11 biến thể trạng + sinh lý cấp cứu.\n\n"
         "2. XỬ LÝ NGOẠI LAI (OUTLIERS):\n   Clip SpO₂ về [50.0, 100.0]% (sửa lỗi 102.7%).\n   Xử lý missing value bằng trung vị Train.\n\n"
         "3. CHUẨN HÓA THANG ĐO:\n   StandardScaler FIT STRICTLY trên Train (1.120 mẫu).\n   Chỉ transform trên Val và Test.\n\n"
         "4. TENSOR ĐẦU RA:\n   Tensor X_scaled (1, 11) sẵn sàng cho XGBoost."),
        ("3 TẦNG CHỐNG RÒ RỈ", COLOR_ROSE,
         "🔒 TẦNG 1: PATIENT-LEVEL SPLIT\n"
         "   Giao tập bệnh nhân: train ∩ val ∩ test = ∅.\n"
         "   Không có ảnh cùng bệnh nhân ở 2 split khác nhau.\n\n"
         "🔒 TẦNG 2: FROZEN VALIDATION THRESHOLD\n"
         "   Ngưỡng quyết định F1 chỉ tìm kiếm trên tập Val.\n"
         "   Đóng băng ngưỡng khi áp dụng đánh giá Test.\n\n"
         "🔒 TẦNG 3: TEMPORAL GUARD (t₀)\n"
         "   Agent kiểm duyệt loại bỏ mọi dữ liệu phát sinh sau mốc khám ban đầu t₀.")
    ]

    for idx, (title, color, body) in enumerate(p_data):
        x = Inches(0.8) + idx * (cw + Inches(0.4))
        add_box(s5, x, top_pos, cw, ch, border_col=color)
        tb = s5.shapes.add_textbox(x + Inches(0.2), top_pos + Inches(0.2), cw - Inches(0.4), ch - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_MAIN
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(10)

        for line in body.split("\n"):
            p = tf.add_paragraph()
            p.text = line
            p.font.name = FONT_CODE if "TENSOR" in line or "SpO₂" in line else FONT_MAIN
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_WHITE if line.startswith("🔒") or line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") else COLOR_SLATE

    # =========================================================================
    # SLIDE 6: DETAILED ARCHITECTURE FLOWCHART (NATIVE CRISP CARDS & ARROWS)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6)
    add_header(s6, "SƠ ĐỒ KIẾN TRÚC & PIPELINE HỆ THỐNG", "Quy Trình Xử Lý 4 Giai Đoạn Từ Dữ Liệu Đầu Vào Đến Hỗ Trợ Lâm Sàng",
               "Phân định rõ ràng: Thu thập dữ liệu -> Mô hình phân tích -> Cân bằng thích ứng -> Quyết định lâm sàng")

    col_w = Inches(2.68)
    col_gap = Inches(0.33)
    y_hdr = Inches(1.80)
    h_hdr = Inches(0.40)
    y_card1 = Inches(2.28)
    h_card1 = Inches(2.32)
    y_card2 = Inches(4.70)
    h_card2 = Inches(2.25)
    h_full = Inches(4.67)

    stage_cols = [
        # Stage 1: Dữ liệu đầu vào
        {
            "tag": "BƯỚC 1: DỮ LIỆU ĐẦU VÀO",
            "tag_color": COLOR_CYAN,
            "cards": [
                {
                    "title": "🩻 Phim X-Quang Ngực (CXR)",
                    "sub": "Tập NIH ChestX-ray14 · 112.120 ảnh",
                    "color": COLOR_CYAN,
                    "bullets": [
                        ("Định dạng", "Ảnh xám chuẩn hóa 224×224 pixels"),
                        ("Nội dung", "14 nhãn tổn thương lồng ngực"),
                        ("Ý nghĩa", "Cung cấp bằng chứng hình thái phổi thực thể")
                    ]
                },
                {
                    "title": "🩺 Sinh Hiệu & Lâm Sàng",
                    "sub": "1.600 hồ sơ bệnh nhân dạng bảng",
                    "color": COLOR_EMERALD,
                    "bullets": [
                        ("Biến số", "11 chỉ số sinh hiệu (SpO₂, Mạch, HA...)"),
                        ("Cấp cứu", "Sốt cao, Nhịp thở, Bạch cầu, CRP, Trop"),
                        ("Ý nghĩa", "Cung cấp diễn tiến sinh lý cấp toàn thân")
                    ]
                }
            ]
        },
        # Stage 2: Mô hình phân tích chuyên sâu
        {
            "tag": "BƯỚC 2: PHÂN TÍCH CHUYÊN SÂU",
            "tag_color": COLOR_EMERALD,
            "cards": [
                {
                    "title": "🖼️ DenseNet-121 (Thị Giác)",
                    "sub": "Mạng tích chập 121 tầng (Pretrained)",
                    "color": COLOR_CYAN,
                    "bullets": [
                        ("Nhiệm vụ", "Quét tổn thương phổi trên phim CXR"),
                        ("Phát hiện", "Viêm phổi, Tràn dịch, Xẹp phổi, Tim to..."),
                        ("Đầu ra", "Vector đặc trưng F_v ∈ ℝ¹⁰²⁴ + 14 xác suất")
                    ]
                },
                {
                    "title": "📊 XGBoost + TreeSHAP",
                    "sub": "Rừng 160 cây quyết định phân tầng",
                    "color": COLOR_EMERALD,
                    "bullets": [
                        ("Nhiệm vụ", "Lượng giá nguy cơ cần can thiệp y tế"),
                        ("Động học", "Tính điểm qSOFA, Shock Index, Hypoxemia"),
                        ("Đầu ra", "Vector đặc trưng F_c ∈ ℝ³² + Trọng số SHAP")
                    ]
                }
            ]
        },
        # Stage 3: Cân bằng thông minh
        {
            "tag": "BƯỚC 3: CÂN BẰNG THÔNG MINH",
            "tag_color": COLOR_AMBER,
            "cards": [
                {
                    "title": "⚖️ Gated Fusion (GMU)",
                    "sub": "Cổng thích ứng trên không gian ℝ²⁵⁶",
                    "color": COLOR_AMBER,
                    "is_tall": True,
                    "bullets": [
                        ("Cơ chế cổng g", "Tự động cân đối mức độ ưu tiên 2 nguồn tin"),
                        ("Tỷ lệ thực tế", "48% Phim X-quang  ·  52% Sinh hiệu"),
                        ("Dung hòa tin", "Bằng chứng thâm nhiễm phổi củng cố"),
                        (" ", "song song cho sốt cao và SpO₂ giảm"),
                        ("Chế độ Fallback", "Nếu thiếu ảnh X-quang, tự động chuyển 100%"),
                        (" ", "sang sinh hiệu, không gây gián đoạn"),
                        ("Ưu điểm", "Tránh lấn át tín hiệu giữa các phương thức")
                    ]
                }
            ]
        },
        # Stage 4: Hỗ trợ ra quyết định
        {
            "tag": "BƯỚC 4: HỖ TRỢ RA QUYẾT ĐỊNH",
            "tag_color": COLOR_BLUE,
            "cards": [
                {
                    "title": "🏥 Báo Cáo Hội Chẩn CDSS",
                    "sub": "5 Agent Chuyên khoa & Phác đồ Bạch Mai",
                    "color": COLOR_BLUE,
                    "is_tall": True,
                    "bullets": [
                        ("Chống rò rỉ", "TemporalGuard loại trừ sự kiện sau t₀"),
                        ("Phân tầng nguy cơ", "Cảnh báo khẩn cấp (Critical) / Ổn định"),
                        ("Phác đồ Bạch Mai", "Gợi ý thuốc chuẩn y khoa + Mã ICD-10"),
                        ("Giải thích bác sĩ", "TreeSHAP chỉ rõ nguyên nhân thúc đẩy rủi ro"),
                        ("Giao diện Web", "Báo cáo trực quan tại http://localhost:8000"),
                        ("Đóng gói", "Container Docker độc lập, sẵn sàng chuyển giao")
                    ]
                }
            ]
        }
    ]

    for idx, stage in enumerate(stage_cols):
        x = Inches(0.8) + idx * (col_w + col_gap)

        # Stage Header Box
        hdr_shape = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_hdr, col_w, h_hdr)
        hdr_shape.fill.solid()
        hdr_shape.fill.fore_color.rgb = CARD_BG
        hdr_shape.line.color.rgb = stage["tag_color"]
        hdr_shape.line.width = Pt(1.5)
        p = hdr_shape.text_frame.paragraphs[0]
        p.text = stage["tag"]
        p.font.name = FONT_MAIN
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = stage["tag_color"]
        p.alignment = PP_ALIGN.CENTER

        # Stage Cards
        cards = stage["cards"]
        if len(cards) == 2:
            card_positions = [(y_card1, h_card1), (y_card2, h_card2)]
        else:
            card_positions = [(y_card1, h_full)]

        for c_idx, c_data in enumerate(cards):
            cy, ch = card_positions[c_idx]
            box = add_box(s6, x, cy, col_w, ch, border_col=c_data["color"], bg_col=CARD_BG)
            tb = s6.shapes.add_textbox(x + Inches(0.12), cy + Inches(0.1), col_w - Inches(0.24), ch - Inches(0.2))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

            p = tf.paragraphs[0]
            p.text = c_data["title"]
            p.font.name = FONT_MAIN
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(2)

            p = tf.add_paragraph()
            p.text = c_data["sub"]
            p.font.name = FONT_MAIN
            p.font.size = Pt(8.5)
            p.font.bold = True
            p.font.color.rgb = c_data["color"]
            p.space_after = Pt(6)

            for b_name, b_desc in c_data["bullets"]:
                p = tf.add_paragraph()
                if b_name.strip():
                    p.text = f"• {b_name}: {b_desc}"
                    p.font.name = FONT_MAIN
                    p.font.size = Pt(8.8)
                    p.font.color.rgb = COLOR_SLATE
                else:
                    p.text = f"   {b_desc}"
                    p.font.name = FONT_MAIN
                    p.font.size = Pt(8.8)
                    p.font.color.rgb = COLOR_SLATE
                p.space_after = Pt(2)

        # Connecting Chevron / Arrow to next stage
        if idx < 3:
            arrow_x = x + col_w + Inches(0.06)
            arrow_y = Inches(4.3)
            tb_arr = s6.shapes.add_textbox(arrow_x, arrow_y, col_gap - Inches(0.12), Inches(0.5))
            tf_arr = tb_arr.text_frame
            p_arr = tf_arr.paragraphs[0]
            p_arr.text = "➔"
            p_arr.font.name = FONT_MAIN
            p_arr.font.size = Pt(18)
            p_arr.font.bold = True
            p_arr.font.color.rgb = COLOR_AMBER if idx == 1 or idx == 2 else COLOR_CYAN
            p_arr.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 7: VISION BRANCH SPECIFICATION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7)
    add_header(s7, "CHI TIẾT NHÁNH THỊ GIÁC (VISION BRANCH)", "DenseNet-121: Kiến Trúc, Trọng Số & Trích Xuất Vector F_v",
               "Đặc tả toán học và kỹ thuật chuyển đổi từ ảnh X-quang sang vector đặc trưng 1024 chiều")

    # Left: CXR Sample Picture
    add_box(s7, Inches(0.8), Inches(1.85), Inches(4.2), Inches(5.1), border_col=COLOR_CYAN)
    cxr_sample = "web/samples/pneumonia_nih.png"
    if os.path.exists(cxr_sample):
        s7.shapes.add_picture(cxr_sample, Inches(1.05), Inches(2.05), width=Inches(3.7), height=Inches(3.7))
    tb = s7.shapes.add_textbox(Inches(1.05), Inches(5.9), Inches(3.7), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Phim CXR Mẫu (NIH ChestX-ray14)\nGrayscale · Center-Crop 224x224"
    p.font.name = FONT_MAIN
    p.font.size = Pt(9.5)
    p.font.color.rgb = COLOR_SLATE
    p.alignment = PP_ALIGN.CENTER

    # Right: Specifications Table
    t_shape = s7.shapes.add_table(5, 2, Inches(5.3), Inches(1.85), Inches(7.233), Inches(5.1))
    table = t_shape.table
    col_widths = [Inches(2.2), Inches(5.033)]
    headers = ["Thuộc tính Kỹ thuật", "Thông số Chi tiết Nhánh Thị giác"]
    data = [
        ["Backbone Mạng", "DenseNet-121 (Dense Convolutional Network, 121 layers)"],
        ["Checkpoint & Nguồn", "TorchXRayVision 'densenet121-res224-nih' (SHA-256: 31a5475c…)"],
        ["Đầu vào (Input)", "Ảnh Grayscale (1, 1, 224, 224), chuẩn hóa [-1024, +1024]"],
        ["Vector Đặc trưng (Output)", "F_v ∈ ℝ¹⁰²⁴ (Trích xuất từ lớp Global Average Pooling trước FC)"],
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(6, 182, 212))

    # =========================================================================
    # SLIDE 8: CLINICAL BRANCH SPECIFICATION
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8)
    add_header(s8, "CHI TIẾT NHÁNH LÂM SÀNG (CLINICAL BRANCH)", "XGBoost & TreeSHAP: Xây Dựng Vector Biểu Diễn F_c ∈ ℝ³²",
               "Cầu nối biểu diễn: Đưa kết quả từ cây quyết định và sinh lý cấp tính sang không gian vector tương thích")

    # Left: Spec Table
    t_shape = s8.shapes.add_table(6, 2, Inches(0.8), Inches(1.85), Inches(6.0), Inches(5.1))
    table = t_shape.table
    col_widths = [Inches(2.0), Inches(4.0)]
    headers = ["Thuộc tính", "Thông số Kỹ thuật Nhánh Lâm sàng"]
    data = [
        ["Thuật toán Mô hình", "XGBoost Classifier (160 estimators, depth 4, lr 0.06, subsample 0.88)"],
        ["Đầu vào 11 biến", "SpO₂, SBP, DBP, HR, Tuổi, Giới tính, Stress, Ngủ, Vận động, Bệnh nền, Triệu chứng"],
        ["8 chỉ số sinh lý cấp", "Nhịp thở (RR), Nhiệt độ, WBC, CRP, Troponin, qSOFA, Shock Index, NLP count"],
        ["Giải thích TreeSHAP", "pred_contribs=True tính chính xác giá trị Shapley ϕ_i trên cây quyết định"],
        ["Cấu trúc Vector F_c", "[0..23]: 11 biến scaled + 11 SHAP + bias + logit\n[24..31]: 8 chỉ số cấp cứu đã chuẩn hóa\n==> TỔNG: Vector F_c ∈ ℝ³²"]
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(16, 185, 129))

    # Right: Chart E3
    shap_img = "reports/presentation_assets/e3_treeshap_importance.png"
    if os.path.exists(shap_img):
        s8.shapes.add_picture(shap_img, Inches(7.0), Inches(1.85), width=Inches(5.533), height=Inches(5.1))
    else:
        add_box(s8, Inches(7.0), Inches(1.85), Inches(5.533), Inches(5.1), border_col=COLOR_EMERALD)

    # =========================================================================
    # SLIDE 9: MULTI-AGENT SWARM PIPELINE (Sequential Agent Orchestration Flow)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9)
    add_header(s9, "HỆ THỐNG ĐIỀU PHỐI MULTI-AGENT SWARM", "Trình Tự & Luồng Điều Phối 5 Tác Tử Y Khoa Khi Tiếp Nhận Input",
               "Cơ chế tiếp nhận tại mốc t₀ -> Phân công song song chuyên biệt -> Hội đồng Gated Fusion -> Phác đồ Bạch Mai")

    # Top Input Ingestion Banner
    box_banner = add_box(s9, Inches(0.8), Inches(1.75), Inches(11.733), Inches(0.48), border_col=COLOR_CYAN, bg_col=CARD_BG)
    tb_b = s9.shapes.add_textbox(Inches(0.95), Inches(1.78), Inches(11.433), Inches(0.42))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "📥 INPUT TIẾP NHẬN TẠI MỐC t₀:  1. Phim X-quang Ngực (CXR)   |   2. 11 Sinh Hiệu & Xét Nghiệm Cấp Cứu   |   3. Bệnh Sử Tiếng Việt Tự Do"
    p.font.name = FONT_MAIN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE

    col_w = Inches(2.68)
    col_gap = Inches(0.33)
    y_hdr = Inches(2.32)
    h_hdr = Inches(0.36)
    y_card1 = Inches(2.76)
    h_card1 = Inches(2.05)
    y_card2 = Inches(4.90)
    h_card2 = Inches(2.05)
    h_full = Inches(4.19)

    agent_stages = [
        # Stage 1: Kiểm soát t0 & Bóc tách thực thể
        {
            "tag": "BƯỚC 1: TIỀN XỬ LÝ & BẢO VỆ",
            "tag_color": COLOR_ROSE,
            "cards": [
                {
                    "title": "🛡️ TemporalLeakageGuard",
                    "sub": "Tác tử Gác Cổng Mốc Thời Gian t₀",
                    "color": COLOR_ROSE,
                    "bullets": [
                        ("Nhiệm vụ", "Chặn mọi dữ liệu phát sinh sau mốc tiếp nhận t₀"),
                        ("Chống rò rỉ", "Bảo đảm zero temporal leakage"),
                        ("Chuyển giao", "Hồ sơ sạch hợp lệ cho các Agent chuyên khoa")
                    ]
                },
                {
                    "title": "📝 TriageTimelineAgent",
                    "sub": "Tác tử Phân Loại & PhoBERT NLP",
                    "color": COLOR_CYAN,
                    "bullets": [
                        ("Mô hình", "PhoBERT NER Tokenizer + Lexicon y tế"),
                        ("Bóc tách", "Triệu chứng, tiền sử bệnh nền, thuốc đã dùng"),
                        ("Thời gian", "Trích xuất mốc khởi phát bệnh (duration hours)")
                    ]
                }
            ]
        },
        # Stage 2: Phân công song song 2 nhánh
        {
            "tag": "BƯỚC 2: PHÂN CÔNG CHUYÊN BIỆT",
            "tag_color": COLOR_EMERALD,
            "cards": [
                {
                    "title": "🩻 CXRDiagnosticAgent",
                    "sub": "Bác sĩ Chẩn Đoán Hình Ảnh Ảo",
                    "color": COLOR_CYAN,
                    "bullets": [
                        ("Mô hình", "DenseNet-121 (TorchXRayVision nih14)"),
                        ("Tác vụ", "Quét 14 bệnh lý phổi (Viêm phổi, Tràn dịch...)"),
                        ("Đầu ra", "Vector thị giác F_v ∈ ℝ¹⁰²⁴ + 14 xác suất tổn thương")
                    ]
                },
                {
                    "title": "🩺 VitalLabDynamicsAgent",
                    "sub": "Bác sĩ Hồi Sức Cấp Cứu Ảo",
                    "color": COLOR_EMERALD,
                    "bullets": [
                        ("Động học", "Tính điểm qSOFA, Shock Index, Hypoxemia"),
                        ("Mô hình", "XGBoost Classifier + TreeSHAP cá thể"),
                        ("Đầu ra", "Vector lâm sàng F_c ∈ ℝ³² + Điểm nguy cơ can thiệp")
                    ]
                }
            ]
        },
        # Stage 3: Hội đồng & Gated Fusion
        {
            "tag": "BƯỚC 3: HỘI CHẨN & GATED FUSION",
            "tag_color": COLOR_AMBER,
            "cards": [
                {
                    "title": "👑 ConsensusOrchestrator",
                    "sub": "Trưởng Ban Hội Đồng Hội Chẩn Đa Phương Thức",
                    "color": COLOR_AMBER,
                    "bullets": [
                        ("Thu thập", "Nhận đồng thời F_v (thị giác) và F_c (lâm sàng)"),
                        ("Gated Fusion", "Cổng GMU tự cân bằng 48% ảnh · 52% sinh hiệu"),
                        ("Dung hòa tin", "Tổn thương phổi củng cố song song cho sốt và SpO₂"),
                        ("Chế độ Fallback", "Nếu thiếu ảnh CXR, tự động gán g = 0 và dựa 100%"),
                        (" ", "vào sinh hiệu, không làm gián đoạn cấp cứu"),
                        ("Điều phối", "Chuyển giao vector F_fusion sang tầng RAG")
                    ]
                }
            ]
        },
        # Stage 4: Đối chiếu phác đồ & CDSS
        {
            "tag": "BƯỚC 4: ĐỐI CHIẾU & QUYẾT ĐỊNH",
            "tag_color": COLOR_BLUE,
            "cards": [
                {
                    "title": "🏥 Hỗ Trợ Quyết Định (CDSS)",
                    "sub": "RAG Phác Đồ BV Bạch Mai & Giao Diện Web",
                    "color": COLOR_BLUE,
                    "bullets": [
                        ("RAG Bạch Mai", "Truy xuất hướng dẫn chẩn đoán điều trị chuẩn"),
                        ("Mã ICD-10", "Gán mã phân loại bệnh học quốc tế"),
                        ("Phân tầng", "Cảnh báo khẩn cấp (Critical) / Can thiệp / Ổn định"),
                        ("Giải thích", "8 thanh TreeSHAP chỉ rõ nguyên nhân thúc đẩy"),
                        ("Hiển thị Web", "Báo cáo trực quan cho bác sĩ tại http://localhost:8000"),
                        ("Triển khai", "Container Docker độc lập, sẵn sàng ứng dụng")
                    ]
                }
            ]
        }
    ]

    for idx, stage in enumerate(agent_stages):
        x = Inches(0.8) + idx * (col_w + col_gap)

        # Stage Header Box
        hdr_shape = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_hdr, col_w, h_hdr)
        hdr_shape.fill.solid()
        hdr_shape.fill.fore_color.rgb = CARD_BG
        hdr_shape.line.color.rgb = stage["tag_color"]
        hdr_shape.line.width = Pt(1.5)
        p = hdr_shape.text_frame.paragraphs[0]
        p.text = stage["tag"]
        p.font.name = FONT_MAIN
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = stage["tag_color"]
        p.alignment = PP_ALIGN.CENTER

        cards = stage["cards"]
        if len(cards) == 2:
            card_positions = [(y_card1, h_card1), (y_card2, h_card2)]
        else:
            card_positions = [(y_card1, h_full)]

        for c_idx, c_data in enumerate(cards):
            cy, ch = card_positions[c_idx]
            box = add_box(s9, x, cy, col_w, ch, border_col=c_data["color"], bg_col=CARD_BG)
            tb = s9.shapes.add_textbox(x + Inches(0.12), cy + Inches(0.08), col_w - Inches(0.24), ch - Inches(0.16))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

            p = tf.paragraphs[0]
            p.text = c_data["title"]
            p.font.name = FONT_MAIN
            p.font.size = Pt(10.5)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(2)

            p = tf.add_paragraph()
            p.text = c_data["sub"]
            p.font.name = FONT_MAIN
            p.font.size = Pt(8.2)
            p.font.bold = True
            p.font.color.rgb = c_data["color"]
            p.space_after = Pt(4)

            for b_name, b_desc in c_data["bullets"]:
                p = tf.add_paragraph()
                if b_name.strip():
                    p.text = f"• {b_name}: {b_desc}"
                    p.font.name = FONT_MAIN
                    p.font.size = Pt(8.4)
                    p.font.color.rgb = COLOR_SLATE
                else:
                    p.text = f"   {b_desc}"
                    p.font.name = FONT_MAIN
                    p.font.size = Pt(8.4)
                    p.font.color.rgb = COLOR_SLATE
                p.space_after = Pt(1.5)

        # Flow Arrow
        if idx < 3:
            arrow_x = x + col_w + Inches(0.06)
            arrow_y = Inches(4.5)
            tb_arr = s9.shapes.add_textbox(arrow_x, arrow_y, col_gap - Inches(0.12), Inches(0.5))
            tf_arr = tb_arr.text_frame
            p_arr = tf_arr.paragraphs[0]
            p_arr.text = "➔"
            p_arr.font.name = FONT_MAIN
            p_arr.font.size = Pt(18)
            p_arr.font.bold = True
            p_arr.font.color.rgb = COLOR_AMBER if idx == 1 or idx == 2 else COLOR_CYAN
            p_arr.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 10: GATED FUSION FORMULATION & FALLBACK
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10)
    add_header(s10, "CƠ CHẾ HỢP NHẤT GATED FEATURE FUSION", "Công Thức Toán Học, Cổng Sigmoid & Cơ Chế Fallback",
               "Chiếu 2 nguồn đặc trưng về không gian ℝ²⁵⁶ và điều tiết tỷ trọng thích ứng theo ca bệnh")

    # Top Formula Card
    add_box(s10, Inches(0.8), Inches(1.85), Inches(11.733), Inches(2.3), border_col=COLOR_AMBER, bg_col=CARD_BG)
    tb = s10.shapes.add_textbox(Inches(1.1), Inches(1.95), Inches(11.133), Inches(2.1))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "CÔNG THỨC TOÁN HỌC GATED MULTIMODAL UNIT (GMU) TRÊN KHÔNG GIAN ℝ²⁵⁶"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(6)

    math_lines = [
        "1. Chiếu đặc trưng (Projection):  F̃_v = GELU(LayerNorm(W_v · F_v + b_v)) ∈ ℝ²⁵⁶  (từ F_v ∈ ℝ¹⁰²⁴)",
        "                                  F̃_c = GELU(LayerNorm(W_c · F_c + b_c)) ∈ ℝ²⁵⁶  (từ F_c ∈ ℝ³²)",
        "2. Cổng thích ứng (Gate):         g = σ(W_g · [ F̃_v ; F̃_c ] + b_g) ∈ (0, 1)²⁵⁶",
        "3. Hợp nhất (Fusion Vector):      F_fusion = g ⊙ F̃_v + (1 − g) ⊙ F̃_c ∈ ℝ²⁵⁶",
        "4. Fallback khi thiếu CXR:        g = 0  ==>  F_fusion = F̃_c  (Tự động chuyển sang Clinical-only an toàn)"
    ]
    for line in math_lines:
        p = tf.add_paragraph()
        p.text = line
        p.font.name = FONT_CODE
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(3)

    # Bottom Table: Comparison between Concat, Decision Fusion and Gated Feature Fusion
    t_shape = s10.shapes.add_table(4, 4, Inches(0.8), Inches(4.35), Inches(11.733), Inches(2.6))
    table = t_shape.table
    col_widths = [Inches(2.5), Inches(2.8), Inches(3.2), Inches(3.233)]
    headers = ["Phương pháp Hợp nhất", "Cơ chế hoạt động", "Ưu điểm", "Hạn chế"]
    data = [
        ["Late Decision Fusion", "Ghép xác suất: P = α·P_vision + (1-α)·P_clinical", "Dễ thực hiện", "Bỏ qua tương tác đặc trưng bậc cao"],
        ["Concatenation Fusion", "Ghép phẳng: F = [F_v ; F_c]", "Giữ nguyên thông tin", "Giả định trọng số cố định, dễ bị át tín hiệu"],
        ["Gated Feature Fusion\n(Đề xuất MedFusion-CXR)", "GMU: Cổng g điều tiết thích ứng từng chiều đặc trưng", "Tự động cân bằng độ tin cậy giữa 2 nguồn + Có Fallback", "Cần dữ liệu ghép cặp để huấn luyện có giám sát W_g"]
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(245, 158, 11))

    # =========================================================================
    # SLIDE 11: EXPERIMENTAL PROTOCOL (E1 to E5)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11)
    add_header(s11, "THIẾT KẾ THỰC NGHIỆM ĐỘC LẬP", "Bảng Kế Hoạch Thực Nghiệm: Mục Tiêu, Dataset & Chỉ Số",
               "Phân định minh bạch giữa các thực nghiệm đã hoàn thành và điều kiện cần cho thực nghiệm multimodal")

    t_shape = s11.shapes.add_table(6, 5, Inches(0.8), Inches(1.85), Inches(11.733), Inches(5.1))
    table = t_shape.table
    col_widths = [Inches(1.0), Inches(2.3), Inches(2.5), Inches(3.2), Inches(2.733)]
    headers = ["Mã", "Tên thực nghiệm", "Tập dữ liệu (Dataset)", "Phương pháp / Mô hình", "Chỉ số đo lường (Metrics)"]
    data = [
        ["E1", "Vision Baseline", "NIH ChestX-ray14\n(Test: 1.314 ảnh / 50 BN)", "DenseNet-121 (Pretrained nih14)\nBootstrap 300 lần theo bệnh nhân", "AUROC (95% CI), AUPRC,\nF1-score, Brier, ECE"],
        ["E2", "Clinical Baseline", "Clinical Tabular Dataset\n(Test: 240 mẫu; 70/15/15)", "XGBoost vs 5 Baselines\n(LightGBM, RF, DT, MLP, LogReg)", "Accuracy, Precision, Recall,\nF1-Score, AUROC, AUPRC"],
        ["E3", "TreeSHAP XAI", "Clinical Test Split\n(240 bệnh nhân)", "XGBoost Booster\n(pred_contribs=True)", "Global Importance |ϕ_i| (%)\n& Trực quan hóa từng ca bệnh"],
        ["E4", "Fusion Architecture", "Dữ liệu ca bệnh mẫu\n(Demo presets)", "Gated Multimodal Unit (GMU ℝ²⁵⁶)\nKiểm tra Fallback Mode khi thiếu ảnh", "Tensor shape (256-d),\nĐộ dài norm ‖F‖₂, Trọng số cổng g"],
        ["E5", "Multimodal Efficacy\n(Future Work)", "Cohort ghép cặp theo BN\n(MIMIC-CXR + MIMIC-IV)", "Huấn luyện có giám sát W_g\nAblation Study (Vision vs Clin vs GMU)", "Mức cải thiện ΔAUROC,\nΔF1-Score trên cùng bệnh nhân"]
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(30, 41, 59))

    # =========================================================================
    # SLIDE 12: EMPIRICAL RESULTS (E1 & E2)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_bg(s12)
    add_header(s12, "KẾT QUẢ THỰC NGHIỆM THỰC TẾ", "Bảng Số Liệu Định Lượng Thực Nghiệm E1 & E2",
               "Báo cáo trung thực số liệu đo lường trực tiếp trên 2 tập test độc lập của hệ thống")

    # Left: E2 Table
    tb_e2 = s12.shapes.add_textbox(Inches(0.8), Inches(1.85), Inches(5.68), Inches(0.4))
    tb_e2.text_frame.paragraphs[0].text = "E2: KẾT QUẢ NHÁNH LÂM SÀNG (TEST SET 240 MẪU)"
    tb_e2.text_frame.paragraphs[0].font.name = FONT_MAIN
    tb_e2.text_frame.paragraphs[0].font.size = Pt(11)
    tb_e2.text_frame.paragraphs[0].font.bold = True
    tb_e2.text_frame.paragraphs[0].font.color.rgb = COLOR_EMERALD

    t_shape = s12.shapes.add_table(7, 5, Inches(0.8), Inches(2.25), Inches(5.68), Inches(4.7))
    t_e2 = t_shape.table
    w_e2 = [Inches(2.0), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.98)]
    h_e2 = ["Mô hình", "F1", "AUROC", "AUPRC", "Brier"]
    d_e2 = [
        ["XGBoost (Đề xuất)", "0.8462", "0.9416", "0.9318", "0.0960"],
        ["LightGBM", "0.8319", "0.9385", "0.9250", "0.1003"],
        ["Random Forest", "0.8111", "0.9382", "0.9304", "0.1042"],
        ["Decision Tree", "0.8416", "0.8927", "0.8605", "0.1149"],
        ["Deep MLP (64x32)", "0.7009", "0.8368", "0.8322", "0.1609"],
        ["Logistic Regression", "0.6991", "0.8412", "0.8413", "0.1537"]
    ]
    format_table(t_e2, w_e2, h_e2, d_e2, header_bg=RGBColor(16, 185, 129))

    # Right: E1 Table (Top Pathologies)
    tb_e1 = s12.shapes.add_textbox(Inches(6.85), Inches(1.85), Inches(5.68), Inches(0.4))
    tb_e1.text_frame.paragraphs[0].text = "E1: KẾT QUẢ DENSENET-121 TRÊN NIH-14 (TEST 1.314 ẢNH)"
    tb_e1.text_frame.paragraphs[0].font.name = FONT_MAIN
    tb_e1.text_frame.paragraphs[0].font.size = Pt(11)
    tb_e1.text_frame.paragraphs[0].font.bold = True
    tb_e1.text_frame.paragraphs[0].font.color.rgb = COLOR_CYAN

    t_shape = s12.shapes.add_table(7, 4, Inches(6.85), Inches(2.25), Inches(5.68), Inches(4.7))
    t_e1 = t_shape.table
    w_e1 = [Inches(1.8), Inches(1.4), Inches(1.24), Inches(1.24)]
    h_e1 = ["Bệnh lý NIH-14", "Số ca dương", "Test AUROC", "Khoảng tin cậy 95%"]
    d_e1 = [
        ["Cardiomegaly (Tim to)", "79 / 1.314", "0.7541", "[0.636, 0.877]"],
        ["Mass (Khối u phổi)", "133 / 1.314", "0.7481", "[0.621, 0.836]"],
        ["Effusion (Tràn dịch)", "268 / 1.314", "0.7049", "[0.632, 0.777]"],
        ["Edema (Phù phổi)", "47 / 1.314", "0.6981", "[0.638, 0.763]"],
        ["Pneumothorax (Tràn khí)", "143 / 1.314", "0.6582", "[0.576, 0.740]"],
        ["MACRO TRUNG BÌNH (14)", "14 Bệnh lý", "0.6482", "Macro AUPRC: 0.1579"]
    ]
    format_table(t_e1, w_e1, h_e1, d_e1, header_bg=RGBColor(6, 182, 212))

    # =========================================================================
    # SLIDE 13: PRODUCT DEMONSTRATION & CDSS DASHBOARD
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_bg(s13)
    add_header(s13, "MINH HỌA SẢN PHẨM THỰC TẾ", "Giao Diện Hội Chẩn CDSS Web Platform (http://localhost:8000)",
               "Tích hợp thời gian thực: Nạp ca bệnh mẫu, đo lường Telemetry cổng và hỗ trợ phác đồ Bạch Mai")

    t_shape = s13.shapes.add_table(5, 4, Inches(0.8), Inches(1.85), Inches(11.733), Inches(5.1))
    table = t_shape.table
    col_widths = [Inches(2.5), Inches(3.2), Inches(3.0), Inches(3.033)]
    headers = ["Module Chức năng", "Input Thao tác Người dùng", "Xử lý Backend Docker", "Output Hiển thị trên UI"]
    data = [
        ["1. Preset Ca Bệnh\n(NIH CXR Thật)",
         "Chọn 1 trong 4 ca bệnh lâm sàng mẫu (Viêm phổi, Tim to, Tràn dịch, Sàng lọc) hoặc Upload ảnh CXR riêng.",
         "Load ảnh PNG 512x512 từ web/samples/, nạp 11 biến sinh hiệu và text bệnh sử đồng bộ.",
         "Hiển thị ảnh trên Lightbox, nút Invert X-ray âm bản, nút Bỏ ảnh test Fallback."],
        ["2. Phân Tích Đa Phương Thức\n(Multimodal Predict)",
         "Bấm 'CHẠY PHÂN TÍCH MEDFUSION-CXR' gửi multipart form (Ảnh + Vitals + Text).",
         "Kích hoạt 5 Agent: PhoBERT bóc tách thực thể -> DenseNet-121 trích F_v -> XGBoost trích F_c -> GMU.",
         "Thời gian suy luận latency ~3.6s (CPU). Trả về JSON chuẩn hóa toàn diện."],
        ["3. Báo Cáo Hợp Nhất\n& Đóng Góp Đa Nguồn",
         "Theo dõi trực tiếp tỷ lệ đóng góp thông tin giữa Ảnh X-quang và Dữ liệu Sinh hiệu.",
         "Tính cổng thích ứng g_vision (48%) vs g_clinical (52%), tự động sinh Nhận định lâm sàng trực quan.",
         "Hộp nhận định lâm sàng bằng tiếng Việt dễ hiểu cho bác sĩ + Khối mở rộng công thức toán cho kỹ sư."],
        ["4. Hỗ Trợ Quyết Định\n& Phác Đồ Bạch Mai",
         "Đối chiếu kết quả phân tích với triệu chứng lâm sàng và chỉ số sinh lý.",
         "Truy xuất Knowledge Base RAG theo hướng dẫn chẩn đoán điều trị Bệnh viện Bạch Mai.",
         "Thẻ định hướng bệnh lý (kèm mã ICD-10) + Phác đồ dùng thuốc + 8 thanh TreeSHAP cá thể."]
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(14, 116, 144))

    # =========================================================================
    # SLIDE 14: LIMITATIONS & FUTURE ROADMAP
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_bg(s14)
    add_header(s14, "GIỚI HẠN & ĐỊNH HƯỚNG PHÁT TRIỂN", "Thừa Nhận Khách Quan Giới Hạn & Lộ Trình Nâng Cấp",
               "Chỉ rõ ranh giới nghiên cứu hiện tại và các bước cần thiết để đưa vào ứng dụng thực tế")

    t_shape = s14.shapes.add_table(5, 3, Inches(0.8), Inches(1.85), Inches(11.733), Inches(5.1))
    table = t_shape.table
    col_widths = [Inches(2.5), Inches(4.6), Inches(4.633)]
    headers = ["Khía cạnh Kỹ thuật", "Giới hạn Hiện tại (Current Limitations)", "Lộ trình Giải quyết (Future Roadmap)"]
    data = [
        ["Dữ liệu Ghép cặp\n(Paired Data)",
         "Hai tập dữ liệu NIH ChestX-ray14 và Clinical Dataset độc lập, chưa liên kết theo từng bệnh nhân.",
         "Thu thập cohort ghép cặp thực tế (MIMIC-CXR + MIMIC-IV) có chung subject_id để ghép nối ảnh và sinh hiệu."],
        ["Trọng số Cổng GMU\n(Gate Weights)",
         "Trọng số W_g và projection layer vận hành ở chế độ nguyên mẫu kiến trúc (Architectural Prototype).",
         "Huấn luyện end-to-end có giám sát trên cohort ghép cặp; thực hiện Ablation Study (Vision vs Clinical vs GMU)."],
        ["Quy mô Kiểm định\n(Evaluation Scale)",
         "E1 mới chạy pilot trên 50 bệnh nhân/split; 1.500/1.600 mẫu lâm sàng là mô phỏng sinh lý.",
         "Mở rộng đánh giá toàn bộ 2.797 bệnh nhân test set của NIH; kiểm định ngoài trên hồ sơ EHR bệnh nhân thực tế."],
        ["Giải thích Thị giác\n(Visual XAI)",
         "Mới giải thích được nhánh lâm sàng bằng TreeSHAP; nhánh ảnh chỉ xuất điểm xác suất.",
         "Tích hợp Grad-CAM / Eigen-CAM khoanh vùng tổn thương trên phim X-quang phối hợp cùng TreeSHAP lâm sàng."]
    ]
    format_table(table, col_widths, headers, data, header_bg=RGBColor(244, 63, 94))

    # =========================================================================
    # SLIDE 15: CONCLUSION & TAKEAWAY
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_bg(s15)
    add_header(s15, "TỔNG KẾT & PHIÊN HỎI ĐÁP", "Ba Đóng Góp Kỹ Thuật Then Chốt Của Đề Tài MedFusion-CXR",
               "Khẳng định giá trị nghiên cứu, tính minh bạch phương pháp luận và khả năng chuyển giao thực tế")

    col_w = Inches(3.64)
    col_h = Inches(3.6)
    top_pos = Inches(1.85)

    takeaways = [
        ("1. ĐÓNG GÓP KIẾN TRÚC", COLOR_CYAN,
         "• Thiết kế hoàn chỉnh kiến trúc đa phương thức MedFusion-CXR.\n"
         "• Kết hợp thành công Deep Learning (DenseNet-121) và Machine Learning (XGBoost).\n"
         "• Đề xuất cơ chế Gated Feature Fusion GMU ℝ²⁵⁶ kèm cơ chế Clinical-only Fallback an toàn."),
        ("2. LIÊM CHÍNH KHOA HỌC", COLOR_EMERALD,
         "• Đánh giá độc lập trung thực trên từng tập dữ liệu.\n"
         "• XGBoost đạt AUROC 94.16%, F1 84.62%.\n"
         "• DenseNet-121 đạt Macro AUROC 0.6482.\n"
         "• Tuyệt đối không tạo kết quả ghép cặp giả mạo khi chưa có dữ liệu hợp lệ."),
        ("3. SẴN SÀNG TRIỂN KHAI", COLOR_AMBER,
         "• Đóng gói trọn vẹn container Docker (medfusion-cxr:latest), đạt trạng thái healthy.\n"
         "• Cung cấp đầy đủ RESTful API v2.2 và Web Dashboard CDSS trực quan.\n"
         "• Vượt qua 39/39 automated unit tests, sẵn sàng chuyển giao nghiên cứu.")
    ]

    for idx, (title, color, body) in enumerate(takeaways):
        x = Inches(0.8) + idx * (col_w + Inches(0.4))
        add_box(s15, x, top_pos, col_w, col_h, border_col=color)
        tb = s15.shapes.add_textbox(x + Inches(0.2), top_pos + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_MAIN
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(10)

        for line in body.split("\n"):
            p = tf.add_paragraph()
            p.text = line
            p.font.name = FONT_MAIN
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_WHITE if line.startswith("•") else COLOR_SLATE

    # Bottom Banner
    add_box(s15, Inches(0.8), Inches(5.65), Inches(11.733), Inches(1.3), border_col=COLOR_CYAN, bg_col=CARD_BG)
    tb = s15.shapes.add_textbox(Inches(1.0), Inches(5.75), Inches(11.333), Inches(1.1))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "XIN TRÂN TRỌNG CẢM ƠN QUÝ THẦY CÔ VÀ HỘI ĐỒNG KHOA HỌC!"
    p.font.name = FONT_MAIN
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = "Sẵn sàng cho Phiên Hỏi Đáp (Q&A)  ·  Hệ thống chạy trực tiếp tại: http://localhost:8000"
    p.font.name = FONT_MAIN
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_CYAN
    p.alignment = PP_ALIGN.CENTER

    # Save
    prs.save(PPTX_ROOT_PATH)
    print(f"Successfully generated crisp PPTX at: {PPTX_ROOT_PATH}")
    try:
        prs.save(PPTX_REPORTS_PATH)
        print(f"Also saved at: {PPTX_REPORTS_PATH}")
    except PermissionError:
        print(f"Note: {PPTX_REPORTS_PATH} is currently open in PowerPoint by user.")


if __name__ == "__main__":
    generate_deck()
