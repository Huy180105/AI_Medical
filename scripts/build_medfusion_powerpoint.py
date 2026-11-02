"""
Generates the official 15-slide PowerPoint deck for MedFusion-CXR.
Follows executive presentation standards: 16:9 widescreen, clean dark-mode
clinical aesthetic, card/bento layouts, high visual density, concise text,
large stat callouts, and embedded high-resolution graphics.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PPTX_PATH = "reports/MedFusion_CXR_Presentation.pptx"
FINAL_PPTX_PATH = "reports/MedFusion_CXR_Presentation_FINAL.pptx"
ARTIFACT_PPTX_PATH = r"C:\Users\Quang Huy\.gemini\antigravity\brain\42b17145-450c-42bc-8abf-662b84f7253c\MedFusion_CXR_Presentation_FINAL.pptx"

# Color Palette: Deep Modern Clinical Tech
BG_COLOR = RGBColor(10, 15, 29)          # #0A0F1D (Dark Navy)
CARD_BG = RGBColor(17, 28, 53)           # #111C35 (Elevated Card)
CARD_BORDER = RGBColor(30, 41, 59)       # #1E293B (Subtle Slate Border)
CARD_HIGHLIGHT = RGBColor(19, 35, 68)    # Highlighted card bg

COLOR_CYAN = RGBColor(6, 182, 212)       # #06B6D4 (Vision & Key Tech)
COLOR_BLUE = RGBColor(59, 130, 246)      # #3B82F6 (Primary Accent)
COLOR_EMERALD = RGBColor(16, 185, 129)   # #10B981 (Clinical & Validated)
COLOR_AMBER = RGBColor(245, 158, 11)     # #F59E0B (Prototype / Alerts)
COLOR_ROSE = RGBColor(244, 63, 94)       # #F43F5E (High Risk / Warnings)
COLOR_WHITE = RGBColor(255, 255, 255)    # #FFFFFF
COLOR_SLATE = RGBColor(148, 163, 184)    # #94A3B8 (Secondary Text)
COLOR_MUTED = RGBColor(100, 116, 139)    # #64748B (Muted)

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    def add_header(slide, eyebrow, title, subtitle):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.15))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Eyebrow
        p0 = tf.paragraphs[0]
        p0.text = eyebrow.upper()
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = COLOR_CYAN
        p0.space_after = Pt(2)

        # Title
        p1 = tf.add_paragraph()
        p1.text = title
        p1.font.name = FONT_HEADING
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_WHITE
        p1.space_after = Pt(3)

        # Subtitle
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.name = FONT_BODY
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = COLOR_SLATE

        # Subtle divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.58), Inches(11.733), Inches(0.015))
        line.fill.solid()
        line.fill.fore_color.rgb = CARD_BORDER
        line.line.color.rgb = CARD_BORDER

    def add_bento_box(slide, left, top, width, height, border_color=CARD_BORDER, bg_color=CARD_BG):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
        return shape

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    # Hero Card
    hero = add_bento_box(slide1, Inches(1.2), Inches(1.2), Inches(10.933), Inches(5.1), border_color=COLOR_CYAN, bg_color=CARD_HIGHLIGHT)
    
    # Title Text Frame
    tb = slide1.shapes.add_textbox(Inches(1.8), Inches(1.6), Inches(9.733), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = "BÁO CÁO NGHIÊN CỨU & KHÓA LUẬN CDSS · CLINICAL AI"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(10)

    p = tf.add_paragraph()
    p.text = "MedFusion-CXR"
    p.font.name = FONT_HEADING
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_after = Pt(8)

    p = tf.add_paragraph()
    p.text = "Hệ thống Đa phương thức & Đa tác tử Hỗ trợ Đánh giá Lâm sàng"
    p.font.name = FONT_HEADING
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(10)

    p = tf.add_paragraph()
    p.text = "Tích hợp Deep Learning Phim X-Quang (DenseNet-121) và Machine Learning Lâm Sàng (XGBoost) qua Cơ Chế Gated Feature Fusion trong Kiến Trúc Multi-Agent"
    p.font.name = FONT_BODY
    p.font.size = Pt(12.5)
    p.font.color.rgb = COLOR_SLATE
    p.space_after = Pt(26)

    # Badges Line
    p = tf.add_paragraph()
    p.text = "⚡ Đóng góp Kỹ thuật:  DenseNet-121 (NIH-14)   |   XGBoost + TreeSHAP   |   Gated Fusion (GMU)   |   5-Agent Swarm"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "PHẦN I · BỐI CẢNH NGHIÊN CỨU", "Phát Biểu Bài Toán & Thách Thức Thực Tiễn", "Vì sao các hệ thống đơn phương thức (Single-Modality) không đáp ứng an toàn trong hỗ trợ ra quyết định lâm sàng?")

    # 3 Cards
    w3 = Inches(3.64)
    h3 = Inches(5.1)
    top3 = Inches(1.85)

    # Card 1: Vision-only
    add_bento_box(slide2, Inches(0.8), top3, w3, h3, border_color=COLOR_CYAN)
    tb = slide2.shapes.add_textbox(Inches(1.05), Inches(2.05), w3 - Inches(0.5), h3 - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🩻 THÁCH THỨC VISION-ONLY"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(14)

    bullets = [
        "Chỉ phân tích hình thái trên phim X-quang, hoàn toàn 'mù' tình trạng sinh lý hiện thời của bệnh nhân.",
        "Không đánh giá được sốc huyết động, suy hô hấp hay phản ứng viêm cấp tính (SpO₂, HR, WBC, CRP).",
        "Hình ảnh tổn thương phổi có thể biểu hiện chậm hơn diễn tiến sinh lý cấp tính."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(10)

    # Card 2: Clinical-only
    add_bento_box(slide2, Inches(4.84), top3, w3, h3, border_color=COLOR_EMERALD)
    tb = slide2.shapes.add_textbox(Inches(5.09), Inches(2.05), w3 - Inches(0.5), h3 - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🩺 HẠN CHẾ CLINICAL-ONLY"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.space_after = Pt(14)

    bullets = [
        "Phản ánh rối loạn sinh hiệu và xét nghiệm toàn thân nhưng thiếu bằng chứng trực quan về tổn thương thực thể.",
        "Khó phân định chính xác căn nguyên (Ví dụ: Suy tim ứ huyết vs. Viêm phổi thâm nhiễm đều có khó thở).",
        "Bỏ lỡ thông tin giải phẫu học quan trọng: mức dịch màng phổi, bóng tim to, xẹp phổi khu trú."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(10)

    # Card 3: Fusion Gap
    add_bento_box(slide2, Inches(8.88), top3, w3, h3, border_color=COLOR_AMBER)
    tb = slide2.shapes.add_textbox(Inches(9.13), Inches(2.05), w3 - Inches(0.5), h3 - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔗 KHOẢNG TRỐNG HỢP NHẤT"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(14)

    bullets = [
        "Ghép nối xác suất đơn giản (Late Decision Fusion) bỏ lỡ tương tác biểu diễn đặc trưng giữa hình ảnh và sinh lý.",
        "Thiếu cơ chế cổng điều tiết thích nghi (Gated Mechanism) theo từng thể bệnh và chất lượng dữ liệu.",
        "Cần kiến trúc hợp nhất cấp đặc trưng (Feature-level) đi kèm khả năng dự phòng khi thiếu ảnh X-quang."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 3: RESEARCH MOTIVATION & OBJECTIVES
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "PHẦN I · BỐI CẢNH NGHIÊN CỨU", "Động Lực Nghiên Cứu & 4 Mục Tiêu Kỹ Thuật", "Thiết kế hệ thống đa phương thức toàn diện kết hợp Deep Learning, Machine Learning và Multi-Agent")

    # 4 Bento Cards (2x2 grid)
    gw = Inches(5.68)
    gh = Inches(2.4)
    xs = [Inches(0.8), Inches(6.85)]
    ys = [Inches(1.85), Inches(4.55)]

    goals = [
        ("1. Xây dựng Nhánh Thị giác (Vision Intelligence)",
         COLOR_CYAN,
         ["Ứng dụng mô hình DenseNet-121 tiền huấn luyện trên 14 nhãn NIH ChestX-ray14.",
          "Trích xuất vector biểu diễn đặc trưng hình thái học lồng ngực F_v ∈ ℝ¹⁰²⁴.",
          "Hiệu chuẩn ngưỡng phân loại tối ưu hóa F1 trên tập Validation."]),
        ("2. Xây dựng Nhánh Lâm sàng (Clinical Intelligence)",
         COLOR_EMERALD,
         ["Huấn luyện XGBoost trên 11 biến sinh hiệu/thể trạng + các chỉ số cấp cứu (qSOFA, Shock Index).",
          "Trích xuất vector biểu diễn lâm sàng F_c ∈ ℝ³² kết hợp với TreeSHAP.",
          "Cung cấp khả năng giải thích nguyên nhân cá thể hóa chính xác trên từng ca bệnh."]),
        ("3. Thiết kế Cơ chế Gated Feature Fusion (GMU)",
         COLOR_AMBER,
         ["Chiếu vector F_v và F_c về không gian biểu diễn tương thích ℝ²⁵⁶.",
          "Thiết kế cổng Sigmoid điều tiết mức độ đóng góp thích ứng: F_fusion = g⊙F̃_v + (1-g)⊙F̃_c.",
          "Tích hợp cơ chế tự động chuyển sang Clinical-only fallback an toàn khi thiếu ảnh."]),
        ("4. Xây dựng Hệ Thống Điều Phối Multi-Agent",
         COLOR_BLUE,
         ["Phân công 5 Agent chuyên khoa: Giám sát thời gian t₀, NLP PhoBERT, Chẩn đoán ảnh, Sinh hiệu, Hội chẩn.",
          "Ngăn chặn triệt để rò rỉ thời gian (Temporal Leakage Guard).",
          "Tổng hợp báo cáo lâm sàng và gợi ý phác đồ điều trị tham khảo theo Hướng dẫn BV Bạch Mai."])
    ]

    for idx, (title, color, bullets) in enumerate(goals):
        x = xs[idx % 2]
        y = ys[idx // 2]
        add_bento_box(slide3, x, y, gw, gh, border_color=color)
        tb = slide3.shapes.add_textbox(x + Inches(0.25), y + Inches(0.2), gw - Inches(0.5), gh - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(8)
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.name = FONT_BODY
            p.font.size = Pt(10.5)
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 4: DATASETS & SCIENTIFIC GOVERNANCE
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "PHẦN I · DỮ LIỆU & QUẢN TRỊ KHOA HỌC", "Nguồn Dữ Liệu & Cam Kết Liêm Chính Khoa Học", "Minh bạch phương pháp luận: Đánh giá độc lập 2 dataset, tuyệt đối không tạo dữ liệu ghép cặp giả mạo")

    # Banner Callout
    banner = add_bento_box(slide4, Inches(0.8), Inches(1.8), Inches(11.733), Inches(0.7), border_color=COLOR_AMBER, bg_color=CARD_HIGHLIGHT)
    tb = slide4.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.333), Inches(0.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "⚠️ CAM KẾT LIÊM CHÍNH KHOA HỌC: Tập ảnh NIH ChestX-ray14 và Tập Lâm sàng hiện tại chưa ghép cặp theo bệnh nhân. Hệ thống đánh giá độc lập từng nhánh và trình bày Gated Fusion ở chế độ Prototype Kiến trúc."
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER

    # 2 Comparison Columns
    col_w = Inches(5.68)
    col_h = Inches(4.35)
    col_top = Inches(2.65)

    # Left: NIH
    add_bento_box(slide4, Inches(0.8), col_top, col_w, col_h, border_color=COLOR_CYAN)
    tb = slide4.shapes.add_textbox(Inches(1.05), col_top + Inches(0.25), col_w - Inches(0.5), col_h - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🩻 TẬP ẢNH X-QUANG: NIH CHESTX-RAY14"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(10)

    nih_bullets = [
        "Quy mô: 112.120 ảnh X-quang lồng ngực từ 30.805 bệnh nhân duy nhất (NIH Clinical Center).",
        "Nhãn bệnh lý: 14 nhãn nhị phân {0, 1} khai phá qua xử lý ngôn ngữ tự nhiên (NLP-mined).",
        "Phân chia bệnh nhân (Patient-Level Split): Train 73.089 / Val 13.435 / Test 25.596 ảnh.",
        "Zero Patient Leakage: Trùng lặp bệnh nhân giữa Train, Val và Test = 0 (Đã kiểm chứng tự động).",
        "Trạng thái đĩa cục bộ: 74.999 ảnh hợp lệ (shards 001–008, ~30 GB) sẵn sàng đánh giá."
    ]
    for b in nih_bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(6)

    # Right: Clinical
    add_bento_box(slide4, Inches(6.85), col_top, col_w, col_h, border_color=COLOR_EMERALD)
    tb = slide4.shapes.add_textbox(Inches(7.1), col_top + Inches(0.25), col_w - Inches(0.5), col_h - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🩺 TẬP LÂM SÀNG: CLINICAL TABULAR DATASET"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.space_after = Pt(10)

    clin_bullets = [
        "Quy mô: 1.600 mẫu bệnh nhân (1.500 mẫu mô phỏng sinh lý + 100 hồ sơ trích xuất lâm sàng).",
        "11 Đặc trưng thể trạng: Tuổi, giới, SpO₂, nhịp tim, HA tâm thu/tâm trương, stress, giấc ngủ, vận động, bệnh nền, mức triệu chứng.",
        "Phân chia phân tầng (Stratified 70/15/15): Train 1.120 / Val 240 / Held-out Test 240 mẫu.",
        "Tỷ lệ dương tính (Cần can thiệp điều trị): 42.25% toàn tập, phân bổ đồng đều giữa các split.",
        "Xử lý ngoại lai: SpO₂ được giới hạn an toàn trong [50.0, 100.0]%; mã hóa biến phân loại chuẩn xác."
    ]
    for b in clin_bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 5: DATA ENGINEERING & PREPROCESSING
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "PHẦN II · PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG", "Kỹ Thuật Dữ Liệu & Kiểm Soát Rò Rỉ Đa Tầng", "Quy trình tiền xử lý độc lập và các rào chắn kỹ thuật ngăn ngừa data leakage triệt để")

    # 3 Flow Columns
    cw = Inches(3.64)
    ch = Inches(5.1)
    ctop = Inches(1.85)

    steps = [
        ("1. Pipeline Xử lý Ảnh NIH",
         COLOR_CYAN,
         ["Chuyển đổi 1 kênh xám (Grayscale), chuẩn hóa cường độ voxel [-1024.0, +1024.0].",
          "Cắt khung vuông trung tâm (Center-crop) độ phân giải chuẩn 224 × 224 pixels.",
          "Kiểm định mã băm SHA-256 checkpoint: 31a5475c… đảm bảo toàn vẹn trọng số.",
          "Cách ly hoàn toàn các file ảnh giả lập đồ họa (Synthetic Graphic Quarantine)."]),
        ("2. Pipeline Chuẩn hóa Lâm sàng",
         COLOR_EMERALD,
         ["StandardScaler chỉ fit strictly trên tập Train (1.120 mẫu) để tránh rò rỉ phân phối (Data snooping).",
          "Kiểm soát giá trị bất thường: Clip SpO₂ về [50, 100]% (phát hiện và sửa lỗi 102.7% và 101.4%).",
          "Tính toán các chỉ số động học cấp tính: qSOFA, Shock Index (HR/SBP), độ giảm oxy máu.",
          "Mã hóa biến giới tính nhị phân không mất mát thông tin."]),
        ("3. Kiểm soát Rò rỉ Đa tầng",
         COLOR_AMBER,
         ["Patient-level Split: Kiểm tra giao tập tập bệnh nhân train ∩ val ∩ test = ∅, không rò rỉ cùng một người.",
          "Freeze Validation Threshold: Ngưỡng quyết định F1 tối ưu được chọn trên Val và đóng băng áp dụng cho Test.",
          "Temporal Guard Agent: Chặn toàn bộ dữ liệu phát sinh sau mốc thời gian tiếp nhận ban đầu t₀.",
          "Tách biệt hoàn toàn pipeline xử lý giữa hai modality độc lập."])
    ]

    for idx, (title, col, bullets) in enumerate(steps):
        x = Inches(0.8) + idx * (cw + Inches(0.4))
        add_bento_box(slide5, x, ctop, cw, ch, border_color=col)
        tb = slide5.shapes.add_textbox(x + Inches(0.25), ctop + Inches(0.25), cw - Inches(0.5), ch - Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(12)
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.name = FONT_BODY
            p.font.size = Pt(10.5)
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 6: OVERALL SYSTEM ARCHITECTURE
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "PHẦN II · PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG", "Kiến Trúc Tổng Thể Hệ Thống MedFusion-CXR", "Tích hợp hai nhánh độc lập qua Gated Feature Fusion và phân cấp điều phối bằng Multi-Agent Swarm")

    arch_img_path = "reports/presentation_assets/architecture_flowchart.png"
    if os.path.exists(arch_img_path):
        slide6.shapes.add_picture(arch_img_path, Inches(0.8), Inches(1.8), width=Inches(11.733), height=Inches(5.1))
    else:
        # Fallback card
        add_bento_box(slide6, Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.1), border_color=COLOR_CYAN)

    # =========================================================================
    # SLIDE 7: VISION INTELLIGENCE (DENSENET-121)
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "PHẦN II · PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG", "Nhánh Trí Tuệ Hình Ảnh: DenseNet-121", "Mô hình mạng nơ-ron tích chập sâu phân tích 14 bệnh lý lồng ngực và trích xuất vector đặc trưng thị giác")

    # Left: Image Card with real NIH sample
    add_bento_box(slide7, Inches(0.8), Inches(1.85), Inches(4.5), Inches(5.1), border_color=COLOR_CYAN)
    sample_img = "web/samples/pneumonia_nih.png"
    if os.path.exists(sample_img):
        slide7.shapes.add_picture(sample_img, Inches(1.15), Inches(2.1), width=Inches(3.8), height=Inches(3.8))
    
    tb = slide7.shapes.add_textbox(Inches(1.15), Inches(6.05), Inches(3.8), Inches(0.7))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Phim X-quang Lồng ngực Mẫu (NIH ChestX-ray14)\nChuẩn hóa Grayscale · Center-Crop 224x224"
    p.font.name = FONT_BODY
    p.font.size = Pt(9.5)
    p.font.color.rgb = COLOR_SLATE
    p.alignment = PP_ALIGN.CENTER

    # Right: Technical Details Card
    add_bento_box(slide7, Inches(5.6), Inches(1.85), Inches(6.933), Inches(5.1), border_color=COLOR_CYAN)
    tb = slide7.shapes.add_textbox(Inches(5.9), Inches(2.15), Inches(6.333), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "CƠ CHẾ HOẠT ĐỘNG NHÁNH THỊ GIÁC (VISION BRANCH)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(12)

    v_details = [
        "Backbone Kiến trúc: DenseNet-121 (Dense Convolutional Network) với kết nối dày đặc bảo toàn gradient.",
        "Trọng số Tiền huấn luyện: TorchXRayVision 'densenet121-res224-nih' (nih-densenet121-nih14.pt, SHA-256 verified).",
        "14 Nhãn Bệnh lý NIH: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.",
        "Trích xuất Biểu diễn Đặc trưng: Hàm extract_features() trích xuất vector đặc trưng thị giác F_v ∈ ℝ¹⁰²⁴ trước lớp fully-connected.",
        "Hiệu chuẩn Ngưỡng (Calibration): Tìm kiếm lưới 91 điểm trên Validation split để chọn ngưỡng tối ưu hóa F1 độc lập cho từng nhãn bệnh."
    ]
    for b in v_details:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 8: CLINICAL INTELLIGENCE (XGBOOST + TREESHAP)
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "PHẦN II · PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG", "Nhánh Trí Tuệ Lâm Sàng: XGBoost & TreeSHAP", "Phân tích nguy cơ lâm sàng, trích xuất biểu diễn 32-d và giải thích quyết định cá thể hóa")

    # Left: Explanation & Vector Representation
    add_bento_box(slide8, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1), border_color=COLOR_EMERALD)
    tb = slide8.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(5.1), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "CƠ CHẾ MÃ HÓA BIỂU DIỄN LÂM SÀNG F_c ∈ ℝ³²"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.space_after = Pt(12)

    c_details = [
        "Mô hình Học máy Cốt lõi: Gradient Boosted Trees (XGBoost) gồm 160 cây, độ sâu tối đa 4, learning rate 0.06.",
        "11 Biến Thể trạng & Sinh hiệu: SpO₂, SBP, DBP, Nhịp tim, Tuổi, Giới tính, Stress, Giấc ngủ, Vận động, Bệnh nền, Mức triệu chứng.",
        "Giải thích TreeSHAP: Hàm get_booster().predict(pred_contribs=True) xuất vector đóng góp phi tuyến chính xác từng thuộc tính.",
        "Vector Biểu diễn 32 chiều: [0..23] gồm 11 biến scaled + 11 SHAP + bias + logit; [24..31] gồm 8 chỉ số cấp cứu (qSOFA, Shock Index, RR, Temp, WBC, CRP, Troponin).",
        "Tính Tương thích: Giải quyết bài toán đưa biểu diễn từ cây quyết định sang mạng nơ-ron chung với DenseNet-121."
    ]
    for b in c_details:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(8)

    # Right: Embedded Chart E3
    shap_img = "reports/presentation_assets/e3_treeshap_importance.png"
    if os.path.exists(shap_img):
        slide8.shapes.add_picture(shap_img, Inches(6.65), Inches(1.85), width=Inches(5.883), height=Inches(5.1))
    else:
        add_bento_box(slide8, Inches(6.65), Inches(1.85), Inches(5.883), Inches(5.1), border_color=COLOR_EMERALD)

    # =========================================================================
    # SLIDE 9: MULTI-AGENT WORKFLOW
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "PHẦN II · PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG", "Hệ Thống Điều Phối Multi-Agent Chuyên Biệt", "Phân công 5 tác tử chuyên khoa vận hành đồng bộ và kiểm soát rò rỉ dữ liệu theo thời gian")

    agents = [
        ("Agent 1: TemporalLeakageGuardAgent", COLOR_ROSE,
         "Giám sát mốc tiếp nhận t₀, loại bỏ toàn bộ dữ liệu ghi nhận sau thời điểm phân tích ban đầu, ngăn chặn Lookahead Bias."),
        ("Agent 2: TriageTimelineAgent (PhoBERT)", COLOR_BLUE,
         "Bóc tách thực thể y khoa (Triệu chứng, Bệnh lý, Thuốc, Cận lâm sàng) và ước lượng thời gian khởi phát bệnh từ văn bản tự do."),
        ("Agent 3: CXRDiagnosticAgent (DenseNet-121)", COLOR_CYAN,
         "Kiểm tra chất lượng ảnh, chạy suy luận 14 nhãn NIH-14, cách ly ảnh synthetic và trích xuất vector đặc trưng F_v ∈ ℝ¹⁰²⁴."),
        ("Agent 4: VitalLabDynamicsAgent (XGBoost)", COLOR_EMERALD,
         "Tính điểm qSOFA, Shock Index, phân tầng mức độ giảm oxy máu, chạy XGBoost + TreeSHAP và xây dựng vector F_c ∈ ℝ³²."),
        ("Agent 5: MultimodalConsensusOrchestrator", COLOR_AMBER,
         "Kích hoạt Gated Fusion, tổng hợp đánh giá đồng thuận, đối chiếu phác đồ điều trị tham khảo theo Hướng dẫn Bệnh viện Bạch Mai.")
    ]

    card_h = Inches(0.92)
    card_gap = Inches(0.12)
    top_start = Inches(1.8)

    for idx, (aname, acol, adesc) in enumerate(agents):
        y = top_start + idx * (card_h + card_gap)
        add_bento_box(slide9, Inches(0.8), y, Inches(11.733), card_h, border_color=acol)
        tb = slide9.shapes.add_textbox(Inches(1.05), y + Inches(0.12), Inches(11.233), card_h - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = aname
        p.font.name = FONT_HEADING
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = acol
        p.space_after = Pt(2)

        p = tf.add_paragraph()
        p.text = adesc
        p.font.name = FONT_BODY
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 10: GATED FUSION PROTOTYPE
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "PHẦN II · PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG", "Cơ Chế Gated Feature Fusion (GMU Prototype)", "Hợp nhất đặc trưng thích ứng và cơ chế dự phòng an toàn (Fallback Mode) khi thiếu modality")

    # Math Hero Card
    mcard = add_bento_box(slide10, Inches(0.8), Inches(1.85), Inches(11.733), Inches(2.2), border_color=COLOR_AMBER, bg_color=CARD_HIGHLIGHT)
    tb = slide10.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.133), Inches(1.9))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "CÔNG THỨC HỢP NHẤT ĐẶC TRƯNG GATED MULTIMODAL UNIT (GMU) TRÊN KHÔNG GIAN ℝ²⁵⁶"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(6)

    formulas = [
        "• Vector thị giác F_v ∈ ℝ¹⁰²⁴ (DenseNet-121 pooling)  &  Vector lâm sàng F_c ∈ ℝ³² (24-d XGBoost+TreeSHAP + 8-d sinh hiệu/lab)",
        "• Phép chiếu không gian tương thích:  F̃_v = Proj_v(F_v) ∈ ℝ²⁵⁶   và   F̃_c = Proj_c(F_c) ∈ ℝ²⁵⁶  (Linear + LayerNorm + GELU)",
        "• Cổng điều tiết trọng số thích ứng:   g = σ(W_g [ F̃_v ; F̃_c ] + b_g) ∈ (0, 1)²⁵⁶",
        "• Vector biểu diễn hợp nhất:   F_fusion = g ⊙ F̃_v + (1 − g) ⊙ F̃_c ∈ ℝ²⁵⁶  (Clinical-only Fallback an toàn: g = 0 ⇒ F_fusion = F̃_c)"
    ]
    for f in formulas:
        p = tf.add_paragraph()
        p.text = f
        p.font.name = "Consolas"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(4)

    # 2 Explanatory Cards
    w2 = Inches(5.68)
    h2 = Inches(2.65)
    top2 = Inches(4.3)

    add_bento_box(slide10, Inches(0.8), top2, w2, h2, border_color=COLOR_CYAN)
    tb = slide10.shapes.add_textbox(Inches(1.05), top2 + Inches(0.2), w2 - Inches(0.5), h2 - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Ý NGHĨA KỸ THUẬT CỦA CƠ CHẾ CỔNG (GATE)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(8)
    bullets = [
        "Khắc phục nhược điểm ghép phẳng (Concatenation) vốn giả định trọng số cố định giữa 2 nguồn.",
        "Cổng g tự động gán trọng số cao hơn cho nhánh có đặc trưng rõ rệt (Tổn thương phổi nặng vs. Rối loạn huyết động).",
        "Giám sát trực tiếp độ dài chuẩn ‖F̃_v‖₂, ‖F̃_c‖₂ và độ phân tán của cổng qua Telemetry API."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(4)

    add_bento_box(slide10, Inches(6.85), top2, w2, h2, border_color=COLOR_AMBER)
    tb = slide10.shapes.add_textbox(Inches(7.1), top2 + Inches(0.2), w2 - Inches(0.5), h2 - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "GIỚI HẠN VỀ MẶT PHƯƠNG PHÁP LUẬN"
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(8)
    bullets = [
        "Trạng thái thực tế: ARCHITECTURAL_PROTOTYPE_UNPAIRED_COHORTS.",
        "Trọng số W_g chưa được huấn luyện có giám sát vì chưa có nhãn lâm sàng trên cùng bệnh nhân với ảnh NIH.",
        "Giá trị gate trên giao diện demo phản ánh khả năng vận hành luồng kiến trúc, không phải hiệu năng lâm sàng đã chứng minh."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 11: EXPERIMENTAL DESIGN
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "PHẦN III · THỰC NGHIỆM & KẾT QUẢ", "Thiết Kế Thực Nghiệm Độc Lập Nghiêm Ngặt", "Phân định minh bạch giữa các thực nghiệm đã hoàn thành và điều kiện cần cho thực nghiệm multimodal")

    exp_boxes = [
        ("Thực nghiệm E1: Đánh giá Nhánh Thị giác (DenseNet-121)", COLOR_CYAN,
         "Dataset: NIH ChestX-ray14 (Test Set: 1.314 ảnh / 50 BN; Val Set: 309 ảnh / 50 BN).\n"
         "Giao thức: Đóng băng ngưỡng Val F1 tối ưu, Bootstrap 300 lần theo mã bệnh nhân tính khoảng tin cậy 95% CI.\n"
         "Chỉ số đo lường: Macro AUROC, Macro AUPRC, Macro F1, Brier Score, Expected Calibration Error (ECE)."),
        ("Thực nghiệm E2: Đánh giá Nhánh Lâm sàng (XGBoost vs 5 Baselines)", COLOR_EMERALD,
         "Dataset: Clinical Tabular Dataset (1.600 mẫu; Train 1.120 / Val 240 / Test 240 mẫu).\n"
         "So sánh 6 mô hình: XGBoost, LightGBM, Random Forest, Decision Tree, Deep MLP (64x32), Logistic Regression.\n"
         "Chỉ số: Test Accuracy, Precision, Recall, F1-Score, AUROC, AUPRC, Brier Score."),
        ("Thực nghiệm E3: Khả năng Giải thích Quyết định (TreeSHAP XAI)", COLOR_BLUE,
         "Phương pháp: Trích xuất chính xác đóng góp đặc trưng phi tuyến từ cây quyết định (pred_contribs=True).\n"
         "Phân tích: Xếp hạng tỷ trọng đóng góp toàn cục (Global feature importance) và đóng góp từng ca bệnh (Local)."),
        ("Thực nghiệm E4: Kiểm thử Tầng Hợp nhất Gated Fusion (Kiến trúc & Luồng)", COLOR_AMBER,
         "Mục tiêu: Xác thực kích thước tensor (1024-d, 32-d -> 256-d), tính liên tục của hàm cổng sigmoid.\n"
         "Kiểm tra Fallback: Xác nhận chuyển đổi an toàn sang Clinical-only khi không có file ảnh đầu vào."),
        ("Thực nghiệm E5: Kiểm chứng Hiệu quả Fusion Đa phương thức (Lộ trình tương lai)", COLOR_MUTED,
         "Điều kiện tiên quyết: Yêu cầu tập dữ liệu lâm sàng ghép cặp theo từng ca bệnh được phê duyệt đạo đức y sinh.\n"
         "Cam kết khoa học: Tuyệt đối không báo cáo mức cải thiện fusion trên tập dữ liệu chưa ghép cặp.")
    ]

    card_h = Inches(0.92)
    card_gap = Inches(0.12)
    top_start = Inches(1.8)

    for idx, (ename, ecol, edesc) in enumerate(exp_boxes):
        y = top_start + idx * (card_h + card_gap)
        add_bento_box(slide11, Inches(0.8), y, Inches(11.733), card_h, border_color=ecol)
        tb = slide11.shapes.add_textbox(Inches(1.05), y + Inches(0.1), Inches(11.233), card_h - Inches(0.15))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ename
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ecol
        p.space_after = Pt(2)

        for line in edesc.split("\n"):
            p = tf.add_paragraph()
            p.text = line
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 12: EMPIRICAL RESULTS (E1 & E2)
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "PHẦN III · THỰC NGHIỆM & KẾT QUẢ", "Kết Quả Thực Nghiệm Độc Lập E1 & E2", "Số liệu định lượng thực tế từ các thực nghiệm đã chạy kiểm chứng trên hệ thống")

    # Left: E2 Chart
    e2_chart_path = "reports/presentation_assets/e2_clinical_benchmark.png"
    if os.path.exists(e2_chart_path):
        slide12.shapes.add_picture(e2_chart_path, Inches(0.8), Inches(1.85), width=Inches(5.68), height=Inches(3.3))
    
    # E2 Stats Card below
    add_bento_box(slide12, Inches(0.8), Inches(5.25), Inches(5.68), Inches(1.7), border_color=COLOR_EMERALD)
    tb = slide12.shapes.add_textbox(Inches(1.0), Inches(5.35), Inches(5.28), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "KẾT LUẬN THỰC NGHIỆM E2 (NHÁNH LÂM SÀNG - 240 MẪU TEST)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.space_after = Pt(4)
    bullets = [
        "XGBoost dẫn đầu toàn diện: AUROC 94.16% | AUPRC 93.18% | Test F1 84.62%.",
        "Vượt trội các thuật toán baseline: MLP (AUROC 83.68%), Logistic Regression (84.12%), Decision Tree (89.27%).",
        "Lưu ý phương pháp luận: 93.7% dữ liệu là mô phỏng sinh lý, cần tiếp tục kiểm định trên EHR thực."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_WHITE

    # Right: E1 Chart
    e1_chart_path = "reports/presentation_assets/e1_nih14_benchmark.png"
    if os.path.exists(e1_chart_path):
        slide12.shapes.add_picture(e1_chart_path, Inches(6.85), Inches(1.85), width=Inches(5.68), height=Inches(3.3))
    
    # E1 Stats Card below
    add_bento_box(slide12, Inches(6.85), Inches(5.25), Inches(5.68), Inches(1.7), border_color=COLOR_CYAN)
    tb = slide12.shapes.add_textbox(Inches(7.05), Inches(5.35), Inches(5.28), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "KẾT LUẬN THỰC NGHIỆM E1 (NHÁNH THỊ GIÁC - NIH-14 PILOT)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(4)
    bullets = [
        "Macro AUROC: 0.6482 | Macro AUPRC: 0.1579 trên 14 nhãn bệnh lý (Pilot Subset ban đầu: 1.314 ảnh Test).",
        "Mặt bệnh nổi bật: Tim to (0.7541 [0.636, 0.877]), Khối u Mass (0.7481), Tràn dịch Effusion (0.7049).",
        "Trạng thái dữ liệu hiện tại: Toàn bộ 112.120 ảnh (25.596 ảnh test chính thức) đã thẩm định 100% trên đĩa cho full benchmark."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 13: PRODUCT DEMONSTRATION
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13)
    add_header(slide13, "PHẦN IV · SẢN PHẨM & KẾT LUẬN", "Minh Họa Sản Phẩm Thực Tế (CDSS Web Platform)", "Tích hợp Backend FastAPI thực thi thật qua 5 Agent, DenseNet-121, XGBoost & TreeSHAP; UI demo trực quan")

    w3 = Inches(3.64)
    h3 = Inches(5.1)
    top3 = Inches(1.85)

    # Demo Card 1
    add_bento_box(slide13, Inches(0.8), top3, w3, h3, border_color=COLOR_CYAN)
    tb = slide13.shapes.add_textbox(Inches(1.05), Inches(2.05), w3 - Inches(0.5), h3 - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. NẠP CA BỆNH 1-CLICK"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(12)
    bullets = [
        "4 Preset ca bệnh lâm sàng mẫu gắn liền với ảnh X-quang NIH thật trong web/samples/.",
        "Ca 1: Viêm phổi cấp & giảm oxy máu.",
        "Ca 2: Tim to / Theo dõi suy tim mất bù.",
        "Ca 3: Tràn dịch màng phổi & khó thở.",
        "Ca 4: Khám sàng lọc định kỳ ổn định.",
        "Hỗ trợ kéo thả ảnh X-quang, đảo ngược âm bản (Invert) và nút 'Bỏ ảnh' kiểm tra Fallback."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(6)

    # Demo Card 2
    add_bento_box(slide13, Inches(4.84), top3, w3, h3, border_color=COLOR_AMBER)
    tb = slide13.shapes.add_textbox(Inches(5.09), Inches(2.05), w3 - Inches(0.5), h3 - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. GATED FUSION TELEMETRY"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(12)
    bullets = [
        "Thanh đo động trực quan hóa tỷ trọng đóng góp của 2 nhánh (ḡ_vision vs 1-ḡ_clinical).",
        "Giám sát chuẩn độ dài vector đặc trưng: ‖F_v‖₂, ‖F_c‖₂ và ‖F_fusion‖₂.",
        "Hiển thị cảnh báo liêm chính khoa học: Gated Fusion vận hành ở chế độ Prototype Kiến trúc.",
        "Khi bỏ ảnh X-quang: Tự động chuyển sang chế độ Clinical-only fallback với g = 0%."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(6)

    # Demo Card 3
    add_bento_box(slide13, Inches(8.88), top3, w3, h3, border_color=COLOR_EMERALD)
    tb = slide13.shapes.add_textbox(Inches(9.13), Inches(2.05), w3 - Inches(0.5), h3 - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. HỖ TRỢ QUYẾT ĐỊNH & PHÁC ĐỒ"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.space_after = Pt(12)
    bullets = [
        "Biểu đồ thanh ngang TreeSHAP cá thể hóa cho từng bệnh nhân (Đỏ: tăng nguy cơ, Xanh: giảm nguy cơ).",
        "Tính toán động học cấp cứu: Điểm qSOFA (1-3), Shock Index (HR/SBP), độ bão hòa Oxy.",
        "Gợi ý phác đồ điều trị tham khảo gắn liền mã ICD-10 theo Hướng dẫn Bệnh viện Bạch Mai.",
        "Tổng hợp kết quả hội chẩn đa phương thức từ 5 Agent chuyên khoa minh bạch từng bước xử lý."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 14: LIMITATIONS & FUTURE WORK
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14)
    add_header(slide14, "PHẦN IV · SẢN PHẨM & KẾT LUẬN", "Giới Hạn Hiện Tại & Hướng Phát Triển Tiếp Theo", "Đánh giá khách quan các hạn chế và đề xuất lộ trình nâng cấp hệ thống trong tương lai")

    # 2 Big Columns
    col_w = Inches(5.68)
    col_h = Inches(5.1)
    col_top = Inches(1.85)

    # Left: Limitations
    add_bento_box(slide14, Inches(0.8), col_top, col_w, col_h, border_color=COLOR_ROSE)
    tb = slide14.shapes.add_textbox(Inches(1.05), col_top + Inches(0.25), col_w - Inches(0.5), col_h - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "⚠️ GIỚI HẠN HIỆN TẠI (LIMITATIONS)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ROSE
    p.space_after = Pt(12)

    limits = [
        "Chưa có Dữ liệu Ghép cặp Bệnh nhân: Tập ảnh NIH và tập lâm sàng hiện tại độc lập; chưa chứng minh được mức cải thiện hiệu năng của Gated Fusion trên tập ghép cặp thực.",
        "Trọng số Cổng GMU chưa Huấn luyện: Trọng số W_g và projection layer vận hành ở chế độ nguyên mẫu kiến trúc (Architectural Prototype).",
        "Quy mô Tập Kiểm định E1: Mới chạy thử nghiệm pilot trên 50 bệnh nhân/split; khoảng tin cậy 95% còn rộng ở các nhãn hiếm (Hernia).",
        "Nguồn dữ liệu Lâm sàng: 1.500/1.600 mẫu lâm sàng là mô phỏng sinh lý, cần được thay thế bằng hồ sơ bệnh án điện tử (EHR) thực tế."
    ]
    for b in limits:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(8)

    # Right: Future Work
    add_bento_box(slide14, Inches(6.85), col_top, col_w, col_h, border_color=COLOR_CYAN)
    tb = slide14.shapes.add_textbox(Inches(7.1), col_top + Inches(0.25), col_w - Inches(0.5), col_h - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🚀 HƯỚNG PHÁT TRIỂN (FUTURE WORK)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_after = Pt(12)

    future = [
        "Hoàn thiện Đánh giá Toàn bộ 112.120 Ảnh NIH: Mở rộng kiểm định DenseNet-121 trên 25.596 ảnh test chính thức, phân tích chi tiết khoảng tin cậy theo phân nhóm tuổi, giới tính và tư thế chụp (PA vs AP).",
        "Cải thiện Tổng quát hóa & Hiệu chỉnh Xác suất: Nghiên cứu các phương pháp Temperature Scaling / Isotonic Regression để tối ưu độ tin cậy của xác suất dự đoán.",
        "Mở rộng Đánh giá Đa nhóm Bệnh nhân: Khảo sát độ nhạy và độ đặc hiệu trên các nhóm bệnh nhân đặc thù và các ca bệnh đồng mắc phức tạp.",
        "Xây dựng Cohort Ghép cặp Hợp lệ trong Tương lai: Khi có nguồn dữ liệu bệnh án điện tử và X-quang ghép cặp được phê duyệt, sẽ tiến hành huấn luyện có giám sát cho tầng Gated Fusion."
    ]
    for b in future:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 15: CONCLUSION & Q&A
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide15)
    add_header(slide15, "PHẦN IV · SẢN PHẨM & KẾT LUẬN", "Tổng Kết Đề Tài & Phiên Hỏi Đáp (Q&A)", "3 Thông điệp then chốt và cam kết chất lượng nghiên cứu của đề tài MedFusion-CXR")

    # 3 Summary Cards
    cw3 = Inches(3.64)
    ch3 = Inches(3.6)
    top3 = Inches(1.85)

    tcards = [
        ("1. Đóng Góp Kiến Trúc", COLOR_CYAN,
         ["Xây dựng thành công kiến trúc đa phương thức MedFusion-CXR kết hợp DenseNet-121 và XGBoost.",
          "Thiết kế cơ chế Gated Feature Fusion GMU ℝ²⁵⁶ điều tiết đóng góp thích ứng.",
          "Hỗ trợ chế độ dự phòng an toàn (Clinical-only fallback) khi thiếu ảnh X-quang."]),
        ("2. Liêm Chính Khoa Học", COLOR_EMERALD,
         ["Đánh giá độc lập trung thực: XGBoost đạt AUROC 94.16%, F1 84.62%; DenseNet-121 đạt Macro AUROC 0.6482.",
          "Minh bạch phân định giữa kiến trúc đề xuất và kết quả đã kiểm chứng thực nghiệm.",
          "Tuyệt đối không tạo dữ liệu ghép cặp giả mạo khi 2 dataset chưa liên kết bệnh nhân."]),
        ("3. Sẵn Sàng Triển Khai", COLOR_AMBER,
         ["Đóng gói container Docker hoàn chỉnh (medfusion-cxr:latest), đạt trạng thái healthy.",
          "Giao diện Web HUD tương tác trực tiếp với 4 ca bệnh mẫu và thanh đo Fusion Telemetry.",
          "Vượt qua 39/39 automated unit tests, sẵn sàng chuyển giao nghiên cứu."])
    ]

    for idx, (title, col, bullets) in enumerate(tcards):
        x = Inches(0.8) + idx * (cw3 + Inches(0.4))
        add_bento_box(slide15, x, top3, cw3, ch3, border_color=col)
        tb = slide15.shapes.add_textbox(x + Inches(0.25), top3 + Inches(0.2), cw3 - Inches(0.5), ch3 - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(10)
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.name = FONT_BODY
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(6)

    # Thank you / Q&A Banner
    q_banner = add_bento_box(slide15, Inches(0.8), Inches(5.65), Inches(11.733), Inches(1.3), border_color=COLOR_CYAN, bg_color=CARD_HIGHLIGHT)
    tb = slide15.shapes.add_textbox(Inches(1.1), Inches(5.75), Inches(11.133), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "XIN TRÂN TRỌNG CẢM ƠN QUÝ THẦY CÔ VÀ HỘI ĐỒNG KHOA HỌC!"
    p.font.name = FONT_HEADING
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = "Sẵn sàng cho Phiên Hỏi Đáp (Q&A)  ·  Mã nguồn & Dashboard: http://localhost:8000"
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_CYAN
    p.alignment = PP_ALIGN.CENTER

    # Save to all required paths
    os.makedirs("reports", exist_ok=True)
    prs.save(OUTPUT_PPTX_PATH)
    print("Saved presentation to:", OUTPUT_PPTX_PATH)
    prs.save(FINAL_PPTX_PATH)
    print("Saved final presentation to:", FINAL_PPTX_PATH)

    try:
        prs.save(ARTIFACT_PPTX_PATH)
        print("Saved presentation to artifact directory:", ARTIFACT_PPTX_PATH)
    except Exception as exc:
        print("Could not save to artifact dir:", exc)


if __name__ == "__main__":
    create_deck()
