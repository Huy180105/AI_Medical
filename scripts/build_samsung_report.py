"""
Builds the complete, rigorous, publication-grade academic report for MedFusion-CXR,
strictly adhering to the Samsung Innovation Campus (SIC) report template structure (13 chapters + preliminary sections).
Generates `reports/Bao_Cao_MedFusion_CXR_Samsung.docx` and syncs to Downloads.
Includes an explicit, beautifully formatted Table of Contents (MỤC LỤC) with dot leaders.
"""

import os
import shutil
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT_PATH = "reports/Bao_Cao_MedFusion_CXR_Samsung.docx"
DOWNLOADS_OUT_PATH = r"C:\Users\Quang Huy\Downloads\Bao_Cao_MedFusion_CXR_Samsung.docx"
os.makedirs("reports", exist_ok=True)

# Visual Styling
COLOR_PRIMARY = RGBColor(14, 116, 144)    # Deep Cyan #0E7490
COLOR_SECONDARY = RGBColor(15, 23, 42)   # Dark Slate #0F172A
COLOR_MUTED = RGBColor(100, 116, 139)    # Slate #64748B
COLOR_DARK = RGBColor(30, 41, 59)        # Dark Text #1E293B

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def format_table(table, col_widths, headers, data, header_bg="0E7490", alt_bg="F8FAFC"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], header_bg)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    # Data rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_col = alt_bg if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or len(str(val)) > 15 else WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(9.5)
                run.font.color.rgb = COLOR_DARK

    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = COLOR_DARK
    return p

def add_body(doc, text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.25
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_DARK
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.italic = italic
    run.font.color.rgb = COLOR_DARK
    return p

def add_bullet(doc, text, bold_prefix=""):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_DARK
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.color.rgb = COLOR_DARK
    return p

def add_figure(doc, img_path, caption, width=5.5):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        run_cap = p_cap.add_run(caption)
        run_cap.font.name = 'Times New Roman'
        run_cap.font.size = Pt(10)
        run_cap.font.italic = True
        run_cap.font.color.rgb = COLOR_MUTED

def add_toc_line(doc, title, page_str, level=1):
    """
    Renders an academic Table of Contents line with dot leader:
    Title ..................................................... Page
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3 if level > 1 else 5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15

    if level == 2:
        p.paragraph_format.left_indent = Inches(0.25)
    elif level == 3:
        p.paragraph_format.left_indent = Inches(0.45)

    # Set tab stop at 6.3 inches (9072 dxa) with dot leader
    pPr = p._p.get_or_add_pPr()
    tabs = OxmlElement('w:tabs')
    tab = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'right')
    tab.set(qn('w:leader'), 'dot')
    tab.set(qn('w:pos'), '9072')
    tabs.append(tab)
    pPr.append(tabs)

    # Title Run
    r1 = p.add_run(title)
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(10.5 if level > 1 else 11)
    if level == 1:
        r1.font.bold = True
        r1.font.color.rgb = COLOR_SECONDARY
    else:
        r1.font.color.rgb = COLOR_DARK

    # Tab and Page Run
    r2 = p.add_run(f"\t{page_str}")
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(10 if level > 1 else 10.5)
    if level == 1:
        r2.font.bold = True
    r2.font.color.rgb = COLOR_DARK


def build_full_report():
    doc = Document()

    # Page Margins: TCVN standard (Left 3cm, Right 2cm, Top 2cm, Bottom 2cm)
    for section in doc.sections:
        section.top_margin = Inches(0.79)     # 2.0 cm
        section.bottom_margin = Inches(0.79)  # 2.0 cm
        section.left_margin = Inches(1.18)    # 3.0 cm
        section.right_margin = Inches(0.79)   # 2.0 cm

    # =========================================================================
    # TRANG BÌA (COVER PAGE)
    # =========================================================================
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(36)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("CHƯƠNG TRÌNH ĐÀO TẠO NHÂN TÀI CÔNG NGHỆ SAMSUNG\nSAMSUNG INNOVATION CAMPUS (SIC)")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = COLOR_SECONDARY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(44)
    r_sub = p_sub.add_run("KHÓA HỌC: TRÍ TUỆ NHÂN TẠO (ARTIFICIAL INTELLIGENCE - CAPSTONE PROJECT)")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = COLOR_MUTED

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(20)
    p_title.paragraph_format.space_after = Pt(16)
    r_title = p_title.add_run("BÁO CÁO TỔNG KẾT ĐỀ TÀI CAPSTONE\n\nMedFusion-CXR:\nHỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH LÂM SÀNG ĐA PHƯƠNG THỨC\nTÍCH HỢP DEEP LEARNING PHIM X-QUANG VÀ MACHINE LEARNING LÂM SÀNG\nQUA CƠ CHẾ GATED FEATURE FUSION TRONG KIẾN TRÚC MULTI-AGENT")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(15.5)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(44)
    p_meta.paragraph_format.space_after = Pt(64)
    r_meta = p_meta.add_run("Học viên thực hiện: Nguyễn Quang Huy\nLớp / Khóa: Samsung AI Capstone 2026\nĐơn vị đào tạo: Samsung Innovation Campus Việt Nam\nThời gian hoàn thành: Tháng 10/2026")
    r_meta.font.name = 'Times New Roman'
    r_meta.font.size = Pt(12)
    r_meta.font.color.rgb = COLOR_DARK

    doc.add_page_break()

    # =========================================================================
    # LỜI CẢM ƠN
    # =========================================================================
    add_heading_1(doc, "LỜI CẢM ƠN")
    add_body(doc, "Trong suốt quá trình tham gia khóa đào tạo Trí tuệ Nhân tạo tại Samsung Innovation Campus (SIC) và thực hiện đề tài tốt nghiệp Capstone 'MedFusion-CXR', em đã nhận được sự quan tâm, chỉ dẫn tận tình từ quý Thầy Cô giảng viên và các chuyên gia công nghệ của Samsung.")
    add_body(doc, "Lời đầu tiên, em xin bày tỏ lòng biết ơn sâu sắc đến Ban Tổ chức Chương trình Samsung Innovation Campus và quý Thầy Cô cố vấn đã trang bị cho em nền tảng kiến thức vững chắc về Machine Learning, Deep Learning, Xử lý ảnh y tế và các nguyên tắc liêm chính khoa học trong xây dựng hệ thống AI.")
    add_body(doc, "Em cũng xin chân thành cảm ơn các bác sĩ chuyên khoa Hồi sức cấp cứu và Chẩn đoán hình ảnh tại các bệnh viện thực hành đã đưa ra những ý kiến đóng góp quý báu về quy trình tiếp nhận ca bệnh thực tế, thang điểm cảnh báo sớm và phác đồ điều trị của Bệnh viện Bạch Mai, giúp đề tài không chỉ dừng lại ở lý thuyết mà gắn liền mật thiết với thực tiễn lâm sàng.")
    add_body(doc, "Dù đã dành nhiều tâm huyết và nỗ lực hoàn thiện hệ thống, đề tài chắc chắn không tránh khỏi những khiếm khuyết. Em rất mong nhận được những ý kiến đóng góp, nhận xét và chỉ dẫn của quý Thầy Cô trong Hội đồng khoa học để đề tài ngày càng hoàn thiện hơn.")

    doc.add_page_break()

    # =========================================================================
    # MỤC LỤC (TABLE OF CONTENTS)
    # =========================================================================
    add_heading_1(doc, "MỤC LỤC")

    toc_entries = [
        ("LỜI CẢM ƠN", "i", 1),
        ("MỤC LỤC", "ii", 1),
        ("DANH MỤC CÁC BẢNG", "iv", 1),
        ("DANH MỤC CÁC HÌNH", "v", 1),
        ("TÓM TẮT ĐỀ TÀI (ABSTRACT)", "vi", 1),
        ("CHƯƠNG 1: BỐI CẢNH, MỤC TIÊU VÀ PHẠM VI", "1", 1),
        ("1.1. Bối cảnh nghiên cứu và nỗi đau thực tế (Clinical Pain Points)", "1", 2),
        ("1.2. Mục tiêu kỹ thuật và phạm vi đề tài", "2", 2),
        ("CHƯƠNG 2: NỀN TẢNG LÝ THUYẾT", "3", 1),
        ("2.1. Tổng quan các phương pháp học sâu và học máy trong Y tế", "3", 2),
        ("2.2. Trích xuất đặc trưng và biểu diễn vector đa phương thức", "3", 2),
        ("2.3. Mạng tích chập DenseNet-121, XGBoost và PhoBERT", "4", 2),
        ("2.4. Cơ chế hợp nhất đặc trưng Gated Multimodal Unit (GMU)", "5", 2),
        ("CHƯƠNG 3: KHẢO SÁT DỮ LIỆU VÀ CHUẨN BỊ MÔI TRƯỜNG", "6", 1),
        ("3.1. Nguồn dữ liệu ban đầu và cam kết liêm chính khoa học", "6", 2),
        ("3.2. Chuẩn bị phần cứng, môi trường và thư viện phần mềm", "7", 2),
        ("3.3. Tổ chức cấu trúc thư mục dự án chuẩn hóa", "7", 2),
        ("CHƯƠNG 4: TIỀN XỬ LÝ DỮ LIỆU VÀ QUẢN LÝ TẬP DỮ LIỆU", "8", 1),
        ("4.1. Tiền xử lý ảnh X-quang: Chuẩn hóa dải Hounsfield và kích thước 224×224", "8", 2),
        ("4.2. Tiền xử lý dữ liệu lâm sàng và bóc tách thực thể tiếng Việt (PhoBERT NER)", "8", 2),
        ("4.3. Phân chia Train/Validation/Test và 3 tầng phòng chống rò rỉ dữ liệu", "9", 2),
        ("CHƯƠNG 5: KHẢO SÁT VÀ XÂY DỰNG CÁC NHÁNH MÔ HÌNH", "10", 1),
        ("5.1. Nhánh Thị giác: DenseNet-121 và trích xuất vector đặc trưng F_v ∈ ℝ¹⁰²⁴", "10", 2),
        ("5.2. Nhánh Lâm sàng: XGBoost + TreeSHAP và xây dựng vector biểu diễn F_c ∈ ℝ³²", "10", 2),
        ("5.3. Tầng Hợp nhất Gated Feature Fusion (GMU ℝ²⁵⁶) và cơ chế Fallback an toàn", "11", 2),
        ("5.4. Các bài học và kinh nghiệm lựa chọn mô hình", "12", 2),
        ("CHƯƠNG 6: HỆ THỐNG ĐIỀU PHỐI ĐA TÁC TỬ (MULTI-AGENT SWARM) VÀ RAG", "13", 1),
        ("6.1. Luồng tiếp nhận đầu vào tại mốc t₀ (TemporalLeakageGuard & TriageTimelineAgent)", "13", 2),
        ("6.2. Phân công song song chuyên biệt (CXRDiagnosticAgent & VitalLabDynamicsAgent)", "13", 2),
        ("6.3. Hội đồng Gated Fusion (ConsensusOrchestrator) & RAG Phác đồ Bệnh viện Bạch Mai", "14", 2),
        ("CHƯƠNG 7: THIẾT KẾ KIẾN TRÚC VÀ TRIỂN KHAI HỆ THỐNG", "15", 1),
        ("7.1. Sơ đồ kiến trúc tổng thể 4 giai đoạn", "15", 2),
        ("7.2. Các endpoint backend API và tích hợp chuẩn y tế FHIR R4", "16", 2),
        ("7.3. Đóng gói hệ thống với Docker Container và Healthcheck", "16", 2),
        ("7.4. Giao diện người dùng CDSS Web Platform", "17", 2),
        ("CHƯƠNG 8: THIẾT KẾ THÍ NGHIỆM VÀ THƯỚC ĐO", "18", 1),
        ("8.1. Nguyên tắc đối chứng và độc lập thực nghiệm (E1 đến E5)", "18", 2),
        ("8.2. Các thước đo: Accuracy, Precision, Recall, F1-Score, AUROC, AUPRC, Brier", "18", 2),
        ("8.3. Phân tích giải thích cá thể hóa bằng TreeSHAP và Bootstrap 95% CI", "19", 2),
        ("CHƯƠNG 9: KẾT QUẢ THỰC NGHIỆM VÀ BIỂU ĐỒ", "20", 1),
        ("9.1. Kết quả thực nghiệm E2: Nhánh lâm sàng trên tập test 240 mẫu", "20", 2),
        ("9.2. Kết quả thực nghiệm E1: Đánh giá DenseNet-121 trên 1.314 ảnh test NIH-14", "21", 2),
        ("9.3. Kết quả thực nghiệm E3: Giải thích TreeSHAP trên các ca bệnh", "22", 2),
        ("9.4. Nhận xét và phân tích định lượng kết quả", "22", 2),
        ("CHƯƠNG 10: ĐÁNH GIÁ THỰC TẾ LÂM SÀNG VÀ GIỚI HẠN CHỨNG CỨ", "23", 1),
        ("10.1. Quy trình thử nghiệm ca bệnh mẫu lâm sàng", "23", 2),
        ("10.2. Tình trạng thực tế của dữ liệu chưa ghép cặp (Unpaired cohorts)", "23", 2),
        ("10.3. Nhận định ranh giới giữa nguyên mẫu kiến trúc và thử nghiệm lâm sàng thực tế", "24", 2),
        ("10.4. Phương án tối thiểu khi triển khai tại cơ sở y tế", "24", 2),
        ("CHƯƠNG 11: NHẬT KÝ KHÓ KHĂN, LỖI VÀ CÁCH KHẮC PHỤC", "25", 1),
        ("11.1. Dữ liệu ảnh lớn và mất cân bằng nhãn bệnh lý", "25", 2),
        ("11.2. Khó khăn về lệch số chiều (1024 vs 32)", "25", 2),
        ("11.3. Khó khăn về rò rỉ dữ liệu qua thời gian (Temporal Data Leakage)", "26", 2),
        ("11.4. Lỗi xử lý ngoại lai sinh hiệu (SpO₂ > 100%)", "26", 2),
        ("11.5. Lỗi thiếu thư viện và xung đột môi trường Python", "26", 2),
        ("11.6. Khó khăn khi đóng gói Docker và tối ưu độ trễ CPU (~3.6s)", "27", 2),
        ("11.7. Lỗi giao diện và đồng bộ trạng thái Web CDSS", "27", 2),
        ("11.8. Bài học về việc không ngụy tạo kết quả ghép cặp giả mạo", "27", 2),
        ("11.9. Giới hạn khi chỉ đánh giá trên tập pilot 50 bệnh nhân", "28", 2),
        ("11.10. Thách thức kiểm định y khoa khi chưa có bác sĩ thực nghiệm độc lập", "28", 2),
        ("11.11. Khắc phục lỗi giao tiếp giữa các tác tử đa tác tử ở giai đoạn đầu", "28", 2),
        ("CHƯƠNG 12: KIỂM THỬ SẢN PHẨM VÀ THẢO LUẬN", "29", 1),
        ("12.1. Các bước đã kiểm thử và xác nhận (39/39 Automated Unit Tests)", "29", 2),
        ("12.2. Bàn luận về đạo đức y tế và bảo vệ quyền riêng tư người bệnh (HIPAA/GDPR)", "30", 2),
        ("CHƯƠNG 13: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", "31", 1),
        ("13.1. Tóm tắt các đóng góp chính của đề tài", "31", 2),
        ("13.2. Hướng phát triển trong tương lai", "31", 2),
        ("TÀI LIỆU THAM KHẢO", "32", 1)
    ]

    for title, page_str, level in toc_entries:
        add_toc_line(doc, title, page_str, level)

    doc.add_page_break()

    # =========================================================================
    # DANH MỤC CÁC BẢNG
    # =========================================================================
    add_heading_1(doc, "DANH MỤC CÁC BẢNG")
    tables_list = [
        ("Bảng 1.1", "Đặc tả kỹ thuật ma trận Input - Method - Output 4 thành phần hệ thống MedFusion-CXR"),
        ("Bảng 3.1", "Kế toán và thống kê 2 tập dữ liệu độc lập (NIH ChestX-ray14 & Clinical Tabular)"),
        ("Bảng 5.1", "Thông số kỹ thuật chi tiết nhánh Thị giác DenseNet-121"),
        ("Bảng 5.2", "Cấu trúc thành phần vector biểu diễn lâm sàng Fc 32 chiều"),
        ("Bảng 5.3", "So sánh 3 phương pháp hợp nhất: Late Decision, Concatenation và Gated Feature Fusion"),
        ("Bảng 7.1", "Danh sách các RESTful API endpoints cốt lõi của MedFusion-CXR"),
        ("Bảng 8.1", "Ma trận thiết kế thực nghiệm độc lập từ E1 đến E5"),
        ("Bảng 9.1", "Kết quả thực nghiệm E2: So sánh hiệu năng XGBoost và 5 mô hình cơ sở trên 240 mẫu test"),
        ("Bảng 9.2", "Kết quả thực nghiệm E1: Đánh giá DenseNet-121 trên 1.314 ảnh test NIH-14 và 95% CI Bootstrap"),
        ("Bảng 12.1", "Kết quả kiểm thử tự động 39/39 Unit Tests của hệ sinh thái MedFusion-CXR")
    ]
    for b_id, b_name in tables_list:
        add_bullet(doc, f"{b_id}: {b_name}")

    # =========================================================================
    # DANH MỤC CÁC HÌNH
    # =========================================================================
    add_heading_1(doc, "DANH MỤC CÁC HÌNH")
    figures_list = [
        ("Hình 6.1", "Sơ đồ kiến trúc tổng thể 4 giai đoạn hệ thống MedFusion-CXR"),
        ("Hình 6.2", "Quy trình điều phối tuần hoàn của 5 Agent y khoa chuyên biệt tại mốc t0"),
        ("Hình 8.1", "Đặc trưng quan trọng cá thể hóa toàn cục của nhánh lâm sàng bằng TreeSHAP"),
        ("Hình 9.1", "Biểu đồ so sánh hiệu năng 6 thuật toán học máy trên tập test lâm sàng 240 mẫu"),
        ("Hình 9.2", "Biểu đồ Forest Plot thể hiện điểm AUROC và khoảng tin cậy 95% của 14 bệnh lý NIH-14"),
        ("Hình 10.1", "Giao diện Hệ thống Hỗ trợ Quyết định Lâm sàng CDSS Web Platform tại localhost:8000")
    ]
    for h_id, h_name in figures_list:
        add_bullet(doc, f"{h_id}: {h_name}")

    # =========================================================================
    # TÓM TẮT ĐỀ TÀI (ABSTRACT)
    # =========================================================================
    add_heading_1(doc, "TÓM TẮT ĐỀ TÀI (ABSTRACT)")
    add_body(doc, "Trong môi trường hồi sức cấp cứu (ICU) và cấp cứu ngoại viện, áp lực thời gian vàng đòi hỏi bác sĩ phải đưa ra quyết định can thiệp chính xác chỉ trong vài phút. Tuy nhiên, các hệ thống AI y tế hiện nay hầu hết vận hành ở dạng đơn phương thức (Single-modality), dẫn đến thế bế tắc lâm sàng: Nhánh thị giác (Vision-only) chỉ nhìn thấy hình thái phổi trên phim X-quang nhưng hoàn toàn mù sinh hiệu; ngược lại, nhánh lâm sàng (Clinical-only) chỉ thấy rối loạn sinh lý cấp tính nhưng thiếu bằng chứng hình thái để phân biệt các hội chứng có biểu hiện tương đồng (như Viêm phổi thùy và Suy tim xung huyết).")
    add_body(doc, "Đề tài 'MedFusion-CXR' đề xuất một kiến trúc hệ thống hỗ trợ quyết định lâm sàng (CDSS) đa phương thức, đa tác tử hoàn chỉnh nhằm khắc phục triệt để khoảng trống công nghệ này. Hệ thống gồm hai nhánh mô hình chuyên biệt: (1) Nhánh Thị giác ứng dụng mạng nơ-ron tích chập DenseNet-121 kế thừa trọng số y khoa TorchXRayVision (nih14) để phát hiện 14 bệnh lý phổi và trích xuất vector đặc trưng F_v in R^1024; (2) Nhánh Lâm sàng ứng dụng thuật toán cây tăng cường gradient XGBoost tích hợp giải thích cục bộ TreeSHAP để lượng giá nguy cơ can thiệp y tế và xây dựng vector biểu diễn lâm sàng F_c in R^32. Hai nguồn đặc trưng được chiếu về không gian tiềm ẩn chung R^256 và dung hòa thông minh qua cơ chế cổng Gated Multimodal Unit (GMU), đi kèm cơ chế Fallback an toàn tự động chuyển 100% sang sinh hiệu khi thiếu ảnh.")
    add_body(doc, "Toàn bộ hệ thống được điều phối bởi 5 tác tử AI chuyên trách (Multi-Agent Swarm) với cơ chế gác cổng TemporalLeakageGuard bảo đảm zero data leakage tại mốc t_0, bóc tách bệnh sử tiếng Việt bằng PhoBERT, đối chiếu phác đồ Bệnh viện Bạch Mai kèm mã ICD-10 và đóng gói hoàn thiện trong Docker Container với giao diện Web CDSS trực quan, đạt độ trễ suy luận chỉ 3.6 giây trên CPU.")

    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 1: BỐI CẢNH, MỤC TIÊU VÀ PHẠM VI
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 1: BỐI CẢNH, MỤC TIÊU VÀ PHẠM VI")
    add_heading_2(doc, "1.1. Bối cảnh nghiên cứu và nỗi đau thực tế (Clinical Pain Points)")
    add_body(doc, "Cấp cứu hô hấp là một trong những tình huống lâm sàng khẩn cấp và căng thẳng nhất tại các cơ sở y tế. Khi bệnh nhân nhập viện trong tình trạng khó thở cấp tính, suy hô hấp diễn tiến nhanh chóng chỉ trong vòng vài phút đến vài chục phút. Bác sĩ trực cấp cứu phải đối mặt với áp lực thời gian cực lớn ('thời gian vàng'), trong khi nguồn dữ liệu người bệnh thường phân mảnh giữa khoa Chẩn đoán hình ảnh và khoa Cấp cứu - Hồi sức tích cực.")
    add_body(doc, "Mặc dù Trí tuệ Nhân tạo (AI) đã đạt được những bước tiến vượt bậc trong phân tích ảnh X-quang phổi và dự đoán diễn tiến bệnh án, hầu hết các giải pháp hiện tại đều vận hành độc lập theo từng phương thức đơn lẻ:")
    add_bullet(doc, "Nhận diện tổn thương thâm nhiễm hoặc tràn dịch trên phim chụp nhưng hoàn toàn không biết bệnh nhân có đang sốt 39.5°C, huyết áp có tụt sốc hay SpO₂ có giảm sâu hay không. Hậu quả là AI có thể cảnh báo quá mức với một nốt xơ phổi cũ lành tính nhưng lại bỏ qua tình trạng suy hô hấp cấp đe dọa tính mạng.", "Bế tắc của AI Thị giác (Vision-only): ")
    add_bullet(doc, "Nhận biết rõ các dấu hiệu rối loạn huyết động và chỉ số cấp cứu nguy kịch, nhưng thiếu bằng chứng hình thái thực thể trong lồng ngực. Điều này khiến mô hình gặp khó khăn lớn trong việc phân biệt các bệnh lý có triệu chứng lâm sàng tương đồng như Viêm phổi nặng và Phù phổi cấp do suy tim.", "Bế tắc của AI Lâm sàng (Clinical-only): ")
    add_body(doc, "Chính sự phân mảnh này tạo ra khoảng trống công nghệ lớn: Y học cấp cứu đòi hỏi một hệ thống có khả năng dung hòa đồng thời cả hình thái tổn thương phổi và diễn tiến sinh lý cấp tính để đưa ra khuyến nghị toàn diện, minh bạch và an toàn.")

    add_heading_2(doc, "1.2. Mục tiêu kỹ thuật và phạm vi đề tài")
    add_body(doc, "Đề tài MedFusion-CXR được thiết kế nhằm đạt được 4 mục tiêu kỹ thuật cốt lõi:")
    add_bullet(doc, "Xây dựng nhánh phân tích hình thái phổi dựa trên kiến trúc DenseNet-121, phân loại 14 bệnh lý lồng ngực chuẩn NIH-14 và trích xuất vector đặc trưng thị giác F_v in R^1024.", "Mục tiêu 1 (Thị giác): ")
    add_bullet(doc, "Xây dựng nhánh phân tích sinh hiệu cấp cứu dựa trên XGBoost kết hợp cơ chế giải thích TreeSHAP, phân tầng nguy cơ can thiệp y tế và kiến tạo vector biểu diễn lâm sàng F_c in R^32.", "Mục tiêu 2 (Lâm sàng): ")
    add_bullet(doc, "Hiện thực hóa cơ chế cổng thích ứng Gated Multimodal Unit (GMU) trên không gian R^256, tự động cân bằng tỷ trọng đóng góp thông tin giữa ảnh và sinh hiệu, đồng thời kích hoạt cơ chế Fallback an toàn khi khuyết thiếu ảnh X-quang.", "Mục tiêu 3 (Hợp nhất GMU): ")
    add_bullet(doc, "Thiết kế hệ điều phối 5 tác tử y khoa (Multi-Agent Swarm) với cơ chế kiểm duyệt mốc t_0 chống rò rỉ dữ liệu, bóc tách thực thể tiếng Việt qua PhoBERT, tích hợp cơ sở tri thức RAG phác đồ Bệnh viện Bạch Mai, mã hóa ICD-10 và đóng gói hoàn chỉnh bằng Docker Container.", "Mục tiêu 4 (Điều phối & Ứng dụng): ")
    add_body(doc, "Bảng 1.1 tóm lược ma trận đặc tả kỹ thuật Input - Method - Output xuyên suốt 4 tầng của đề tài:")

    # Table 1.1
    t1_headers = ["Thành phần", "Input đầu vào cụ thể", "Phương pháp cốt lõi", "Output đầu ra cụ thể"]
    t1_widths = [1.5, 1.8, 1.8, 1.8]
    t1_data = [
        ["Nhánh Thị giác (Vision Branch)", "Phim CXR Grayscale 224×224 chuẩn hóa [-1024, +1024]", "DenseNet-121 (TorchXRayVision nih14)", "Vector F_v ∈ ℝ¹⁰²⁴, 14 xác suất bệnh lý NIH-14"],
        ["Nhánh Lâm sàng (Clinical Branch)", "11 biến sinh hiệu/thể trạng + 8 chỉ số cấp cứu", "XGBoost Classifier + TreeSHAP XAI", "Vector F_c ∈ ℝ³², Xác suất P(treatment), 11 giá trị SHAP"],
        ["Tầng Hợp nhất (Gated Fusion)", "Vector F_v ∈ ℝ¹⁰²⁴ và Vector F_c ∈ ℝ³²", "Linear + LayerNorm + GELU + Sigmoid Gate", "Vector hợp nhất F_fusion ∈ ℝ²⁵⁶, Tỷ trọng g (48/52)"],
        ["Lớp Điều phối (Multi-Agent)", "Dữ liệu tiếp nhận ban đầu mốc t₀, Bệnh sử text", "5 Agent Swarm + PhoBERT + RAG Bạch Mai", "Báo cáo CDSS, Phân tầng nguy cơ, Gợi ý thuốc, ICD-10"]
    ]
    t1 = doc.add_table(rows=len(t1_data)+1, cols=4)
    format_table(t1, t1_widths, t1_headers, t1_data)

    # =========================================================================
    # CHƯƠNG 2: NỀN TẢNG LÝ THUYẾT
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 2: NỀN TẢNG LÝ THUYẾT")
    add_heading_2(doc, "2.1. Tổng quan các phương pháp học sâu và học máy trong Y tế")
    add_body(doc, "Trong chẩn đoán hình ảnh lồng ngực, mạng nơ-ron tích chập (Convolutional Neural Networks - CNN) đã chứng minh ưu thế vượt trội trong việc tự động phát hiện các mẫu hình bệnh học phức tạp như đám mờ phế nang, thâm nhiễm kẽ, tràn dịch và vôi hóa. Ngược lại, đối với dữ liệu dạng bảng (Tabular clinical data) bao gồm các biến đo lường sinh học, huyết động và xét nghiệm sinh hóa, các thuật toán cây quyết định tăng cường gradient (Gradient Boosted Decision Trees - GBDT), tiêu biểu là XGBoost, luôn vượt trội hơn mạng nơ-ron sâu nhờ khả năng xử lý tốt mối quan hệ phi tuyến, dữ liệu khuyết thiếu và không bị ảnh hưởng tiêu cực bởi phân phối lệch.")

    add_heading_2(doc, "2.2. Trích xuất đặc trưng và biểu diễn vector đa phương thức")
    add_body(doc, "Thách thức lớn nhất trong AI y tế đa phương thức là sự bất đồng nhất về bản chất và số chiều giữa các nguồn dữ liệu. Dữ liệu hình ảnh ban đầu là ma trận điểm ảnh không gian hai chiều, trong khi dữ liệu sinh hiệu là các đại lượng vật lý vô hướng rời rạc. Để có thể kết hợp, hai nguồn dữ liệu phải được trích xuất thành các vector biểu diễn đặc trưng tiềm ẩn (Latent feature embeddings) nắm giữ thông tin ngữ nghĩa cốt lõi của ca bệnh.")

    add_heading_2(doc, "2.3. Mạng tích chập DenseNet-121, XGBoost và PhoBERT")
    add_body(doc, "Kiến trúc DenseNet-121 (Dense Convolutional Network) giải quyết triệt để vấn đề suy giảm gradient (vanishing-gradient) và suy giảm đặc trưng thông qua cơ chế kết nối dày đặc (Dense Connectivity):")
    add_body(doc, "x_l = H_l([x_0, x_1, ..., x_{l-1}])", bold_prefix="Công thức kết nối DenseNet: ", italic=True)
    add_body(doc, "Mỗi tầng tích chập nhận đầu vào là sự kết hợp nối tiếp (concatenation) của toàn bộ các bản đồ đặc trưng sinh ra từ các tầng trước đó, giúp tối đa hóa khả năng tái sử dụng đặc trưng hình thái.")
    add_body(doc, "Thuật toán XGBoost tối ưu hóa hàm mục tiêu bậc hai kết hợp giữa hàm mất mát (loss function) và thành phần kiểm soát độ phức tạp mô hình (regularization), giúp ngăn chặn overfitting trên dữ liệu sinh hiệu cấp cứu.")
    add_body(doc, "Mô hình PhoBERT (Pre-trained language models for Vietnamese) dựa trên kiến trúc RoBERTa, được tiền huấn luyện trên 20GB văn bản tiếng Việt, đóng vai trò trích xuất thực thể tên riêng (NER) đối với các triệu chứng lâm sàng và mốc thời gian trong bệnh án.")

    add_heading_2(doc, "2.4. Cơ chế hợp nhất đặc trưng Gated Multimodal Unit (GMU)")
    add_body(doc, "So với phương pháp cộng xác suất ở đầu ra (Late Decision Fusion) hay ghép nối vector thô (Concatenation), cơ chế Gated Multimodal Unit (GMU) mang lại khả năng điều tiết trọng số thích ứng theo từng chiều đặc trưng:")
    add_bullet(doc, "F̃_v = GELU(LayerNorm(W_v · F_v + b_v)) ∈ ℝ²⁵⁶  (từ F_v ∈ ℝ¹⁰²⁴)", "1. Chiếu đặc trưng Thị giác: ")
    add_bullet(doc, "F̃_c = GELU(LayerNorm(W_c · F_c + b_c)) ∈ ℝ²⁵⁶  (từ F_c ∈ ℝ³²)", "2. Chiếu đặc trưng Lâm sàng: ")
    add_bullet(doc, "g = σ(W_g · [F̃_v ; F̃_c] + b_g) ∈ (0, 1)²⁵⁶", "3. Cổng thích ứng Sigmoid: ")
    add_bullet(doc, "F_fusion = g ⊙ F̃_v + (1 − g) ⊙ F̃_c ∈ ℝ²⁵⁶", "4. Vector hợp nhất cuối cùng: ")
    add_body(doc, "Trong đó phép ⊙ là tích Hadamard (element-wise product). Khi khuyết thiếu ảnh X-quang, cổng tự động đóng về 0 (g = 0), công thức suy biến thành F_fusion = F̃_c, bảo đảm hệ thống vận hành an toàn ở chế độ Fallback.")

    # =========================================================================
    # CHƯƠNG 3: KHẢO SÁT DỮ LIỆU VÀ CHUẨN BỊ MÔI TRƯỜNG
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 3: KHẢO SÁT DỮ LIỆU VÀ CHUẨN BỊ MÔI TRƯỜNG")
    add_heading_2(doc, "3.1. Nguồn dữ liệu ban đầu và cam kết liêm chính khoa học")
    add_body(doc, "Hệ thống MedFusion-CXR sử dụng hai tập dữ liệu độc lập chuẩn mực được kiểm soát nguồn gốc nghiêm ngặt:")
    add_bullet(doc, "Bao gồm 112.120 ảnh X-quang ngực chụp sau-trước (PA) và trước-sau (AP) của 30.805 bệnh nhân duy nhất từ Viện Y tế Quốc gia Hoa Kỳ (NIH Clinical Center, Wang et al., CVPR 2017). Tập dữ liệu được gán nhãn 14 bệnh lý lồng ngực thông qua xử lý ngôn ngữ tự nhiên từ báo cáo X-quang gốc.", "1. Tập dữ liệu NIH ChestX-ray14: ")
    add_bullet(doc, "Bao gồm 1.600 hồ sơ bệnh nhân (1.500 mẫu mô phỏng sinh lý cấp cứu dựa trên nguyên lý huyết động và 100 ca lâm sàng trích xuất thực tế), ghi nhận 11 biến thể trạng và sinh hiệu cấp cứu cùng nhãn mục tiêu treatment_needed (tỷ lệ dương tính toàn cục 42.25%).", "2. Tập dữ liệu Clinical Tabular Dataset: ")
    add_body(doc, "Cam kết liêm chính phương pháp luận: Do hai tập dữ liệu này không chung mã bệnh nhân, nhóm nghiên cứu tuyệt đối không ghép cặp giả tạo giữa ảnh của bệnh nhân này với sinh hiệu của bệnh nhân khác để báo cáo kết quả ảo. Hai nhánh được huấn luyện và đánh giá hoàn toàn độc lập trên tập test riêng biệt.")

    # Table 3.1
    t3_headers = ["Tiêu chí quản trị", "Dataset 1: NIH ChestX-ray14", "Dataset 2: Clinical Tabular"]
    t3_widths = [2.2, 2.4, 2.4]
    t3_data = [
        ["Nguồn gốc & Đơn vị", "NIH Clinical Center (Wang et al., CVPR 2017)", "Mô phỏng sinh lý (1.500) + Bệnh án (100)"],
        ["Tổng quy mô mẫu", "112.120 ảnh X-quang / 30.805 bệnh nhân", "1.600 hồ sơ bệnh nhân dạng bảng"],
        ["Định dạng Input", "Ảnh Grayscale PNG (Kích thước gốc 1024×1024)", "11 trường số và phân loại (CSV format)"],
        ["Nhãn mục tiêu (Target)", "14 nhãn bệnh lý lồng ngực nhị phân {0, 1}", "Nhãn nhị phân treatment_needed (42.25% dương)"],
        ["Phân chia tập (Splits)", "Train: 73.089 | Val: 13.435 | Test: 25.596 ảnh", "Train: 1.120 (70%) | Val: 240 (15%) | Test: 240 (15%)"],
        ["Kiểm soát rò rỉ bệnh nhân", "Patient-Level Split (Overlap bệnh nhân = 0)", "Stratified Split theo nhãn, scaler fit trên Train"]
    ]
    t3 = doc.add_table(rows=len(t3_data)+1, cols=3)
    format_table(t3, t3_widths, t3_headers, t3_data)

    add_heading_2(doc, "3.2. Chuẩn bị phần cứng, môi trường và thư viện phần mềm")
    add_body(doc, "Môi trường thực nghiệm được xây dựng trên hệ thống máy trạm tiêu chuẩn: CPU Intel Core i7 / AMD Ryzen, RAM 16GB, không đòi hỏi GPU cao cấp. Môi trường phần mềm bao gồm Python 3.11, PyTorch 2.2, TorchXRayVision 1.5.5, XGBoost 2.0, Scikit-learn 1.4, Transformers 4.38, FastAPI 0.110, Uvicorn và Docker Engine 25.0.")

    add_heading_2(doc, "3.3. Tổ chức cấu trúc thư mục dự án chuẩn hóa")
    add_body(doc, "Dự án được cấu trúc theo mô hình phân lớp rõ ràng (Clean Architecture):")
    add_bullet(doc, "src/api/: Các router điều phối RESTful API (multimodal, vision, clinical, treatment, fhir, xai).")
    add_bullet(doc, "src/vision/: Kiến trúc mạng DenseNet-121, checkpoint adapter, khối Gated Feature Fusion.")
    add_bullet(doc, "src/services/: Dịch vụ nạp mô hình XGBoost, tính toán TreeSHAP và động học sinh lý.")
    add_bullet(doc, "src/agents/: 5 Agent y khoa chuyên biệt và tác tử gác cổng TemporalLeakageGuard.")
    add_bullet(doc, "web/: Giao diện Web Platform CDSS, mã nguồn HTML5, CSS3 hiện đại và JavaScript thuần.")

    # =========================================================================
    # CHƯƠNG 4: TIỀN XỬ LÝ DỮ LIỆU VÀ QUẢN LÝ TẬP DỮ LIỆU
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 4: TIỀN XỬ LÝ DỮ LIỆU VÀ QUẢN LÝ TẬP DỮ LIỆU")
    add_heading_2(doc, "4.1. Tiền xử lý ảnh X-quang: Chuẩn hóa dải Hounsfield và kích thước 224×224")
    add_body(doc, "Hình ảnh X-quang ngực trải qua pipeline tiền xử lý 4 bước nghiêm ngặt:")
    add_bullet(doc, "Chuyển đổi ảnh màu RGB sang 1 kênh xám (Grayscale), triệt tiêu các nhiễu màu nhân tạo do chụp màn hình hoặc nén ảnh.")
    add_bullet(doc, "Rescale cường độ điểm ảnh về thang đo đơn vị Hounsfield [-1024.0, +1024.0] theo công thức X_norm = (X / 255.0) * 2048.0 - 1024.0, tương thích tuyệt đối với checkpoint TorchXRayVision.")
    add_bullet(doc, "Cắt tâm vuông (Center-Crop) và resize về kích thước tiêu chuẩn 224 × 224 pixels.")
    add_bullet(doc, "Đóng gói thành Tensor PyTorch kích thước (1, 1, 224, 224) sẵn sàng đưa vào DenseNet-121.")

    add_heading_2(doc, "4.2. Tiền xử lý dữ liệu lâm sàng và bóc tách thực thể tiếng Việt (PhoBERT NER)")
    add_body(doc, "Dữ liệu sinh hiệu lâm sàng được làm sạch ngoại lai (outliers): Giá trị độ bão hòa oxy SpO₂ đọc lỗi vượt ngưỡng (như 102.7%) được cắt về thang chuẩn [50.0, 100.0]%. Các giá trị khuyết thiếu được điền bằng giá trị trung vị (median) của tập Train.")
    add_body(doc, "Đặc biệt, lớp chuẩn hóa StandardScaler chỉ được fit nghiêm ngặt trên 1.120 mẫu của tập Train. Tập Validation và Test chỉ áp dụng hàm transform, ngăn chặn tuyệt đối hiện tượng rò rỉ phân phối (data distribution leakage).")
    add_body(doc, "Đối với phần bệnh sử tự do tiếng Việt, tác tử TriageTimelineAgent sử dụng mô hình PhoBERT kết hợp bộ từ điển y khoa để nhận diện thực thể triệu chứng, tiền sử bệnh nền và thời gian khởi phát.")

    add_heading_2(doc, "4.3. Phân chia Train/Validation/Test và 3 tầng phòng chống rò rỉ dữ liệu")
    add_body(doc, "Hệ thống thiết lập 3 tầng kiểm soát rò rỉ dữ liệu (Data Leakage Guardrails) chưa từng có trong các đồ án thông thường:")
    add_bullet(doc, "Gao tập mã bệnh nhân giữa các tập Train, Validation và Test bằng rỗng (train ∩ val ∩ test = ∅). Toàn bộ ảnh của cùng một người bệnh chỉ được phép nằm trong duy nhất một tập, loại bỏ triệt để hiện tượng mô hình học thuộc đặc điểm cá nhân của bệnh nhân.", "Tầng 1 (Patient-Level Split): ")
    add_bullet(doc, "Ngưỡng quyết định F1 tối ưu chỉ được tìm kiếm trên tập Validation. Ngưỡng này được 'đóng băng' cố định khi đánh giá trên tập Test độc lập, không cho phép tối ưu ngược từ tập Test.", "Tầng 2 (Frozen Validation Threshold): ")
    add_bullet(doc, "Tác tử TemporalLeakageGuard kiểm tra tem thời gian của toàn bộ hồ sơ, loại bỏ ngay lập tức bất kỳ dữ liệu xét nghiệm hay ghi chú nào phát sinh sau thời điểm tiếp nhận khám ban đầu t₀.", "Tầng 3 (Temporal Guardrail mốc t₀): ")

    # =========================================================================
    # CHƯƠNG 5: KHẢO SÁT VÀ XÂY DỰNG CÁC NHÁNH MÔ HÌNH
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 5: KHẢO SÁT VÀ XÂY DỰNG CÁC NHÁNH MÔ HÌNH")
    add_heading_2(doc, "5.1. Nhánh Thị giác: DenseNet-121 và trích xuất vector đặc trưng F_v ∈ ℝ¹⁰²⁴")
    add_body(doc, "Mô hình thị giác kế thừa checkpoint 'densenet121-res224-nih' (SHA-256: 31a5475c...) của TorchXRayVision. Trước lớp phân loại Fully-Connected cuối cùng, hệ thống trích xuất tầng Global Average Pooling để thu về vector đặc trưng thị giác F_v in R^1024. Vector này mã hóa toàn bộ thông tin hình thái học của lồng ngực và được chuẩn hóa trước khi chuyển giao sang tầng Gated Fusion.")

    add_heading_2(doc, "5.2. Nhánh Lâm sàng: XGBoost + TreeSHAP và xây dựng vector biểu diễn F_c ∈ ℝ³²")
    add_body(doc, "Mô hình XGBoost được huấn luyện với 160 cây quyết định, độ sâu tối đa 4, tốc độ học 0.06 và hệ số subsample 0.88. Để tương thích với mô hình Deep Learning, vector lâm sàng F_c in R^32 được cấu trúc thành 4 khối dữ liệu:")
    add_bullet(doc, "Khối 1 (Chiều 0 đến 10): 11 biến thể trạng và sinh hiệu đã chuẩn hóa qua StandardScaler.")
    add_bullet(doc, "Khối 2 (Chiều 11 đến 21): 11 giá trị giải thích TreeSHAP đóng góp cá thể hóa của từng biến.")
    add_bullet(doc, "Khối 3 (Chiều 22 đến 23): Điểm độ lệch cơ sở (base bias) và logit xác suất thô của XGBoost.")
    add_bullet(doc, "Khối 4 (Chiều 24 đến 31): 8 chỉ số cấp cứu động học (Nhịp thở, Nhiệt độ, Bạch cầu, CRP, Troponin, điểm qSOFA, Shock Index và số lượng triệu chứng).")

    add_heading_2(doc, "5.3. Tầng Hợp nhất Gated Feature Fusion (GMU ℝ²⁵⁶) và cơ chế Fallback an toàn")
    add_body(doc, "Cơ chế GMU giải quyết bài toán lệch số chiều giữa 1024 chiều của ảnh và 32 chiều của lâm sàng. Nếu ghép nối thô, ảnh sẽ chiếm 97% thông tin và lấn át hoàn toàn sinh hiệu. GMU chiếu cả hai về không gian tiềm ẩn R^256, dùng cổng Sigmoid để cân bằng độ mạnh tín hiệu. Khi không có ảnh X-quang, cổng tự động đóng về 0, chuyển 100% sang sinh hiệu mà không gây dừng hệ thống.")

    # =========================================================================
    # CHƯƠNG 6: HỆ THỐNG ĐIỀU PHỐI ĐA TÁC TỬ (MULTI-AGENT SWARM) VÀ RAG
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 6: HỆ THỐNG ĐIỀU PHỐI ĐA TÁC TỬ (MULTI-AGENT SWARM) VÀ RAG")
    add_heading_2(doc, "6.1. Luồng tiếp nhận đầu vào tại mốc t₀ (TemporalLeakageGuard & TriageTimelineAgent)")
    add_body(doc, "Khi một ca bệnh mới được tiếp nhận, dữ liệu bao gồm ảnh X-quang, bảng sinh hiệu và bệnh sử tự do được chuyển đến tác tử bảo vệ TemporalLeakageGuard để rà soát tem thời gian. Đồng thời, tác tử TriageTimelineAgent kích hoạt PhoBERT NER để bóc tách các thực thể bệnh học tiếng Việt.")

    add_heading_2(doc, "6.2. Phân công song song chuyên biệt (CXRDiagnosticAgent & VitalLabDynamicsAgent)")
    add_body(doc, "Hai tác tử chuyên khoa vận hành song song bất đồng bộ:")
    add_bullet(doc, "CXRDiagnosticAgent: Đóng vai trò Bác sĩ Chẩn đoán hình ảnh ảo, kích hoạt DenseNet-121 trích xuất vector F_v và quét 14 bệnh lý phổi.")
    add_bullet(doc, "VitalLabDynamicsAgent: Đóng vai trò Bác sĩ Cấp cứu ảo, tính toán các chỉ số sốc (qSOFA, Shock Index) và kích hoạt XGBoost trích xuất vector F_c.")

    add_heading_2(doc, "6.3. Hội đồng Gated Fusion (ConsensusOrchestrator) & RAG Phác đồ Bệnh viện Bạch Mai")
    add_body(doc, "Tác tử ConsensusOrchestrator đóng vai trò Chủ tọa Hội đồng hội chẩn, thu thập F_v và F_c đưa qua cổng GMU. Sau đó, hệ thống truy xuất cơ sở tri thức RAG theo Hướng dẫn Chẩn đoán và Điều trị Bệnh nội khoa của Bệnh viện Bạch Mai, gán mã ICD-10 và đưa ra khuyến nghị dùng thuốc chuẩn y khoa.")
    add_figure(doc, "reports/presentation_assets/agent_orchestration_flow.png", "Hình 6.2: Sơ đồ trình tự và luồng điều phối của 5 Agent y khoa chuyên biệt tại mốc tiếp nhận t0")

    # =========================================================================
    # CHƯƠNG 7: THIẾT KẾ KIẾN TRÚC VÀ TRIỂN KHAI HỆ THỐNG
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 7: THIẾT KẾ KIẾN TRÚC VÀ TRIỂN KHAI HỆ THỐNG")
    add_heading_2(doc, "7.1. Sơ đồ kiến trúc tổng thể 4 giai đoạn")
    add_body(doc, "Hệ thống MedFusion-CXR được thiết kế theo quy trình 4 giai đoạn rõ ràng: (1) Thu thập & Tiền xử lý đa nguồn; (2) Phân tích chuyên sâu 2 nhánh; (3) Hợp nhất thích ứng Gated Fusion; (4) Hỗ trợ ra quyết định lâm sàng CDSS.")
    add_figure(doc, "reports/presentation_assets/architecture_flowchart.png", "Hình 7.1: Sơ đồ kiến trúc tổng thể 4 giai đoạn của hệ sinh thái MedFusion-CXR")

    add_heading_2(doc, "7.2. Các endpoint backend API và tích hợp chuẩn y tế FHIR R4")
    add_body(doc, "Backend được xây dựng trên nền tảng FastAPI với tài liệu OpenAPI Swagger tự động. Hệ thống hỗ trợ chuyển đổi dữ liệu lâm sàng sang chuẩn trao đổi thông tin y tế quốc tế HL7 FHIR R4 (Fast Healthcare Interoperability Resources) qua router /api/v1/fhir.")

    add_heading_2(doc, "7.3. Đóng gói hệ thống với Docker Container và Healthcheck")
    add_body(doc, "Toàn bộ hệ thống được đóng gói trong một Docker image duy nhất (medfusion-cxr:latest) dựa trên Python 3.11-slim. Dockerfile tích hợp cơ chế kiểm tra sức khỏe tự động (HEALTHCHECK) kiểm tra định kỳ endpoint /api/v1/health mỗi 30 giây.")

    add_heading_2(doc, "7.4. Giao diện người dùng CDSS Web Platform")
    add_body(doc, "Giao diện Web tương tác tại http://localhost:8000 được thiết kế theo phong cách Dark Clinical Tech hiện đại, hỗ trợ bác sĩ nạp ca bệnh mẫu có sẵn ảnh X-quang thật, hiển thị thước đo đóng góp 48% ảnh / 52% sinh hiệu, nhận định lâm sàng bằng tiếng Việt và 8 thanh biểu đồ TreeSHAP.")

    # =========================================================================
    # CHƯƠNG 8: THIẾT KẾ THÍ NGHIỆM VÀ THƯỚC ĐO
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 8: THIẾT KẾ THÍ NGHIỆM VÀ THƯỚC ĐO")
    add_heading_2(doc, "8.1. Nguyên tắc đối chứng và độc lập thực nghiệm (E1 đến E5)")
    add_body(doc, "Nhóm nghiên cứu thiết lập ma trận 5 thực nghiệm từ E1 đến E5 để đánh giá khách quan từng cấu phần hệ thống:")

    # Table 8.1
    t8_headers = ["Mã", "Tên thực nghiệm", "Tập dữ liệu", "Phương pháp / Mô hình", "Chỉ số đo lường"]
    t8_widths = [0.8, 1.8, 1.8, 1.8, 1.8]
    t8_data = [
        ["E1", "Vision Baseline", "NIH ChestX-ray14 (Test: 1.314 ảnh)", "DenseNet-121 (nih14) Bootstrap 300 lần", "AUROC (95% CI), AUPRC, F1, Brier"],
        ["E2", "Clinical Baseline", "Clinical Tabular (Test: 240 mẫu)", "XGBoost vs 5 Baselines (LightGBM, RF...)", "Accuracy, F1, AUROC, AUPRC, Brier"],
        ["E3", "TreeSHAP XAI", "Clinical Test (240 mẫu)", "XGBoost Booster (pred_contribs=True)", "Global Importance |ϕ_i|, Local Force"],
        ["E4", "Fusion Architecture", "Ca bệnh mẫu lâm sàng", "GMU ℝ²⁵⁶, Fallback Mode verification", "Tensor shape (256-d), Norm L2, Cổng g"],
        ["E5", "Multimodal Efficacy", "Cohort ghép cặp (MIMIC-CXR+IV)", "Huấn luyện có giám sát W_g (Future Work)", "ΔAUROC, ΔF1 trên cùng bệnh nhân"]
    ]
    t8 = doc.add_table(rows=len(t8_data)+1, cols=5)
    format_table(t8, t8_widths, t8_headers, t8_data)

    add_heading_2(doc, "8.2. Các thước đo: Accuracy, Precision, Recall, F1-Score, AUROC, AUPRC, Brier Score")
    add_body(doc, "Hệ thống sử dụng đồng thời F1-Score (đo lường sự cân bằng giữa phát hiện ca nguy cấp và hạn chế báo động giả), AUROC (khả năng phân tách ca bệnh ở mọi ngưỡng cắt), AUPRC (chỉ số vàng khi dữ liệu mất cân bằng) và Brier Score (đo lường sai số bình phương xác suất dự đoán, phản ánh độ tin cậy của xác suất).")

    # =========================================================================
    # CHƯƠNG 9: KẾT QUẢ THỰC NGHIỆM VÀ BIỂU ĐỒ
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 9: KẾT QUẢ THỰC NGHIỆM VÀ BIỂU ĐỒ")
    add_heading_2(doc, "9.1. Kết quả thực nghiệm E2: Nhánh lâm sàng trên tập test 240 mẫu")
    add_body(doc, "Trên tập test 240 mẫu độc lập, mô hình XGBoost đạt hiệu năng vượt trội toàn diện so với 5 thuật toán cơ sở:")

    # Table 9.1
    t91_headers = ["Mô hình", "Accuracy", "F1-Score", "AUROC", "AUPRC", "Brier Score"]
    t91_widths = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    t91_data = [
        ["XGBoost (Đề xuất)", "0.8667", "0.8462", "0.9416", "0.9318", "0.0960"],
        ["LightGBM", "0.8542", "0.8319", "0.9385", "0.9250", "0.1003"],
        ["Random Forest", "0.8375", "0.8111", "0.9382", "0.9304", "0.1042"],
        ["Decision Tree", "0.8583", "0.8416", "0.8927", "0.8605", "0.1149"],
        ["Deep MLP (64×32)", "0.7417", "0.7009", "0.8368", "0.8322", "0.1609"],
        ["Logistic Regression", "0.7333", "0.6991", "0.8412", "0.8413", "0.1537"]
    ]
    t91 = doc.add_table(rows=len(t91_data)+1, cols=6)
    format_table(t91, t91_widths, t91_headers, t91_data)
    add_figure(doc, "reports/presentation_assets/e2_clinical_benchmark.png", "Hình 9.1: Biểu đồ so sánh hiệu năng 6 thuật toán học máy trên tập test lâm sàng 240 mẫu")

    add_heading_2(doc, "9.2. Kết quả thực nghiệm E1: Đánh giá DenseNet-121 trên 1.314 ảnh test NIH-14")
    add_body(doc, "Trên tập test 1.314 ảnh thuộc 50 bệnh nhân độc lập, DenseNet-121 thể hiện khả năng nhận diện tin cậy đối với các tổn thương hình thái lớn:")

    # Table 9.2
    t92_headers = ["Bệnh lý lồng ngực NIH-14", "Số ca dương tính", "Test AUROC", "Khoảng tin cậy 95% (Bootstrap)"]
    t92_widths = [2.5, 1.5, 1.5, 2.0]
    t92_data = [
        ["Cardiomegaly (Tim to)", "79 / 1.314", "0.7541", "[0.636, 0.877]"],
        ["Mass (Khối u phổi)", "133 / 1.314", "0.7481", "[0.621, 0.836]"],
        ["Effusion (Tràn dịch màng phổi)", "268 / 1.314", "0.7049", "[0.632, 0.777]"],
        ["Edema (Phù phổi cấp)", "47 / 1.314", "0.6981", "[0.638, 0.763]"],
        ["Pneumothorax (Tràn khí màng phổi)", "143 / 1.314", "0.6582", "[0.576, 0.740]"],
        ["MACRO TRUNG BÌNH (14 BỆNH LÝ)", "14 Bệnh lý", "0.6482", "Macro AUPRC: 0.1579 | Brier: 0.072"]
    ]
    t92 = doc.add_table(rows=len(t92_data)+1, cols=4)
    format_table(t92, t92_widths, t92_headers, t92_data)
    add_figure(doc, "reports/presentation_assets/e1_nih14_benchmark.png", "Hình 9.2: Biểu đồ thể hiện điểm AUROC và khoảng tin cậy 95% của các bệnh lý lồng ngực trên tập NIH-14")

    add_heading_2(doc, "9.3. Kết quả thực nghiệm E3: Giải thích TreeSHAP trên các ca bệnh")
    add_body(doc, "Phân tích độ quan trọng toàn cục bằng TreeSHAP chỉ ra rằng: Độ bão hòa oxy SpO₂ (|ϕ| = 31.8%), Huyết áp tâm thu (|ϕ| = 24.5%), Điểm sốc Shock Index (|ϕ| = 14.2%) và Nhịp tim (|ϕ| = 11.6%) là 4 yếu tố chi phối mạnh mẽ nhất đến quyết định can thiệp y tế.")
    add_figure(doc, "reports/presentation_assets/e3_treeshap_importance.png", "Hình 9.3: Biểu đồ phân tích độ quan trọng đặc trưng toàn cục bằng TreeSHAP trên nhánh lâm sàng")

    # =========================================================================
    # CHƯƠNG 10: ĐÁNH GIÁ THỰC TẾ LÂM SÀNG VÀ GIỚI HẠN CHỨNG CỨ
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 10: ĐÁNH GIÁ THỰC TẾ LÂM SÀNG VÀ GIỚI HẠN CHỨNG CỨ")
    add_heading_2(doc, "10.1. Quy trình thử nghiệm ca bệnh mẫu lâm sàng")
    add_body(doc, "Hệ thống được kiểm thử thực tế trên 4 ca bệnh lâm sàng mẫu tích hợp sẵn ảnh X-quang thật: (1) Ca Viêm phổi cấp tính; (2) Ca Tim to và suy tim xung huyết; (3) Ca Tràn dịch màng phổi lượng nhiều; (4) Ca Sàng lọc bình thường. Thời gian phản hồi trung bình đo được là 3.6 giây trên CPU tiêu chuẩn.")

    add_heading_2(doc, "10.2. Tình trạng thực tế của dữ liệu chưa ghép cặp (Unpaired cohorts)")
    add_body(doc, "Thừa nhận khoa học khách quan: Tập dữ liệu NIH ChestX-ray14 và Clinical Tabular Dataset là hai tập độc lập, chưa liên kết theo mã bệnh nhân thực tế. Do đó, tầng Gated Fusion hiện vận hành ở chế độ Nguyên mẫu Kiến trúc (Architectural Prototype). Nhóm nghiên cứu không công bố số liệu cải thiện hiệu năng ghép cặp giả mạo khi chưa có tập dữ liệu ghép cặp thực sự.")

    # =========================================================================
    # CHƯƠNG 11: NHẬT KÝ KHÓ KHĂN, LỖI VÀ CÁCH KHẮC PHỤC
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 11: NHẬT KÝ KHÓ KHĂN, LỖI VÀ CÁCH KHẮC PHỤC")
    add_body(doc, "Trong suốt quá trình nghiên cứu và phát triển hệ thống MedFusion-CXR, nhóm đã ghi nhận và giải quyết triệt để 11 khó khăn và lỗi kỹ thuật điển hình:")
    add_bullet(doc, "Tập ảnh NIH có dung lượng hơn 40GB với sự mất cân bằng cực lớn (bệnh Hernia chỉ chiếm 0.2% mẫu). Nhóm đã áp dụng chiến lược phân tầng theo bệnh nhân và sử dụng ngưỡng F1-optimal trên tập Val.", "1. Khó khăn về dung lượng dữ liệu ảnh: ")
    add_bullet(doc, "Vector ảnh 1024 chiều áp đảo hoàn toàn vector sinh hiệu 32 chiều nếu ghép nối trực tiếp. Nhóm đã thiết kế lớp chiếu trung gian đưa cả hai về không gian 256 chiều trước khi đưa vào cổng Sigmoid.", "2. Khó khăn về lệch số chiều: ")
    add_bullet(doc, "Bệnh án y tế có nguy cơ rò rỉ dữ liệu tương lai. Nhóm đã xây dựng tác tử TemporalLeakageGuard chặn toàn bộ biến số sau mốc t₀.", "3. Rò rỉ dữ liệu qua thời gian: ")
    add_bullet(doc, "Dữ liệu sinh hiệu thực tế có trường hợp cảm biến SpO₂ đo lỗi lên 102.7%. Nhóm đã viết hàm kiểm duyệt clip giá trị về [50.0, 100.0]%.", "4. Lỗi xử lý ngoại lai sinh hiệu: ")
    add_bullet(doc, "Xung đột giữa PyTorch, TorchXRayVision và thư viện scikit-learn. Nhóm đã giải quyết bằng việc đóng băng phiên bản trong virtual environment và Dockerfile.", "5. Xung đột môi trường Python: ")
    add_bullet(doc, "Mô hình Deep Learning thường chậm trên máy tính không có GPU. Nhóm đã tối ưu hóa luồng suy luận của DenseNet-121 với kích thước 224×224 xám, đưa độ trễ toàn hệ thống xuống còn 3.6 giây trên CPU.", "6. Tối ưu hóa độ trễ thời gian thực: ")
    add_bullet(doc, "Giao diện ban đầu hiển thị các công thức toán L2-norm gây khó hiểu cho bác sĩ. Nhóm đã thiết kế lại thành thanh đo tỷ lệ phần trăm 48/52 kèm nhận định lâm sàng bằng tiếng Việt dễ hiểu.", "7. Tối ưu trải nghiệm người dùng CDSS: ")
    add_bullet(doc, "Nhiều nghiên cứu trước đây tự ghép ngẫu nhiên ảnh của người A với sinh hiệu người B để khoe kết quả đa phương thức cao. Nhóm đã kiên quyết giữ vững đạo đức khoa học: tách biệt thực nghiệm E1 và E2, công bố trung thực giới hạn kiến trúc.", "8. Giữ vững liêm chính khoa học: ")

    # =========================================================================
    # CHƯƠNG 12: KIỂM THỬ SẢN PHẨM VÀ THẢO LUẬN
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 12: KIỂM THỬ SẢN PHẨM VÀ THẢO LUẬN")
    add_heading_2(doc, "12.1. Các bước đã kiểm thử và xác nhận (39/39 Automated Unit Tests)")
    add_body(doc, "Toàn bộ hệ thống đã vượt qua 39 bài kiểm thử đơn vị tự động (pytest), bao gồm kiểm thử tiền xử lý ảnh, kiểm thử vector lâm sàng 32 chiều, kiểm thử cổng GMU và chế độ Fallback, kiểm thử router API và định dạng chuẩn FHIR R4.")

    add_heading_2(doc, "12.2. Bàn luận về đạo đức y tế và bảo vệ quyền riêng tư người bệnh")
    add_body(doc, "Hệ thống tuân thủ nghiêm ngặt các nguyên tắc bảo mật y tế HIPAA và GDPR. Toàn bộ thông tin định danh cá nhân (PII) đều được ẩn danh hóa trước khi xử lý. Hệ thống định vị rõ ràng là công cụ hỗ trợ quyết định (CDSS), bác sĩ điều trị luôn là người giữ quyền quyết định y khoa cuối cùng.")

    # =========================================================================
    # CHƯƠNG 13: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 13: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    add_heading_2(doc, "13.1. Tóm tắt các đóng góp chính của đề tài")
    add_body(doc, "Đề tài MedFusion-CXR đã hoàn thành xuất sắc 3 đóng góp kỹ thuật trọng tâm:")
    add_bullet(doc, "Xây dựng thành công kiến trúc đa phương thức kết hợp DenseNet-121, XGBoost và cổng GMU R^256 có cơ chế Fallback an toàn.", "1. Đóng góp về Kiến trúc: ")
    add_bullet(doc, "Đánh giá thực nghiệm độc lập trung thực (XGBoost đạt F1 0.8462, AUROC 0.9416; DenseNet-121 đạt Macro AUROC 0.6482) với 3 tầng chống rò rỉ dữ liệu.", "2. Đóng góp về Phương pháp luận: ")
    add_bullet(doc, "Đóng gói hoàn chỉnh sản phẩm dạng Docker Container, tích hợp RESTful API v2.2, hỗ trợ chuẩn FHIR R4 và giao diện Web CDSS trực quan.", "3. Đóng góp về Ứng dụng thực tế: ")

    add_heading_2(doc, "13.2. Hướng phát triển trong tương lai")
    add_body(doc, "Trong giai đoạn tiếp theo, nhóm nghiên cứu đặt mục tiêu:")
    add_bullet(doc, "Kết nối với kho dữ liệu ghép cặp quốc tế MIMIC-CXR và MIMIC-IV để huấn luyện có giám sát ma trận trọng số cổng W_g và thực hiện nghiên cứu bóc tách (Ablation Study).")
    add_bullet(doc, "Tích hợp kỹ thuật giải thích thị giác Grad-CAM / Eigen-CAM để trực quan hóa vùng tổn thương ngay trên phim chụp song song với TreeSHAP lâm sàng.")
    add_bullet(doc, "Tiến hành thử nghiệm lâm sàng đối chứng tại các khoa Cấp cứu để đánh giá mức độ giảm tải áp lực cho y bác sĩ.")

    # =========================================================================
    # TÀI LIỆU THAM KHẢO
    # =========================================================================
    add_heading_1(doc, "TÀI LIỆU THAM KHẢO")
    refs = [
        "[1] Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 2097-2106.",
        "[2] Cohen, J. P., Viviano, J. D., Bertin, P., Paul, P., Bertany, P., et al. (2022). TorchXRayVision: A library of chest X-ray datasets and models. Medical Imaging with Deep Learning (MIDL). arXiv:2111.00596.",
        "[3] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 785-794.",
        "[4] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS 2017), 30.",
        "[5] Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1037-1042.",
        "[6] Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). Gated multimodal units for information fusion. In ICLR 2017 Workshop.",
        "[7] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 4700-4708.",
        "[8] Bệnh viện Bạch Mai. (2022). Hướng dẫn chẩn đoán và điều trị bệnh nội khoa. Nhà xuất bản Y học, Hà Nội."
    ]
    for r in refs:
        add_body(doc, r)

    doc.save(DOCX_OUT_PATH)
    print(f"Successfully generated full report at: {DOCX_OUT_PATH}")

    try:
        shutil.copy2(DOCX_OUT_PATH, DOWNLOADS_OUT_PATH)
        print(f"Successfully copied to: {DOWNLOADS_OUT_PATH}")
    except Exception as e:
        print(f"Note: Could not copy to Downloads: {e}")

if __name__ == "__main__":
    build_full_report()
