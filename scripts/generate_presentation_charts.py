"""
Generates high-resolution dark-themed presentation graphics for MedFusion-CXR slides.
All colors and metrics match the validated results in reports/nih_benchmark_report.json
and models/treatment_prediction/benchmark_report.json.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "reports/presentation_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visual styling constants
DARK_BG = "#0D1527"
CARD_BG = "#131E36"
TEXT_COLOR = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
CYAN = "#06B6D4"
BLUE = "#3B82F6"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#F43F5E"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]


def generate_e2_clinical_chart():
    """Generates grouped bar chart for 6 Clinical Models (E2)."""
    with open("models/treatment_prediction/benchmark_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models_data = data.get("benchmark_results", {})
    # Sort models by F1 score
    sorted_models = sorted(models_data.items(), key=lambda kv: kv[1]["f1_score"], reverse=True)

    names = [kv[0].replace("Deep Neural Network (MLP)", "MLP Neural Net") for kv in sorted_models]
    auroc = [kv[1]["roc_auc"] * 100 for kv in sorted_models]
    auprc = [kv[1]["auprc"] * 100 for kv in sorted_models]
    f1 = [kv[1]["f1_score"] * 100 for kv in sorted_models]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    r1 = ax.bar(x - width, auroc, width, label="AUROC (%)", color=CYAN, edgecolor="#0891b2", linewidth=1.2, zorder=3)
    r2 = ax.bar(x, auprc, width, label="AUPRC (%)", color=BLUE, edgecolor="#2563eb", linewidth=1.2, zorder=3)
    r3 = ax.bar(x + width, f1, width, label="F1-Score (%)", color=EMERALD, edgecolor="#059669", linewidth=1.2, zorder=3)

    # Highlight champion
    ax.axvspan(-0.45, 0.45, color=CYAN, alpha=0.15, zorder=1, label="★ Champion Model")

    ax.set_title("E2: So sánh 6 Mô hình trên Tập Lâm sàng (Held-out Test 240 mẫu)", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="600", color=TEXT_COLOR)
    ax.set_ylim(60, 100)
    ax.set_ylabel("Hiệu năng (%)", fontsize=10, color=TEXT_MUTED)
    ax.grid(axis="y", color="#1E293B", linestyle="--", alpha=0.7, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#1E293B")

    ax.tick_params(colors=TEXT_MUTED, which="both")
    ax.legend(facecolor=DARK_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=9.5, loc="lower right")

    # Value tags for XGBoost
    for bar in [r1[0], r2[0], r3[0]]:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#FFFFFF")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "e2_clinical_benchmark.png")
    plt.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved:", out_path)


def generate_e1_nih14_chart():
    """Generates AUROC bar chart for 14 NIH ChestX-ray14 Pathologies with 95% CIs."""
    with open("reports/nih_benchmark_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    p_res = data.get("per_pathology_results", {})
    sorted_labels = sorted(p_res.items(), key=lambda kv: kv[1]["test_results"]["auroc"] or 0, reverse=True)

    labels = [kv[0].replace("_", " ") for kv in sorted_labels]
    aurocs = [kv[1]["test_results"]["auroc"] for kv in sorted_labels]
    ci_lows = [kv[1]["test_results"]["auroc_ci"][0] for kv in sorted_labels]
    ci_highs = [kv[1]["test_results"]["auroc_ci"][1] for kv in sorted_labels]

    yerr = [
        np.array(aurocs) - np.array(ci_lows),
        np.array(ci_highs) - np.array(aurocs),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    colors = [CYAN if a >= 0.70 else (BLUE if a >= 0.60 else "#64748B") for a in aurocs]
    x = np.arange(len(labels))

    bars = ax.bar(x, aurocs, yerr=yerr, capsize=3.5, color=colors, edgecolor="#0284c7", linewidth=1.1, ecolor=AMBER, zorder=3)

    # Macro mean line
    macro_mean = data.get("macro_summary", {}).get("macro_auroc", 0.6482)
    ax.axhline(macro_mean, color=ROSE, linestyle="--", linewidth=1.8, label=f"Macro AUROC = {macro_mean:.4f} (14 nhãn)", zorder=4)

    ax.set_title("E1: Đánh giá Pretrained DenseNet-121 trên 14 Bệnh lý NIH ChestX-ray14 (Test Set)", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, fontweight="600", color=TEXT_COLOR)
    ax.set_ylim(0.40, 0.95)
    ax.set_ylabel("Test AUROC (Khoảng tin cậy 95%)", fontsize=10, color=TEXT_MUTED)
    ax.grid(axis="y", color="#1E293B", linestyle="--", alpha=0.7, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#1E293B")

    ax.tick_params(colors=TEXT_MUTED, which="both")
    ax.legend(facecolor=DARK_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=10, loc="upper right")

    # Add numeric labels on top 3
    for idx in range(3):
        h = aurocs[idx]
        ax.annotate(f"{h:.3f}", xy=(x[idx], h + 0.02), ha="center", fontsize=8.5, fontweight="bold", color="#38BDF8")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "e1_nih14_benchmark.png")
    plt.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved:", out_path)


def generate_e3_treeshap_chart():
    """Generates horizontal bar chart of TreeSHAP Feature Attributions."""
    with open("models/treatment_prediction/benchmark_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    importances = data.get("feature_importances", {})
    sorted_items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)

    labels = [kv[0] for kv in sorted_items]
    values = [kv[1] * 100 for kv in sorted_items]

    # Invert so highest is at the top
    labels = labels[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    colors = [CYAN if v >= 14.0 else (BLUE if v >= 8.0 else "#475569") for v in values]

    bars = ax.barh(np.arange(len(labels)), values, color=colors, edgecolor="#0284c7", linewidth=1.1, zorder=3, height=0.65)

    ax.set_title("E3: Tỷ trọng Đóng góp Toàn cục TreeSHAP |ϕ_i| (XGBoost Test Split)", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="600", color=TEXT_COLOR)
    ax.set_xlabel("Đóng góp tương đối (%)", fontsize=10, color=TEXT_MUTED)
    ax.set_xlim(0, 30)
    ax.grid(axis="x", color="#1E293B", linestyle="--", alpha=0.7, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#1E293B")

    ax.tick_params(colors=TEXT_MUTED, which="both")

    for bar in bars:
        w = bar.get_width()
        if w > 0.5:
            ax.annotate(f"{w:.1f}%",
                        xy=(w + 0.5, bar.get_y() + bar.get_height() / 2),
                        va="center", fontsize=8.5, fontweight="bold", color="#F8FAFC")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "e3_treeshap_importance.png")
    plt.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved:", out_path)


def generate_architecture_diagram():
    """Generates crystal-clear, high-contrast, beautiful system flowchart with 4 distinct stages.
       Strictly uses intuitive Vietnamese clinical-tech explanations without cluttered math.
    """
    fig, ax = plt.subplots(figsize=(13.333, 6.4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    def draw_card(x, y, w, h, step_tag, title, bullets, border_col, bg_col="#111C35"):
        rect = plt.Rectangle((x, y), w, h, facecolor=bg_col, edgecolor=border_col, linewidth=2.0, zorder=2)
        ax.add_patch(rect)
        # Step Tag Pill
        ax.text(x + 0.015, y + h - 0.035, step_tag, fontsize=8.0, fontweight="bold", color=border_col,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=DARK_BG, edgecolor=border_col, alpha=0.95), zorder=3)
        # Title
        ax.text(x + 0.015, y + h - 0.082, title, fontsize=11.5, fontweight="bold", color="#FFFFFF", zorder=3)
        # Divider Line
        ax.plot([x + 0.015, x + w - 0.015], [y + h - 0.105, y + h - 0.105], color=border_col, lw=0.8, alpha=0.6, zorder=3)
        # Bullets
        for idx, (b_title, b_desc) in enumerate(bullets):
            ax.text(x + 0.015, y + h - 0.145 - (idx * 0.065), f"• {b_title}:", fontsize=9.2, fontweight="bold", color="#F8FAFC", zorder=3)
            ax.text(x + 0.025, y + h - 0.175 - (idx * 0.065), b_desc, fontsize=8.5, color="#94A3B8", zorder=3)

    col_w = 0.222
    y_top = 0.52
    y_bot = 0.06
    card_h = 0.41

    # -------------------------------------------------------------
    # GIAI ĐOẠN 1: DỮ LIỆU ĐẦU VÀO
    # -------------------------------------------------------------
    col1_x = 0.025
    draw_card(col1_x, y_top, col_w, card_h,
              "MODALITY 1 · HÌNH ẢNH", "Phim X-Quang Ngực (CXR)",
              [("Nguồn dữ liệu", "Tập NIH ChestX-ray14 (112.120 ảnh)"),
               ("Định dạng xử lý", "Ảnh xám chuẩn hóa 224x224 pixels"),
               ("Ý nghĩa lâm sàng", "Cung cấp hình thái tổn thương phổi")],
              CYAN)

    draw_card(col1_x, y_bot, col_w, card_h,
              "MODALITY 2 · SINH HIỆU", "Dữ Liệu Lâm Sàng Cấp Cứu",
              [("Nguồn dữ liệu", "1.600 hồ sơ bệnh nhân dạng bảng"),
               ("11 Biến sinh hiệu", "Mạch, Huyết áp, SpO2, Nhịp thở, Sốt..."),
               ("Ý nghĩa lâm sàng", "Cung cấp diễn tiến sinh lý cấp toàn thân")],
              EMERALD)

    # -------------------------------------------------------------
    # GIAI ĐOẠN 2: MÔ HÌNH PHÂN TÍCH CHUYÊN BIỆT
    # -------------------------------------------------------------
    col2_x = 0.275
    draw_card(col2_x, y_top, col_w, card_h,
              "NHÁNH THỊ GIÁC", "DenseNet-121 (121 Tầng)",
              [("Cơ chế phân tích", "Quét tổn thương phổi trên phim X-quang"),
               ("14 Bệnh lý NIH", "Phát hiện Viêm phổi, Tràn dịch, Xẹp phổi..."),
               ("Đặc trưng trích xuất", "Vector đặc trưng thị giác F_v (1024 chiều)")],
              CYAN)

    draw_card(col2_x, y_bot, col_w, card_h,
              "NHÁNH LÂM SÀNG", "XGBoost + TreeSHAP",
              [("Cơ chế phân tích", "Rừng 160 cây quyết định phân tầng"),
               ("Chỉ số cấp cứu", "Định lượng qSOFA, Shock Index, Labs"),
               ("Giải thích nguyên nhân", "TreeSHAP chỉ rõ yếu tố nguy cơ cá thể")],
              EMERALD)

    # -------------------------------------------------------------
    # GIAI ĐOẠN 3: CÂN BẰNG THÔNG MINH
    # -------------------------------------------------------------
    col3_x = 0.525
    draw_card(col3_x, 0.16, 0.218, 0.77,
              "HỢP NHẤT ĐẶC TRƯNG", "Gated Fusion (GMU)",
              [("Cổng thích ứng g", "Tự động cân bằng tỷ trọng 2 nguồn tin"),
               ("Tỷ lệ phân bổ thực tế", "48% Phim X-quang  ·  52% Sinh hiệu"),
               ("Dung hòa thông tin", "Dấu hiệu thâm nhiễm phổi bổ trợ cho"),
               (" ", "tình trạng sốt cao và tụt SpO2"),
               ("Chế độ an toàn Fallback", "Nếu chưa có ảnh CXR, tự động chuyển"),
               (" ", "100% sang sinh hiệu, không gây gián đoạn")],
              AMBER)

    # -------------------------------------------------------------
    # GIAI ĐOẠN 4: HỖ TRỢ RA QUYẾT ĐỊNH
    # -------------------------------------------------------------
    col4_x = 0.768
    draw_card(col4_x, 0.16, 0.210, 0.77,
              "ĐẦU RA HỆ THỐNG", "Hỗ Trợ Quyết Định (CDSS)",
              [("Điều phối 5 Agent", "Kiểm duyệt rò rỉ t0, trích xuất triệu chứng"),
               ("Phân tầng nguy cơ", "Cảnh báo khẩn cấp / Cần can thiệp y tế"),
               ("Phác đồ BV Bạch Mai", "Gợi ý thuốc, hướng xử trí + mã ICD-10"),
               ("Giải thích cho bác sĩ", "Biểu đồ TreeSHAP cá thể từng ca bệnh"),
               ("Báo cáo hội chẩn", "Xuất khuyến nghị toàn diện trên Web UI")],
              BLUE)

    # Connecting Flow Arrows
    def draw_arrow(x1, y1, x2, y2, color):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4, mutation_scale=14), zorder=4)

    # 1A -> 2A
    draw_arrow(col1_x + col_w, y_top + card_h/2, col2_x, y_top + card_h/2, CYAN)
    # 1B -> 2B
    draw_arrow(col1_x + col_w, y_bot + card_h/2, col2_x, y_bot + card_h/2, EMERALD)
    # 2A -> 3
    draw_arrow(col2_x + col_w, y_top + card_h/2, col3_x, 0.65, CYAN)
    # 2B -> 3
    draw_arrow(col2_x + col_w, y_bot + card_h/2, col3_x, 0.38, EMERALD)
    # 3 -> 4
    draw_arrow(col3_x + 0.218, 0.54, col4_x, 0.54, AMBER)

    # Stage Headers on Top
    stage_headers = [
        ("BƯỚC 1: DỮ LIỆU ĐẦU VÀO", col1_x + col_w/2, "#38BDF8"),
        ("BƯỚC 2: PHÂN TÍCH CHUYÊN SÂU", col2_x + col_w/2, "#34D399"),
        ("BƯỚC 3: CÂN BẰNG THÔNG MINH", col3_x + 0.218/2, AMBER),
        ("BƯỚC 4: HỖ TRỢ RA QUYẾT ĐỊNH", col4_x + 0.210/2, "#60A5FA"),
    ]
    for text, cx, color in stage_headers:
        ax.text(cx, 0.965, text, ha="center", va="top", fontsize=10.0, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F172A", edgecolor=color, alpha=0.9))

    # Super Title
    ax.text(0.50, 1.025, "MEDFUSION-CXR · SƠ ĐỒ PIPELINE HOÀN CHỈNH TỪ ĐẦU VÀO ĐẾN HỖ TRỢ LÂM SÀNG",
            ha="center", va="top", fontsize=13.0, fontweight="bold", color="#FFFFFF")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "architecture_flowchart.png")
    plt.savefig(out_path, dpi=240, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved:", out_path)


def generate_agent_orchestration_diagram():
    """Generates sequential multi-agent orchestration diagram showing the step-by-step
       execution pipeline when input arrives at t0.
    """
    fig, ax = plt.subplots(figsize=(13.333, 6.4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    def draw_card(x, y, w, h, step_tag, title, bullets, border_col, bg_col="#111C35"):
        rect = plt.Rectangle((x, y), w, h, facecolor=bg_col, edgecolor=border_col, linewidth=1.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.012, y + h - 0.035, step_tag, fontsize=7.8, fontweight="bold", color=border_col,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=DARK_BG, edgecolor=border_col, alpha=0.95), zorder=3)
        ax.text(x + 0.012, y + h - 0.078, title, fontsize=10.5, fontweight="bold", color="#FFFFFF", zorder=3)
        ax.plot([x + 0.012, x + w - 0.012], [y + h - 0.098, y + h - 0.098], color=border_col, lw=0.8, alpha=0.5, zorder=3)
        for idx, (b_title, b_desc) in enumerate(bullets):
            ax.text(x + 0.012, y + h - 0.135 - (idx * 0.060), f"• {b_title}:", fontsize=8.8, fontweight="bold", color="#F8FAFC", zorder=3)
            ax.text(x + 0.020, y + h - 0.162 - (idx * 0.060), b_desc, fontsize=8.2, color="#94A3B8", zorder=3)

    # 1. Top Input Banner
    banner_rect = plt.Rectangle((0.025, 0.85), 0.950, 0.10, facecolor="#131E36", edgecolor=CYAN, linewidth=1.5, zorder=2)
    ax.add_patch(banner_rect)
    ax.text(0.040, 0.925, "INPUT TIẾP NHẬN TẠI MỐC t₀:", fontsize=10.5, fontweight="bold", color=CYAN, zorder=3)
    ax.text(0.245, 0.925, "1. Phim X-Quang Ngực (CXR)  |  2. 11 Sinh Hiệu & Labs Cấp Cứu  |  3. Bệnh Sử Tiếng Việt Tự Do",
            fontsize=9.8, fontweight="600", color="#FFFFFF", zorder=3)
    ax.text(0.040, 0.885, "Quy trình điều phối 5 Agent chuyên khoa kích hoạt tuần tự theo mốc thời gian thực tiếp nhận bệnh nhân.",
            fontsize=8.2, color="#94A3B8", zorder=3)

    # 4 Columns
    col_w = 0.222
    col1_x = 0.025
    col2_x = 0.275
    col3_x = 0.525
    col4_x = 0.768

    # Column 1: Triage & Guard
    draw_card(col1_x, 0.44, col_w, 0.38,
              "AGENT 1 · BẢO VỆ DỮ LIỆU", "TemporalLeakageGuard",
              [("Kiểm soát mốc t0", "Loại trừ dữ liệu phát sinh sau tiếp nhận"),
               ("Chống rò rỉ", "Bảo đảm zero temporal leakage"),
               ("Chuyển giao", "Cung cấp hồ sơ sạch cho các agent sau")],
              ROSE)

    draw_card(col1_x, 0.04, col_w, 0.38,
              "AGENT 2 · BÓC TÁCH TRIỆU CHỨNG", "TriageTimelineAgent",
              [("PhoBERT NER", "Nhận diện triệu chứng, bệnh nền, thuốc"),
               ("Thời gian bệnh", "Trích xuất mốc khởi phát (duration hours)"),
               ("Chuyển giao", "Bộ thực thể y tế chuẩn hóa ngữ cảnh")],
              CYAN)

    # Column 2: Parallel Specialists
    draw_card(col2_x, 0.44, col_w, 0.38,
              "AGENT 3 · CHẨN ĐOÁN HÌNH ẢNH", "CXRDiagnosticAgent",
              [("DenseNet-121", "Quét tổn thương trên phim X-quang"),
               ("14 Nhãn NIH", "Phát hiện Viêm phổi, Tràn dịch, Xẹp phổi..."),
               ("Đầu ra trích xuất", "Vector thị giác F_v (1024 chiều) + Xác suất")],
              CYAN)

    draw_card(col2_x, 0.04, col_w, 0.38,
              "AGENT 4 · ĐỘNG HỌC LÂM SÀNG", "VitalLabDynamicsAgent",
              [("Động học sinh lý", "Tính điểm qSOFA, Shock Index, Hypoxemia"),
               ("XGBoost + SHAP", "Định lượng xác suất can thiệp y tế"),
               ("Đầu ra trích xuất", "Vector lâm sàng F_c (32 chiều) + Đóng góp")],
              EMERALD)

    # Column 3: Master Orchestrator
    draw_card(col3_x, 0.04, 0.218, 0.78,
              "AGENT 5 · TRƯỞNG BAN HỘI CHẨN", "MultimodalConsensusOrchestrator",
              [("Thu thập đa nguồn", "Nhận đồng thời F_v, F_c và triệu chứng"),
               ("Gated Fusion GMU", "Tự động cân bằng 48% ảnh · 52% sinh hiệu"),
               ("Dung hòa thông tin", "Tổn thương phổi củng cố song song"),
               (" ", "cho tình trạng sốt cao và SpO2 giảm"),
               ("Chế độ Fallback", "Nếu thiếu phim CXR, tự động dựa 100%"),
               (" ", "vào sinh hiệu, không làm gián đoạn cấp cứu"),
               ("Ra quyết định", "Chuyển giao vector F_fusion sang RAG")],
              AMBER)

    # Column 4: Clinical Delivery
    draw_card(col4_x, 0.04, 0.210, 0.78,
              "KẾT QUẢ ĐẦU RA", "Báo Cáo Hội Chẩn CDSS",
              [("Phác đồ Bạch Mai", "RAG truy xuất hướng dẫn điều trị chuẩn"),
               ("Mã bệnh ICD-10", "Gán mã bệnh học quốc tế tương ứng"),
               ("Phân tầng nguy cơ", "Cảnh báo khẩn cấp (Critical) / Ổn định"),
               ("Giải thích bác sĩ", "8 thanh TreeSHAP cá thể hóa từng ca"),
               ("Hiển thị Web UI", "Báo cáo trực quan cho bác sĩ tại"),
               (" ", "http://localhost:8000")],
              BLUE)

    # Flow Arrows
    def draw_arrow(x1, y1, x2, y2, color):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=13), zorder=4)

    # From Banner down to Col 1
    draw_arrow(0.136, 0.86, 0.136, 0.82, CYAN)

    # Col 1 -> Col 2 (Fork)
    draw_arrow(col1_x + col_w, 0.63, col2_x, 0.63, CYAN)
    draw_arrow(col1_x + col_w, 0.23, col2_x, 0.23, EMERALD)

    # Col 2 -> Col 3 (Join)
    draw_arrow(col2_x + col_w, 0.63, col3_x, 0.55, CYAN)
    draw_arrow(col2_x + col_w, 0.23, col3_x, 0.35, EMERALD)

    # Col 3 -> Col 4
    draw_arrow(col3_x + 0.218, 0.43, col4_x, 0.43, AMBER)

    # Super Title
    ax.text(0.50, 1.025, "MEDFUSION-CXR · TRÌNH TỰ ĐIỀU PHỐI 5 AGENT KHI TIẾP NHẬN INPUT ĐA PHƯƠNG THỨC",
            ha="center", va="top", fontsize=12.5, fontweight="bold", color="#FFFFFF")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "agent_orchestration_flow.png")
    plt.savefig(out_path, dpi=240, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    generate_e2_clinical_chart()
    generate_e1_nih14_chart()
    generate_e3_treeshap_chart()
    generate_architecture_diagram()
    generate_agent_orchestration_diagram()
    print("All presentation asset charts successfully generated!")
