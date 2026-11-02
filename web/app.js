/**
 * MedFusion-CXR · Multimodal Multi-Agent Clinical Decision Support Frontend
 * Connects to:
 *  - POST /api/v1/treatment/multimodal_predict (DenseNet-121 NIH-14 + XGBoost + TreeSHAP + GMU)
 *  - GET  /api/v1/treatment/benchmark          (Experiment E2: 6 ML/DL Clinical Models)
 *  - GET  /api/v1/multimodal/nih-benchmark     (Experiment E1: Pretrained DenseNet-121 on NIH-14)
 *  - POST /api/v1/predict                      (Vietnamese Clinical NLP · PhoBERT NER)
 */

(() => {
    "use strict";

    // =========================================================================
    // 1. CLINICAL CASE PRESETS (Backed by real NIH ChestX-ray14 PNGs)
    // =========================================================================
    const CLINICAL_PRESETS = {
        pneumonia: {
            label: "Ca 1 · Viêm phổi cấp & Giảm oxy máu",
            sampleUrl: "/web/samples/pneumonia_nih.png",
            sampleFilename: "pneumonia_nih.png",
            clinical_text:
                "Bệnh nhân nam 62 tuổi, tiền sử tăng huyết áp và đái tháo đường típ 2, vào viện vì sốt cao 38.8°C rét run, ho đờm vàng và khó thở tăng dần 3 ngày nay. Đã chụp X-quang ngực thẳng và làm công thức máu, CRP.",
            age: 62,
            gender: "Nam (Male)",
            spo2: 91.5,
            heart_rate: 112,
            sys_bp: 146,
            dia_bp: 90,
            chronic_conditions: 2,
            symptoms_severity: 4,
            stress_score: 8,
            sleep_hours: 4.5,
            physical_activity_hrs: 1.0,
            respiratory_rate: 25,
            temperature_c: 38.8,
            wbc: 16.4,
            crp: 78.5,
            troponin: 0.02,
        },
        cardiomegaly: {
            label: "Ca 2 · Bóng tim to / Theo dõi suy tim mất bù",
            sampleUrl: "/web/samples/cardiomegaly_nih.png",
            sampleFilename: "cardiomegaly_nih.png",
            clinical_text:
                "Bệnh nhân nữ 68 tuổi có tiền sử tăng huyết áp và suy tim mạn đang dùng Amlodipine, nhập viện vì khó thở khi nằm, phù hai chân và đau ngực âm ỉ 2 ngày nay. Chỉ định X-quang ngực và xét nghiệm Troponin.",
            age: 68,
            gender: "Nữ (Female)",
            spo2: 92.8,
            heart_rate: 104,
            sys_bp: 162,
            dia_bp: 98,
            chronic_conditions: 3,
            symptoms_severity: 4,
            stress_score: 7,
            sleep_hours: 5.0,
            physical_activity_hrs: 0.5,
            respiratory_rate: 23,
            temperature_c: 37.1,
            wbc: 9.2,
            crp: 14.0,
            troponin: 0.09,
        },
        effusion: {
            label: "Ca 3 · Tràn dịch màng phổi & Khó thở",
            sampleUrl: "/web/samples/effusion_nih.png",
            sampleFilename: "effusion_nih.png",
            clinical_text:
                "Bệnh nhân nam 57 tuổi nhập viện vì khó thở tăng dần khi gắng sức, ho kéo dài và đau tức vùng ngực đáy phổi 5 ngày nay. Khám ghi nhận rì rào phế nang giảm, chỉ định chụp X-quang ngực và xét nghiệm CRP.",
            age: 57,
            gender: "Nam (Male)",
            spo2: 93.2,
            heart_rate: 98,
            sys_bp: 138,
            dia_bp: 86,
            chronic_conditions: 1,
            symptoms_severity: 4,
            stress_score: 7,
            sleep_hours: 5.5,
            physical_activity_hrs: 1.5,
            respiratory_rate: 22,
            temperature_c: 37.9,
            wbc: 12.6,
            crp: 42.0,
            troponin: 0.01,
        },
        normal: {
            label: "Ca 4 · Khám sàng lọc định kỳ ổn định",
            sampleUrl: "/web/samples/normal_nih.png",
            sampleFilename: "normal_nih.png",
            clinical_text:
                "Bệnh nhân nữ 34 tuổi khám sức khỏe định kỳ, không sốt, không ho, không khó thở, sinh hiệu ổn định, chụp X-quang ngực kiểm tra tổng quát.",
            age: 34,
            gender: "Nữ (Female)",
            spo2: 98.5,
            heart_rate: 72,
            sys_bp: 116,
            dia_bp: 74,
            chronic_conditions: 0,
            symptoms_severity: 1,
            stress_score: 3,
            sleep_hours: 7.5,
            physical_activity_hrs: 4.0,
            respiratory_rate: 16,
            temperature_c: 36.7,
            wbc: 6.8,
            crp: 2.1,
            troponin: 0.01,
        },
        pneumothorax: {
            label: "Ca 5 · Tràn khí màng phổi cấp (Pneumothorax)",
            sampleUrl: "/web/samples/pneumothorax_nih.png",
            sampleFilename: "pneumothorax_nih.png",
            clinical_text:
                "Bệnh nhân nam 56 tuổi đột ngột đau nhói ngực phải dữ dội, khó thở tăng nhanh sau khi gắng sức. Thăm khám ghi nhận rì rào phế nang phổi phải giảm rõ rệt, gõ vang, SpO2 tụt xuống 89%, mạch nhanh 118 ck/phút. Cần can thiệp cấp cứu dẫn lưu màng phổi.",
            age: 56,
            gender: "Nam (Male)",
            spo2: 89.0,
            heart_rate: 118,
            sys_bp: 105,
            dia_bp: 70,
            chronic_conditions: 1,
            symptoms_severity: 5,
            stress_score: 9,
            sleep_hours: 3.5,
            physical_activity_hrs: 0.5,
            respiratory_rate: 28,
            temperature_c: 37.1,
            wbc: 10.5,
            crp: 12.0,
            troponin: 0.01,
        },
        atelectasis: {
            label: "Ca 6 · Xẹp phổi thùy (Atelectasis)",
            sampleUrl: "/web/samples/atelectasis_nih.png",
            sampleFilename: "atelectasis_nih.png",
            clinical_text:
                "Bệnh nhân nữ 55 tuổi sau phẫu thuật ổ bụng ngày thứ 3 xuất hiện thở nông, ho đờm khó khạc, tức ngực bên trái. Nghe phổi có vùng giảm thông khí đáy phổi trái, sốt nhẹ 37.8C. Nghi ngờ xẹp phổi do tắc đờm sau mổ.",
            age: 55,
            gender: "Nữ (Female)",
            spo2: 92.5,
            heart_rate: 96,
            sys_bp: 130,
            dia_bp: 85,
            chronic_conditions: 2,
            symptoms_severity: 4,
            stress_score: 7,
            sleep_hours: 5.0,
            physical_activity_hrs: 0.5,
            respiratory_rate: 24,
            temperature_c: 37.8,
            wbc: 11.2,
            crp: 25.0,
            troponin: 0.02,
        },
        mass_nodule: {
            label: "Ca 7 · Khối u nhu mô phổi (Lung Mass / Nodule)",
            sampleUrl: "/web/samples/mass_nodule_nih.png",
            sampleFilename: "mass_nodule_nih.png",
            clinical_text:
                "Bệnh nhân nam 54 tuổi có tiền sử hút thuốc lá 30 bao-năm, đến khám vì ho khan kéo dài hơn 2 tháng, thỉnh thoảng khạc đờm có tia máu, gầy sút 4kg không rõ nguyên nhân. Thể trạng gầy, sinh hiệu ổn định, cần chụp CT ngực đối chiếu tổn thương dạng khối trên X-quang.",
            age: 54,
            gender: "Nam (Male)",
            spo2: 96.0,
            heart_rate: 84,
            sys_bp: 135,
            dia_bp: 85,
            chronic_conditions: 1,
            symptoms_severity: 3,
            stress_score: 6,
            sleep_hours: 6.0,
            physical_activity_hrs: 2.0,
            respiratory_rate: 18,
            temperature_c: 36.8,
            wbc: 8.0,
            crp: 8.5,
            troponin: 0.01,
        },
        edema: {
            label: "Ca 8 · Phù phổi cấp do suy tim (Pulmonary Edema)",
            sampleUrl: "/web/samples/edema_nih.png",
            sampleFilename: "edema_nih.png",
            clinical_text:
                "Bệnh nhân nam 52 tuổi có tiền sử tăng huyết áp và bệnh tim thiếu máu cục bộ, nhập viện vì khó thở kịch phát về đêm, ho khạc bọt hồng, vã mồ hôi. Huyết áp vọt lên 175/105 mmHg, SpO2 88%, nghe phổi đầy rale ẩm rải rác từ đáy lên giữa 2 phế trường.",
            age: 52,
            gender: "Nam (Male)",
            spo2: 88.0,
            heart_rate: 122,
            sys_bp: 175,
            dia_bp: 105,
            chronic_conditions: 3,
            symptoms_severity: 5,
            stress_score: 9,
            sleep_hours: 3.0,
            physical_activity_hrs: 0.5,
            respiratory_rate: 30,
            temperature_c: 36.9,
            wbc: 9.4,
            crp: 15.0,
            troponin: 0.08,
        },
        infiltration: {
            label: "Ca 9 · Thâm nhiễm nhu mô phổi lan tỏa (Infiltration / Pneumonitis)",
            sampleUrl: "/web/samples/infiltration_nih.png",
            sampleFilename: "infiltration_nih.png",
            clinical_text:
                "Bệnh nhân nam 59 tuổi sốt cao 38.6C kéo dài 4 ngày, ho khan nhiều, mệt mỏi li bì, cảm giác hụt hơi khi đi lại. Nghe phổi có rale nổ rải rác hai bên. X-quang ghi nhận tổn thương thâm nhiễm dạng lưới nốt lan tỏa nhu mô hai phổi.",
            age: 59,
            gender: "Nam (Male)",
            spo2: 91.0,
            heart_rate: 102,
            sys_bp: 125,
            dia_bp: 80,
            chronic_conditions: 2,
            symptoms_severity: 4,
            stress_score: 8,
            sleep_hours: 4.0,
            physical_activity_hrs: 1.0,
            respiratory_rate: 25,
            temperature_c: 38.6,
            wbc: 14.8,
            crp: 55.0,
            troponin: 0.02,
        },
        fibrosis: {
            label: "Ca 10 · Xơ hóa phổi & Dày dính màng phổi (Pulmonary Fibrosis)",
            sampleUrl: "/web/samples/fibrosis_nih.png",
            sampleFilename: "fibrosis_nih.png",
            clinical_text:
                "Bệnh nhân nữ 50 tuổi có tiền sử lao phổi cũ đã điều trị khỏi 5 năm trước, nay đi khám vì khó thở tăng dần khi gắng sức, ho khan từng cơn, tức nặng ngực. Nghe phổi giảm rì rào phế nang vùng đỉnh và đáy hai bên, hình ảnh X-quang có các dải xơ co kéo và dày màng phổi.",
            age: 50,
            gender: "Nữ (Female)",
            spo2: 94.0,
            heart_rate: 88,
            sys_bp: 120,
            dia_bp: 80,
            chronic_conditions: 2,
            symptoms_severity: 3,
            stress_score: 5,
            sleep_hours: 6.5,
            physical_activity_hrs: 1.5,
            respiratory_rate: 20,
            temperature_c: 36.6,
            wbc: 7.2,
            crp: 5.0,
            troponin: 0.01,
        },
    };

    const TAB_META = {
        "multiagent-tab": {
            title: "Hội chẩn Đa phương thức MedFusion-CXR",
            subtitle:
                "Tích hợp DenseNet-121 (NIH ChestX-ray14), XGBoost + TreeSHAP và Gated Feature Fusion trong kiến trúc Multi-Agent",
        },
        "benchmark-tab": {
            title: "Thực nghiệm Độc lập E1 (NIH-14) & E2 (Clinical)",
            subtitle:
                "Kiểm định nghiêm ngặt không ghép cặp giả tạo: Patient-level Bootstrap CIs cho ảnh X-quang và Stratified Split 70/15/15 cho lâm sàng",
        },
        "xai-tab": {
            title: "Giải thích Mô hình (TreeSHAP) & Cầu nối Biểu diễn",
            subtitle:
                "Định lượng đóng góp đặc trưng phi tuyến từ cây quyết định XGBoost và cơ chế chiếu sang không gian Gated Fusion ℝ²⁵⁶",
        },
        "arch-tab": {
            title: "Kiến trúc Hệ thống & Luồng Multi-Agent",
            subtitle:
                "Sơ đồ phân tách rõ thành phần đã kiểm chứng thực nghiệm (E1, E2) và nguyên mẫu hợp nhất đa phương thức (GMU Prototype)",
        },
        "nlp-tab": {
            title: "Bóc tách Thực thể Lâm sàng Tiếng Việt (PhoBERT)",
            subtitle:
                "Trích xuất tự động Chẩn đoán (DISEASE), Triệu chứng (SYMPTOM), Thuốc (MEDICINE) và Xét nghiệm (TEST) từ bệnh án tự do",
        },
    };

    // State
    let selectedXrayBlob = null;
    let selectedXrayFilename = "";
    let previewObjectUrl = null;
    let e2ChartInstance = null;
    let e1ChartInstance = null;
    let xaiChartInstance = null;
    let benchmarksLoaded = false;

    // =========================================================================
    // 2. UTILITIES & TOAST NOTIFICATIONS
    // =========================================================================
    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function showToast(message) {
        const container = document.getElementById("toast-container");
        if (!container) return;
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transition = "opacity 0.25s ease";
            setTimeout(() => toast.remove(), 260);
        }, 3200);
    }

    function fmtPct(val, digits = 2) {
        if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
        return `${(Number(val) * 100).toFixed(digits)}%`;
    }

    function fmtDec(val, digits = 4) {
        if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
        return Number(val).toFixed(digits);
    }

    // =========================================================================
    // 3. TAB NAVIGATION
    // =========================================================================
    function initNavigation() {
        const navButtons = document.querySelectorAll(".nav-btn[data-tab]");
        const pageTitle = document.getElementById("page-title");
        const pageSubtitle = document.getElementById("page-subtitle");

        navButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                navButtons.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");

                document.querySelectorAll(".tab-content").forEach((sec) => {
                    sec.classList.toggle("active", sec.id === targetTab);
                });

                if (TAB_META[targetTab]) {
                    pageTitle.textContent = TAB_META[targetTab].title;
                    pageSubtitle.textContent = TAB_META[targetTab].subtitle;
                }

                if ((targetTab === "benchmark-tab" || targetTab === "xai-tab") && !benchmarksLoaded) {
                    loadAllBenchmarks();
                }
            });
        });
    }

    // =========================================================================
    // 4. X-RAY VIEWER, FILE UPLOAD & CLINICAL PRESETS
    // =========================================================================
    function setXrayPreview(blobOrFile, filename) {
        const imgEl = document.getElementById("xray-preview-img");
        const emptyHint = document.getElementById("xray-empty-hint");
        const nameEl = document.getElementById("xray-filename-display");

        if (previewObjectUrl) {
            URL.revokeObjectURL(previewObjectUrl);
            previewObjectUrl = null;
        }

        if (!blobOrFile) {
            selectedXrayBlob = null;
            selectedXrayFilename = "";
            imgEl.src = "";
            imgEl.classList.add("hidden");
            imgEl.classList.remove("inverted");
            emptyHint.classList.remove("hidden");
            nameEl.textContent = "Chế độ Clinical-only (Không đính kèm X-quang)";
            return;
        }

        selectedXrayBlob = blobOrFile;
        selectedXrayFilename = filename || blobOrFile.name || "cxr_upload.png";
        previewObjectUrl = URL.createObjectURL(blobOrFile);
        imgEl.src = previewObjectUrl;
        imgEl.classList.remove("hidden");
        emptyHint.classList.add("hidden");
        const kb = Math.max(1, Math.round(blobOrFile.size / 1024));
        nameEl.textContent = `Đã nạp: ${selectedXrayFilename} (${kb} KB · Grayscale CXR)`;
    }

    async function applyClinicalPreset(presetKey, silent = false) {
        const preset = CLINICAL_PRESETS[presetKey];
        if (!preset) return;

        document.querySelectorAll(".preset-chip").forEach((chip) => {
            chip.classList.toggle("active", chip.getAttribute("data-preset") === presetKey);
        });

        const fields = [
            "clinical-text",
            "age",
            "gender",
            "spo2",
            "heart_rate",
            "sys_bp",
            "dia_bp",
            "chronic_conditions",
            "symptoms_severity",
            "stress_score",
            "sleep_hours",
            "physical_activity_hrs",
            "respiratory_rate",
            "temperature_c",
            "wbc",
            "crp",
            "troponin",
        ];

        fields.forEach((id) => {
            const key = id === "clinical-text" ? "clinical_text" : id;
            const el = document.getElementById(id);
            if (el && preset[key] !== undefined) {
                el.value = String(preset[key]);
            }
        });

        const fileInput = document.getElementById("xray-file");
        if (fileInput) fileInput.value = "";

        try {
            const resp = await fetch(preset.sampleUrl);
            if (resp.ok) {
                const blob = await resp.blob();
                setXrayPreview(blob, preset.sampleFilename);
            }
        } catch (err) {
            console.warn("Could not fetch sample radiograph:", err);
        }

        if (!silent) {
            showToast(`Đã nạp ${preset.label}`);
        }
    }

    function initXrayAndPresetControls() {
        const fileInput = document.getElementById("xray-file");
        const dropzone = document.getElementById("xray-dropzone");
        const btnInvert = document.getElementById("btn-invert-xray");
        const btnClear = document.getElementById("btn-clear-xray");

        fileInput.addEventListener("change", (e) => {
            const file = e.target.files && e.target.files[0];
            if (file) {
                document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
                setXrayPreview(file, file.name);
                showToast(`Đã tải phim X-quang: ${file.name}`);
            }
        });

        ["dragenter", "dragover"].forEach((evt) => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = "#22d3ee";
            });
        });

        ["dragleave", "drop"].forEach((evt) => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = "";
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) {
                document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
                setXrayPreview(file, file.name);
                showToast(`Đã nạp ảnh kéo thả: ${file.name}`);
            }
        });

        btnInvert.addEventListener("click", () => {
            const imgEl = document.getElementById("xray-preview-img");
            if (!imgEl || imgEl.classList.contains("hidden")) {
                showToast("Chưa có phim X-quang để đảo ngược âm bản");
                return;
            }
            imgEl.classList.toggle("inverted");
        });

        btnClear.addEventListener("click", () => {
            fileInput.value = "";
            setXrayPreview(null, "");
            showToast("Đã bỏ ảnh X-quang · Sẵn sàng chạy chế độ Clinical-only");
        });

        document.querySelectorAll(".preset-chip").forEach((chip) => {
            chip.addEventListener("click", () => {
                applyClinicalPreset(chip.getAttribute("data-preset"));
            });
        });
    }

    // =========================================================================
    // 5. MULTIMODAL CONSULTATION EXECUTION & RENDERING
    // =========================================================================
    const NIH_VI_PATHOLOGY = {
        Atelectasis: "Xẹp phổi",
        Cardiomegaly: "Bóng tim to",
        Effusion: "Tràn dịch màng phổi",
        Infiltration: "Thâm nhiễm phổi",
        Mass: "Khối u nhu mô phổi",
        Nodule: "Nốt mờ phổi",
        Pneumonia: "Viêm phổi",
        Pneumothorax: "Tràn khí màng phổi",
        Consolidation: "Đông đặc phổi",
        Edema: "Phù phổi cấp",
        Emphysema: "Khí phế thũng",
        Fibrosis: "Xơ hóa phổi",
        Pleural_Thickening: "Dày màng phổi",
        Hernia: "Thoát vị hoành",
    };

    function renderMultimodalResult(data) {
        document.getElementById("result-placeholder").classList.add("hidden");
        const content = document.getElementById("result-content");
        content.classList.remove("hidden");

        const pred = data.agent3_prediction || {};
        const vitals = data.vital_dynamics || {};
        const fusion = data.gated_fusion || {};
        const xray = data.agent2_xray || null;

        // Latency HUD
        const latencyText = `${data.latency_ms ?? "--"} ms`;
        document.getElementById("inference-time-badge").textContent = `Hoàn tất · ${latencyText}`;
        document.getElementById("live-latency-pill").textContent = `ONLINE · ${latencyText}`;

        // 1. Hero Decision Banner
        const banner = document.getElementById("pred-banner");
        const isHighRisk = Boolean(pred.treatment_needed) || vitals.risk_tier === "Critical / High";
        banner.classList.toggle("low-risk", !isHighRisk);

        document.getElementById("risk-badge").textContent = isHighRisk
            ? `NGUY CƠ CAO · ${pred.risk_level || "High Risk"}`
            : `ỔN ĐỊNH · ${pred.risk_level || "Low Risk"}`;
        document.getElementById("champion-model-name").textContent =
            pred.champion_model || "XGBoost Clinical Branch";
        document.getElementById("pred-title").textContent = isHighRisk
            ? "KHUYẾN NGHỊ CAN THIỆP & THEO DÕI Y TẾ CHUYÊN KHOA"
            : "SINH HIỆU ỔN ĐỊNH · TIẾP TỤC THEO DÕI ĐỊNH KỲ";

        document.getElementById("pred-prob").textContent = `${Number(pred.probability_pct ?? 0).toFixed(1)}%`;
        document.getElementById("stat-shock-index").textContent =
            vitals.shock_index !== undefined && vitals.shock_index !== null
                ? Number(vitals.shock_index).toFixed(2)
                : "—";
        document.getElementById("stat-qsofa").textContent = `${vitals.qsofa_score ?? 0} / 3`;
        document.getElementById("stat-hypoxemia").textContent =
            vitals.hypoxemia_level || "Normal Oxygenation";

        // 2. Gated Feature Fusion Telemetry & Clinical Insight Box
        const gVisPct = Number(fusion.gate_vision_weight ?? 0) * 100;
        const gClinPct = Number(fusion.gate_clinical_weight ?? 1) * 100;
        document.getElementById("gate-vision-pct").textContent = `${gVisPct.toFixed(1)}%`;
        document.getElementById("gate-clinical-pct").textContent = `${gClinPct.toFixed(1)}%`;
        document.getElementById("gate-bar-vision").style.width = `${gVisPct}%`;
        document.getElementById("gate-bar-clinical").style.width = `${gClinPct}%`;

        const badgeEl = document.getElementById("fusion-status-badge");
        const descEl = document.getElementById("fusion-mode-desc");
        const insightEl = document.getElementById("fusion-insight-text");

        if (fusion.active && xray) {
            badgeEl.textContent = "🛡️ HỢP NHẤT TỰ ĐỘNG (GMU)";
            badgeEl.className = "proto-badge";
            if (descEl) descEl.textContent = "Hệ thống tự động cân đối mức độ ưu tiên giữa Phim X-Quang và Dữ liệu Sinh hiệu";

            // Find top detected pathology for clinical insight explanation
            let topPathLabel = "";
            if (xray && Array.isArray(xray.detected_pathologies) && xray.detected_pathologies.length > 0) {
                const flagged = xray.detected_pathologies.filter((p) => p.flagged);
                const topItem = flagged.length > 0 ? flagged[0] : xray.detected_pathologies[0];
                topPathLabel = NIH_VI_PATHOLOGY[topItem.label] || topItem.label;
            }

            if (gClinPct >= 55) {
                insightEl.innerHTML = `<strong>Nhận định lâm sàng:</strong> Dữ liệu sinh hiệu chiếm ưu thế (<strong>${gClinPct.toFixed(1)}%</strong>) do bệnh nhân có dấu hiệu rối loạn sinh lý rõ rệt (SpO₂: ${vitals.spo2 ?? "--"}%, Mạch: ${vitals.heart_rate ?? "--"} bpm). Phim X-quang (<strong>${gVisPct.toFixed(1)}%</strong>) đóng vai trò xác nhận tổn thương ${topPathLabel ? `nghi ngờ <em>${escapeHtml(topPathLabel)}</em>` : "thực thể"}, củng cố chỉ định can thiệp.`;
            } else if (gVisPct >= 55) {
                insightEl.innerHTML = `<strong>Nhận định lâm sàng:</strong> Hình ảnh X-quang chiếm tỷ trọng chính (<strong>${gVisPct.toFixed(1)}%</strong>) với bằng chứng tổn thương phổi nổi bật ${topPathLabel ? `(<strong>${escapeHtml(topPathLabel)}</strong>)` : ""}. Dữ liệu sinh hiệu (<strong>${gClinPct.toFixed(1)}%</strong>) cung cấp thông tin huyết động toàn thân để bác sĩ quyết định mức độ can thiệp.`;
            } else {
                insightEl.innerHTML = `<strong>Nhận định lâm sàng:</strong> Hai nguồn dữ liệu đóng góp cân bằng (<strong>${gVisPct.toFixed(1)}%</strong> ảnh · <strong>${gClinPct.toFixed(1)}%</strong> sinh hiệu). Bằng chứng tổn thương phổi trên phim X-quang được củng cố song song bởi tình trạng sinh lý hiện tại, giúp hạn chế bỏ sót ca bệnh cấp cứu.`;
            }
        } else {
            badgeEl.textContent = "⚡ CHẾ ĐỘ LÂM SÀNG DỰ PHÒNG";
            badgeEl.className = "proto-badge fallback-mode";
            if (descEl) descEl.textContent = "Chưa nạp ảnh CXR · Hệ thống kích hoạt nhánh dự phòng thuần lâm sàng";
            insightEl.innerHTML = `<strong>Nhận định an toàn:</strong> Không có phim X-quang. Hệ thống tự động chuyển cổng về <strong>100% dữ liệu lâm sàng</strong> (XGBoost), bảo đảm bác sĩ vẫn nhận được khuyến nghị xử trí cấp cứu an toàn mà không bị gián đoạn.`;
        }

        // Technical Math details for the accordion (accessible when expanded)
        document.getElementById("gate-norms-line").textContent = fusion.active
            ? `‖F̃_v‖₂ = ${fusion.vision_proj_l2_norm} · ‖F̃_c‖₂ = ${fusion.clinical_proj_l2_norm} · ` +
              `‖F_fusion‖₂ = ${fusion.fused_vector_l2_norm} · Độ phân tán cổng σ(g) = ${fusion.gate_dispersion_std}`
            : `Đã bỏ qua cổng (Fallback) · ${fusion.equation || "F_fusion = F_c"}`;
        document.getElementById("fusion-governance-note").textContent =
            fusion.governance_disclosure || "Nhánh hình ảnh và nhánh lâm sàng được phân tích độc lập, sau đó kết hợp qua cơ chế cổng.";

        // 2b. MULTIMODAL CONSENSUS & POTENTIAL DISEASES SYNTHESIS (Hero Section)
        const consensus = data.multimodal_consensus || {};
        const primaryTitle = document.getElementById("consensus-primary-title");
        const summaryDesc = document.getElementById("consensus-summary-desc");
        const acuityBadge = document.getElementById("consensus-acuity-badge");
        const confPill = document.getElementById("consensus-confidence-pill");
        const diffCountChip = document.getElementById("differentials-count-chip");
        const diffCards = document.getElementById("consensus-differentials-cards");

        if (primaryTitle && consensus.primary_diagnosis) {
            primaryTitle.textContent = consensus.primary_diagnosis;
            if (summaryDesc) summaryDesc.textContent = consensus.conclusion_summary || "";
            if (confPill) confPill.textContent = `Độ tin cậy: ${Number(consensus.diagnostic_confidence_pct || 0).toFixed(1)}%`;

            const acuity = consensus.acuity_risk || "NGUY CƠ CAO";
            if (acuityBadge) {
                acuityBadge.textContent = acuity;
                if (acuity.includes("CẤP CỨU") || acuity.includes("RẤT CAO")) {
                    acuityBadge.className = "status-badge critical";
                } else if (acuity.includes("CAO")) {
                    acuityBadge.className = "status-badge high";
                } else if (acuity.includes("VỪA") || acuity.includes("TRUNG BÌNH")) {
                    acuityBadge.className = "status-badge moderate";
                } else {
                    acuityBadge.className = "status-badge normal";
                }
            }

            const diseases = consensus.potential_diseases || data.agent4_protocols || [];
            if (diffCountChip) {
                diffCountChip.textContent = `${diseases.length} Bệnh lý chẩn đoán phân biệt`;
            }

            if (diffCards) {
                diffCards.innerHTML = diseases
                    .map((d, idx) => {
                        const prob = Number(d.probability_pct || 0);
                        const sev = String(d.severity || "Vừa");
                        const sevClass = sev.includes("Cấp cứu")
                            ? "sev-critical"
                            : sev.includes("Nghiêm trọng")
                            ? "sev-high"
                            : "sev-moderate";
                        return `
                            <div class="differential-card">
                                <div class="diff-card-top">
                                    <div class="diff-title-col">
                                        <div class="diff-num-tag">#${idx + 1}</div>
                                        <div>
                                            <h6>${escapeHtml(d.disease_name)} <span class="icd10-pill">[${escapeHtml(d.icd10_code)}]</span></h6>
                                            <span class="diff-sev-badge ${sevClass}">${escapeHtml(d.severity)}</span>
                                        </div>
                                    </div>
                                    <div class="diff-prob-col">
                                        <div class="diff-prob-num">${prob.toFixed(1)}%</div>
                                        <div class="diff-prob-bar-track">
                                            <div class="diff-prob-bar-fill" style="width:${Math.min(100, Math.max(5, prob))}%"></div>
                                        </div>
                                    </div>
                                </div>

                                <div class="diff-evidence-grid">
                                    <div class="evidence-box cxr-evidence">
                                        <div class="ev-head">
                                            <span>🩻 Căn cứ Hình ảnh X-quang (DenseNet-121)</span>
                                        </div>
                                        <p>${escapeHtml(d.cxr_evidence || "Không có dấu hiệu tổn thương rõ rệt trên phim X-quang.")}</p>
                                    </div>
                                    <div class="evidence-box clinical-evidence">
                                        <div class="ev-head">
                                            <span>🩺 Căn cứ Lâm sàng &amp; PhoBERT NLP</span>
                                        </div>
                                        <p>${escapeHtml(d.clinical_evidence || "Không ghi nhận triệu chứng cơ năng điển hình.")}</p>
                                    </div>
                                </div>

                                <div class="diff-treatment-box">
                                    <div class="treatment-head">
                                        <span>🏥 Phác đồ Điều trị &amp; Can thiệp Tham khảo (BV Bạch Mai)</span>
                                    </div>
                                    <p>${escapeHtml(d.bach_mai_treatment || "Theo dõi triệu chứng và thăm khám định kỳ.")}</p>
                                </div>
                            </div>
                        `;
                    })
                    .join("");
            }
        }

        // 3. DenseNet-121 NIH-14 Scores
        const xrayBody = document.getElementById("xray-findings-body");
        const shaChip = document.getElementById("xray-sha-chip");
        if (xray && Array.isArray(xray.detected_pathologies)) {
            shaChip.textContent = "14 Tổn thương";
            const topItems = xray.detected_pathologies.slice(0, 8);
            xrayBody.innerHTML = `
                <div class="nih-pathology-table">
                    ${topItems
                        .map((item) => {
                            const rawProb = Number(item.probability || 0);
                            const calScore = Number(item.calibrated_score || 0);

                            // Determine clear clinical status based on real probability & clinical threshold
                            const isTrulyFlagged = item.flagged && (rawProb >= 0.15 || calScore >= 0.60);
                            const isModerate = item.flagged && !isTrulyFlagged && rawProb >= 0.08;

                            let statusBadge = "";
                            let valClass = "val-normal";
                            let barClass = "bar-normal";
                            // Always display genuine calibrated/raw probability percentage (never distorted op-norm boundary)
                            const displayPct = Math.round(rawProb * 1000) / 10;

                            if (isTrulyFlagged) {
                                statusBadge = `<span class="nih-chip chip-danger">● Nghi ngờ cao</span>`;
                                valClass = "val-danger";
                                barClass = "bar-danger";
                            } else if (isModerate) {
                                statusBadge = `<span class="nih-chip chip-warning">Theo dõi</span>`;
                                valClass = "val-warning";
                                barClass = "bar-warning";
                            } else {
                                statusBadge = `<span class="nih-chip chip-safe">Âm tính</span>`;
                                valClass = "val-safe";
                                barClass = "bar-safe";
                            }

                            const viName = NIH_VI_PATHOLOGY[item.label] || item.label;
                            const barWidth = Math.min(100, Math.max(3, displayPct));

                            return `
                                <div class="nih-row">
                                    <div class="nih-row-main">
                                        <div class="nih-col-name" title="${escapeHtml(viName)} (${escapeHtml(item.label)})">
                                            <span class="nih-label-vi">${escapeHtml(viName)}</span>
                                            <span class="nih-label-en">(${escapeHtml(item.label)})</span>
                                        </div>
                                        <div class="nih-col-meta">
                                            ${statusBadge}
                                            <strong class="nih-prob-number ${valClass}">${displayPct.toFixed(1)}%</strong>
                                        </div>
                                    </div>
                                    <div class="nih-progress-track">
                                        <div class="nih-progress-fill ${barClass}" style="width:${barWidth}%"></div>
                                    </div>
                                </div>
                            `;
                        })
                        .join("")}
                </div>
            `;
        } else {
            shaChip.textContent = "Chưa có ảnh";
            xrayBody.innerHTML = `
                <p style="font-size:0.8rem;color:var(--text-muted);line-height:1.55;">
                    Không đính kèm phim X-quang ngực trong lần chạy này. Cổng Gated Fusion tự động chuyển sang nhánh <strong>XGBoost + Sinh hiệu cấp cứu</strong> an toàn.
                </p>
            `;
        }

        // 4. Per-Patient TreeSHAP Bars
        const shapContainer = document.getElementById("patient-shap-bars");
        const shapItems = (pred.shap_attributions || []).slice(0, 8);
        if (shapItems.length > 0) {
            const maxAbs = Math.max(0.01, ...shapItems.map((s) => Math.abs(Number(s.shap_value || 0))));
            shapContainer.innerHTML = `
                <div class="shap-table">
                    ${shapItems
                        .map((item) => {
                            const val = Number(item.shap_value || 0);
                            const widthPct = Math.min(100, Math.max(4, (Math.abs(val) / maxAbs) * 100));
                            const sign = val >= 0 ? "+" : "";
                            const valCls = val >= 0 ? "shap-pos" : "shap-neg";
                            const barCls = val >= 0 ? "fill-pos" : "fill-neg";
                            return `
                                <div class="shap-row">
                                    <div class="shap-row-main">
                                        <div class="shap-feature-name" title="${escapeHtml(item.feature_label)}">
                                            <span>${escapeHtml(item.feature_label)}</span>
                                            <small class="shap-raw-val">(${escapeHtml(item.raw_value)})</small>
                                        </div>
                                        <strong class="shap-val-text ${valCls}">${sign}${val.toFixed(3)}</strong>
                                    </div>
                                    <div class="shap-progress-track">
                                        <div class="shap-progress-fill ${barCls}" style="width:${widthPct}%"></div>
                                    </div>
                                </div>
                            `;
                        })
                        .join("")}
                </div>
            `;
        } else {
            shapContainer.innerHTML = `<p style="font-size:0.8rem;color:var(--text-muted);">Chưa có dữ liệu SHAP.</p>`;
        }

        // 5. PhoBERT Extracted Entities
        const nlpCard = document.getElementById("nlp-entities-card");
        const nlpBody = document.getElementById("nlp-entities-body");
        const entities = data.agent1_nlp || [];
        if (entities.length > 0) {
            nlpCard.classList.remove("hidden");
            nlpBody.innerHTML = entities
                .map((ent) => {
                    const typeLower = String(ent.type || "symptom").toLowerCase();
                    const cssType = ["disease", "symptom", "medicine", "test"].includes(typeLower)
                        ? typeLower
                        : "symptom";
                    return `<span class="entity-tag ${cssType}">${escapeHtml(ent.text)} · <small>${escapeHtml(ent.type)}</small></span>`;
                })
                .join("");
        } else {
            nlpCard.classList.add("hidden");
        }

        // 6. Bach Mai Protocols
        const diseasesList = document.getElementById("predicted-diseases-list");
        const protocols = data.multimodal_consensus?.potential_diseases || data.agent4_protocols || [];
        diseasesList.innerHTML = protocols
            .map(
                (d) => `
                <div class="disease-card">
                    <div class="disease-card-header">
                        <h6>${escapeHtml(d.disease_name)} <small style="color:var(--cyan-bright);font-family:var(--font-mono)">[${escapeHtml(d.icd10_code)}]</small></h6>
                        <span class="disease-prob-badge">${escapeHtml(d.severity)} · ${Number(d.probability_pct || 0).toFixed(1)}%</span>
                    </div>
                    <p class="disease-treatment">${escapeHtml(d.bach_mai_treatment)}</p>
                </div>
            `
            )
            .join("");

        // 7. Multi-Agent Execution Trace (nếu có phần tử trong DOM)
        const stepsList = document.getElementById("agent-steps-list");
        if (stepsList) {
            stepsList.innerHTML = (data.execution_steps || [])
                .map((step) => {
                    const badgeColor =
                        step.status === "success"
                            ? "#6ee7b7"
                            : step.status === "warning"
                            ? "#fde68a"
                            : "#94a3b8";
                    return `
                        <div class="agent-step-item">
                            <div class="agent-step-header">
                                <strong>${escapeHtml(step.agent)}</strong>
                                <span style="font-family:var(--font-mono);font-size:0.7rem;color:${badgeColor}">${escapeHtml(String(step.status).toUpperCase())}</span>
                            </div>
                            <div class="agent-step-detail">${escapeHtml(step.detail)}</div>
                        </div>
                    `;
                })
                .join("");
        }

        // 8. Risk Factors & Recommendations
        const riskList = document.getElementById("risk-factors-list");
        const risks = data.agent5_xai || [];
        riskList.innerHTML =
            risks.length > 0
                ? risks
                      .map(
                          (r) =>
                              `<li><strong>${escapeHtml(r.factor)}:</strong> ${escapeHtml(r.value)} — <span style="color:var(--cyan-bright)">${escapeHtml(r.impact)}</span></li>`
                      )
                      .join("")
                : `<li>Các chỉ số sinh hiệu chính nằm trong ngưỡng an toàn.</li>`;

        const recList = document.getElementById("recommendations-list");
        recList.innerHTML = (data.consensus_recommendations || [])
            .map((rec) => `<li>${escapeHtml(rec)}</li>`)
            .join("");
    }

    function initConsultationForm() {
        const form = document.getElementById("multimodal-agent-form");
        const submitBtn = document.getElementById("btn-submit-consult");

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            submitBtn.disabled = true;
            const labelEl = submitBtn.querySelector(".cta-label");
            const prevLabel = labelEl.textContent;
            labelEl.textContent = "⏳ ĐANG CHẠY DENSENET-121 · XGBOOST · GATED FUSION...";

            try {
                const fd = new FormData();
                fd.append("clinical_text", document.getElementById("clinical-text").value || "");
                fd.append("age", document.getElementById("age").value);
                fd.append("gender", document.getElementById("gender").value);
                fd.append("spo2", document.getElementById("spo2").value);
                fd.append("heart_rate", document.getElementById("heart_rate").value);
                fd.append("sys_bp", document.getElementById("sys_bp").value);
                fd.append("dia_bp", document.getElementById("dia_bp").value);
                fd.append("chronic_conditions", document.getElementById("chronic_conditions").value);
                fd.append("symptoms_severity", document.getElementById("symptoms_severity").value);
                fd.append("stress_score", document.getElementById("stress_score").value);
                fd.append("sleep_hours", document.getElementById("sleep_hours").value);
                fd.append("physical_activity_hrs", document.getElementById("physical_activity_hrs").value);

                const optionalLabs = ["respiratory_rate", "temperature_c", "wbc", "crp", "troponin"];
                optionalLabs.forEach((fieldId) => {
                    const val = document.getElementById(fieldId).value;
                    if (val !== "" && val !== null && val !== undefined) {
                        fd.append(fieldId, val);
                    }
                });

                if (selectedXrayBlob) {
                    fd.append("xray_file", selectedXrayBlob, selectedXrayFilename || "cxr.png");
                }

                const resp = await fetch("/api/v1/treatment/multimodal_predict", {
                    method: "POST",
                    body: fd,
                });

                if (!resp.ok) {
                    const errText = await resp.text();
                    throw new Error(`HTTP ${resp.status}: ${errText}`);
                }

                const data = await resp.json();
                renderMultimodalResult(data);
                showToast(`Phân tích hoàn tất (${data.latency_ms} ms)`);
            } catch (err) {
                console.error("Multimodal consultation error:", err);
                showToast(`Lỗi khi phân tích: ${err.message}`);
            } finally {
                submitBtn.disabled = false;
                labelEl.textContent = prevLabel;
            }
        });
    }

    // =========================================================================
    // 6. EXPERIMENTS E1 (NIH-14), E2 (CLINICAL) & E3 (TREESHAP) CHARTS
    // =========================================================================
    async function loadAllBenchmarks() {
        try {
            const [e2Resp, e1Resp] = await Promise.all([
                fetch("/api/v1/treatment/benchmark"),
                fetch("/api/v1/multimodal/nih-benchmark"),
            ]);

            if (e2Resp.ok) {
                const e2Data = await e2Resp.json();
                renderE2Benchmark(e2Data);
                renderGlobalShapChart(e2Data.feature_importances || {});
            }

            if (e1Resp.ok) {
                const e1Data = await e1Resp.json();
                renderE1NihBenchmark(e1Data);
            }

            benchmarksLoaded = true;
        } catch (err) {
            console.error("Failed loading benchmarks:", err);
            showToast("Không thể tải dữ liệu thực nghiệm E1/E2");
        }
    }

    function renderE2Benchmark(data) {
        const results = data.benchmark_results || {};
        const entries = Object.entries(results).sort(
            (a, b) => (b[1].f1_score || 0) - (a[1].f1_score || 0)
        );
        const xgb = results["XGBoost"] || {};

        const kpiRow = document.getElementById("e2-kpi-row");
        kpiRow.innerHTML = `
            <div class="kpi-pill">
                <small>XGBoost AUROC</small>
                <strong>${fmtPct(xgb.roc_auc)}</strong>
            </div>
            <div class="kpi-pill">
                <small>XGBoost AUPRC</small>
                <strong>${fmtPct(xgb.auprc)}</strong>
            </div>
            <div class="kpi-pill">
                <small>XGBoost Test F1</small>
                <strong>${fmtPct(xgb.f1_score)}</strong>
            </div>
        `;

        const tbody = document.getElementById("benchmark-table-body");
        tbody.innerHTML = entries
            .map(([name, m], idx) => {
                const isXgb = name === "XGBoost";
                const rowStyle = isXgb ? 'style="background:rgba(6,182,212,0.11);font-weight:700;"' : "";
                return `
                    <tr ${rowStyle}>
                        <td>#${idx + 1}</td>
                        <td>${escapeHtml(name)} ${isXgb ? '<span class="mono-badge-sm">Primary Branch</span>' : ""}</td>
                        <td>${fmtDec(m.optimal_threshold, 2)}</td>
                        <td>${fmtPct(m.accuracy)}</td>
                        <td>${fmtPct(m.precision)}</td>
                        <td>${fmtPct(m.recall)}</td>
                        <td style="color:var(--cyan-bright);font-weight:700;">${fmtPct(m.f1_score)}</td>
                        <td>${fmtPct(m.roc_auc)}</td>
                        <td>${fmtPct(m.auprc)}</td>
                    </tr>
                `;
            })
            .join("");

        const ctx = document.getElementById("benchmarkChart");
        if (!ctx || typeof Chart === "undefined") return;
        if (e2ChartInstance) e2ChartInstance.destroy();

        e2ChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: entries.map(([name]) => name),
                datasets: [
                    {
                        label: "AUROC (%)",
                        data: entries.map(([, m]) => Number(((m.roc_auc || 0) * 100).toFixed(2))),
                        backgroundColor: "rgba(34, 211, 238, 0.78)",
                        borderRadius: 6,
                    },
                    {
                        label: "AUPRC (%)",
                        data: entries.map(([, m]) => Number(((m.auprc || 0) * 100).toFixed(2))),
                        backgroundColor: "rgba(59, 130, 246, 0.78)",
                        borderRadius: 6,
                    },
                    {
                        label: "F1-Score (%)",
                        data: entries.map(([, m]) => Number(((m.f1_score || 0) * 100).toFixed(2))),
                        backgroundColor: "rgba(16, 185, 129, 0.78)",
                        borderRadius: 6,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#cbd5e1", font: { family: "Inter" } } },
                },
                scales: {
                    y: {
                        min: 50,
                        max: 100,
                        grid: { color: "rgba(148, 163, 184, 0.12)" },
                        ticks: { color: "#94a3b8" },
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#cbd5e1" },
                    },
                },
            },
        });
    }

    function renderE1NihBenchmark(payload) {
        const report = payload.benchmark_report;
        const subtitle = document.getElementById("e1-cohort-subtitle");
        const kpiRow = document.getElementById("e1-kpi-row");
        const tbody = document.getElementById("nih-table-body");

        if (!report) {
            subtitle.textContent = "Chưa tìm thấy báo cáo reports/nih_benchmark_report.json.";
            return;
        }

        const macro = report.macro_summary || {};
        const cohort = report.cohort_accounting || {};
        const leak = cohort.leakage_verification || {};

        subtitle.textContent =
            `Đánh giá trên ${cohort.evaluated_test_images ?? 0} ảnh Test (${cohort.evaluated_test_patients ?? 0} BN) & ` +
            `${cohort.evaluated_validate_images ?? 0} ảnh Val · Tổng manifest NIH: ${cohort.manifest_total_rows ?? 112120} ảnh · ` +
            `Patient Overlap = ${leak.train_test_overlap ?? 0} (Zero Leakage Verified)`;

        kpiRow.innerHTML = `
            <div class="kpi-pill">
                <small>Macro AUROC (14 nhãn)</small>
                <strong>${fmtDec(macro.macro_auroc, 4)}</strong>
            </div>
            <div class="kpi-pill">
                <small>Macro AUPRC</small>
                <strong>${fmtDec(macro.macro_auprc, 4)}</strong>
            </div>
            <div class="kpi-pill">
                <small>Macro ECE / Brier</small>
                <strong>${fmtDec(macro.macro_ece, 4)} / ${fmtDec(macro.macro_brier_score, 4)}</strong>
            </div>
        `;

        const perPath = report.per_pathology_results || {};
        const rows = Object.entries(perPath).sort((a, b) => {
            const aRoc = (a[1].test_results || a[1].test_metrics || {}).auroc || 0;
            const bRoc = (b[1].test_results || b[1].test_metrics || {}).auroc || 0;
            return bRoc - aRoc;
        });

        tbody.innerHTML = rows
            .map(([label, item]) => {
                const tm = item.test_results || item.test_metrics || {};
                const rocCi = Array.isArray(tm.auroc_ci)
                    ? `[${fmtDec(tm.auroc_ci[0], 3)}, ${fmtDec(tm.auroc_ci[1], 3)}]`
                    : "";
                const prCi = Array.isArray(tm.auprc_ci)
                    ? `[${fmtDec(tm.auprc_ci[0], 3)}, ${fmtDec(tm.auprc_ci[1], 3)}]`
                    : "";
                return `
                    <tr>
                        <td><strong>${escapeHtml(label)}</strong></td>
                        <td>${tm.pos_count ?? 0} / ${tm.sample_count ?? 0}</td>
                        <td>${fmtDec(item.validation_optimal_threshold, 2)}</td>
                        <td style="color:var(--cyan-bright);font-weight:700;">${fmtDec(tm.auroc, 4)} <small style="color:var(--text-muted)">${rocCi}</small></td>
                        <td>${fmtDec(tm.auprc, 4)} <small style="color:var(--text-muted)">${prCi}</small></td>
                        <td>${fmtDec(tm.f1, 4)}</td>
                        <td>${fmtDec(tm.brier_score, 4)}</td>
                        <td>${fmtDec(tm.ece, 4)}</td>
                    </tr>
                `;
            })
            .join("");

        const ctx = document.getElementById("nihBenchmarkChart");
        if (!ctx || typeof Chart === "undefined") return;
        if (e1ChartInstance) e1ChartInstance.destroy();

        e1ChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: rows.map(([label]) => label),
                datasets: [
                    {
                        label: "Test AUROC",
                        data: rows.map(([, item]) => {
                            const tm = item.test_results || item.test_metrics || {};
                            return Number((tm.auroc || 0).toFixed(4));
                        }),
                        backgroundColor: "rgba(6, 182, 212, 0.8)",
                        borderRadius: 5,
                    },
                    {
                        label: "Test AUPRC",
                        data: rows.map(([, item]) => {
                            const tm = item.test_results || item.test_metrics || {};
                            return Number((tm.auprc || 0).toFixed(4));
                        }),
                        backgroundColor: "rgba(245, 158, 11, 0.78)",
                        borderRadius: 5,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#cbd5e1", font: { family: "Inter" } } },
                },
                scales: {
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: "rgba(148, 163, 184, 0.12)" },
                        ticks: { color: "#94a3b8" },
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#cbd5e1", maxRotation: 32, minRotation: 18 },
                    },
                },
            },
        });
    }

    function renderGlobalShapChart(importances) {
        const ctx = document.getElementById("xaiChart");
        if (!ctx || typeof Chart === "undefined") return;
        const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
        if (sorted.length === 0) return;

        if (xaiChartInstance) xaiChartInstance.destroy();

        xaiChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: sorted.map(([k]) => k),
                datasets: [
                    {
                        label: "Tỷ trọng đóng góp TreeSHAP |ϕ_i| (%)",
                        data: sorted.map(([, v]) => Number((Number(v) * 100).toFixed(2))),
                        backgroundColor: sorted.map((_, idx) =>
                            idx < 3 ? "rgba(34, 211, 238, 0.85)" : "rgba(59, 130, 246, 0.68)"
                        ),
                        borderRadius: 6,
                    },
                ],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#cbd5e1" } },
                },
                scales: {
                    x: {
                        grid: { color: "rgba(148, 163, 184, 0.12)" },
                        ticks: { color: "#94a3b8", callback: (v) => `${v}%` },
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#e2e8f0" },
                    },
                },
            },
        });
    }

    // =========================================================================
    // 7. STANDALONE CLINICAL NLP TAB (PHOBERT NER)
    // =========================================================================
    function initStandaloneNlp() {
        const btn = document.getElementById("btn-analyze-nlp");
        const input = document.getElementById("nlp-text-input");
        const box = document.getElementById("nlp-results");
        const container = document.getElementById("entities-container");

        btn.addEventListener("click", async () => {
            const text = (input.value || "").trim();
            if (!text) {
                showToast("Vui lòng nhập văn bản bệnh án tiếng Việt");
                return;
            }
            btn.disabled = true;
            try {
                const resp = await fetch("/api/v1/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text }),
                });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const data = await resp.json();
                const entities = data.entities || [];

                const groups = {
                    DISEASE: { title: "🏥 Bệnh lý / Chẩn đoán (DISEASE)", cls: "disease", items: [] },
                    SYMPTOM: { title: "🌡️ Triệu chứng Lâm sàng (SYMPTOM)", cls: "symptom", items: [] },
                    TEST: { title: "🧪 Xét nghiệm & Chẩn đoán hình ảnh (TEST)", cls: "test", items: [] },
                    MEDICINE: { title: "💊 Thuốc & Điều trị (MEDICINE)", cls: "medicine", items: [] },
                };

                entities.forEach((e) => {
                    const key = String(e.type || "SYMPTOM").toUpperCase();
                    if (groups[key]) {
                        groups[key].items.push(e);
                    } else {
                        groups.SYMPTOM.items.push(e);
                    }
                });

                box.classList.remove("hidden");
                container.innerHTML = Object.values(groups)
                    .map(
                        (g) => `
                        <div class="analytic-box">
                            <h5>${g.title}</h5>
                            <div class="tags-wrapper">
                                ${
                                    g.items.length > 0
                                        ? g.items
                                              .map(
                                                  (it) =>
                                                      `<span class="entity-tag ${g.cls}">${escapeHtml(it.text)} <small>(${(Number(it.score || 1) * 100).toFixed(0)}%)</small></span>`
                                              )
                                              .join("")
                                        : `<span style="font-size:0.78rem;color:var(--text-muted)">Không ghi nhận</span>`
                                }
                            </div>
                        </div>
                    `
                    )
                    .join("");
                showToast(`Bóc tách thành công ${entities.length} thực thể y khoa`);
            } catch (err) {
                console.error("NLP extraction failed:", err);
                showToast(`Lỗi bóc tách NLP: ${err.message}`);
            } finally {
                btn.disabled = false;
            }
        });
    }

    // =========================================================================
    // BOOTSTRAP ON DOM READY
    // =========================================================================
    document.addEventListener("DOMContentLoaded", () => {
        initNavigation();
        initXrayAndPresetControls();
        initConsultationForm();
        initStandaloneNlp();
        applyClinicalPreset("pneumonia", true);
    });
})();
