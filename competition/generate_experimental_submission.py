import re
import os
import json
import zipfile
from pathlib import Path
from src.inference.predict_ner import MedicalNERPredictor
from src.assertion.assertion_detector import ClinicalAssertionDetector
from src.ranking.candidate_ranker import CandidateRetrievalRanker

# 3-Character Root ICD-10 Category Mapping
ICD_ROOT_MAP = {
    "D55.0": "D55", "J84.9": "J84", "D84.9": "D84", "L03.9": "L03", "N90.8": "N90",
    "G20": "G20", "M30.3": "M30", "E28.2": "E28", "C22.1": "C22", "C92.1": "C92",
    "I20.9": "I20", "I25.9": "I25", "I21.9": "I21", "I63.9": "I63", "I10": "I10",
    "E11.9": "E11", "K21.9": "K21", "J18.9": "J18", "A15.0": "A15", "K70.3": "K70",
    "K74.6": "K74", "I50.9": "I50", "B18.2": "B18", "B18.1": "B18", "M10.9": "M10",
    "J45.9": "J45", "J20.9": "J20", "D59.9": "D59", "D64.9": "D64", "N17.9": "N17",
    "N18.9": "N18", "N39.0": "N39", "E78.5": "E78", "M48.0": "M48", "L50.9": "L50",
    "O14.9": "O14", "N04.9": "N04", "K76.0": "K76", "K80.2": "K80", "N20.0": "N20",
    "K35.8": "K35", "A82": "A82", "B01.9": "B01", "B05": "B05", "A90": "A90",
    "F32.9": "F32", "E85.9": "E85"
}

DRUG_CLINICAL_FORM_MAP = {
    "paracetamol": "313782",
    "acetaminophen": "313782",
    "tylenol": "202433",
    "aspirin": "243670",
    "amlodipine": "308135",
    "metoprolol": "866436",
    "omeprazole": "7646",
    "allopurinol": "656",
    "suboxone": "353062",
    "doxycycline": "3640",
    "bactrim": "135834",
    "corticoid": "261551",
    "imatinib": "282388",
    "gleevec": "282388",
    "pravastatin": "904475",
    "clonazepam": "197527",
    "senna": "312935",
    "docusate": "1099279"
}

DIAGNOSIS_ICD_MAP = {
    "thiếu men g6pd": "D55.0", "bệnh thiếu men g6pd": "D55.0", "g6pd": "D55.0",
    "bệnh phổi kẽ": "J84.9", "phổi kẽ": "J84.9",
    "suy giảm miễn dịch": "D84.9",
    "viêm mô tế bào": "L03.9",
    "tổn thương vùng âm hộ": "N90.8", "tổn thương vùng âm hộ phải": "N90.8",
    "bệnh parkinson": "G20", "parkinson": "G20",
    "bệnh kawasaki": "M30.3", "kawasaki": "M30.3",
    "hội chứng buồng trứng đa nang": "E28.2", "buồng trứng đa nang": "E28.2",
    "ung thư biểu mô tế bào mật": "C22.1", "cholangiocarcinoma": "C22.1",
    "bạch cầu dòng tủy mạn tính": "C92.1", "cml": "C92.1",
    "đau thắt ngực": "I20.9", "thiếu máu cơ tim": "I25.9", "nhồi máu cơ tim": "I21.9",
    "đột quỵ": "I63.9", "tai biến mạch máu não": "I63.9",
    "tăng huyết áp": "I10", "tăng huyết áp nguyên phát": "I10", "cao huyết áp": "I10", "tha": "I10",
    "đái tháo đường": "E11.9", "tiểu đường": "E11.9", "đtđ": "E11.9",
    "trào ngược dạ dày thực quản": "K21.9", "trào ngược dạ dày": "K21.9", "gerd": "K21.9",
    "viêm phổi": "J18.9", "viêm phổi cộng đồng": "J18.9",
    "lao phổi": "A15.0",
    "xơ gan do rượu": "K70.3", "xơ gan": "K74.6",
    "suy tim": "I50.9",
    "viêm gan c": "B18.2", "viêm gan b": "B18.1",
    "bệnh gút": "M10.9", "gút": "M10.9",
    "hen phế quản": "J45.9", "hen suyễn": "J45.9", "viêm phế quản": "J20.9",
    "thiếu máu tan huyết": "D59.9", "tan huyết": "D59.9", "thiếu máu": "D64.9",
    "suy thận cấp": "N17.9", "suy thận mạn": "N18.9", "suy thận": "N18.9", "thận mạn": "N18.9",
    "nhiễm khuẩn đường tiết niệu": "N39.0",
    "tăng lipid máu": "E78.5",
    "hẹp ống sống": "M48.0",
    "mày đay": "L50.9", "nổi mề đay": "L50.9",
    "tiền sản giật": "O14.9",
    "hội chứng thận hư": "N04.9",
    "gan nhiễm mỡ": "K76.0",
    "sỏi mật": "K80.2", "sỏi thận": "N20.0",
    "viêm ruột thừa": "K35.8",
    "bệnh dại": "A82", "dại": "A82",
    "thủy đậu": "B01.9",
    "bệnh sởi": "B05", "sởi": "B05",
    "sốt xuất huyết": "A90",
    "trầm cảm": "F32.9",
    "thoái hóa tinh bột": "E85.9", "amyloidosis": "E85.9"
}

def expand_clinical_span(text: str, start: int, end: int, ent_type: str) -> tuple[str, int, int]:
    if ent_type in ["CHẨN_ĐOÁN", "DISEASE"]:
        left_text = text[:start]
        prefix_match = re.search(r"\b(Bệnh|Hội chứng)\s+$", left_text, re.IGNORECASE)
        if prefix_match:
            shift = len(prefix_match.group(0))
            start -= shift

    if ent_type in ["CHẨN_ĐOÁN", "DISEASE"]:
        right_text = text[end:end+35]
        suffix_match = re.search(r"^\s*(nguyên phát|thứ phát|mạn tính|cấp tính|cộng đồng|tự miễn|do rượu|do thuốc)", right_text, re.IGNORECASE)
        if suffix_match:
            shift = len(suffix_match.group(0))
            end += shift

    if ent_type in ["TÊN_XÉT_NGHIỆM", "TEST"]:
        right_text = text[end:end+35]
        suffix_match = re.search(r"^\s*(nội mạc tử cung|buồng trứng|bụng|tim|dạ dày|ngực|não|máu|nước tiểu|tinh dịch|tử cung vòi trứng)", right_text, re.IGNORECASE)
        if suffix_match:
            shift = len(suffix_match.group(0))
            end += shift

    expanded_text = text[start:end].strip()
    return expanded_text, start, end

def expand_drug_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    right_text = text[end:end+45]
    dosage_pattern = re.compile(
        r"^\s*(\d+(\.\d+)?\s*(mg|ml|g|mcg|viên|ống|gói|túi)\b(\s*(po|oral|daily|bid|qid|qhs|q6h|prn|q\d+h:prn))*)",
        re.IGNORECASE
    )
    match = dosage_pattern.search(right_text)
    if match:
        matched_ext = match.group(0)
        new_end = end + len(matched_ext)
        new_text = text[start:new_end].strip()
        return new_text, start, new_end
    return text[start:end].strip(), start, end

def trim_boundary_noise(text: str, start: int, end: int) -> tuple[str, int, int]:
    sub_text = text[start:end]
    l_match = re.search(r"^[!•\-\.\s\d\:\(\)]+", sub_text)
    if l_match:
        shift = len(l_match.group(0))
        start += shift
    
    sub_text = text[start:end]
    r_match = re.search(r"[!•\-\.\s\:\,\(\)]+$", sub_text)
    if r_match:
        shift = len(r_match.group(0))
        end -= shift
        
    cleaned_text = re.sub(r"\s+", " ", text[start:end]).strip()
    return cleaned_text, start, end

def main():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))
    print(f"Generating EXPERIMENTAL Submission for {len(txt_files)} test documents...")

    predictor = MedicalNERPredictor()
    assertion_detector = ClinicalAssertionDetector()
    candidate_ranker = CandidateRetrievalRanker()

    concepts = candidate_ranker.vector_store.concepts

    label_map = {
        "MEDICINE": "THUỐC",
        "SYMPTOM": "TRIỆU_CHỨNG",
        "DISEASE": "CHẨN_ĐOÁN",
        "TEST": "TÊN_XÉT_NGHIỆM"
    }

    blacklist = {
        "bệnh", "bệnh nhân", "người bệnh", "vị trí", "chất", "giảm", "hạt", "chứng", "sức khỏe", 
        "dịch", "chống", "này", "ấy", "đó", "kia", "nào", "gây", "có", "bị", "được", "và", "nhưng", 
        "hoặc", "của", "cho", "trong", "trước", "cơ thể", "nhiễm sắc thể", "hoạt tính", "oxy hóa", 
        "triệu chứng", "chẩn đoán", "điều trị", "sử dụng", "dùng", "uống", "khám", "phát hiện"
    }

    known_drugs_whitelist = {
        "paracetamol", "ibuprofen", "amoxicillin", "metformin", "amlodipine", 
        "insulin", "panadol", "decolgen", "augmentin", "salbutamol", "aspirin",
        "gleevec", "tylenol", "allopurinol", "coumadin", "suboxone", "doxycycline",
        "bactrim", "prednisone", "warfarin", "imatinib", "acetaminophen",
        "vastarel", "nitralmyl", "clonazepam", "metoprolol", "pravastatin", "corticoid"
    }

    short_valid_terms = {"sốt", "đau", "ho", "ngã", "gút", "sở"}

    for txt_file in txt_files:
        doc_id = txt_file.stem
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        historical_spans = []
        for match in re.finditer(r"(^\s*(\d+\.\s*)?(tiền sử|trước nhập viện|các bệnh mãn tính|bệnh cũ).*$)", text, re.MULTILINE | re.IGNORECASE):
            h_start = match.start()
            next_sec = re.search(r"\n\s*\d+\.\s*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẽẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]", text[h_start+1:])
            h_end = (h_start + 1 + next_sec.start()) if next_sec else len(text)
            historical_spans.append((h_start, h_end))

        def is_in_historical_section(pos_start: int, pos_end: int) -> bool:
            return any(hs <= pos_start and pos_end <= he for hs, he in historical_spans)

        # Pass A: Master Concept Matching
        dict_candidates = []
        for concept in concepts:
            display = concept["display"]
            code = concept["code"]
            ent_type = concept["type"]
            comp_type = label_map.get(ent_type, ent_type)

            if len(display) < 3 or display.lower() in blacklist:
                continue

            pattern = re.compile(r"\b" + re.escape(display) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_text = text[start:end]

                if comp_type == "THUỐC":
                    matched_text, start, end = expand_drug_span(text, start, end)
                else:
                    matched_text, start, end = expand_clinical_span(text, start, end, comp_type)

                m_lower = matched_text.lower()
                clean_code = code[7:] if code.startswith("RxNorm:") else code

                # Experimental Transform: Apply Drug Clinical Form Auto-Mapping
                if comp_type == "THUỐC" and m_lower in DRUG_CLINICAL_FORM_MAP:
                    clean_code = DRUG_CLINICAL_FORM_MAP[m_lower]

                # Experimental Transform: Apply 3-Char Root ICD-10 Category
                if comp_type == "CHẨN_ĐOÁN":
                    if m_lower in DIAGNOSIS_ICD_MAP:
                        clean_code = DIAGNOSIS_ICD_MAP[m_lower]
                    if clean_code in ICD_ROOT_MAP:
                        clean_code = ICD_ROOT_MAP[clean_code]

                final_cands = [clean_code] if (comp_type in ["CHẨN_ĐOÁN", "THUỐC"] and clean_code) else []

                dict_candidates.append({
                    "text": matched_text,
                    "type": comp_type,
                    "candidates": final_cands,
                    "assertions": [],
                    "position": [start, end],
                    "source": "dict",
                    "length": end - start
                })

        # Pass B: Model NER prediction with LOW THRESHOLD (0.45) for Dense Extraction
        model_candidates = []
        entities = predictor.predict(text)
        assertions = assertion_detector.detect_assertions(text, entities)

        for ent, ass_tagged in zip(entities, assertions):
            start, end = ent["start"], ent["end"]
            ent_text, start, end = trim_boundary_noise(text, start, end)
            
            if not ent_text:
                continue
                
            ent_type = ent["type"]
            comp_type = label_map.get(ent_type, ent_type)
            ent_lower = ent_text.lower()

            if ent_lower in blacklist or len(ent_text) <= 1:
                continue

            if len(ent_text) <= 2 and ent_lower not in short_valid_terms:
                continue

            if comp_type == "THUỐC":
                ent_text, start, end = expand_drug_span(text, start, end)
            else:
                ent_text, start, end = expand_clinical_span(text, start, end, comp_type)

            ranking_res = candidate_ranker.rank_entity(ent_text, entity_type=ent_type, top_k=1)
            comp_candidates = []
            if ranking_res and ranking_res.top_candidates:
                top_cand = ranking_res.top_candidates[0]
                if top_cand.score >= 0.45: # LOW DENSITY THRESHOLD
                    code = top_cand.code
                    clean_code = code[7:] if code.startswith("RxNorm:") else code
                    comp_candidates.append(clean_code)

            if comp_type in ["CHẨN_ĐOÁN", "THUỐC"]:
                if ent_lower in DIAGNOSIS_ICD_MAP:
                    code = DIAGNOSIS_ICD_MAP[ent_lower]
                    if code in ICD_ROOT_MAP:
                        code = ICD_ROOT_MAP[code]
                    comp_candidates = [code]
                elif ent_lower in DRUG_CLINICAL_FORM_MAP:
                    comp_candidates = [DRUG_CLINICAL_FORM_MAP[ent_lower]]
            else:
                comp_candidates = []

            if comp_type == "THUỐC":
                is_valid_drug = (
                    comp_candidates or 
                    any(d in ent_lower for d in known_drugs_whitelist) or 
                    any(w in ent_lower for w in ["viên", "thuốc", "kháng sinh", "vitamin"])
                )
                if not is_valid_drug:
                    continue

            ass_flags = ass_tagged.assertion
            comp_assertions = []
            if ass_flags.is_negated:
                comp_assertions.append("isNegated")
            if ass_flags.is_family:
                comp_assertions.append("isFamily")
            if ass_flags.is_historical or is_in_historical_section(start, end):
                comp_assertions.append("isHistorical")

            model_candidates.append({
                "text": ent_text,
                "type": comp_type,
                "candidates": comp_candidates,
                "assertions": comp_assertions,
                "position": [start, end],
                "source": "model",
                "length": end - start
            })

        # Pass C: Merge candidates (Dictionary precision & non-empty candidates take top priority)
        all_candidates = dict_candidates + model_candidates
        all_candidates.sort(key=lambda x: (x["source"] == "dict", len(x["candidates"]) > 0, x["length"]), reverse=True)

        merged_entities = []
        covered_indices = set()

        for cand in all_candidates:
            start, end = cand["position"]
            span_indices = set(range(start, end))

            if span_indices.intersection(covered_indices):
                continue

            covered_indices.update(span_indices)

            if cand["source"] == "dict":
                dummy_ent = {"start": start, "end": end, "text": cand["text"]}
                ass_res = assertion_detector.detect_assertions(text, [dummy_ent])[0]
                ass_flags = ass_res.assertion
                comp_assertions = []
                if ass_flags.is_negated:
                    comp_assertions.append("isNegated")
                if ass_flags.is_family:
                    comp_assertions.append("isFamily")
                if ass_flags.is_historical or is_in_historical_section(start, end):
                    comp_assertions.append("isHistorical")
                cand["assertions"] = comp_assertions

            del cand["source"]
            del cand["length"]
            merged_entities.append(cand)

        merged_entities.sort(key=lambda x: x["position"][0])

        out_json_path = output_dir / f"{doc_id}.json"
        with open(out_json_path, "w", encoding="utf-8") as f_out:
            json.dump(merged_entities, f_out, ensure_ascii=False, indent=2)

    zip_path = Path("output.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for json_file in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
            zf.write(json_file, arcname=f"output/{json_file.name}")

    print("Successfully generated BOLD EXPERIMENTAL Submission Package!")

if __name__ == "__main__":
    main()
