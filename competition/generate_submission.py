import re
import os
import json
import zipfile
from pathlib import Path
from src.inference.predict_ner import MedicalNERPredictor
from src.assertion.assertion_detector import ClinicalAssertionDetector
from src.ranking.candidate_ranker import CandidateRetrievalRanker

# Base Blacklist
blacklist = {
    "bệnh", "bệnh nhân", "người bệnh", "vị trí", "chất", "giảm", "hạt", "chứng", "sức khỏe", 
    "dịch", "chống", "này", "ấy", "đó", "kia", "nào", "gây", "có", "bị", "được", "và", "nhưng", 
    "hoặc", "của", "cho", "trong", "trước", "cơ thể", "triệu chứng", "chẩn đoán", "điều trị", 
    "sử dụng", "dùng", "uống", "khám", "phát hiện", "nhãn khoa", "da liễu", "trung ương"
}

def is_blacklisted(text: str) -> bool:
    text_lower = text.lower().strip()
    if text_lower in blacklist or len(text_lower) <= 2:
        return True
    
    # Substring matches to prevent complex genetic/mothball false positives
    substring_blacklist = [
        "nhiễm sắc thể", "gen lặn", "đột biến gen", "di truyền", "đột biến",
        "đậu tằm", "băng phiến", "long não", "mong manh", "phá hủy", "tác nhân", 
        "chuyển hóa", "chó", "mèo", "đồ lót", "date]", "tinh bột", "acid", 
        "creatinine", "glucose", "ct", "men", "thiếu men", "lắng đọng", "đại thực bào",
        "tổ thương", "mô bệnh học", "lyssavirus", "lyssaviridae", "thiết bị", "thương mại",
        "nghệ tách", "tinh dầu", "bột nghệ", "yakult", "cà phê", "đơn vị", "chỉ số", "băng",
        "phiến", "mực", "lưu shunt", "nhu mô", "cánh tay", "ngừa thai", "tổn thương",
        "bách huyết", "doppler", "anti", "hở", "llq", "pcp", "troponin", "inr", "paroxysmal"
    ]
    for bad in substring_blacklist:
        if bad in text_lower:
            return True
    return False

# Verified Disease ICD-10 Ontology Mapping
VERIFIED_DIAGNOSIS_ICD_MAP = {
    "thiếu men g6pd": "D55.0", "bệnh thiếu men g6pd": "D55.0", "g6pd": "D55.0",
    "bệnh phổi kẽ": "J84.9", "phổi kẽ": "J84.9",
    "suy gián miễn dịch": "D84.9",
    "viêm mô tế bào": "L03.9",
    "tổn thương vùng âm hộ": "N90.8", "tổn thương vùng âm hộ phải": "N90.8",
    "bệnh parkinson": "G20", "parkinson": "G20",
    "bệnh kawasaki": "M30.3", "kawasaki": "M30.3",
    "hội chứng buồng trứng đa nang": "E28.2", "buồng trứng đa nang": "E28.2",
    "ung thư biểu mô tế bào mật": "C22.1", "cholangiocarcinoma": "C22.1",
    "bạch cầu dòng tủy mạn tính": "C92.1", "cml": "C92.1",
    "bạch cầu dòng tủy mãn tính": "C92.1",
    "đau thắt ngực": "I20.9", "thiếu máu cơ tim": "I25.9", "nhồi máu cơ tim": "I21.9",
    "đột quỵ": "I63.9", "tai biến mạch máu não": "I63.9",
    "tăng huyết áp": "I10", "tăng huyết áp nguyên phát": "I10", "cao huyết áp": "I10", "tha": "I10",
    "đái tháo đường": "E11.9", "tiểu đường": "E11.9", "đtđ": "E11.9", "đái tháo đường típ 2": "E11.9",
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
    "mày đay": "L50.9", "nổi mề đay": "L50.9", "mày đay vô căn": "L50.9", "mày đay mạn": "L50.8",
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
    "thoái hóa tinh bột": "E85.9", "amyloidosis": "E85.9",
    "viêm sung huyết hang vị dạ dày": "K29.5",
    "viêm hang vị sung huyết": "K29.5",
    "giả gout": "M11.2",
    "rung nhĩ": "I48.91",
    "mụn trứng cá": "L70.0",
    "dị ứng": "T78.40",
    "nấm bẹn": "B35.6",
    "béo phì": "E66.9",
    "run tay": "R25.1",
    "rối loạn cảm xúc": "F34.8",
    "não úng tủy": "G91.9",
    "não úng thủy": "G91.9",
    "bệnh mạch vành": "I25.10",
    "áp xe phổi": "J85.2",
    "thuyên tắc phổi": "I26.9",
    "u xơ tuyến vú": "D24.9",
    "u nang tuyến vú": "N60.0",
    "hội chứng ruột kích thích": "K58.9",
    "loét tá tràng": "K26.9",
    "xơ vữa động mạch": "I70.90",
    "tụ máu": "T14.8",
    "nghiện rượu": "F10.20",
    "ảo thanh": "R44.0",
    "viêm nha chu": "K05.3",
    "tăng nhãn áp": "H40.9"
}

# Verified Specific Medicine RxNorm Map
VERIFIED_DRUG_RXNORM_MAP = {
    "amlodipine": "308135", "amlodipine 10 mg": "308135",
    "aspirin": "243670", "aspirin 81 mg": "243670",
    "metoprolol": "866436", "metoprolol 50 mg": "866436",
    "acetaminophen": "313782", "paracetamol": "313782",
    "pravastatin": "904475", "pravastatin 40 mg": "904475",
    "docusate": "1099279", "docusate 100 mg": "1099279",
    "senna": "312935", "senna 8.6 mg": "312935",
    "clonazepam": "197527", "clonazepam 0.5 mg": "197527",
    "gleevec": "282388", "imatinib": "282388",
    "tylenol": "202433",
    "omeprazole": "7646",
    "allopurinol": "656",
    "suboxone": "353062",
    "vitamin k": "11246",
    "doxycycline": "3640",
    "bactrim": "135834",
    "corticoid": "261551",
    "b12": "11252",
    "torsemide": "10737",
    "coumadin": "202421",
    "crestor": "377884",
    "cotrimoxazol": "259238",
    "doxycyclin": "3640",
    "nitroglycerin": "7431",
    "furosemide": "4603",
    "insulin glargine": "274785",
    "rosuvastatin": "301542",
    "carvedilol": "20352",
    "clonidine": "2599",
    "vitamin b12": "11252"
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
    l_match = re.search(r"[!•\-\.\s\d\:\(\)]+", sub_text)
    if l_match and sub_text.startswith(l_match.group(0)):
        shift = len(l_match.group(0))
        start += shift
    
    sub_text = text[start:end]
    r_match = re.search(r"[!•\-\.\s\:\,\(\)]+$", sub_text)
    if r_match:
        shift = len(r_match.group(0))
        end -= shift
        
    cleaned_text = re.sub(r"\s+", " ", text[start:end]).strip()
    return cleaned_text, start, end

def extract_test_results(text: str, merged_entities: list[dict]) -> list[dict]:
    """Scans for test results immediately following any TÊN_XÉT_NGHIỆM."""
    results = []
    result_pattern = re.compile(
        r"^\s*([\:\=\s]+)?(\d+[\,\.]\d+|\b(âm tính|dương tính|tăng|giảm|bình thường)\b)",
        re.IGNORECASE
    )

    for ent in merged_entities:
        if ent["type"] != "TÊN_XÉT_NGHIỆM":
            continue
        start, end = ent["position"]
        lookahead = text[end:end+25]
        match = result_pattern.search(lookahead)
        if match:
            matched_text = match.group(2).strip()
            match_start = end + lookahead.find(matched_text)
            match_end = match_start + len(matched_text)
            
            results.append({
                "text": matched_text,
                "type": "KẾ_QUẢ_XÉT_NGHIỆM",
                "candidates": [],
                "assertions": [],
                "position": [match_start, match_end]
            })
    return results

def main():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))
    print(f"Running HYBRID Pipeline with Precise Single-Candidate Mapping on {len(txt_files)} test documents...")

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

    known_drugs_whitelist = {
        "paracetamol", "ibuprofen", "amoxicillin", "metformin", "amlodipine", 
        "insulin", "panadol", "decolgen", "augmentin", "salbutamol", "aspirin",
        "gleevec", "tylenol", "allopurinol", "coumadin", "suboxone", "doxycycline",
        "bactrim", "prednisone", "warfarin", "imatinib", "acetaminophen",
        "vastarel", "nitralmyl", "clonazepam", "metoprolol", "pravastatin", "corticoid",
        "crestor", "cotrimoxazol", "doxycyclin", "nitroglycerin", "furosemide", 
        "rosuvastatin", "carvedilol", "clonidine"
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

        # PASS A: Dictionary Concept Matching
        dict_candidates = []
        for concept in concepts:
            display = concept["display"]
            code = concept["code"]
            ent_type = concept["type"]
            comp_type = label_map.get(ent_type, ent_type)

            if is_blacklisted(display):
                continue

            pattern = re.compile(r"\b" + re.escape(display) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_text = text[start:end]

                if comp_type == "THUỐC":
                    matched_text, start, end = expand_drug_span(text, start, end)
                else:
                    matched_text, start, end = expand_clinical_span(text, start, end, comp_type)

                if is_blacklisted(matched_text):
                    continue

                m_lower = matched_text.lower()
                clean_code = code[7:] if code.startswith("RxNorm:") else code

                if "g6pd" in m_lower or "glucose-6-phosphate" in m_lower:
                    comp_type = "CHẨN_ĐOÁN"

                # Assign exactly 1 candidate code to prevent union penalties
                final_cands = []
                if comp_type in ["CHẨN_ĐOÁN", "THUỐC"]:
                    if comp_type == "CHẨN_ĐOÁN" and m_lower in VERIFIED_DIAGNOSIS_ICD_MAP:
                        final_cands = [VERIFIED_DIAGNOSIS_ICD_MAP[m_lower]]
                    elif comp_type == "THUỐC" and m_lower in VERIFIED_DRUG_RXNORM_MAP:
                        final_cands = [VERIFIED_DRUG_RXNORM_MAP[m_lower]]
                    else:
                        ranking_res = candidate_ranker.rank_entity(matched_text, entity_type=ent_type, top_k=1)
                        if ranking_res and ranking_res.top_candidates:
                            tc = ranking_res.top_candidates[0]
                            if tc.score >= 0.65:
                                cc = tc.code[7:] if tc.code.startswith("RxNorm:") else tc.code
                                final_cands = [cc]

                dict_candidates.append({
                    "text": matched_text,
                    "type": comp_type,
                    "candidates": final_cands,
                    "assertions": [],
                    "position": [start, end],
                    "source": "dict",
                    "length": end - start
                })

        # PASS B: Model NER prediction
        model_candidates = []
        entities = predictor.predict(text)
        assertions = assertion_detector.detect_assertions(text, entities)

        for ent, ass_tagged in zip(entities, assertions):
            start, end = ent["start"], ent["end"]
            ent_text, start, end = trim_boundary_noise(text, start, end)
            
            if not ent_text or is_blacklisted(ent_text):
                continue
                
            ent_type = ent["type"]
            comp_type = label_map.get(ent_type, ent_type)
            ent_lower = ent_text.lower()

            if "g6pd" in ent_lower or "glucose-6-phosphate" in ent_lower:
                comp_type = "CHẨN_ĐOÁN"

            if comp_type == "THUỐC":
                ent_text, start, end = expand_drug_span(text, start, end)
            else:
                ent_text, start, end = expand_clinical_span(text, start, end, comp_type)

            if is_blacklisted(ent_text):
                continue

            ent_lower = ent_text.lower()

            # Assign exactly 1 candidate code
            comp_candidates = []
            if comp_type in ["CHẨN_ĐOÁN", "THUỐC"]:
                if comp_type == "CHẨN_ĐOÁN" and ent_lower in VERIFIED_DIAGNOSIS_ICD_MAP:
                    comp_candidates = [VERIFIED_DIAGNOSIS_ICD_MAP[ent_lower]]
                elif comp_type == "THUỐC" and ent_lower in VERIFIED_DRUG_RXNORM_MAP:
                    comp_candidates = [VERIFIED_DRUG_RXNORM_MAP[ent_lower]]
                else:
                    ranking_res = candidate_ranker.rank_entity(ent_text, entity_type=ent_type, top_k=1)
                    if ranking_res and ranking_res.top_candidates:
                        tc = ranking_res.top_candidates[0]
                        if tc.score >= 0.65:
                            cc = tc.code[7:] if tc.code.startswith("RxNorm:") else tc.code
                            comp_candidates = [cc]

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

        # PASS C: Merge candidates (Dictionary precision & non-empty candidates take top priority)
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

        # PASS D: Extract KẾT_QUẢ_XÉT_NGHIỆM (Test Results) based on TÊN_XÉT_NGHIỆM
        test_results = extract_test_results(text, merged_entities)
        merged_entities.extend(test_results)

        merged_entities.sort(key=lambda x: x["position"][0])

        out_json_path = output_dir / f"{doc_id}.json"
        with open(out_json_path, "w", encoding="utf-8") as f_out:
            json.dump(merged_entities, f_out, ensure_ascii=False, indent=2)

    zip_path = Path("output.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for json_file in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
            zf.write(json_file, arcname=f"output/{json_file.name}")

    print("Successfully generated Dense Semantic Single-Candidate Submission Package!")

if __name__ == "__main__":
    main()
