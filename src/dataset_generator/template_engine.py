import random
from typing import Any


class ClinicalTemplateEngine:
    """
    Synthesizes realistic Vietnamese clinical document templates across medical specialties
    (Respiratory, Cardiology, Endocrinology, Nephrology) with ground-truth annotations.
    """

    PATIENT_PROFILES = [
        {"gender": "Nam", "age_min": 30, "age_max": 75},
        {"gender": "Nữ", "age_min": 25, "age_max": 80}
    ]

    SYMPTOM_GROUPS = [
        {"symptom": "ho kéo dài và ho đờm xanh", "type": "SYMPTOM", "disease": "viêm phổi thùy", "icd": "J18.9", "drug": "Amoxicillin", "rx": "RxNorm:7052"},
        {"symptom": "đau ngực dữ dội và khó thở cấp", "type": "SYMPTOM", "disease": "nhồi máu cơ tim cấp", "icd": "I21.9", "drug": "Aspirin", "rx": "RxNorm:1191"},
        {"symptom": "sốt cao 39.5 độ C và sốt xuất huyết", "type": "SYMPTOM", "disease": "sốt xuất huyết Dengue", "icd": "A90", "drug": "Paracetamol", "rx": "RxNorm:161"},
        {"symptom": "ợ chua và trào ngược dạ dày", "type": "SYMPTOM", "disease": "trào ngược dạ dày thực quản", "icd": "K21.9", "drug": "Omeprazole", "rx": "RxNorm:7646"}
    ]

    LAB_TESTS = [
        {"test": "chụp X-quang ngực", "type": "TEST"},
        {"test": "xét nghiệm công thức máu WBC", "type": "TEST"},
        {"test": "xét nghiệm chức năng thận Creatinine", "type": "TEST"},
        {"test": "điện tâm đồ ECG", "type": "TEST"}
    ]

    def generate_template_instance(self, instance_id: int) -> dict[str, Any]:
        """
        Generates a single annotated clinical note template.
        """
        profile = random.choice(self.PATIENT_PROFILES)
        age = random.randint(profile["age_min"], profile["age_max"])
        grp = random.choice(self.SYMPTOM_GROUPS)
        lab = random.choice(self.LAB_TESTS)

        text = (
            f"Bệnh nhân {profile['gender'].lower()} {age} tuổi, vào viện vì {grp['symptom']}. "
            f"Bệnh nhân nghi ngờ {grp['disease']}, được chỉ định làm {lab['test']}. "
            f"Bác sĩ kê đơn điều trị bằng {grp['drug']}."
        )

        entities = [
            {"text": grp["symptom"], "type": "SYMPTOM"},
            {"text": grp["disease"], "type": "DISEASE", "icd10_code": grp["icd"]},
            {"text": lab["test"], "type": "TEST"},
            {"text": grp["drug"], "type": "MEDICINE", "rxnorm_code": grp["rx"]}
        ]

        assertions = [
            {"text": grp["symptom"], "is_negated": False, "is_family": False, "is_historical": False, "is_uncertain": False, "is_conditional": False},
            {"text": grp["disease"], "is_negated": False, "is_family": False, "is_historical": False, "is_uncertain": True, "is_conditional": False}
        ]

        relations = [
            {"subject": grp["drug"], "relation_type": "TREATS", "object": grp["disease"]}
        ]

        return {
            "instance_id": f"synth_{instance_id:06d}",
            "text": text,
            "entities": entities,
            "assertions": assertions,
            "relations": relations
        }
