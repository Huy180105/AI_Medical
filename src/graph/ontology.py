from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DISEASE = "Disease"
    DRUG = "Drug"
    SYMPTOM = "Symptom"
    LAB = "Lab"
    PROCEDURE = "Procedure"
    ICD10 = "ICD10"
    GUIDELINE = "Guideline"
    CONDITION = "Condition"
    COMPLICATION = "Complication"


class EdgeType(str, Enum):
    HAS_SYMPTOM = "has_symptom"
    TREATS = "treats"
    REQUIRES_TEST = "requires_test"
    CONTRAINDICATED_FOR = "contraindicated_for"
    MAPPED_TO = "mapped_to"
    RELATED_TO = "related_to"
    HAS_COMPLICATION = "has_complication"
    GUIDED_BY = "guided_by"
    MENTIONS = "mentions"


class MedicalOntology:
    SYMPTOM_ALIASES: dict[str, list[str]] = {
        "fever": ["fever", "sot", "high temperature"],
        "cough": ["cough", "ho"],
        "sore throat": ["sore throat", "dau hong"],
        "runny nose": ["runny nose", "so mui"],
        "chest pain": ["chest pain", "dau nguc"],
        "shortness of breath": ["shortness of breath", "dyspnea", "kho tho"],
        "headache": ["headache", "dau dau"],
        "nausea": ["nausea", "buon non"],
        "vomiting": ["vomiting", "non"],
        "diarrhea": ["diarrhea", "tieu chay"],
        "pain": ["pain", "dau"],
    }

    DISEASE_ALIASES: dict[str, list[str]] = {
        "Acute upper respiratory infection": ["respiratory infection", "upper respiratory infection", "viral infection"],
        "Pneumonia unspecified": ["pneumonia", "lower respiratory infection"],
        "Fever unspecified": ["febrile illness", "fever"],
        "Cough": ["cough"],
        "Migraine unspecified": ["migraine", "headache"],
        "Kidney disease": ["kidney disease", "renal disease"],
        "Liver disease": ["liver disease"],
        "Pregnancy": ["pregnancy", "pregnant"],
        "Gastric ulcer": ["gastric ulcer", "ulcer"],
        "Dehydration": ["dehydration"],
        "Bacterial infection": ["bacterial infection"],
    }

    LAB_ALIASES: dict[str, list[str]] = {
        "Oxygen saturation": ["oxygen saturation", "spo2"],
        "Respiratory rate": ["respiratory rate"],
        "Chest X-ray": ["chest x-ray", "xray", "abnormal lung findings"],
        "Complete blood count": ["complete blood count", "cbc"],
        "Liver function test": ["liver disease", "liver function"],
        "Renal function test": ["renal disease", "kidney disease", "renal function"],
    }

    PROCEDURE_ALIASES: dict[str, list[str]] = {
        "Clinical follow up": ["clinical follow up", "follow up"],
        "Hydration assessment": ["hydration", "dehydration"],
    }

    COMPLICATION_ALIASES: dict[str, list[str]] = {
        "Dyspnea": ["dyspnea", "shortness of breath"],
        "Hypoxemia": ["oxygen saturation below", "low oxygen"],
        "Dehydration": ["dehydration"],
    }

    @classmethod
    def canonicalize(cls, value: str, candidates: dict[str, list[str]]) -> str | None:
        normalized = cls.normalize_text(value)
        for canonical, aliases in candidates.items():
            if normalized == cls.normalize_text(canonical):
                return canonical
            if any(normalized == cls.normalize_text(alias) for alias in aliases):
                return canonical
        return None

    @staticmethod
    def normalize_text(value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())

    @classmethod
    def contains_any(cls, text: str, aliases: list[str]) -> bool:
        normalized = cls.normalize_text(text)
        return any(cls.normalize_text(alias) in normalized for alias in aliases)

    @classmethod
    def node_id(cls, node_type: NodeType, name: str) -> str:
        slug = cls.normalize_text(name).replace(" ", "_")
        return f"{node_type.value}:{slug}"

    @classmethod
    def node_payload(cls, node_type: NodeType, name: str, **attributes: Any) -> dict[str, Any]:
        return {
            "id": cls.node_id(node_type, name),
            "name": name,
            "type": node_type.value,
            **attributes,
        }
