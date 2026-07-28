from typing import Any


class ICD10Service:
    def normalize(self, entity: dict[str, Any]) -> dict[str, Any]:
        text = str(entity.get("text", "")).strip()
        lowered = text.lower()
        mapping = {
            "sot": ("R50.9", "Fever unspecified"),
            "fever": ("R50.9", "Fever unspecified"),
            "ho": ("R05", "Cough"),
            "cough": ("R05", "Cough"),
            "viem phoi": ("J18.9", "Pneumonia unspecified"),
            "pneumonia": ("J18.9", "Pneumonia unspecified"),
        }
        code, name = mapping.get(lowered, ("", text))
        return {
            "original": text,
            "type": entity.get("type", ""),
            "code_system": "ICD-10" if code else "",
            "code": code,
            "name": name,
            "confidence": entity.get("score", 0.0),
        }
