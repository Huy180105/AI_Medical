from typing import Any


class RxNormService:
    def normalize(self, entity: dict[str, Any]) -> dict[str, Any]:
        text = str(entity.get("text", "")).strip()
        lowered = text.lower()
        mapping = {
            "paracetamol": ("RX-PARA", "Paracetamol"),
            "acetaminophen": ("RX-PARA", "Paracetamol"),
            "ibuprofen": ("RX-IBU", "Ibuprofen"),
            "amoxicillin": ("RX-AMOX", "Amoxicillin"),
        }
        code, name = mapping.get(lowered, ("", text))
        return {
            "original": text,
            "type": entity.get("type", ""),
            "code_system": "RxNorm" if code else "",
            "code": code,
            "name": name,
            "confidence": entity.get("score", 0.0),
        }
