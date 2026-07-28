import random
import re
from typing import Any


class ClinicalNoiseInjector:
    """
    Injects realistic Vietnamese clinical abbreviations, typos, shorthand variations, and noise.
    """

    CLINICAL_ABBREVIATIONS = {
        "Bệnh nhân": "BN",
        "bệnh nhân": "BN",
        "huyết áp": "HA",
        "Huyết áp": "HA",
        "đái tháo đường": "ĐTD",
        "Đái tháo đường": "ĐTD",
        "chẩn đoán": "CĐ",
        "Chẩn đoán": "CĐ",
        "bác sĩ": "BS",
        "Bác sĩ": "BS",
        "tiền sử": "TS",
        "xét nghiệm": "XN",
        "chụp X-quang": "XQ"
    }

    def inject_noise(self, template_instance: dict[str, Any], noise_probability: float = 0.5) -> dict[str, Any]:
        """
        Applies clinical abbreviation replacements and noise to template text.
        """
        text = template_instance["text"]
        entities = template_instance["entities"]

        if random.random() > noise_probability:
            return template_instance

        modified_text = text

        # Apply Clinical Abbreviations
        for full_word, abbr in self.CLINICAL_ABBREVIATIONS.items():
            if random.random() < 0.6:  # 60% chance to substitute abbreviation
                modified_text = re.sub(r"\b" + re.escape(full_word) + r"\b", abbr, modified_text)

        # Update character offsets for entities if text changed
        updated_entities = []
        for ent in entities:
            ent_text = ent["text"]
            pos = modified_text.find(ent_text)
            if pos != -1:
                new_ent = dict(ent)
                new_ent["start"] = pos
                new_ent["end"] = pos + len(ent_text)
                updated_entities.append(new_ent)
            else:
                updated_entities.append(dict(ent))

        noised_instance = dict(template_instance)
        noised_instance["text"] = modified_text
        noised_instance["entities"] = updated_entities
        noised_instance["has_clinical_noise"] = True

        return noised_instance
