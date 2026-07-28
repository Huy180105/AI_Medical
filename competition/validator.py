from typing import Any


class CompetitionJSONValidator:
    """
    Validates generated JSON documents against the Viettel AI Race competition schema specifications.
    """

    REQUIRED_KEYS = [
        "document_id",
        "text",
        "entities",
        "assertions",
        "relations",
        "icd10_codes",
        "rxnorm_codes"
    ]

    def validate_json_record(self, record: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validates a single competition JSON dictionary.
        Returns (is_valid, list_of_errors).
        """
        errors = []

        # 1. Key existence checks
        for k in self.REQUIRED_KEYS:
            if k not in record:
                errors.append(f"Missing required key: '{k}'")

        if errors:
            return False, errors

        # 2. Type checks
        if not isinstance(record["document_id"], str) or not record["document_id"]:
            errors.append("'document_id' must be a non-empty string.")

        if not isinstance(record["text"], str):
            errors.append("'text' must be a string.")

        if not isinstance(record["entities"], list):
            errors.append("'entities' must be a list.")
        else:
            for idx, ent in enumerate(record["entities"]):
                if not isinstance(ent, dict):
                    errors.append(f"Entity at index {idx} must be a dictionary.")
                else:
                    if "text" not in ent or "type" not in ent:
                        errors.append(f"Entity at index {idx} missing 'text' or 'type'.")

        if not isinstance(record["assertions"], list):
            errors.append("'assertions' must be a list.")

        if not isinstance(record["relations"], list):
            errors.append("'relations' must be a list.")

        if not isinstance(record["icd10_codes"], list):
            errors.append("'icd10_codes' must be a list.")

        if not isinstance(record["rxnorm_codes"], list):
            errors.append("'rxnorm_codes' must be a list.")

        is_valid = len(errors) == 0
        return is_valid, errors
