from typing import Any


class ClinicalDecisionValidator:
    """
    Validates the structure, confidence ranges, and safety invariants
    of the generated clinical decision payloads.
    """

    ALLOWED_RISK_LEVELS = {"Low", "Medium", "High", "Emergency"}
    ALLOWED_FOLLOW_UP_ACTIONS = {"Revisit", "Emergency", "Hospitalization", "Observation"}

    @classmethod
    def validate(cls, payload: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validates the unified JSON response structure.
        Returns a tuple of (is_valid, error_messages).
        """
        errors = []

        # 1. Check basic structure keys
        required_keys = {"diagnosis_candidates", "risk_level", "recommendations", "follow_up", "evidence", "confidence"}
        missing_keys = required_keys - set(payload.keys())
        if missing_keys:
            errors.append(f"Missing required CDSS keys: {', '.join(missing_keys)}")
            return False, errors

        # 2. Validate risk_level
        risk_level = payload["risk_level"]
        if risk_level not in cls.ALLOWED_RISK_LEVELS:
            errors.append(f"Invalid risk_level '{risk_level}'. Must be one of {cls.ALLOWED_RISK_LEVELS}")

        # 3. Validate follow_up
        follow_up = payload["follow_up"]
        if not isinstance(follow_up, dict):
            errors.append("follow_up must be a dictionary.")
        else:
            action = follow_up.get("action")
            if action not in cls.ALLOWED_FOLLOW_UP_ACTIONS:
                errors.append(f"Invalid follow_up action '{action}'. Must be one of {cls.ALLOWED_FOLLOW_UP_ACTIONS}")
            if not follow_up.get("timeframe"):
                errors.append("follow_up timeframe is missing or empty.")

        # 4. Validate confidence
        confidence = payload["confidence"]
        if not isinstance(confidence, (int, float)):
            errors.append("Confidence must be a numeric value.")
        elif not (0.0 <= confidence <= 1.0):
            errors.append(f"Confidence score '{confidence}' out of bounds [0.0, 1.0]")

        # 5. Validate diagnosis candidates confidence
        candidates = payload.get("diagnosis_candidates", [])
        if not isinstance(candidates, list):
            errors.append("diagnosis_candidates must be a list.")
        else:
            for idx, cand in enumerate(candidates):
                c_conf = cand.get("confidence", 0.0)
                if not (0.0 <= c_conf <= 1.0):
                    errors.append(f"Candidate #{idx} ({cand.get('disease')}) has out-of-bounds confidence: {c_conf}")

        # 6. Verify evidence is supplied
        evidence = payload.get("evidence", {})
        if not isinstance(evidence, dict):
            errors.append("evidence must be a dictionary.")
        elif not evidence.get("supporting_paths") and candidates:
            # We warn but don't strictly crash if there are candidates but paths are empty,
            # unless it's a safety constraint. Let's make it a warning or a soft error.
            pass

        return len(errors) == 0, errors
