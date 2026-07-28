import re
from typing import Any
from src.assertion.assertion_models import ClinicalAssertion, AssertionTaggedEntity
from src.assertion.patterns import (
    REGEX_TERMINATORS,
    REGEX_NEGATION,
    REGEX_FAMILY,
    REGEX_HISTORICAL,
    REGEX_UNCERTAINTY,
    REGEX_CONDITIONAL
)


class ClinicalAssertionDetector:
    """
    Context-aware Clinical Assertion Engine for Vietnamese medical text.
    Classifies entity status flags: is_negated, is_family, is_historical, is_uncertain, is_conditional.
    """

    def __init__(self, context_window_chars: int = 60) -> None:
        self.context_window_chars = context_window_chars

    def detect_assertions(self, text: str, entities: list[dict[str, Any]]) -> list[AssertionTaggedEntity]:
        """
        Processes text and entity offsets, returning enriched AssertionTaggedEntity models.
        """
        tagged_entities = []

        for ent in entities:
            ent_text = ent.get("text", "")
            ent_type = ent.get("type", "UNKNOWN")
            
            # Find entity position in text if start/end not provided
            start = ent.get("start", -1)
            end = ent.get("end", -1)

            if start == -1 or end == -1:
                pos = text.find(ent_text)
                if pos != -1:
                    start = pos
                    end = pos + len(ent_text)
                else:
                    start = 0
                    end = len(ent_text)

            # Extract left and right context windows
            left_window_start = max(0, start - self.context_window_chars)
            left_context = text[left_window_start:start]

            right_window_end = min(len(text), end + self.context_window_chars)
            right_context = text[end:right_window_end]

            # Determine assertion flags
            assertion = self._evaluate_context(left_context, right_context)

            tagged = AssertionTaggedEntity(
                text=ent_text,
                type=ent_type,
                start=start,
                end=end,
                assertion=assertion
            )
            tagged_entities.append(tagged)

        return tagged_entities

    def _evaluate_context(self, left_context: str, right_context: str) -> ClinicalAssertion:
        """
        Evaluates left and right context windows against regex trigger patterns and scope boundaries.
        """
        assertion = ClinicalAssertion()

        # Truncate left context at nearest scope terminator
        left_cleaned = self._truncate_at_terminator(left_context, from_left=True)
        # Truncate right context at nearest scope terminator
        right_cleaned = self._truncate_at_terminator(right_context, from_left=False)

        combined_window = f"{left_cleaned} ENTITY {right_cleaned}"

        # 1. Check Negation
        if REGEX_NEGATION.search(left_cleaned):
            assertion.is_negated = True

        # 2. Check Family History
        if REGEX_FAMILY.search(left_cleaned):
            assertion.is_family = True

        # 3. Check Historical Context
        if REGEX_HISTORICAL.search(left_cleaned):
            assertion.is_historical = True

        # 4. Check Uncertainty
        if REGEX_UNCERTAINTY.search(left_cleaned):
            assertion.is_uncertain = True

        # 5. Check Conditional
        if REGEX_CONDITIONAL.search(left_cleaned):
            assertion.is_conditional = True

        return assertion

    def _truncate_at_terminator(self, context_str: str, from_left: bool) -> str:
        """
        Cleans context by cutting at sentence boundaries or conjunction terminators.
        """
        matches = list(REGEX_TERMINATORS.finditer(context_str))
        if not matches:
            return context_str

        if from_left:
            # Cut at the rightmost terminator in left context
            last_match = matches[-1]
            return context_str[last_match.end():]
        else:
            # Cut at the leftmost terminator in right context
            first_match = matches[0]
            return context_str[:first_match.start()]
