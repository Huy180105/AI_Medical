from src.relation.relation_models import ClinicalRelation


class RelationPostProcessor:
    """
    Deduplicates and filters extracted clinical relations.
    """

    @staticmethod
    def filter_and_deduplicate(relations: list[ClinicalRelation], min_confidence: float = 0.5) -> list[ClinicalRelation]:
        """
        Removes duplicates and relations below min_confidence.
        """
        seen_keys = set()
        filtered = []

        for rel in relations:
            if rel.confidence < min_confidence:
                continue

            key = (rel.subject.text.lower(), rel.relation_type, rel.object.text.lower())
            if key not in seen_keys:
                seen_keys.add(key)
                filtered.append(rel)

        return filtered
