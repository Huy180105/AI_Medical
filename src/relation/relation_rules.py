import re

# Allowed Entity Type Pairs for each Relation Category
DOMAIN_CONSTRAINTS = {
    "TREATS": {
        "subject_types": {"MEDICINE", "DRUG", "PROCEDURE", "TREATMENT"},
        "object_types": {"DISEASE", "SYMPTOM", "CONDITION"}
    },
    "CAUSED_BY": {
        "subject_types": {"SYMPTOM", "DISEASE", "CONDITION"},
        "object_types": {"DISEASE", "CONDITION", "CAUSE"}
    },
    "HAS_SYMPTOM": {
        "subject_types": {"DISEASE", "CONDITION"},
        "object_types": {"SYMPTOM", "SIGN"}
    },
    "HAS_TEST_RESULT": {
        "subject_types": {"TEST", "PROCEDURE", "LAB"},
        "object_types": {"VALUE", "RESULT", "MEASUREMENT", "NUM"}
    },
    "CONTRAINDICATED_FOR": {
        "subject_types": {"MEDICINE", "DRUG"},
        "object_types": {"DISEASE", "CONDITION"}
    }
}

# Relation Verbal Patterns
PATTERNS_TREATS = [
    r"\bđiều trị\b", r"\bchữa\b", r"\bdùng để\b", r"\bđược chỉ định\b",
    r"\bgiảm\b", r"\bkháng\b", r"\bđẩy lùi\b", r"\bđược dùng\b"
]

PATTERNS_CAUSED_BY = [
    r"\bdo\b", r"\bgây ra bởi\b", r"\bnguyên nhân do\b", r"\bdẫn đến từ\b",
    r"\bbiến chứng của\b", r"\bdo nhiễm\b", r"\bdo bị\b"
]

PATTERNS_HAS_SYMPTOM = [
    r"\bcó triệu chứng\b", r"\bbiểu hiện\b", r"\bkèm theo\b", r"\bgồm có\b",
    r"\bxuất hiện\b", r"\bkèm\b", r"\bdấu hiệu\b"
]

PATTERNS_HAS_TEST_RESULT = [
    r"\bkết quả\b", r"\bcho kết quả\b", r"\bđạt\b", r"\bchỉ số\b", r"=", r":"
]

PATTERNS_CONTRAINDICATED_FOR = [
    r"\bchống chỉ định\b", r"\bkhông dùng cho\b", r"\btránh dùng\b",
    r"\bnguy hiểm ở\b", r"\bkhông nên dùng\b"
]


def compile_regex_list(patterns: list[str]) -> re.Pattern:
    combined = "|".join(patterns)
    return re.compile(combined, re.IGNORECASE | re.UNICODE)


REGEX_TREATS = compile_regex_list(PATTERNS_TREATS)
REGEX_CAUSED_BY = compile_regex_list(PATTERNS_CAUSED_BY)
REGEX_HAS_SYMPTOM = compile_regex_list(PATTERNS_HAS_SYMPTOM)
REGEX_HAS_TEST_RESULT = compile_regex_list(PATTERNS_HAS_TEST_RESULT)
REGEX_CONTRAINDICATED_FOR = compile_regex_list(PATTERNS_CONTRAINDICATED_FOR)
