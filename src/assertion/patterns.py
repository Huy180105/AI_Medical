import re

# Scope terminators that stop assertion propagation
SCOPE_TERMINATORS = [
    r"\bnhưng\b", r"\btuy nhiên\b", r"\bngoại trừ\b", r"\bngoài ra\b",
    r"\bchỉ có\b", r"\bhiện tại\b", r"[.;!?]"
]

# Negation Triggers (Chỉ dấu phủ định)
NEGATION_PATTERNS = [
    r"\bkhông\b", r"\bchưa\b", r"\bkhông thấy\b", r"\bkhông có\b",
    r"\bbác bỏ\b", r"\bphủ nhận\b", r"\bâm tính\b", r"\bkhông phát hiện\b",
    r"\bchưa phát hiện\b", r"\bhoàn toàn không\b", r"\bkhông ghi nhận\b"
]

# Family History Triggers (Chỉ dấu tiền sử gia đình)
FAMILY_PATTERNS = [
    r"\bmẹ\b", r"\bbố\b", r"\bcha\b", r"\bông nội\b", r"\bông ngoại\b",
    r"\bbà nội\b", r"\bbà ngoại\b", r"\banh trai\b", r"\bchị gái\b", r"\bem gái\b",
    r"\bgia đình\b", r"\btiền sử gia đình\b", r"\btiền căn gia đình\b", r"\bdi truyền\b"
]

# Historical Context Triggers (Chỉ dấu tiền sử bản thân / quá khứ)
HISTORICAL_PATTERNS = [
    r"\btiền sử\b", r"\btiền căn\b", r"\bđã từng\b", r"\btrước đây\b",
    r"\bquá khứ\b", r"\bđã mổ\b", r"\bđã điều trị\b", r"\bnăm ngoái\b",
    r"\btháng trước\b", r"\bcách đây\b", r"\blịch sử\b"
]

# Uncertainty Triggers (Chỉ dấu nghi ngờ / giả định)
UNCERTAINTY_PATTERNS = [
    r"\bnghi ngờ\b", r"\btheo dõi\b", r"\bchưa loại trừ\b", r"\bkhả năng\b",
    r"\bnghi\b", r"\bcó thể\b", r"\bchưa rõ\b", r"\bchưa xác định\b", r"\bchẩn đoán phân biệt\b"
]

# Conditional Triggers (Chỉ dấu điều kiện)
CONDITIONAL_PATTERNS = [
    r"\bnếu\b", r"\bkhi nào\b", r"\btrong trường hợp\b", r"\bnếu có\b", r"\bphòng khi\b"
]


def compile_regex_list(patterns: list[str]) -> re.Pattern:
    """Compiles a list of regex pattern strings into a single regex object."""
    combined = "|".join(patterns)
    return re.compile(combined, re.IGNORECASE | re.UNICODE)


REGEX_TERMINATORS = compile_regex_list(SCOPE_TERMINATORS)
REGEX_NEGATION = compile_regex_list(NEGATION_PATTERNS)
REGEX_FAMILY = compile_regex_list(FAMILY_PATTERNS)
REGEX_HISTORICAL = compile_regex_list(HISTORICAL_PATTERNS)
REGEX_UNCERTAINTY = compile_regex_list(UNCERTAINTY_PATTERNS)
REGEX_CONDITIONAL = compile_regex_list(CONDITIONAL_PATTERNS)
