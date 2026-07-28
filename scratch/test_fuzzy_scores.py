import re
import sys

def levenshtein_similarity(s1: str, s2: str) -> float:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    dist = previous_row[-1]
    return 1.0 - dist / max(len(s1), len(s2))

def token_set_ratio(s1: str, s2: str) -> float:
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    diff1 = t1.difference(t2)
    diff2 = t2.difference(t1)
    intersection = t1.intersection(t2)
    
    s_inter = " ".join(sorted(list(intersection)))
    s_diff1 = " ".join(sorted(list(diff1)))
    s_diff2 = " ".join(sorted(list(diff2)))
    
    res1 = s_inter + " " + s_diff1 if s_diff1 else s_inter
    res2 = s_inter + " " + s_diff2 if s_diff2 else s_inter
    
    return max(
        levenshtein_similarity(s_inter, res1),
        levenshtein_similarity(s_inter, res2),
        levenshtein_similarity(res1, res2)
    )

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    t1 = "cao HA"
    t2 = "tăng huyết áp"
    print(f"Levenshtein sim '{t1}' vs '{t2}': {levenshtein_similarity(t1, t2):.4f}")
    print(f"Token Set Ratio '{t1}' vs '{t2}': {token_set_ratio(t1, t2):.4f}")
