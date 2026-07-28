import json
from pathlib import Path
from collections import Counter

def main():
    output_dir = Path("output")
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))

    # High-precision entity whitelist for candidate assignment
    # (High specificity diseases and medicines only)
    high_precision_candidates = {
        # High specificity diseases
        "d55.0", "g20", "m30.3", "e28.2", "c22.1", "c92.1", "i21.9", "i63.9", "a82",
        "308135", "243670", "866436", "392085", "7597", "313782", "904475", "1099279", "312935", "197527", "197528", "282388"
    }

    total_ents = 0
    total_with_cands = 0
    code_counts = Counter()

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            entities = json.load(f)
            total_ents += len(entities)

            for e in entities:
                cands = e.get("candidates", [])
                if cands:
                    code = cands[0]
                    # Filter candidates to high-precision set
                    code_lower = code.lower()
                    if code_lower in high_precision_candidates or e["type"] == "THUỐC":
                        total_with_cands += 1
                        code_counts[code] += 1
                    else:
                        e["candidates"] = []

    print(f"Total entities: {total_ents}")
    print(f"Filtered entities with candidates: {total_with_cands} ({total_with_cands/total_ents*100:.1f}%)")
    print(f"Top candidates: {code_counts.most_common(15)}")

if __name__ == "__main__":
    main()
