import os
import json
import re
from pathlib import Path
from collections import Counter

def main():
    input_dir = Path("input")
    txt_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))
    print(f"Total test files: {len(txt_files)}")

    titles = []
    all_text = ""
    for p in txt_files:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
            all_text += "\n" + text
            first_line = text.split("\n")[0].strip() if text else ""
            titles.append((p.name, first_line[:60]))

    lines = []
    lines.append("\n--- First 30 File Titles ---")
    for fname, title in titles[:30]:
        lines.append(f"{fname:<8}: {title}")

    words = re.findall(r"\b\w+\b", all_text.lower())
    lines.append(f"\nTotal word count across all test files: {len(words)}")
    
    # Load current output files and check stats
    output_dir = Path("output")
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem))
    total_ents = 0
    type_counts = Counter()
    cand_counts = Counter()
    has_cand_count = 0

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            ents = json.load(f)
            total_ents += len(ents)
            for e in ents:
                type_counts[e["type"]] += 1
                if e["candidates"]:
                    has_cand_count += 1
                    cand_counts[e["candidates"][0]] += 1

    out_text = "\n".join(lines)
    Path("scratch/corpus_stats.txt").write_text(out_text, encoding="utf-8")
    print("Saved stats to scratch/corpus_stats.txt")

if __name__ == "__main__":
    main()
