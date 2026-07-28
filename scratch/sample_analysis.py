import json
from pathlib import Path

def main():
    sample_ids = ["3", "5", "7", "13", "18", "24"]
    lines = []
    
    for sid in sample_ids:
        txt_path = Path(f"input/{sid}.txt")
        json_path = Path(f"output/{sid}.json")
        
        if not txt_path.exists() or not json_path.exists():
            continue
            
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        with open(json_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
            
        lines.append(f"==================================================")
        lines.append(f"DOCUMENT {sid}.txt ({len(raw_text)} chars)")
        lines.append(f"Raw Text Excerpt:\n{raw_text[:300]}...")
        lines.append(f"\nExtracted Entities ({len(preds)} total):")
        for p in preds:
            lines.append(f"  - '{p['text']}' [{p['type']}] cand={p['candidates']} ass={p['assertions']}")
        lines.append("\n")

    Path("scratch/sample_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Saved sample analysis to scratch/sample_analysis.txt")

if __name__ == "__main__":
    main()
