import json
from pathlib import Path

import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for doc_id in [1, 15, 20]:
        txt_path = Path(f"input/{doc_id}.txt")
        json_path = Path(f"output/{doc_id}.json")
        
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        with open(json_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
            
        print(f"=========================================")
        print(f"DOCUMENT {doc_id}.txt (Length: {len(text)} chars)")
        print(f"=========================================")
        print(text[:500])
        print(f"\nCurrently extracted entities ({len(entities)}):")
        for e in entities:
            cands = e.get("candidates", [])
            print(f"  - [{e['type']}] '{e['text']}' -> cands: {cands}, ass: {e['assertions']}")
        print("\n")

if __name__ == "__main__":
    main()
