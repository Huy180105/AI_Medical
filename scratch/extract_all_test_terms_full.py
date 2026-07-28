import json
import sys
from pathlib import Path
from collections import Counter
from src.inference.predict_ner import MedicalNERPredictor

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    input_dir = Path("input")
    txt_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))
    
    predictor = MedicalNERPredictor()
    disease_terms = Counter()
    drug_terms = Counter()
    
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        entities = predictor.predict(text)
        for ent in entities:
            txt = ent["text"].lower().strip()
            etype = ent["type"]
            if etype == "DISEASE":
                disease_terms[txt] += 1
            elif etype == "MEDICINE":
                drug_terms[txt] += 1
                
    print("TOP 100 DISEASE TERMS:")
    for term, count in disease_terms.most_common(100):
        print(f"  - '{term}': {count}")
        
    print("\nTOP 100 DRUG TERMS:")
    for term, count in drug_terms.most_common(100):
        print(f"  - '{term}': {count}")

if __name__ == "__main__":
    main()
