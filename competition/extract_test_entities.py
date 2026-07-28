import json
from pathlib import Path
from src.inference.predict_ner import MedicalNERPredictor

def main():
    input_dir = Path("input")
    txt_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))
    
    predictor = MedicalNERPredictor()
    
    unique_entities = {}
    
    print(f"Extracting entities from {len(txt_files)} files...")
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        entities = predictor.predict(text)
        for ent in entities:
            text_ent = ent["text"].strip()
            ent_type = ent["type"]
            if not text_ent:
                continue
            
            if text_ent not in unique_entities:
                unique_entities[text_ent] = {
                    "type": ent_type,
                    "count": 0,
                    "samples": []
                }
            unique_entities[text_ent]["count"] += 1
            if len(unique_entities[text_ent]["samples"]) < 3:
                unique_entities[text_ent]["samples"].append(txt_file.name)
                
    # Save results
    output_path = Path("competition/extracted_test_entities.json")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(unique_entities, f_out, ensure_ascii=False, indent=2)
        
    print(f"Extraction complete! Found {len(unique_entities)} unique entities. Saved to {output_path}")

if __name__ == "__main__":
    main()
