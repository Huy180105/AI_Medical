import re
import json
from pathlib import Path

def main():
    # Test sample 1.txt and verify candidate assignment for drugs and diseases
    from competition.generate_submission import expand_drug_span
    
    sample_text = "Danh sách thuốc trước nhập viện: 1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily"
    print("Sample text:", sample_text)
    
    # Test expand_drug_span
    match = re.search(r"amlodipine", sample_text)
    if match:
        start, end = match.span()
        expanded_text, s, e = expand_drug_span(sample_text, start, end)
        print(f"Expanded drug: '{expanded_text}' [{s}, {e}]")

if __name__ == "__main__":
    main()
