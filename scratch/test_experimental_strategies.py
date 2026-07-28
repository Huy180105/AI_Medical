import re
import json
from pathlib import Path

# 3-Character Root ICD-10 Category Mapping
ICD_ROOT_MAP = {
    "D55.0": "D55", "J84.9": "J84", "D84.9": "D84", "L03.9": "L03", "N90.8": "N90",
    "G20": "G20", "M30.3": "M30", "E28.2": "E28", "C22.1": "C22", "C92.1": "C92",
    "I20.9": "I20", "I25.9": "I25", "I21.9": "I21", "I63.9": "I63", "I10": "I10",
    "E11.9": "E11", "K21.9": "K21", "J18.9": "J18", "A15.0": "A15", "K70.3": "K70",
    "K74.6": "K74", "I50.9": "I50", "B18.2": "B18", "B18.1": "B18", "M10.9": "M10",
    "J45.9": "J45", "J20.9": "J20", "D59.9": "D59", "D64.9": "D64", "N17.9": "N17",
    "N18.9": "N18", "N39.0": "N39", "E78.5": "E78", "M48.0": "M48", "L50.9": "L50",
    "O14.9": "O14", "N04.9": "N04", "K76.0": "K76", "K80.2": "K80", "N20.0": "N20",
    "K35.8": "K35", "A82": "A82", "B01.9": "B01", "B05": "B05", "A90": "A90",
    "F32.9": "F32", "E85.9": "E85"
}

# Generic Drug to Default Clinical Form RxCUI
DRUG_CLINICAL_FORM_MAP = {
    "paracetamol": "313782",
    "acetaminophen": "313782",
    "tylenol": "202433",
    "aspirin": "243670",
    "amlodipine": "308135",
    "metoprolol": "866436",
    "omeprazole": "7646",
    "allopurinol": "656",
    "suboxone": "353062",
    "doxycycline": "3640",
    "bactrim": "135834",
    "corticoid": "261551",
    "imatinib": "282388",
    "gleevec": "282388",
    "pravastatin": "904475",
    "clonazepam": "197527",
    "senna": "312935",
    "docusate": "1099279"
}

def apply_experimental_transform(cand_code: str, etype: str) -> str:
    """
    Transforms candidate code according to bold experimental hypotheses.
    """
    if etype == "CHẨN_ĐOÁN" and cand_code in ICD_ROOT_MAP:
        return ICD_ROOT_MAP[cand_code]
    return cand_code

if __name__ == "__main__":
    print("Experimental strategy helpers loaded successfully!")
