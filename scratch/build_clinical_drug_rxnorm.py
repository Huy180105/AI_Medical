import json

def get_clinical_drug_rxnorm_map():
    # RxCUI Clinical Drug (SCD/SBD) mapping for drug + strength combinations
    clinical_drugs = [
        {"code": "308135", "display": "amlodipine 10 mg", "type": "MEDICINE"},
        {"code": "308135", "display": "amlodipine 10mg", "type": "MEDICINE"},
        {"code": "243670", "display": "aspirin 81 mg", "type": "MEDICINE"},
        {"code": "243670", "display": "aspirin 81mg", "type": "MEDICINE"},
        {"code": "866436", "display": "metoprolol succinate xl 50 mg", "type": "MEDICINE"},
        {"code": "866436", "display": "metoprolol 50 mg", "type": "MEDICINE"},
        {"code": "392085", "display": "guaifenesin", "type": "MEDICINE"},
        {"code": "7597", "display": "nystatin", "type": "MEDICINE"},
        {"code": "313782", "display": "acetaminophen 325-650 mg", "type": "MEDICINE"},
        {"code": "313782", "display": "acetaminophen 325 mg", "type": "MEDICINE"},
        {"code": "313782", "display": "paracetamol 500 mg", "type": "MEDICINE"},
        {"code": "313782", "display": "paracetamol 500mg", "type": "MEDICINE"},
        {"code": "904475", "display": "pravastatin 40 mg", "type": "MEDICINE"},
        {"code": "904475", "display": "pravastatin 40mg", "type": "MEDICINE"},
        {"code": "1099279", "display": "docusate sodium 100 mg", "type": "MEDICINE"},
        {"code": "1099279", "display": "docusate 100 mg", "type": "MEDICINE"},
        {"code": "312935", "display": "senna 8.6 mg", "type": "MEDICINE"},
        {"code": "312935", "display": "senna 8.6mg", "type": "MEDICINE"},
        {"code": "197527", "display": "clonazepam 0.5 mg", "type": "MEDICINE"},
        {"code": "197527", "display": "clonazepam 0.5mg", "type": "MEDICINE"},
        {"code": "197528", "display": "clonazepam 1.5 mg", "type": "MEDICINE"},
        {"code": "197528", "display": "clonazepam 1.5mg", "type": "MEDICINE"},
        {"code": "197526", "display": "clonazepam 1 mg", "type": "MEDICINE"},
        {"code": "197526", "display": "clonazepam 1mg", "type": "MEDICINE"},
        {"code": "197525", "display": "clonazepam 2 mg", "type": "MEDICINE"},
        {"code": "197525", "display": "clonazepam 2mg", "type": "MEDICINE"},
        {"code": "311207", "display": "ibuprofen 400 mg", "type": "MEDICINE"},
        {"code": "311207", "display": "ibuprofen 400mg", "type": "MEDICINE"},
        {"code": "310965", "display": "amoxicillin 500 mg", "type": "MEDICINE"},
        {"code": "310965", "display": "amoxicillin 500mg", "type": "MEDICINE"},
        {"code": "860975", "display": "metformin 500 mg", "type": "MEDICINE"},
        {"code": "860975", "display": "metformin 500mg", "type": "MEDICINE"},
        {"code": "860979", "display": "metformin 850 mg", "type": "MEDICINE"},
        {"code": "860983", "display": "metformin 1000 mg", "type": "MEDICINE"},
        {"code": "197361", "display": "atorvastatin 10 mg", "type": "MEDICINE"},
        {"code": "197362", "display": "atorvastatin 20 mg", "type": "MEDICINE"},
        {"code": "197363", "display": "atorvastatin 40 mg", "type": "MEDICINE"},
        {"code": "197364", "display": "atorvastatin 80 mg", "type": "MEDICINE"},
        {"code": "312435", "display": "omeprazole 20 mg", "type": "MEDICINE"},
        {"code": "312435", "display": "omeprazole 20mg", "type": "MEDICINE"}
    ]
    return clinical_drugs

if __name__ == "__main__":
    m = get_clinical_drug_rxnorm_map()
    print(f"Generated {len(m)} clinical drug RxCUI concepts!")
