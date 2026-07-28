import json

from src.knowledge.loader import KnowledgeLoader


def test_loader_reads_csv_and_json_documents(tmp_path):
    csv_path = tmp_path / "icd10.csv"
    csv_path.write_text(
        "code,title,description,category,keywords\n"
        "R50.9,Fever,Fever symptom,symptom,fever;sot\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "medical_guideline.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "id": "GL-1",
                    "title": "Respiratory guidance",
                    "text": "Assess fever and cough.",
                    "category": "guideline",
                    "keywords": ["fever", "cough"],
                }
            ]
        ),
        encoding="utf-8",
    )

    documents = KnowledgeLoader(str(tmp_path)).load_documents()

    assert len(documents) == 2
    assert {document.metadata["source_type"] for document in documents} == {"icd10", "medical_guideline"}
    assert any("Fever" in document.text for document in documents)
