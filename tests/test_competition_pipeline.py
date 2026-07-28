from pathlib import Path
import zipfile
import pytest

from competition.pipeline import CompetitionPipeline
from competition.validator import CompetitionJSONValidator
from competition.zip_submission import BatchSubmissionProcessor


def test_competition_pipeline_process_text():
    pipeline = CompetitionPipeline()
    raw_text = "Bệnh nhân nam 56 tuổi bị ho kéo dài và sốt cao, được bác sĩ chỉ định dùng Paracetamol."
    
    record = pipeline.process_text(document_id="doc_001", raw_text=raw_text)
    
    assert record["document_id"] == "doc_001"
    assert "entities" in record
    assert "assertions" in record
    assert "relations" in record
    assert "icd10_codes" in record or "rxnorm_codes" in record


def test_competition_json_validator():
    validator = CompetitionJSONValidator()
    valid_record = {
        "document_id": "doc_001",
        "text": "sample text",
        "entities": [{"text": "sốt cao", "type": "SYMPTOM"}],
        "assertions": [],
        "relations": [],
        "icd10_codes": [],
        "rxnorm_codes": []
    }
    
    is_valid, errors = validator.validate_json_record(valid_record)
    assert is_valid is True
    assert len(errors) == 0

    invalid_record = {"text": "sample text"}
    is_valid, errors = validator.validate_json_record(invalid_record)
    assert is_valid is False
    assert len(errors) > 0


def test_batch_submission_processor(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    zip_path = tmp_path / "submission.zip"
    
    input_dir.mkdir()
    
    # Create sample input txt files
    (input_dir / "sample1.txt").write_text("Bệnh nhân bị trào ngược dạ dày.", encoding="utf-8")
    (input_dir / "sample2.txt").write_text("Bệnh nhân dùng Aspirin 100mg.", encoding="utf-8")

    processor = BatchSubmissionProcessor()
    json_files = processor.process_directory(input_dir, output_dir)
    
    assert len(json_files) == 2
    for jf in json_files:
        assert jf.exists()

    archive_path = processor.create_submission_zip(json_files, zip_path)
    assert archive_path.exists()
    
    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        assert "sample1.json" in namelist
        assert "sample2.json" in namelist
