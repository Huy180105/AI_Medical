from pathlib import Path
import pytest
from src.training.clinical_trainer import ClinicalFineTuningTrainer


def test_clinical_fine_tuning_trainer(tmp_path: Path):
    trainer = ClinicalFineTuningTrainer(
        model_name="vinai/phobert-base",
        epochs=2,
        patience=2,
        checkpoint_dir=tmp_path
    )
    
    train_data = [{"text": "Sample train text"}]
    valid_data = [{"text": "Sample valid text"}]
    
    result = trainer.train_pipeline(train_data, valid_data)
    
    assert result["model_name"] == "vinai/phobert-base"
    assert result["best_f1_score"] > 0.0
    assert Path(result["best_checkpoint_path"]).exists()

