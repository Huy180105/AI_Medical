from pathlib import Path
import pytest
from src.dataset_generator.template_engine import ClinicalTemplateEngine
from src.dataset_generator.clinical_noise_injector import ClinicalNoiseInjector
from src.dataset_generator.synthetic_generator import SyntheticDatasetGenerator


def test_template_engine():
    engine = ClinicalTemplateEngine()
    inst = engine.generate_template_instance(1)

    assert "text" in inst
    assert len(inst["entities"]) >= 4
    assert len(inst["assertions"]) >= 2
    assert len(inst["relations"]) >= 1


def test_clinical_noise_injector():
    injector = ClinicalNoiseInjector()
    template_inst = {
        "text": "Bệnh nhân bị đái tháo đường và huyết áp cao.",
        "entities": [{"text": "đái tháo đường", "type": "DISEASE"}]
    }

    noised = injector.inject_noise(template_inst, noise_probability=1.0)
    assert noised["text"] != template_inst["text"]
    # Check abbreviation replacement (BN, ĐTD, HA)
    assert "BN" in noised["text"] or "HA" in noised["text"] or "ĐTD" in noised["text"]


def test_synthetic_dataset_generator():
    generator = SyntheticDatasetGenerator()
    dataset = generator.generate_dataset(num_samples=10, noise_rate=0.5)
    
    assert len(dataset) == 10
    assert "entities" in dataset[0]

    json_path = generator.export_dataset_json(num_samples=5, filename="test_synthetic.json")
    assert Path(json_path).exists()

    # Cleanup test JSON
    Path(json_path).unlink(missing_ok=True)
