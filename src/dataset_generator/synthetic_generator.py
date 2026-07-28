import json
from pathlib import Path
from typing import Any
from src.dataset_generator.template_engine import ClinicalTemplateEngine
from src.dataset_generator.clinical_noise_injector import ClinicalNoiseInjector
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYNTHETIC_DIR = Path("data/synthetic")
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


class SyntheticDatasetGenerator:
    """
    Master Synthetic Dataset Generator for Vietnamese Clinical NLP and Competition Training.
    Synthesizes clinical notes with ground-truth NER, Assertion, Relation, and ICD-10/RxNorm codes.
    """

    def __init__(self, output_dir: Path = SYNTHETIC_DIR) -> None:
        self.template_engine = ClinicalTemplateEngine()
        self.noise_injector = ClinicalNoiseInjector()
        self.output_dir = output_dir

    def generate_dataset(self, num_samples: int = 100, noise_rate: float = 0.5) -> list[dict[str, Any]]:
        """
        Synthesizes num_samples annotated clinical instances.
        """
        logger.info("Synthesizing %d clinical document instances (noise rate: %.2f)...", num_samples, noise_rate)
        dataset = []

        for i in range(1, num_samples + 1):
            template_inst = self.template_engine.generate_template_instance(i)
            noised_inst = self.noise_injector.inject_noise(template_inst, noise_probability=noise_rate)
            dataset.append(noised_inst)

        return dataset

    def export_dataset_json(self, num_samples: int = 100, filename: str = "synthetic_clinical_dataset.json") -> str:
        """
        Generates dataset and writes it to JSON file.
        """
        dataset = self.generate_dataset(num_samples=num_samples)
        file_path = self.output_dir / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        logger.info("Successfully exported %d synthetic clinical records to '%s'.", len(dataset), file_path)
        return str(file_path)
