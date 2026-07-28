import argparse
import json
from pathlib import Path
import zipfile
from typing import Any

from competition.pipeline import CompetitionPipeline
from competition.validator import CompetitionJSONValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchSubmissionProcessor:
    """
    Batch processor ingesting directory of clinical text (.txt) files, running pipeline,
    validating output JSONs, and building a single submission ZIP archive.
    """

    def __init__(self) -> None:
        self.pipeline = CompetitionPipeline()
        self.validator = CompetitionJSONValidator()

    def process_directory(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """
        Processes all .txt files in input_dir and writes validated .json files to output_dir.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = list(input_dir.glob("*.txt"))
        logger.info("Found %d clinical text files in '%s'.", len(txt_files), input_dir)

        output_json_paths = []

        for txt_file in txt_files:
            doc_id = txt_file.stem
            with open(txt_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

            record = self.pipeline.process_text(document_id=doc_id, raw_text=raw_text)
            
            # Validate JSON schema
            is_valid, errors = self.validator.validate_json_record(record)
            if not is_valid:
                logger.warning("Validation warnings for document '%s': %s", doc_id, errors)

            output_file = output_dir / f"{doc_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            output_json_paths.append(output_file)

        return output_json_paths

    def create_submission_zip(self, json_files: list[Path], zip_output_path: Path) -> Path:
        """
        Packages a list of output JSON files into a zip archive.
        """
        zip_output_path = Path(zip_output_path)
        zip_output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for json_file in json_files:
                zf.write(json_file, arcname=json_file.name)

        logger.info("Successfully created submission archive '%s' containing %d files.", zip_output_path, len(json_files))
        return zip_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Viettel AI Race Batch Submission Processor")
    parser.add_argument("--input_dir", type=str, default="data/competition/input", help="Path to input directory containing .txt files")
    parser.add_argument("--output_dir", type=str, default="data/competition/output", help="Path to output directory for .json files")
    parser.add_argument("--output_zip", type=str, default="submission.zip", help="Path to output .zip archive")
    args = parser.parse_args()

    processor = BatchSubmissionProcessor()
    json_paths = processor.process_directory(Path(args.input_dir), Path(args.output_dir))
    processor.create_submission_zip(json_paths, Path(args.output_zip))


if __name__ == "__main__":
    main()
