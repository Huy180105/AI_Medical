import argparse
import json
from pathlib import Path

from src.agent import MedicalAgent
from src.inference.predict_ner import MedicalNERPredictor
from src.mlops.benchmark import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Medical AI Agent benchmark and export reports.")
    parser.add_argument("--samples", type=str, required=True, help="Path to JSON file containing a list of input texts.")
    parser.add_argument("--output-dir", type=str, default="experiments/benchmarks", help="Report output directory.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup samples before measurement.")
    args = parser.parse_args()

    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not all(isinstance(item, str) for item in samples):
        raise ValueError("--samples must point to a JSON array of strings.")

    predictor = MedicalNERPredictor()
    agent = MedicalAgent(ner_predictor=predictor)
    report = BenchmarkRunner().run_agent_benchmark(
        agent=agent,
        samples=samples,
        warmup_runs=args.warmup_runs,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["latency"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
