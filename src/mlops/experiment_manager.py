import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import Config


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    config_path: Path
    metrics_path: Path
    plots_dir: Path
    checkpoint_dir: Path
    artifacts_dir: Path


class ExperimentManager:
    def __init__(self, experiments_dir: str | None = None) -> None:
        self.experiments_dir = Path(experiments_dir or Config.EXPERIMENTS_DIR)

    def create_experiment(self, name: str | None = None, config: dict[str, Any] | None = None) -> ExperimentPaths:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_name = name or f"run-{timestamp}"
        root = self.experiments_dir / experiment_name
        paths = ExperimentPaths(
            root=root,
            config_path=root / "config.json",
            metrics_path=root / "metrics.json",
            plots_dir=root / "plots",
            checkpoint_dir=root / "checkpoint",
            artifacts_dir=root / "artifacts",
        )
        for directory in [paths.root, paths.plots_dir, paths.checkpoint_dir, paths.artifacts_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.write_config(paths, config or {})
        self.write_metrics(paths, {})
        return paths

    def write_config(self, paths: ExperimentPaths, config: dict[str, Any]) -> None:
        payload = {
            "experiment": {key: str(value) for key, value in asdict(paths).items()},
            "config": config,
        }
        paths.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def write_metrics(self, paths: ExperimentPaths, metrics: dict[str, Any]) -> None:
        paths.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

    def copy_checkpoint(self, paths: ExperimentPaths, checkpoint_path: str) -> Path:
        source = Path(checkpoint_path)
        destination = paths.checkpoint_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return destination
