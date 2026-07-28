from pathlib import Path
from typing import Any

from src.mlops.experiment_manager import ExperimentManager
from src.mlops.mlflow_tracker import MLflowTracker
from src.mlops.model_registry import ModelRegistry
from src.mlops.wandb_tracker import WandBTracker
from src.utils.config import Config


class TrainingRunTracker:
    def __init__(
        self,
        experiment_manager: ExperimentManager | None = None,
        mlflow_tracker: MLflowTracker | None = None,
        wandb_tracker: WandBTracker | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.mlflow_tracker = mlflow_tracker or MLflowTracker()
        self.wandb_tracker = wandb_tracker or WandBTracker()
        self.model_registry = model_registry or ModelRegistry()

    def track_completed_training(
        self,
        run_name: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        artifact_paths: list[str] | None = None,
        model_uri: str | None = None,
        register_model: bool = True,
    ) -> dict[str, Any]:
        paths = self.experiment_manager.create_experiment(name=run_name, config=config)
        self.experiment_manager.write_metrics(paths, metrics)

        for artifact_path in artifact_paths or []:
            path = Path(artifact_path)
            if path.exists() and (path.name.startswith("checkpoint") or path.is_file()):
                self.experiment_manager.copy_checkpoint(paths, str(path))

        with self.mlflow_tracker.run(run_name=run_name, params=config) as run_id:
            self.mlflow_tracker.log_training_metrics(metrics)
            self.mlflow_tracker.log_artifacts([str(paths.root), *(artifact_paths or [])])
            registered = None
            if register_model and model_uri:
                registered = self.model_registry.register(
                    model_uri=model_uri,
                    description=f"Registered from run {run_id}",
                    aliases=["latest"],
                )

        self.wandb_tracker.start(run_name=run_name, config=config)
        self.wandb_tracker.log(metrics)
        self.wandb_tracker.finish()

        return {
            "experiment_dir": str(paths.root),
            "mlflow_experiment": Config.MLFLOW_EXPERIMENT_NAME,
            "registered_model": registered.__dict__ if registered else None,
        }
