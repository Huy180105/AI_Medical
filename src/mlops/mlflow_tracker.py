import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.mlops.gpu_monitor import GPUMonitor
from src.utils.config import Config


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        gpu_monitor: GPUMonitor | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or Config.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        self.gpu_monitor = gpu_monitor or GPUMonitor()

    @contextmanager
    def run(self, run_name: str | None = None, params: dict[str, Any] | None = None) -> Iterator[str]:
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        start = time.perf_counter()
        with mlflow.start_run(run_name=run_name) as active_run:
            if params:
                mlflow.log_params(params)
            try:
                yield active_run.info.run_id
            finally:
                mlflow.log_metric("training_time_seconds", round(time.perf_counter() - start, 4))
                for index, gpu in enumerate(self.gpu_monitor.collect()):
                    for key, value in gpu.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"gpu_{index}_{key}", value)

    def log_training_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        import mlflow

        for key, value in metrics.items():
            mlflow.log_metric(key, float(value), step=step)

    def log_artifacts(self, artifact_paths: list[str]) -> None:
        import mlflow

        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            if path.exists():
                if path.is_dir():
                    mlflow.log_artifacts(str(path))
                else:
                    mlflow.log_artifact(str(path))

    def log_transformers_model(self, model: Any, tokenizer: Any, artifact_path: str = "model") -> str:
        import mlflow

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path=artifact_path,
        )
        return f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
