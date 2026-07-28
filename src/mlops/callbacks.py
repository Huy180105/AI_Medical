import time
from typing import Any

from src.mlops.gpu_monitor import GPUMonitor
from src.mlops.mlflow_tracker import MLflowTracker
from src.mlops.wandb_tracker import WandBTracker


class MLOpsTrainerCallback:
    """Hugging Face Trainer-compatible callback for optional MLOps tracking."""

    def __init__(
        self,
        mlflow_tracker: MLflowTracker | None = None,
        wandb_tracker: WandBTracker | None = None,
        gpu_monitor: GPUMonitor | None = None,
    ) -> None:
        self.mlflow_tracker = mlflow_tracker or MLflowTracker()
        self.wandb_tracker = wandb_tracker or WandBTracker()
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self._start_time = time.perf_counter()

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        config = {
            "learning_rate": getattr(args, "learning_rate", None),
            "epochs": getattr(args, "num_train_epochs", None),
            "train_batch_size": getattr(args, "per_device_train_batch_size", None),
            "eval_batch_size": getattr(args, "per_device_eval_batch_size", None),
        }
        self.wandb_tracker.start(run_name=getattr(args, "run_name", None), config=config)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        metrics = self._numeric_metrics(logs or {})
        if not metrics:
            return
        step = getattr(state, "global_step", None)
        self.mlflow_tracker.log_training_metrics(metrics, step=step)
        self.wandb_tracker.log(metrics, step=step)

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        normalized = self._numeric_metrics(metrics or {})
        step = getattr(state, "global_step", None)
        self.mlflow_tracker.log_training_metrics(normalized, step=step)
        self.wandb_tracker.log(normalized, step=step)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        elapsed = round(time.perf_counter() - self._start_time, 4)
        final_metrics: dict[str, float] = {"training_time_seconds": elapsed}
        for index, gpu in enumerate(self.gpu_monitor.collect()):
            for key, value in gpu.items():
                if isinstance(value, (int, float)):
                    final_metrics[f"gpu_{index}_{key}"] = float(value)
        self.mlflow_tracker.log_training_metrics(final_metrics, step=getattr(state, "global_step", None))
        self.wandb_tracker.log(final_metrics, step=getattr(state, "global_step", None))
        self.wandb_tracker.finish()

    def _numeric_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        return {
            key.replace("eval_", "validation_"): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
