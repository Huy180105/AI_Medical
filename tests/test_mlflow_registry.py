from src.mlops.experiment_manager import ExperimentManager
from src.mlops.model_registry import RegisteredModelInfo
from src.mlops.training_tracker import TrainingRunTracker


class FakeMLflowTracker:
    class RunContext:
        def __enter__(self):
            return "run-1"

        def __exit__(self, exc_type, exc, traceback):
            return False

    def run(self, run_name=None, params=None):
        return self.RunContext()

    def log_training_metrics(self, metrics, step=None):
        self.metrics = metrics

    def log_artifacts(self, artifact_paths):
        self.artifact_paths = artifact_paths


class FakeWandBTracker:
    def start(self, run_name=None, config=None):
        self.config = config

    def log(self, metrics, step=None):
        self.metrics = metrics

    def finish(self):
        self.finished = True


class FakeRegistry:
    def register(self, model_uri, description=None, aliases=None):
        return RegisteredModelInfo(name="model", version="1", source=model_uri)


def test_training_tracker_creates_experiment_and_registers_model(tmp_path):
    tracker = TrainingRunTracker(
        experiment_manager=ExperimentManager(str(tmp_path)),
        mlflow_tracker=FakeMLflowTracker(),
        wandb_tracker=FakeWandBTracker(),
        model_registry=FakeRegistry(),
    )

    result = tracker.track_completed_training(
        run_name="test-run",
        config={"epochs": 1},
        metrics={"loss": 0.1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        model_uri="runs:/run-1/model",
    )

    assert (tmp_path / "test-run" / "config.json").exists()
    assert result["registered_model"]["version"] == "1"
