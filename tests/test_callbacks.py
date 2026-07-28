from src.mlops.callbacks import MLOpsTrainerCallback


class FakeTracker:
    def __init__(self):
        self.logged = []

    def log_training_metrics(self, metrics, step=None):
        self.logged.append((metrics, step))


class FakeWandB:
    def __init__(self):
        self.logged = []

    def start(self, run_name=None, config=None):
        self.config = config

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


class State:
    global_step = 3


def test_callback_logs_numeric_training_metrics():
    mlflow = FakeTracker()
    wandb = FakeWandB()
    callback = MLOpsTrainerCallback(mlflow_tracker=mlflow, wandb_tracker=wandb)

    callback.on_log(None, State(), None, logs={"loss": 0.2, "learning_rate": 1e-5, "ignored": "x"})

    assert mlflow.logged[0] == ({"loss": 0.2, "learning_rate": 1e-05}, 3)
    assert wandb.logged[0][0]["loss"] == 0.2
