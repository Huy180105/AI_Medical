from typing import Any

from src.utils.config import Config


class WandBTracker:
    def __init__(
        self,
        project: str | None = None,
        mode: str | None = None,
    ) -> None:
        self.project = project or Config.WANDB_PROJECT
        self.mode = mode or Config.WANDB_MODE
        self._run = None

    def start(self, run_name: str | None = None, config: dict[str, Any] | None = None):
        import wandb

        self._run = wandb.init(project=self.project, name=run_name, config=config or {}, mode=self.mode)
        return self._run

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
