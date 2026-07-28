from dataclasses import dataclass
from typing import Any

from src.utils.config import Config


@dataclass(frozen=True)
class RegisteredModelInfo:
    name: str
    version: str
    source: str
    stage: str | None = None


class ModelRegistry:
    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or Config.MLFLOW_TRACKING_URI
        self.model_name = model_name or Config.MLFLOW_REGISTERED_MODEL_NAME

    def register(self, model_uri: str, description: str | None = None, aliases: list[str] | None = None) -> RegisteredModelInfo:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        result = mlflow.register_model(model_uri=model_uri, name=self.model_name)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        if description:
            client.update_model_version(
                name=self.model_name,
                version=result.version,
                description=description,
            )
        for alias in aliases or ["latest"]:
            client.set_registered_model_alias(self.model_name, alias, result.version)
        return RegisteredModelInfo(name=self.model_name, version=result.version, source=model_uri)

    def latest(self, alias: str = "latest") -> dict[str, Any]:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self.tracking_uri)
        version = client.get_model_version_by_alias(self.model_name, alias)
        return {
            "name": version.name,
            "version": version.version,
            "source": version.source,
            "run_id": version.run_id,
            "status": version.status,
            "aliases": list(version.aliases),
        }

    def rollback(self, target_version: str, alias: str = "latest") -> dict[str, Any]:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.set_registered_model_alias(self.model_name, alias, target_version)
        return self.latest(alias=alias)
