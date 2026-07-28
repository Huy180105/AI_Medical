# Medical AI Agent MLOps Runbook

## Services

Start the production-shaped stack:

```powershell
docker compose up --build
```

Service endpoints:

- FastAPI app: `http://localhost:8000`
- Metrics API: `http://localhost:9000/metrics`
- MLflow UI: `http://localhost:5000`
- Redis: `localhost:6379`
- Postgres: `localhost:5432`

## MLflow

Environment variables:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `MLFLOW_REGISTERED_MODEL_NAME`

The `src.mlops.mlflow_tracker.MLflowTracker` module tracks:

- training loss
- validation metrics
- precision
- recall
- F1
- GPU memory
- training time
- artifacts
- model artifacts

## Model Registry

Register a model URI:

```powershell
python scripts/register_model.py register --model-uri runs:/<run_id>/model --description "PhoBERT NER production candidate"
```

Inspect latest:

```powershell
python scripts/register_model.py latest
```

Rollback the `latest` alias:

```powershell
python scripts/register_model.py rollback --version <version>
```

## Experiments

Experiment outputs are stored under `experiments/`.

Each experiment has:

- `config.json`
- `metrics.json`
- `plots/`
- `checkpoint/`
- `artifacts/`

## Benchmarks

Create a JSON file containing a list of sample medical texts, then run:

```powershell
python scripts/run_benchmark.py --samples samples.json --output-dir experiments/benchmarks
```

Generated reports:

- `benchmark.json`
- `benchmark.csv`
- `benchmark.html`

## Metrics API

The metrics service is intentionally separate from the main medical API so monitoring can evolve without changing AI inference logic.

`GET /metrics` returns:

- GPU metrics
- CPU metrics
- RAM metrics
- Redis health
- Postgres health
- latency probe result
