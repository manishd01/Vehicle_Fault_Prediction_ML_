import json
import os
from pathlib import Path

import mlflow

DATA_ROOT = Path("/app/data")
METRICS_DIR = DATA_ROOT / "metrics"
MODEL_DIR = DATA_ROOT / "models"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)


def log_experiment(metrics_path, model_dir):
    with open(metrics_path, encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, dict):
        print(f"Skipping unexpected metrics format: {metrics_path}")
        return

    model_name = results.get("model", metrics_path.stem)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("source_metrics_file", metrics_path.name)
        mlflow.log_param("model_name", model_name)

        for key, value in results.items():
            if key in {"parameters", "feature_columns", "features"}:
                continue

            if isinstance(value, (str, int, float, bool)):
                if key in {"model", "selection_metric"}:
                    mlflow.log_param(key, value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, value)

        parameters = results.get("parameters", {})
        if isinstance(parameters, dict):
            mlflow.log_params({key: str(value) for key, value in parameters.items()})

        feature_columns = results.get("feature_columns") or results.get("features")

        if feature_columns:
            mlflow.log_param("feature_count", len(feature_columns))

        mlflow.log_artifact(str(metrics_path))

        artifact_candidates = {
            "random_forest": model_dir / "random_forest.joblib",
            "tuned_random_forest": model_dir / "random_forest_tuned.joblib",
            "pytorch_feedforward_neural_network": model_dir / "deep_learning.pt",
        }

        artifact_path = artifact_candidates.get(model_name)

        if artifact_path and artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="model_files")

        print(f"Logged MLflow run: {model_name}")


def track_experiments(config=None):
    config = config or {}

    metrics_dir = Path(config.get("metrics_dir", METRICS_DIR))
    model_dir = Path(config.get("model_dir", MODEL_DIR))
    tracking_uri = config.get("tracking_uri", TRACKING_URI)
    experiment_name = config.get("experiment_name", "vehicle-fault-prediction")

    print("\n========== STEP 16: MLFLOW TRACKING ==========")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    metrics_files = sorted(metrics_dir.glob("*.json"))

    if not metrics_files:
        raise FileNotFoundError(f"No JSON metric files found in {metrics_dir}")

    for metrics_path in metrics_files:
        try:
            log_experiment(metrics_path, model_dir)
        except Exception as error:
            print(f"Could not log {metrics_path.name}: {error}")

    print(f"\nMLflow tracking URI: {tracking_uri}")
    print("========== STEP 16 COMPLETED ==========")


def main():
    track_experiments()


if __name__ == "__main__":
    main()
