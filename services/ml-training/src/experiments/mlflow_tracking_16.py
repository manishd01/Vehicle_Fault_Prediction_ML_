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


def log_experiment(metrics_path: Path):
    with open(metrics_path, encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, dict):
        print(f"Skipping unexpected metrics format: {metrics_path}")
        return

    model_name = results.get(
        "model",
        metrics_path.stem,
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("source_metrics_file", metrics_path.name)
        mlflow.log_param("model_name", model_name)

        # Log scalar parameters.
        for key, value in results.items():
            if key in {"parameters", "feature_columns", "features"}:
                continue

            if isinstance(value, (str, int, float, bool)):
                if key in {
                    "model",
                    "selection_metric",
                }:
                    mlflow.log_param(key, value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, value)

        parameters = results.get("parameters", {})
        if isinstance(parameters, dict):
            mlflow.log_params({key: str(value) for key, value in parameters.items()})

        feature_columns = results.get("feature_columns") or results.get("features")

        if feature_columns:
            mlflow.log_param(
                "feature_count",
                len(feature_columns),
            )

        mlflow.log_artifact(str(metrics_path))

        # Attach a model artifact when available.
        artifact_candidates = {
            "random_forest": MODEL_DIR / "random_forest.joblib",
            "tuned_random_forest": MODEL_DIR / "random_forest_tuned.joblib",
            "pytorch_feedforward_neural_network": MODEL_DIR / "deep_learning.pt",
        }

        artifact_path = artifact_candidates.get(model_name)

        if artifact_path and artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="model_files")

        print(f"Logged MLflow run: {model_name}")


def main():
    print("\n========== STEP 16: MLFLOW TRACKING ==========")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("vehicle-fault-prediction")

    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Metrics directory not found: {METRICS_DIR}")

    metrics_files = sorted(METRICS_DIR.glob("*.json"))

    if not metrics_files:
        raise FileNotFoundError(f"No JSON metric files found in {METRICS_DIR}")

    for metrics_path in metrics_files:
        try:
            log_experiment(metrics_path)
        except Exception as error:
            print(f"Could not log {metrics_path.name}: {error}")

    print(f"\nMLflow tracking URI: {TRACKING_URI}")
    print("========== STEP 16 COMPLETED ==========")


if __name__ == "__main__":
    main()
