from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SERVICE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/app/data")

STAGES = {
    11: {
        "name": "Baseline Logistic Regression",
        "script": "src/models/baseline_11.py",
    },
    12: {
        "name": "Random Forest",
        "script": "src/models/random_forest_12.py",
    },
    13: {
        "name": "Deep Learning",
        "script": "src/models/deep_learning_13.py",
    },
    14: {
        "name": "Model Evaluation",
        "script": "src/evaluation/evaluate_14.py",
    },
    15: {
        "name": "Hyperparameter Tuning",
        "script": "src/models/tuning_15.py",
    },
    16: {
        "name": "MLflow Tracking",
        "script": "src/experiments/mlflow_tracking_16.py",
    },
    17: {
        "name": "ONNX Export",
        "script": "src/export/export_onnx_17.py",
    },
    18: {
        "name": "ONNX Benchmarking",
        "script": "src/benchmarking/benchmark_onnx_18.py",
    },
}

LOG_DIR = DATA_ROOT / "pipeline_runs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================


def log(message: str, level: str = "INFO", **details) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **details,
    }

    print(
        f"[{record['timestamp']}] [{level}] {message}",
        flush=True,
    )

    run_id = os.environ.get("PIPELINE_RUN_ID", "manual_ml_run")
    log_path = LOG_DIR / f"ml_training_{run_id}.jsonl"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


# ============================================================
# DATASET VALIDATION
# ============================================================


def validate_split(split_name: str) -> None:
    split_path = DATA_ROOT / "processed" / split_name

    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_name} dataset: {split_path}")

    if split_path.is_file():
        valid = split_path.suffix.lower() == ".parquet"
    else:
        valid = any(split_path.rglob("*.parquet"))

    if not valid:
        raise FileNotFoundError(f"No Parquet data found in {split_path}")


def validate_required_datasets(stages: list[int]) -> None:
    # Training and validation are needed by model training.
    if any(stage in stages for stage in [11, 12, 13, 14, 15]):
        validate_split("train")
        validate_split("validation")

    # Stage 18 benchmarks against the held-out test dataset.
    if 18 in stages:
        validate_split("test")

    log("Required ML datasets validated.")


# ============================================================
# STAGE EXECUTION
# ============================================================


def execute_stage(stage_number: int) -> None:
    stage = STAGES[stage_number]
    script_path = SERVICE_ROOT / stage["script"]

    if not script_path.exists():
        raise FileNotFoundError(f"ML stage script not found: {script_path}")

    log(
        f"Starting ML stage {stage_number}: {stage['name']}",
        stage=stage_number,
        script=stage["script"],
    )

    environment = os.environ.copy()

    # Most existing scripts expect /app/data.
    # Preserve that path because Docker Compose mounts project data there.
    environment["DATA_ROOT"] = "/app/data"
    environment["PYTHONPATH"] = "/app"

    environment.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SERVICE_ROOT,
        env=environment,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ML stage {stage_number} ({stage['name']}) failed "
            f"with exit code {result.returncode}."
        )

    log(
        f"Completed ML stage {stage_number}: {stage['name']}",
        stage=stage_number,
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Vehicle Fault Prediction ML stages."
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=sorted(STAGES.keys()),
        action="append",
        help=(
            "Run a specific stage. Can be supplied more than once. "
            "If omitted, all eight ML stages run."
        ),
    )

    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip initial Parquet dataset validation.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    selected_stages = sorted(set(args.stage)) if args.stage else list(STAGES.keys())

    log(
        "ML training execution started.",
        selected_stages=selected_stages,
    )

    try:
        if not args.skip_dataset_validation:
            validate_required_datasets(selected_stages)

        for stage_number in selected_stages:
            execute_stage(stage_number)

        log(
            "All requested ML stages completed successfully.",
            completed_stages=selected_stages,
        )

        return 0

    except KeyboardInterrupt:
        log("ML training interrupted by user.", level="WARNING")
        return 130

    except Exception as exc:
        log(
            "ML training failed.",
            level="ERROR",
            error=str(exc),
        )

        print(f"\nML TRAINING FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
