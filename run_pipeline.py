# # PS M:\projects\Resum_project\MLProject\Vehicle_Fault_Prediction> python .\run_pipeline.py ".\data\raw\sample_small_meterpt3.csv"
# python .\run_pipeline.py --mode local --input .\data\raw\sample_small_meterpt3.csv    ----coorrectone:

# # # Local execution
# # python .\run_pipeline.py .\data\raw\sample_small_meterpt3.csv --mode local

# # # Databricks execution
# # python .\run_pipeline.py .\data\raw\sample_small_meterpt3.csv --mode databricks


from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
RUNS_ROOT = DATA_ROOT / "pipeline_runs"

PYSPARK_SERVICE = "pyspark"
ML_SERVICE = "ml-training"

DEFAULT_INPUT = DATA_ROOT / "raw" / "MetroPT3(AirCompressor).csv"

LOCAL_JOBS = [
    (1, "01_ingest.py", "Raw CSV ingestion"),
    (2, "02_inspect_parquet.py", "Parquet inspection"),
    (3, "03_validate_data.py", "Data validation"),
    (4, "04_data_quality.py", "Data quality checks"),
    (5, "05_time_series_analysis.py", "Time-series analysis"),
    (6, "06_time_series_preparation.py", "Time-series preparation"),
    (7, "07_target_creation.py", "Fault target creation"),
    (8, "08_feature_engineering.py", "Feature engineering"),
    (9, "09_feature_analysis.py", "Feature analysis"),
    (10, "10_train_val_test_split.py", "Train/validation/test split"),
]

ML_STAGES = [
    (11, "Baseline model", "src/models/baseline_11.py"),
    (12, "Random Forest model", "src/models/random_forest_12.py"),
    (13, "Deep learning model", "src/models/deep_learning_13.py"),
    (14, "Model evaluation", "src/evaluation/evaluate_14.py"),
    (15, "Hyperparameter tuning", "src/models/tuning_15.py"),
    (16, "MLflow tracking", "src/experiments/mlflow_tracking_16.py"),
    (17, "ONNX export", "src/export/export_onnx_17.py"),
    (18, "ONNX benchmarking", "src/benchmarking/benchmark_onnx_18.py"),
]

DATABRICKS_NOTEBOOKS = [
    (1, "01_bronze_ingestion"),
    (2, "02_silver_quality"),
    (3, "03_silver_timeseries"),
    (4, "04_gold_labels"),
    (5, "05_gold_features"),
    (6, "06_gold_dataset"),
]


# ============================================================
# LOGGING AND CHECKPOINTS
# ============================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Pipeline:
    def __init__(self, mode: str, run_id: str, checkpoint: dict[str, Any]):
        self.mode = mode
        self.run_id = run_id
        self.checkpoint = checkpoint

        RUNS_ROOT.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = RUNS_ROOT / "pipeline_checkpoint.json"
        self.log_path = RUNS_ROOT / f"{run_id}.jsonl"

    def log(self, level: str, message: str, **details: Any) -> None:
        record = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "mode": self.mode,
            "level": level,
            "message": message,
            **details,
        }

        print(
            f"[{record['timestamp']}] [{level}] {message}",
            flush=True,
        )

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=str) + "\n")

    def save_checkpoint(self) -> None:
        self.checkpoint["updated_at"] = utc_now()

        temporary_path = self.checkpoint_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self.checkpoint, indent=4, default=str),
            encoding="utf-8",
        )
        temporary_path.replace(self.checkpoint_path)

    def stage_status(self, stage: int, status: str, **details: Any) -> None:
        key = str(stage)

        self.checkpoint.setdefault("stages", {})
        self.checkpoint["stages"][key] = {
            "status": status,
            "updated_at": utc_now(),
            **details,
        }

        self.save_checkpoint()

    def stage_completed(self, stage: int) -> bool:
        return (
            self.checkpoint.get("stages", {}).get(str(stage), {}).get("status")
            == "COMPLETED"
        )


# ============================================================
# INPUT FINGERPRINT
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_input_fingerprint(path: Path) -> str:
    return sha256_file(path)


def validate_input_path(path: Path) -> Path:
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("The local ingestion job currently expects a CSV file.")

    # The CSV must be inside the project data directory because Docker
    # mounts PROJECT_ROOT/data to /app/data.
    try:
        path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "The input CSV must be inside the project's data directory.\n"
            f"Project data directory: {DATA_ROOT}\n"
            f"Provided file: {path}"
        ) from exc

    return path


def container_data_path(host_path: Path) -> str:
    relative_path = host_path.resolve().relative_to(DATA_ROOT.resolve())
    return f"/app/data/{relative_path.as_posix()}"


# ============================================================
# DOCKER COMMAND EXECUTION
# ============================================================


def run_command(
    command: list[str],
    pipeline: Pipeline,
    description: str,
) -> None:
    pipeline.log(
        "INFO",
        f"Starting: {description}",
        command=command,
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}.")

    pipeline.log("INFO", f"Completed: {description}")


def run_local_pyspark_job(
    job_file: str,
    input_container_path: str,
    pipeline: Pipeline,
) -> None:
    environment_args = [
        "-e",
        "DATA_ROOT=/app/data",
        "-e",
        f"PARQUET_PATH={input_container_path}",
    ]

    if job_file == "01_ingest.py":
        environment_args.extend(
            [
                "-e",
                f"CSV_PATH={pipeline.checkpoint['input_container_path']}",
            ]
        )

    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        *environment_args,
        PYSPARK_SERVICE,
        "python",
        f"jobs/{job_file}",
    ]

    run_command(command, pipeline, job_file)


def run_local_ml_stage(stage: int, pipeline: Pipeline) -> None:
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        "-e",
        "DATA_ROOT=/app/data",
        "-e",
        "PYTHONPATH=/app",
        "-e",
        f"PIPELINE_RUN_ID={pipeline.run_id}",
        "-e",
        "MLFLOW_TRACKING_URI=http://mlflow:5000",
        ML_SERVICE,
        "python",
        "run_ml_training.py",
        "--stage",
        str(stage),
    ]

    run_command(command, pipeline, f"ML stage {stage}")


# ============================================================
# DATASET CONTRACT
# ============================================================


def validate_ml_datasets() -> None:
    required_splits = ["train", "validation", "test"]

    for split in required_splits:
        split_path = PROCESSED_ROOT / split

        if not split_path.exists():
            raise FileNotFoundError(f"Required ML dataset is missing: {split_path}")

        parquet_files = (
            list(split_path.rglob("*.parquet")) if split_path.is_dir() else [split_path]
        )

        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for the '{split}' dataset: " f"{split_path}"
            )

    print("ML input datasets validated:")
    for split in required_splits:
        print(f"  - {PROCESSED_ROOT / split}")


# ============================================================
# DATABRICKS JOB EXECUTION
# ============================================================


def databricks_request(
    host: str,
    token: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{host.rstrip('/')}/api/2.1/{endpoint}"

    body = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Databricks API error {exc.code}: {response_body}") from exc


def run_databricks_job(pipeline: Pipeline) -> None:
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    job_id = os.environ.get("DATABRICKS_JOB_ID", "").strip()

    if not host or not token or not job_id:
        raise RuntimeError(
            "Databricks mode requires DATABRICKS_HOST, "
            "DATABRICKS_TOKEN and DATABRICKS_JOB_ID environment variables."
        )

    if not pipeline.checkpoint.get("databricks_run_id"):
        result = databricks_request(
            host,
            token,
            "jobs/run-now",
            {"job_id": int(job_id)},
        )

        run_id = result.get("run_id")
        if not run_id:
            raise RuntimeError(f"Databricks did not return a run_id: {result}")

        pipeline.checkpoint["databricks_run_id"] = run_id
        pipeline.save_checkpoint()

        pipeline.log(
            "INFO",
            "Databricks job submitted",
            databricks_run_id=run_id,
        )

    run_id = int(pipeline.checkpoint["databricks_run_id"])

    while True:
        result = databricks_request(
            host,
            token,
            f"jobs/runs/get?run_id={run_id}",
        )

        state = result.get("state", {})
        life_cycle = state.get("life_cycle_state")
        result_state = state.get("result_state")
        state_message = state.get("state_message", "")

        pipeline.log(
            "INFO",
            "Databricks job status",
            lifecycle=life_cycle,
            result=result_state,
            message=state_message,
        )

        if life_cycle == "TERMINATED":
            if result_state != "SUCCESS":
                raise RuntimeError(
                    "Databricks job failed: " f"{result_state} - {state_message}"
                )
            break

        if life_cycle in {"INTERNAL_ERROR", "SKIPPED"}:
            raise RuntimeError(f"Databricks job ended unexpectedly: {state}")

        time.sleep(20)

    # The six notebooks currently write Delta tables in Unity Catalog.
    # The common ML scripts need local Parquet split directories.
    # Supply a downloaded/exported directory containing train/,
    # validation/ and test/ to bridge those two storage systems.
    export_dir_value = os.environ.get("DATABRICKS_EXPORT_DIR", "").strip()

    if not export_dir_value:
        raise RuntimeError(
            "Databricks notebooks completed, but the ML input export "
            "has not been configured.\n"
            "Set DATABRICKS_EXPORT_DIR to a local directory containing "
            "train/, validation/ and test/ Parquet datasets.\n"
            "The current Databricks notebooks write Delta tables; they "
            "do not automatically copy them into the local Docker data volume."
        )

    export_dir = Path(export_dir_value).expanduser().resolve()

    for split in ["train", "validation", "test"]:
        source = export_dir / split
        destination = PROCESSED_ROOT / split

        if not source.exists():
            raise FileNotFoundError(f"Databricks export is missing: {source}")

        if destination.exists():
            shutil.rmtree(destination)

        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)

    for stage, notebook in DATABRICKS_NOTEBOOKS:
        pipeline.stage_status(
            stage,
            "COMPLETED",
            notebook=notebook,
            databricks_run_id=run_id,
        )

    pipeline.log(
        "INFO",
        "Databricks processing and dataset export completed.",
    )


# ============================================================
# PIPELINE STAGE EXECUTION
# ============================================================


def execute_local_data_stages(
    pipeline: Pipeline,
    input_container_path: str,
    start_stage: int,
) -> None:
    base_parquet_path = "/app/data/processed/metropt"

    for stage, job_file, description in LOCAL_JOBS:
        if stage < start_stage:
            continue

        if pipeline.stage_completed(stage):
            pipeline.log("INFO", f"Skipping completed stage {stage}")
            continue

        pipeline.stage_status(stage, "RUNNING", job=job_file)
        pipeline.log("INFO", f"Stage {stage}: {description}")

        try:
            run_local_pyspark_job(
                job_file,
                base_parquet_path,
                pipeline,
            )

            pipeline.stage_status(
                stage,
                "COMPLETED",
                job=job_file,
            )

        except Exception as exc:
            pipeline.stage_status(
                stage,
                "FAILED",
                job=job_file,
                error=str(exc),
            )
            raise


def execute_ml_stages(
    pipeline: Pipeline,
    start_stage: int,
) -> None:
    validate_ml_datasets()

    for stage, description, script in ML_STAGES:
        if stage < start_stage:
            continue

        if pipeline.stage_completed(stage):
            pipeline.log("INFO", f"Skipping completed stage {stage}")
            continue

        pipeline.stage_status(
            stage,
            "RUNNING",
            script=script,
        )

        pipeline.log("INFO", f"Stage {stage}: {description}")

        try:
            run_local_ml_stage(stage, pipeline)

            pipeline.stage_status(
                stage,
                "COMPLETED",
                script=script,
            )

        except Exception as exc:
            pipeline.stage_status(
                stage,
                "FAILED",
                script=script,
                error=str(exc),
            )
            raise


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Fault Prediction master pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=["local", "databricks"],
        default="local",
        help="Choose local PySpark or Databricks processing.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV path for local mode.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest pipeline checkpoint.",
    )

    parser.add_argument(
        "--from-stage",
        type=int,
        choices=list(range(1, 19)),
        help="Restart execution from this logical stage.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional custom pipeline run ID.",
    )

    return parser.parse_args()


def create_checkpoint(
    mode: str,
    run_id: str,
    fingerprint: str,
    input_path: Path | None,
    input_container_path: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "status": "RUNNING",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input_path": str(input_path) if input_path else None,
        "input_fingerprint": fingerprint,
        "input_container_path": input_container_path,
        "stages": {},
    }


def load_or_create_checkpoint(
    args: argparse.Namespace,
    fingerprint: str,
    input_path: Path | None,
    input_container_path: str | None,
) -> tuple[dict[str, Any], str]:
    checkpoint_path = RUNS_ROOT / "pipeline_checkpoint.json"

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "No pipeline checkpoint exists. Run without --resume first."
            )

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        if checkpoint.get("mode") != args.mode:
            raise ValueError("Cannot resume a checkpoint created in a different mode.")

        if checkpoint.get("input_fingerprint") != fingerprint:
            raise ValueError(
                "Input fingerprint changed. Start a new run instead of "
                "resuming with different input data."
            )

        run_id = checkpoint["run_id"]
        checkpoint["status"] = "RUNNING"

        return checkpoint, run_id

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = create_checkpoint(
        mode=args.mode,
        run_id=run_id,
        fingerprint=fingerprint,
        input_path=input_path,
        input_container_path=input_container_path,
    )

    return checkpoint, run_id


def main() -> int:
    args = parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    input_path: Path | None = None
    input_container_path: str | None = None

    if args.mode == "local":
        input_path = validate_input_path(args.input)
        input_container_path = container_data_path(input_path)
        fingerprint = get_input_fingerprint(input_path)
    else:
        # Databricks mode is fingerprinted by its configured job ID.
        job_id = os.environ.get("DATABRICKS_JOB_ID", "")
        if not job_id:
            raise RuntimeError("Set DATABRICKS_JOB_ID before starting Databricks mode.")
        fingerprint = hashlib.sha256(job_id.encode("utf-8")).hexdigest()

    checkpoint, run_id = load_or_create_checkpoint(
        args,
        fingerprint,
        input_path,
        input_container_path,
    )

    pipeline = Pipeline(args.mode, run_id, checkpoint)

    start_stage = args.from_stage or 1

    pipeline.log(
        "INFO",
        "Pipeline started",
        start_stage=start_stage,
        input_path=str(input_path) if input_path else None,
    )

    try:
        if args.mode == "local":
            if start_stage <= 10:
                execute_local_data_stages(
                    pipeline,
                    input_container_path,
                    start_stage,
                )

            if start_stage <= 11:
                execute_ml_stages(pipeline, max(start_stage, 11))

        else:
            if start_stage <= 6:
                run_databricks_job(pipeline)

            if start_stage <= 11:
                execute_ml_stages(pipeline, max(start_stage, 11))

        pipeline.checkpoint["status"] = "COMPLETED"
        pipeline.checkpoint["completed_at"] = utc_now()
        pipeline.save_checkpoint()

        pipeline.log("INFO", "Pipeline completed successfully.")
        return 0

    except KeyboardInterrupt:
        pipeline.checkpoint["status"] = "INTERRUPTED"
        pipeline.checkpoint["interrupted_at"] = utc_now()
        pipeline.save_checkpoint()
        pipeline.log("WARNING", "Pipeline interrupted by user.")
        return 130

    except Exception as exc:
        pipeline.checkpoint["status"] = "FAILED"
        pipeline.checkpoint["failed_at"] = utc_now()
        pipeline.checkpoint["error"] = str(exc)
        pipeline.save_checkpoint()

        pipeline.log(
            "ERROR",
            "Pipeline failed",
            error=str(exc),
        )

        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------

# from __future__ import annotations

# import argparse
# import hashlib
# import json
# import shutil
# import subprocess
# import sys
# import time
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any

# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# ROOT = Path(__file__).resolve().parent
# DATA_DIR = ROOT / "data_Sample"
# RUNS_DIR = DATA_DIR / "pipeline_runs"
# METRICS_DIR = DATA_DIR / "metrics"

# CONTAINER_DATA_ROOT = "/app/data"
# CONTAINER_PARQUET_PATH = f"{CONTAINER_DATA_ROOT}/processed/metropt"

# TOTAL_STAGES = 18
# EXECUTION_MODES = ("local", "databricks")

# # ---------------------how ot run in command->
# parser.add_argument(
#     "--mode",
#     choices=EXECUTION_MODES,
#     default="local",
#     help="Data processing execution mode.",
# )

# # ============================================================
# # 2. PIPELINE STAGES
# # ============================================================

# PYSPARK_JOBS = [
#     ("Data ingestion", "jobs/01_ingest.py"),
#     ("Parquet inspection", "jobs/02_inspect_parquet.py"),
#     ("Data validation", "jobs/03_validate_data.py"),
#     ("Data quality checks", "jobs/04_data_quality.py"),
#     ("Time-series analysis", "jobs/05_time_series_analysis.py"),
#     ("Time-series preparation", "jobs/06_time_series_preparation.py"),
#     ("Fault-label creation", "jobs/07_target_creation.py"),
#     ("Feature engineering", "jobs/08_feature_engineering.py"),
#     ("Feature analysis", "jobs/09_feature_analysis.py"),
#     ("Train/validation/test split", "jobs/10_train_val_test_split.py"),
# ]

# ML_STAGES = [
#     (
#         "Baseline model",
#         ["python", "src/models/baseline_11.py"],
#     ),
#     (
#         "Random Forest model",
#         ["python", "src/models/random_forest_12.py"],
#     ),
#     (
#         "Deep-learning model",
#         ["python", "-m", "src.models.deep_learning_13"],
#     ),
#     (
#         "Model evaluation",
#         ["python", "-m", "src.evaluation.evaluate_14"],
#     ),
#     (
#         "Random Forest tuning",
#         ["python", "-m", "src.models.tuning_15"],
#     ),
#     (
#         "MLflow tracking",
#         ["python", "-m", "src.experiments.mlflow_tracking_16"],
#     ),
#     (
#         "ONNX export",
#         ["python", "-m", "src.export.export_onnx_17"],
#     ),
#     (
#         "ONNX benchmark",
#         ["python", "-m", "src.benchmarking.benchmark_onnx_18"],
#     ),
# ]

# # Expected outputs used to validate cached stages.
# # Check these paths against the outputs written by your scripts.
# STAGE_OUTPUTS = {
#     1: ["processed/metropt"],
#     2: ["processed/metropt"],
#     3: ["processed/metropt"],
#     4: ["processed/metropt"],
#     5: [],
#     6: ["processed/metropt_timeseries"],
#     7: ["processed/metropt_labeled"],
#     8: ["processed/metropt_features"],
#     9: ["processed/metropt_features"],
#     10: [
#         "processed/train",
#         "processed/validation",
#         "processed/test",
#     ],
#     # 11: [],  # The distribution report lives inside the run directory.
#     11: [
#         "metrics/baseline_logistic_regression.json",
#         "models/baseline_logistic_regression.joblib",
#     ],
#     12: [
#         "metrics/random_forest.json",
#         "models/random_forest.joblib",
#     ],
#     13: [
#         "metrics/deep_learning.json",
#         "models/deep_learning.pt",
#     ],
#     14: ["metrics/evaluation_14.json"],
#     15: [
#         "metrics/random_forest_tuned.json",
#     ],
#     16: [],  # MLflow tracking may not create a deterministic output file.
#     17: [],  # ONNX export may be skipped when only a fallback model exists.
#     18: ["metrics/onnx_benchmark_18.json"],
# }

# METRIC_FILES = [
#     "baseline_logistic_regression.json",
#     "random_forest.json",
#     "deep_learning.json",
#     "evaluation_14.json",
#     "random_forest_tuned.json",
#     "onnx_benchmark_18.json",
# ]


# # ============================================================
# # 3. COMMAND-LINE ARGUMENTS
# # ============================================================


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Run the Vehicle Fault Prediction pipeline "
#             "with persistent checkpoints and resume support."
#         )
#     )

#     parser.add_argument(
#         "input",
#         nargs="?",
#         type=Path,
#         help="Input CSV, XLSX, or XLSM file for a new run.",
#     )

#     parser.add_argument(
#         "--sheet",
#         help="Excel sheet name or zero-based index.",
#     )

#     parser.add_argument(
#         "--resume",
#         metavar="RUN_ID",
#         help="Resume a previous run using its saved checkpoint.",
#     )

#     parser.add_argument(
#         "--from-step",
#         type=int,
#         choices=range(1, TOTAL_STAGES + 1),
#         help="Request a starting stage when resuming.",
#     )

#     parser.add_argument(
#         "--start-api",
#         action="store_true",
#         help="Start the API after all pipeline stages succeed.",
#     )

#     args = parser.parse_args()

#     if args.resume and args.input:
#         parser.error("Do not provide an input file with --resume.")

#     if not args.resume and not args.input:
#         parser.error("Provide an input file or use --resume RUN_ID.")

#     if args.from_step and not args.resume:
#         parser.error("--from-step requires --resume.")

#     return args


# # ============================================================
# # 4. CHECKPOINT AND FINGERPRINT HELPERS
# # ============================================================


# def utc_now() -> str:
#     return datetime.now(timezone.utc).isoformat()


# def sha256_file(path: Path) -> str:
#     digest = hashlib.sha256()

#     with path.open("rb") as file:
#         for chunk in iter(lambda: file.read(1024 * 1024), b""):
#             digest.update(chunk)

#     return digest.hexdigest()


# def save_json(path: Path, payload: dict[str, Any]) -> None:
#     """Atomically save JSON to reduce the risk of a corrupt checkpoint."""
#     path.parent.mkdir(parents=True, exist_ok=True)

#     temporary_path = path.with_suffix(path.suffix + ".tmp")
#     temporary_path.write_text(
#         json.dumps(payload, indent=2, default=str),
#         encoding="utf-8",
#     )
#     temporary_path.replace(path)


# def load_json(path: Path) -> dict[str, Any]:
#     with path.open("r", encoding="utf-8") as file:
#         return json.load(file)


# def code_fingerprint(paths: list[Path]) -> str | None:
#     """Fingerprint the stage's code files."""
#     digest = hashlib.sha256()

#     for path in paths:
#         if not path.is_file():
#             return None

#         digest.update(str(path).encode("utf-8"))
#         digest.update(sha256_file(path).encode("ascii"))

#     return digest.hexdigest()


# def new_checkpoint(run_id: str, staged_input: Path) -> dict[str, Any]:
#     return {
#         "run_id": run_id,
#         "status": "RUNNING",
#         "started_at_utc": utc_now(),
#         "updated_at_utc": utc_now(),
#         "input": {
#             "path": str(staged_input.resolve()),
#             "sha256": sha256_file(staged_input),
#         },
#         "last_completed_stage": 0,
#         "next_stage": 1,
#         "stages": {},
#         "failure": None,
#     }


# def save_checkpoint(
#     checkpoint_path: Path,
#     checkpoint: dict[str, Any],
# ) -> None:
#     checkpoint["updated_at_utc"] = utc_now()
#     save_json(checkpoint_path, checkpoint)


# def mark_running(
#     checkpoint_path: Path,
#     checkpoint: dict[str, Any],
#     number: int,
#     name: str,
#     fingerprint: str | None,
# ) -> None:
#     checkpoint["status"] = "RUNNING"
#     checkpoint["next_stage"] = number
#     checkpoint["failure"] = None

#     previous = checkpoint["stages"].get(str(number), {})
#     attempts = previous.get("attempts", 0) + 1

#     checkpoint["stages"][str(number)] = {
#         "number": number,
#         "name": name,
#         "status": "RUNNING",
#         "attempts": attempts,
#         "started_at_utc": utc_now(),
#         "code_sha256": fingerprint,
#     }

#     save_checkpoint(checkpoint_path, checkpoint)


# def mark_finished(
#     checkpoint_path: Path,
#     checkpoint: dict[str, Any],
#     number: int,
#     name: str,
#     success: bool,
#     fingerprint: str | None,
#     error: str | None = None,
#     output_paths: list[str] | None = None,
#     stage_status: str | None = None,
# ) -> None:
#     previous = checkpoint["stages"].get(str(number), {})

#     checkpoint["stages"][str(number)] = {
#         **previous,
#         "number": number,
#         "name": name,
#         "status": stage_status or ("SUCCESS" if success else "FAILED"),
#         "code_sha256": fingerprint,
#         "outputs": output_paths or [],
#         "error": error,
#         "finished_at_utc": utc_now(),
#     }

#     if success:
#         checkpoint["last_completed_stage"] = max(
#             checkpoint.get("last_completed_stage", 0),
#             number,
#         )
#         checkpoint["next_stage"] = number + 1 if number < TOTAL_STAGES else None
#         checkpoint["failure"] = None
#         checkpoint["status"] = "RUNNING"
#     else:
#         checkpoint["next_stage"] = number
#         checkpoint["status"] = "FAILED"
#         checkpoint["failure"] = {
#             "stage": number,
#             "name": name,
#             "message": error or "Stage failed",
#         }

#     save_checkpoint(checkpoint_path, checkpoint)


# # ============================================================
# # 5. INPUT STAGING
# # ============================================================


# def prepare_input(
#     source: Path,
#     run_dir: Path,
# ) -> tuple[Path, Path | None]:
#     source = source.expanduser().resolve()

#     if not source.is_file():
#         raise FileNotFoundError(f"Input file not found: {source}")

#     extension = source.suffix.lower()

#     if extension not in {".csv", ".xlsx", ".xlsm"}:
#         raise ValueError("Supported input formats: CSV, XLSX, XLSM.")

#     run_dir.mkdir(parents=True, exist_ok=True)

#     if extension == ".csv":
#         staged = run_dir / "input.csv"
#         shutil.copy2(source, staged)
#         return staged, None

#     staged = run_dir / f"input{extension}"
#     shutil.copy2(source, staged)

#     # The CSV will be generated inside the container.
#     return staged, run_dir / "input.csv"


# def container_data_path(path: Path) -> str:
#     return (
#         f"{CONTAINER_DATA_ROOT}/"
#         f"{path.resolve().relative_to(DATA_DIR.resolve()).as_posix()}"
#     )


# # ============================================================
# # 6. DOCKER COMPOSE HELPERS
# # ============================================================


# def write_compose_override(
#     run_dir: Path,
#     run_id: str,
# ) -> Path:
#     suffix = run_id.lower()
#     override_path = run_dir / "docker-compose.sample.yml"

#     content = f"""
# services:
#   ml-training:
#     container_name: vehicle-fault-training-sample-{suffix}
#     environment:
#       PYTHONDONTWRITEBYTECODE: "1"
#     volumes:
#       - ./data_Sample:/app/data
#       - ./data_Sample/models:/app/models

#   api:
#     container_name: vehicle-fault-api-sample-{suffix}
#     ports: !override
#       - "8001:8000"
#     volumes:
#       - ./data_Sample/models/onnx:/app/models/onnx:ro

#   pyspark:
#     container_name: vehicle-fault-pyspark-sample-{suffix}
#     environment:
#       PYTHONDONTWRITEBYTECODE: "1"
#     volumes:
#       - ./data_Sample:/app/data

#   mlflow:
#     container_name: vehicle-fault-mlflow-sample-{suffix}
#     ports: !override []
#     volumes:
#       - ./data_Sample/mlflow-data:/mlruns
# """

#     override_path.write_text(content, encoding="utf-8")
#     return override_path


# def compose_prefix(
#     override_path: Path,
#     run_id: str,
# ) -> list[str]:
#     relative_override = override_path.resolve().relative_to(ROOT).as_posix()

#     return [
#         "docker",
#         "compose",
#         "--project-name",
#         f"vehicle-fault-prediction-sample-{run_id.lower()}",
#         "-f",
#         "docker-compose.yml",
#         "-f",
#         relative_override,
#     ]


# def run_command(
#     command: list[str],
#     cwd: Path = ROOT,
# ) -> tuple[int, str]:
#     """Run a command while displaying its output live."""
#     print("$", " ".join(command), flush=True)

#     try:
#         process = subprocess.Popen(
#             command,
#             cwd=cwd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             encoding="utf-8",
#             errors="replace",
#             bufsize=1,
#         )
#     except OSError as error:
#         print(f"Could not start command: {error}")
#         return 127, str(error)

#     output_lines: list[str] = []

#     assert process.stdout is not None

#     for line in process.stdout:
#         print(line, end="", flush=True)
#         output_lines.append(line)

#     return_code = process.wait()
#     return return_code, "".join(output_lines)


# def cleanup_compose_project(compose_args: list[str]) -> None:
#     """Remove this run's containers and network, preserving data and volumes."""
#     print("\nCleaning up pipeline containers...", flush=True)

#     code, output = run_command(
#         [
#             *compose_args,
#             "down",
#             "--remove-orphans",
#         ]
#     )

#     if code == 0:
#         print("Pipeline containers and network removed.", flush=True)
#     else:
#         print(
#             "WARNING: Could not fully clean up pipeline containers:\n"
#             f"{output[-3000:]}",
#             file=sys.stderr,
#             flush=True,
#         )


# def docker_exec_command(
#     compose_args: list[str],
#     service: str,
#     command: list[str],
#     environment: dict[str, str] | None = None,
# ) -> list[str]:
#     result = [*compose_args, "exec", "-T"]

#     for key, value in (environment or {}).items():
#         result.extend(["-e", f"{key}={value}"])

#     result.append(service)
#     result.extend(command)

#     return result


# # ============================================================
# # 7. STAGE DEFINITIONS
# # ============================================================


# def get_stage_definitions(
#     run_dir: Path,
# ) -> list[dict[str, Any]]:
#     definitions: list[dict[str, Any]] = []

#     for name, script in PYSPARK_JOBS:
#         code_path = ROOT / "services" / "pyspark" / script

#         definitions.append(
#             {
#                 "number": len(definitions) + 1,
#                 "name": name,
#                 "service": "pyspark",
#                 "command": ["python", script],
#                 "environment_kind": "pyspark",
#                 "code_paths": [code_path],
#                 "outputs": STAGE_OUTPUTS[len(definitions) + 1],
#             }
#         )

#     # Stage 11: check that every split has both classes.
#     distribution_script = r"""
# import json
# import sys
# from pyspark.sql import SparkSession

# spark = (
#     SparkSession.builder
#     .appName("PipelineSplitDistributionCheck")
#     .getOrCreate()
# )


# try:
#     root = "/app/data/processed"
#     result = {}
#     warnings = []

#     for split in ("train", "validation", "test"):
#         path = f"{root}/{split}"

#         rows = (
#             spark.read.parquet(path)
#             .groupBy("fault_label")
#             .count()
#             .collect()
#         )

#         counts = {
#             int(row["fault_label"]): int(row["count"])
#             for row in rows
#         }

#         normal = counts.get(0, 0)
#         fault = counts.get(1, 0)

#         result[split] = {
#             "rows": normal + fault,
#             "normal": normal,
#             "fault": fault,
#             "status": (
#                 "passed"
#                 if normal > 0 and fault > 0
#                 else "warning"
#             ),
#         }

#         print(
#             f"{split}: normal={normal}, fault={fault}",
#             flush=True,
#         )

#         if normal == 0 and fault == 0:
#             warnings.append(
#                 f"{split} is empty. Check the dataset."
#             )
#         elif normal == 0 or fault == 0:
#             warnings.append(
#                 f"{split} contains only one class: "
#                 f"normal={normal}, fault={fault}. "
#                 "Some training or evaluation tasks may be unavailable."
#             )

#     with open(sys.argv[1], "w", encoding="utf-8") as file:
#         json.dump(result, file, indent=2)

#     if warnings:
#         print("\n========== CLASS DISTRIBUTION WARNINGS ==========")
#         for warning in warnings:
#             print(f"WARNING: {warning}", flush=True)

#     print(
#         "\nClass-distribution validation completed. "
#         "Review any warnings above.",
#         flush=True,
#     )

# finally:
#     spark.stop()

# """

#     # definitions.append(
#     #     {
#     #         "number": 11,
#     #         "name": "Fault-label distribution gate",
#     #         "service": "pyspark",
#     #         "command": [
#     #             "python",
#     #             "-c",
#     #             distribution_script,
#     #             container_data_path(run_dir / "split_distribution.json"),
#     #         ],
#     #         "environment_kind": "pyspark",
#     #         "code_paths": [],
#     #         "outputs": [f"pipeline_runs/{run_dir.name}/split_distribution.json"],
#     #     }
#     # )

#     # Attach the distribution check to Stage 10.
#     # Stage 10 will run the split script first, then this check.
#     definitions[-1]["post_command"] = [
#         "python",
#         "-c",
#         distribution_script,
#         container_data_path(run_dir / "split_distribution.json"),
#     ]

#     for name, command in ML_STAGES:
#         module = command[-1]

#         if command[1].endswith(".py"):
#             code_path = ROOT / "services" / "ml-training" / command[1]
#         else:
#             code_path = (
#                 ROOT
#                 / "services"
#                 / "ml-training"
#                 / Path(*module.split(".")).with_suffix(".py")
#             )

#         definitions.append(
#             {
#                 "number": len(definitions) + 1,
#                 "name": name,
#                 "service": "ml-training",
#                 "command": command,
#                 "environment_kind": "ml",
#                 "code_paths": [code_path],
#                 "outputs": STAGE_OUTPUTS[len(definitions) + 1],
#             }
#         )

#     return definitions


# def stage_output_paths(stage: dict[str, Any]) -> list[Path]:
#     return [DATA_DIR / item for item in stage["outputs"]]


# def outputs_exist(stage: dict[str, Any]) -> bool:
#     for path in stage_output_paths(stage):
#         if not path.exists():
#             return False

#         if path.is_file() and path.stat().st_size == 0:
#             return False

#         if path.is_dir() and not any(path.iterdir()):
#             return False

#     return True


# def validate_metric_outputs(stage: dict[str, Any]) -> str | None:
#     """Reject skipped, failed, or invalid metrics reports."""

#     for relative_path in stage["outputs"]:
#         path = DATA_DIR / relative_path

#         # Validate JSON reports stored in the metrics directory.
#         if "metrics" not in Path(relative_path).parts:
#             continue

#         if path.suffix.lower() != ".json" or not path.is_file():
#             continue

#         try:
#             payload = load_json(path)
#         except (OSError, json.JSONDecodeError) as error:
#             return f"Invalid metrics JSON at {path}: {error}"

#         status = payload.get("status")

#         if isinstance(status, str) and status.lower() in {
#             "skipped",
#             "failed",
#             "error",
#         }:
#             return (
#                 f"Stage '{stage['name']}' produced a metrics report "
#                 f"with status='{status}': {path}"
#             )

#     return None


# def can_reuse_stage(
#     checkpoint: dict[str, Any],
#     stage: dict[str, Any],
# ) -> bool:
#     number = stage["number"]
#     saved = checkpoint.get("stages", {}).get(str(number), {})

#     if saved.get("status") not in {"SUCCESS", "SKIPPED_WITH_WARNING"}:
#         return False

#     current_fingerprint = code_fingerprint(stage["code_paths"])
#     saved_fingerprint = saved.get("code_sha256")

#     # Stages with source files must have a matching fingerprint.
#     if stage["code_paths"]:
#         if current_fingerprint is None:
#             return False
#         if current_fingerprint != saved_fingerprint:
#             return False

#     # Do not trust a cached stage if mapped outputs have disappeared.
#     if not outputs_exist(stage):
#         return False

#     # Stages with no mapped outputs are reusable only on recorded success.
#     return True


# # ============================================================
# # 8. REPORTS
# # ============================================================


# def save_report(
#     report_path: Path,
#     run_id: str,
#     checkpoint: dict[str, Any],
#     started_at: str,
#     invocation_seconds: float,
#     failure: str | None = None,
# ) -> None:
#     report = {
#         "run_id": run_id,
#         "status": "FAILED" if failure else checkpoint["status"],
#         "started_at_utc": started_at,
#         "finished_at_utc": utc_now(),
#         "elapsed_seconds_this_invocation": round(
#             time.perf_counter() - invocation_seconds,
#             2,
#         ),
#         "last_completed_stage": checkpoint.get("last_completed_stage", 0),
#         "next_stage": checkpoint.get("next_stage"),
#         "stages": checkpoint.get("stages", {}),
#         "input": checkpoint.get("input", {}),
#         "failure": failure or checkpoint.get("failure"),
#     }

#     save_json(report_path, report)


# # ============================================================
# # 9. MAIN PIPELINE
# # ============================================================


# def main() -> int:
#     args = parse_args()
#     invocation_seconds = time.perf_counter()

#     DATA_DIR.mkdir(parents=True, exist_ok=True)
#     RUNS_DIR.mkdir(parents=True, exist_ok=True)
#     METRICS_DIR.mkdir(parents=True, exist_ok=True)
#     (DATA_DIR / "models" / "onnx").mkdir(parents=True, exist_ok=True)
#     (DATA_DIR / "mlflow-data").mkdir(parents=True, exist_ok=True)

#     if args.resume:
#         run_id = args.resume
#         run_dir = RUNS_DIR / run_id

#         if not run_dir.is_dir():
#             print(
#                 f"ERROR: Run directory not found: {run_dir}",
#                 file=sys.stderr,
#             )
#             return 2

#         checkpoint_path = run_dir / "checkpoint.json"
#         report_path = run_dir / "run_report.json"
#         override_path = run_dir / "docker-compose.sample.yml"

#         try:
#             checkpoint = load_json(checkpoint_path)
#         except (OSError, json.JSONDecodeError) as error:
#             print(f"ERROR: Cannot load checkpoint: {error}", file=sys.stderr)
#             return 2

#         if checkpoint.get("run_id") != run_id:
#             print("ERROR: Run ID does not match checkpoint.", file=sys.stderr)
#             return 2

#         staged_input = Path(checkpoint["input"]["path"])

#         if not staged_input.is_file():
#             print(
#                 f"ERROR: Saved input is missing: {staged_input}",
#                 file=sys.stderr,
#             )
#             return 2

#         if sha256_file(staged_input) != checkpoint["input"]["sha256"]:
#             print(
#                 "ERROR: Saved input changed since the run started.",
#                 file=sys.stderr,
#             )
#             return 2

#         if not override_path.is_file():
#             print(
#                 f"ERROR: Compose override is missing: {override_path}",
#                 file=sys.stderr,
#             )
#             return 2

#         # For Excel runs, reuse the staged CSV if it was already created.
#         csv_input = (
#             run_dir / "input.csv"
#             if staged_input.suffix.lower() in {".xlsx", ".xlsm"}
#             else staged_input
#         )

#         started_at = checkpoint.get("started_at_utc", utc_now())

#     else:
#         run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
#         run_dir = RUNS_DIR / run_id
#         run_dir.mkdir(parents=True, exist_ok=True)

#         staged_input, converted_csv = prepare_input(args.input, run_dir)
#         csv_input = converted_csv or staged_input

#         checkpoint_path = run_dir / "checkpoint.json"
#         report_path = run_dir / "run_report.json"
#         override_path = write_compose_override(run_dir, run_id)

#         checkpoint = new_checkpoint(run_id, staged_input)
#         checkpoint["input"]["csv_path"] = str(csv_input.resolve())
#         checkpoint["input"]["excel_sheet"] = args.sheet
#         checkpoint["input"]["csv_sha256"] = (
#             sha256_file(csv_input) if csv_input.is_file() else None
#         )

#         started_at = checkpoint["started_at_utc"]
#         save_checkpoint(checkpoint_path, checkpoint)

#     compose_args = compose_prefix(override_path, run_id)
#     stages = get_stage_definitions(run_dir)

#     print("=" * 65)
#     print(" VEHICLE FAULT PREDICTION PIPELINE")
#     print("=" * 65)
#     print(f"Run ID:     {run_id}")
#     print(f"Run folder: {run_dir}")
#     print(f"Checkpoint: {checkpoint_path}")

#     # NEW: Preserve the API when startup succeeds.
#     keep_api_running = False
#     try:
#         # Start the containers under this run's unique Compose project.
#         code, output = run_command(
#             [
#                 *compose_args,
#                 "up",
#                 "-d",
#                 "pyspark",
#                 "ml-training",
#                 "mlflow",
#             ]
#         )

#         if code != 0:
#             raise RuntimeError(f"Could not start pipeline services:\n{output}")

#         # Convert Excel only if no converted CSV has been saved yet.
#         if staged_input.suffix.lower() in {".xlsx", ".xlsm"}:
#             if not csv_input.is_file() or csv_input.stat().st_size == 0:
#                 sheet = args.sheet or checkpoint["input"].get("excel_sheet") or "0"

#                 conversion_code = (
#                     "import pandas as pd, sys; "
#                     "sheet = int(sys.argv[3]) if sys.argv[3].isdigit() "
#                     "else sys.argv[3]; "
#                     "df = pd.read_excel("
#                     "sys.argv[1], sheet_name=sheet, engine='openpyxl'); "
#                     "df.to_csv(sys.argv[2], index=False); "
#                     "print(f'Converted {len(df)} rows and "
#                     "{len(df.columns)} columns')"
#                 )

#                 command = docker_exec_command(
#                     compose_args,
#                     "ml-training",
#                     [
#                         "python",
#                         "-c",
#                         conversion_code,
#                         container_data_path(staged_input),
#                         container_data_path(csv_input),
#                         str(sheet),
#                     ],
#                 )

#                 # code, output = run_command(command)

#                 # if code != 0:
#                 #     raise RuntimeError(f"Excel-to-CSV conversion failed:\n{output}")

#                 code, output = run_command(command)

#                 error = None

#                 if code != 0:
#                     error = f"Command exited with code {code}.\n{output[-5000:]}"

#                 # Stage 10: run the distribution check after splitting succeeds.
#                 if error is None and stage.get("post_command"):
#                     post_command = docker_exec_command(
#                         compose_args,
#                         stage["service"],
#                         stage["post_command"],
#                         environment,
#                     )

#                     post_code, post_output = run_command(post_command)
#                     output += "\n" + post_output

#                     if post_code != 0:
#                         error = (
#                             "Stage 10 split-distribution check failed.\n"
#                             f"{post_output[-5000:]}"
#                         )

#             # Save the converted CSV fingerprint for resume validation.
#             current_csv_hash = sha256_file(csv_input)
#             saved_csv_hash = checkpoint["input"].get("csv_sha256")

#             if saved_csv_hash and current_csv_hash != saved_csv_hash:
#                 raise ValueError(
#                     "Converted CSV fingerprint differs from the " "previous checkpoint."
#                 )

#             checkpoint["input"]["csv_sha256"] = current_csv_hash
#             save_checkpoint(checkpoint_path, checkpoint)

#         if not csv_input.is_file() or csv_input.stat().st_size == 0:
#             raise FileNotFoundError(f"Pipeline CSV is missing or empty: {csv_input}")

#         pyspark_environment = {
#             "CSV_PATH": container_data_path(csv_input),
#             "PARQUET_PATH": CONTAINER_PARQUET_PATH,
#             "DATA_ROOT": CONTAINER_DATA_ROOT,
#         }

#         # Find the earliest stage that is not safely reusable.
#         # Find the earliest stage that is not safely reusable.
#         earliest_invalid = TOTAL_STAGES + 1

#         for stage in stages:
#             if args.resume and can_reuse_stage(checkpoint, stage):
#                 continue

#             earliest_invalid = stage["number"]
#             break

#         if args.from_step:
#             if args.from_step > earliest_invalid:
#                 raise ValueError(
#                     f"Stage {earliest_invalid} is the earliest stage "
#                     "that is not safely reusable. Cannot skip it."
#                 )

#             start_stage = args.from_step
#         else:
#             start_stage = earliest_invalid

#         # NEW: Report only stages that will actually be skipped.
#         if args.resume:
#             for stage in stages:
#                 if stage["number"] >= start_stage:
#                     break

#                 print(
#                     f"[{stage['number']}/{TOTAL_STAGES}] "
#                     f"{stage['name']} ... SKIPPED"
#                 )
#         if start_stage > TOTAL_STAGES:
#             print("\nAll stages are already complete and reusable.")
#         else:
#             print(
#                 f"\nResuming from stage {start_stage}: "
#                 f"{stages[start_stage - 1]['name']}"
#             )

#         for stage in stages:
#             number = stage["number"]

#             if number < start_stage:
#                 continue

#             # A later stage is not allowed to run while an earlier
#             # stage is incomplete or has invalid cached outputs.
#             fingerprint = code_fingerprint(stage["code_paths"])

#             mark_running(
#                 checkpoint_path,
#                 checkpoint,
#                 number,
#                 stage["name"],
#                 fingerprint,
#             )

#             print(
#                 f"\n[{number}/{TOTAL_STAGES}] " f"{stage['name']} ... RUNNING",
#                 flush=True,
#             )

#             stage_clock = time.perf_counter()

#             environment = (
#                 pyspark_environment
#                 if stage["environment_kind"] == "pyspark"
#                 else {"PYTHONDONTWRITEBYTECODE": "1"}
#             )

#             # Stage 19 requires the ONNX model generated by Stage 18.
#             # If it is missing, skip benchmarking with a warning.
#             if number == 18:
#                 onnx_path = DATA_DIR / "models" / "onnx" / "random_forest.onnx"

#                 if not onnx_path.is_file():
#                     stage_status = "SKIPPED_WITH_WARNING"
#                     error = None

#                     print(
#                         "WARNING: ONNX benchmark skipped because "
#                         "random_forest.onnx was not generated by Stage 18."
#                     )

#                     code = 0
#                     output = ""
#                 else:
#                     stage_status = None
#                     code, output = run_command(
#                         docker_exec_command(
#                             compose_args,
#                             stage["service"],
#                             stage["command"],
#                             environment,
#                         )
#                     )
#             else:
#                 stage_status = None
#                 code, output = run_command(
#                     docker_exec_command(
#                         compose_args,
#                         stage["service"],
#                         stage["command"],
#                         environment,
#                     )
#                 )

#             error = None

#             if code != 0:
#                 error = f"Command exited with code {code}.\n{output[-5000:]}"

#             # NEW: Run the Stage 10 distribution check after the split succeeds.
#             if error is None and stage.get("post_command"):
#                 post_command = docker_exec_command(
#                     compose_args,
#                     stage["service"],
#                     stage["post_command"],
#                     environment,
#                 )

#                 post_code, post_output = run_command(post_command)

#                 output += "\n\n[Post-command output]\n" + post_output

#                 if post_code != 0:
#                     error = (
#                         "Stage 10 split-distribution check failed.\n"
#                         f"{post_output[-5000:]}"
#                     )

#             if stage["name"] == "MLflow tracking" and "Could not log " in output:
#                 error = "MLflow tracking reported a logging failure."

#             # Stage 10 (continue)must leave its distribution report behind.
#             if number == 10:
#                 distribution_path = run_dir / "split_distribution.json"

#                 if not distribution_path.is_file():
#                     error = "Fault-label distribution report was not created."
#                 elif distribution_path.stat().st_size == 0:
#                     error = "Fault-label distribution report is empty."

#             # Validate mapped outputs after a successful command.
#             # Validate mapped outputs after a successful command.

#             # Validate mapped outputs after a successful command.
#             if (
#                 error is None
#                 and stage_status != "SKIPPED_WITH_WARNING"
#                 and not outputs_exist(stage)
#             ):
#                 error = (
#                     "Stage exited successfully but expected output(s) "
#                     f"are missing: {stage['outputs']}"
#                 )

#             # Validate metric JSON files.
#             if error is None and stage_status != "SKIPPED_WITH_WARNING":
#                 error = validate_metric_outputs(stage)

#             # Default status for a normal stage.
#             stage_status = None

#             # Stage 18: Handle an intentional ONNX export skip.
#             if error is None and number == 17:
#                 onnx_dir = DATA_DIR / "models" / "onnx"
#                 onnx_path = onnx_dir / "random_forest.onnx"
#                 metadata_path = onnx_dir / "random_forest_metadata.json"

#                 if not onnx_path.is_file() or not metadata_path.is_file():
#                     stage_status = "SKIPPED_WITH_WARNING"
#                     print(
#                         "WARNING: ONNX files were not generated. "
#                         "Check the ONNX exporter output. "
#                         "The ONNX benchmark cannot run without the model."
#                     )

#             # Stage 16: Handle a deliberate tuning skip.
#             if error is None and number == 15:
#                 tuning_metrics_path = DATA_DIR / "metrics" / "random_forest_tuned.json"

#                 if tuning_metrics_path.is_file():
#                     tuning_metrics = load_json(tuning_metrics_path)

#                     if tuning_metrics.get("status") == "skipped_missing_class":
#                         stage_status = "SKIPPED_WITH_WARNING"
#                         print(
#                             "WARNING: Random Forest tuning was skipped "
#                             "because the training data lacks both classes."
#                         )

#             success = error is None

#             elapsed = round(time.perf_counter() - stage_clock, 2)

#             mark_finished(
#                 checkpoint_path=checkpoint_path,
#                 checkpoint=checkpoint,
#                 number=number,
#                 name=stage["name"],
#                 success=success,
#                 fingerprint=fingerprint,
#                 error=error,
#                 output_paths=[str(path) for path in stage_output_paths(stage)],
#                 stage_status=stage_status,  # FIXED
#             )

#             checkpoint["stages"][str(number)]["elapsed_seconds"] = elapsed
#             save_checkpoint(checkpoint_path, checkpoint)

#             save_report(
#                 report_path,
#                 run_id,
#                 checkpoint,
#                 started_at,
#                 invocation_seconds,
#                 error,
#             )

#             displayed_status = (
#                 "SKIPPED WITH WARNING"
#                 if stage_status == "SKIPPED_WITH_WARNING"
#                 else "SUCCESS" if success else "FAILED"
#             )

#             print(
#                 f"Stage {number}: {displayed_status} ({elapsed:.1f}s)",
#                 flush=True,
#             )

#             if not success:
#                 print(
#                     "\nPipeline stopped. Completed stages have been "
#                     "saved to checkpoint.json."
#                 )
#                 print("Resume with:\n" f"python .\\run_pipeline.py --resume {run_id}")
#                 return 1

#         checkpoint["status"] = "SUCCESS"
#         checkpoint["failure"] = None
#         checkpoint["next_stage"] = None
#         checkpoint["finished_at_utc"] = utc_now()
#         save_checkpoint(checkpoint_path, checkpoint)

#         if args.start_api:
#             code, output = run_command(
#                 [
#                     *compose_args,
#                     "up",
#                     "-d",
#                     "api",
#                 ]
#             )

#             if code != 0:
#                 checkpoint["status"] = "FAILED"
#                 checkpoint["failure"] = {
#                     "stage": "API startup",
#                     "message": output[-5000:],
#                 }
#                 save_checkpoint(checkpoint_path, checkpoint)

#                 save_report(
#                     report_path,
#                     run_id,
#                     checkpoint,
#                     started_at,
#                     invocation_seconds,
#                     "Could not start API service.",
#                 )
#                 return 1
#             # NEW: The API is running; preserve its container and network.
#             keep_api_running = True

#         save_report(
#             report_path,
#             run_id,
#             checkpoint,
#             started_at,
#             invocation_seconds,
#         )

#         print("\n" + "=" * 65)
#         print(" PIPELINE COMPLETED SUCCESSFULLY")
#         print("=" * 65)
#         print(f"Checkpoint: {checkpoint_path}")
#         print(f"Report:     {report_path}")

#         return 0

#     except Exception as error:
#         checkpoint["status"] = "FAILED"
#         checkpoint["failure"] = {
#             "message": str(error),
#             "time_utc": utc_now(),
#         }

#         if checkpoint.get("next_stage") is None:
#             checkpoint["next_stage"] = checkpoint.get("last_completed_stage", 0) + 1

#         save_checkpoint(checkpoint_path, checkpoint)

#         save_report(
#             report_path,
#             run_id,
#             checkpoint,
#             started_at,
#             invocation_seconds,
#             str(error),
#         )

#         print(f"\nERROR: {error}", file=sys.stderr)
#         print(
#             "\nTo retry this run, execute:\n"
#             f"python .\\run_pipeline.py --resume {run_id}",
#             file=sys.stderr,
#         )

#         return 1
#     finally:
#         if keep_api_running:
#             print(
#                 "\nKeeping the API running. "
#                 "Stopping the pipeline's other services...",
#                 flush=True,
#             )

#             code, output = run_command(
#                 [
#                     *compose_args,
#                     "stop",
#                     "pyspark",
#                     "ml-training",
#                     "mlflow",
#                 ]
#             )

#             if code != 0:
#                 print(
#                     "WARNING: Could not stop all auxiliary services:\n"
#                     f"{output[-3000:]}",
#                     file=sys.stderr,
#                     flush=True,
#                 )
#         else:
#             cleanup_compose_project(compose_args)


# if __name__ == "__main__":
#     raise SystemExit(main())


# # """Run the existing PySpark and ML-training pipeline in Docker Compose."""
