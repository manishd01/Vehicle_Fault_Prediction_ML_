from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_Sample"
RUNS_DIR = DATA_DIR / "pipeline_runs"
METRICS_DIR = DATA_DIR / "metrics"

CONTAINER_DATA_ROOT = "/app/data"
CONTAINER_PARQUET_PATH = f"{CONTAINER_DATA_ROOT}/processed/metropt"

TOTAL_STAGES = 19


# ============================================================
# 2. PIPELINE STAGES
# ============================================================

PYSPARK_JOBS = [
    ("Data ingestion", "jobs/01_ingest.py"),
    ("Parquet inspection", "jobs/02_inspect_parquet.py"),
    ("Data validation", "jobs/03_validate_data.py"),
    ("Data quality checks", "jobs/04_data_quality.py"),
    ("Time-series analysis", "jobs/05_time_series_analysis.py"),
    ("Time-series preparation", "jobs/06_time_series_preparation.py"),
    ("Fault-label creation", "jobs/07_target_creation.py"),
    ("Feature engineering", "jobs/08_feature_engineering.py"),
    ("Feature analysis", "jobs/09_feature_analysis.py"),
    ("Train/validation/test split", "jobs/10_train_val_test_split.py"),
]

ML_STAGES = [
    (
        "Baseline model",
        ["python", "src/models/baseline_11.py"],
    ),
    (
        "Random Forest model",
        ["python", "src/models/random_forest_12.py"],
    ),
    (
        "Deep-learning model",
        ["python", "-m", "src.models.deep_learning_13"],
    ),
    (
        "Model evaluation",
        ["python", "-m", "src.evaluation.evaluate_14"],
    ),
    (
        "Random Forest tuning",
        ["python", "-m", "src.models.tuning_15"],
    ),
    (
        "MLflow tracking",
        ["python", "-m", "src.experiments.mlflow_tracking_16"],
    ),
    (
        "ONNX export",
        ["python", "-m", "src.export.export_onnx_17"],
    ),
    (
        "ONNX benchmark",
        ["python", "-m", "src.benchmarking.benchmark_onnx_18"],
    ),
]

# Expected outputs used to validate cached stages.
# Check these paths against the outputs written by your scripts.
STAGE_OUTPUTS = {
    1: ["processed/metropt"],
    2: ["processed/metropt"],
    3: ["processed/metropt"],
    4: ["processed/metropt"],
    5: [],
    6: ["processed/metropt_timeseries"],
    7: ["processed/metropt_labeled"],
    8: ["processed/metropt_features"],
    9: ["processed/metropt_features"],
    10: [
        "processed/train",
        "processed/validation",
        "processed/test",
    ],
    11: [],  # The distribution report lives inside the run directory.
    12: [
        "metrics/baseline_logistic_regression.json",
        "models/baseline_logistic_regression.joblib",
    ],
    13: [
        "metrics/random_forest.json",
        "models/random_forest.joblib",
    ],
    14: [
        "metrics/deep_learning.json",
        "models/deep_learning.pt",
    ],
    15: ["metrics/evaluation_14.json"],
    16: [
        "metrics/random_forest_tuned.json",
        "models/random_forest_tuned.joblib",
    ],
    17: [],  # MLflow tracking may not create a deterministic output file.
    18: [
        "models/onnx/random_forest.onnx",
        "models/onnx/random_forest_metadata.json",
    ],
    19: ["metrics/onnx_benchmark_18.json"],
}

METRIC_FILES = [
    "baseline_logistic_regression.json",
    "random_forest.json",
    "deep_learning.json",
    "evaluation_14.json",
    "random_forest_tuned.json",
    "onnx_benchmark_18.json",
]


# ============================================================
# 3. COMMAND-LINE ARGUMENTS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Vehicle Fault Prediction pipeline "
            "with persistent checkpoints and resume support."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input CSV, XLSX, or XLSM file for a new run.",
    )

    parser.add_argument(
        "--sheet",
        help="Excel sheet name or zero-based index.",
    )

    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume a previous run using its saved checkpoint.",
    )

    parser.add_argument(
        "--from-step",
        type=int,
        choices=range(1, TOTAL_STAGES + 1),
        help="Request a starting stage when resuming.",
    )

    parser.add_argument(
        "--start-api",
        action="store_true",
        help="Start the API after all pipeline stages succeed.",
    )

    args = parser.parse_args()

    if args.resume and args.input:
        parser.error("Do not provide an input file with --resume.")

    if not args.resume and not args.input:
        parser.error("Provide an input file or use --resume RUN_ID.")

    if args.from_step and not args.resume:
        parser.error("--from-step requires --resume.")

    return args


# ============================================================
# 4. CHECKPOINT AND FINGERPRINT HELPERS
# ============================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save JSON to reduce the risk of a corrupt checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def code_fingerprint(paths: list[Path]) -> str | None:
    """Fingerprint the stage's code files."""
    digest = hashlib.sha256()

    for path in paths:
        if not path.is_file():
            return None

        digest.update(str(path).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))

    return digest.hexdigest()


def new_checkpoint(run_id: str, staged_input: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "input": {
            "path": str(staged_input.resolve()),
            "sha256": sha256_file(staged_input),
        },
        "last_completed_stage": 0,
        "next_stage": 1,
        "stages": {},
        "failure": None,
    }


def save_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> None:
    checkpoint["updated_at_utc"] = utc_now()
    save_json(checkpoint_path, checkpoint)


def mark_running(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    number: int,
    name: str,
    fingerprint: str | None,
) -> None:
    checkpoint["status"] = "RUNNING"
    checkpoint["next_stage"] = number
    checkpoint["failure"] = None

    previous = checkpoint["stages"].get(str(number), {})
    attempts = previous.get("attempts", 0) + 1

    checkpoint["stages"][str(number)] = {
        "number": number,
        "name": name,
        "status": "RUNNING",
        "attempts": attempts,
        "started_at_utc": utc_now(),
        "code_sha256": fingerprint,
    }

    save_checkpoint(checkpoint_path, checkpoint)


def mark_finished(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    number: int,
    name: str,
    success: bool,
    fingerprint: str | None,
    error: str | None = None,
    output_paths: list[str] | None = None,
) -> None:
    previous = checkpoint["stages"].get(str(number), {})

    checkpoint["stages"][str(number)] = {
        **previous,
        "number": number,
        "name": name,
        "status": "SUCCESS" if success else "FAILED",
        "code_sha256": fingerprint,
        "outputs": output_paths or [],
        "error": error,
        "finished_at_utc": utc_now(),
    }

    if success:
        checkpoint["last_completed_stage"] = max(
            checkpoint.get("last_completed_stage", 0),
            number,
        )
        checkpoint["next_stage"] = number + 1 if number < TOTAL_STAGES else None
        checkpoint["failure"] = None
        checkpoint["status"] = "RUNNING"
    else:
        checkpoint["next_stage"] = number
        checkpoint["status"] = "FAILED"
        checkpoint["failure"] = {
            "stage": number,
            "name": name,
            "message": error or "Stage failed",
        }

    save_checkpoint(checkpoint_path, checkpoint)


# ============================================================
# 5. INPUT STAGING
# ============================================================


def prepare_input(
    source: Path,
    run_dir: Path,
) -> tuple[Path, Path | None]:
    source = source.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    extension = source.suffix.lower()

    if extension not in {".csv", ".xlsx", ".xlsm"}:
        raise ValueError("Supported input formats: CSV, XLSX, XLSM.")

    run_dir.mkdir(parents=True, exist_ok=True)

    if extension == ".csv":
        staged = run_dir / "input.csv"
        shutil.copy2(source, staged)
        return staged, None

    staged = run_dir / f"input{extension}"
    shutil.copy2(source, staged)

    # The CSV will be generated inside the container.
    return staged, run_dir / "input.csv"


def container_data_path(path: Path) -> str:
    return (
        f"{CONTAINER_DATA_ROOT}/"
        f"{path.resolve().relative_to(DATA_DIR.resolve()).as_posix()}"
    )


# ============================================================
# 6. DOCKER COMPOSE HELPERS
# ============================================================


def write_compose_override(
    run_dir: Path,
    run_id: str,
) -> Path:
    suffix = run_id.lower()
    override_path = run_dir / "docker-compose.sample.yml"

    content = f"""
services:
  ml-training:
    container_name: vehicle-fault-training-sample-{suffix}
    environment:
      PYTHONDONTWRITEBYTECODE: "1"
    volumes:
      - ./data_Sample:/app/data
      - ./data_Sample/models:/app/models

  api:
    container_name: vehicle-fault-api-sample-{suffix}
    ports: !override
      - "8001:8000"
    volumes:
      - ./data_Sample/models/onnx:/app/models/onnx:ro

  pyspark:
    container_name: vehicle-fault-pyspark-sample-{suffix}
    environment:
      PYTHONDONTWRITEBYTECODE: "1"
    volumes:
      - ./data_Sample:/app/data

  mlflow:
    container_name: vehicle-fault-mlflow-sample-{suffix}
    ports: !override []
    volumes:
      - ./data_Sample/mlflow-data:/mlruns
"""

    override_path.write_text(content, encoding="utf-8")
    return override_path


def compose_prefix(
    override_path: Path,
    run_id: str,
) -> list[str]:
    relative_override = override_path.resolve().relative_to(ROOT).as_posix()

    return [
        "docker",
        "compose",
        "--project-name",
        f"vehicle-fault-prediction-sample-{run_id.lower()}",
        "-f",
        "docker-compose.yml",
        "-f",
        relative_override,
    ]


def run_command(
    command: list[str],
    cwd: Path = ROOT,
) -> tuple[int, str]:
    """Run a command while displaying its output live."""
    print("$", " ".join(command), flush=True)

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as error:
        print(f"Could not start command: {error}")
        return 127, str(error)

    output_lines: list[str] = []

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    return_code = process.wait()
    return return_code, "".join(output_lines)


def cleanup_compose_project(compose_args: list[str]) -> None:
    """Remove this run's containers and network, preserving data and volumes."""
    print("\nCleaning up pipeline containers...", flush=True)

    code, output = run_command(
        [
            *compose_args,
            "down",
            "--remove-orphans",
        ]
    )

    if code == 0:
        print("Pipeline containers and network removed.", flush=True)
    else:
        print(
            "WARNING: Could not fully clean up pipeline containers:\n"
            f"{output[-3000:]}",
            file=sys.stderr,
            flush=True,
        )


def docker_exec_command(
    compose_args: list[str],
    service: str,
    command: list[str],
    environment: dict[str, str] | None = None,
) -> list[str]:
    result = [*compose_args, "exec", "-T"]

    for key, value in (environment or {}).items():
        result.extend(["-e", f"{key}={value}"])

    result.append(service)
    result.extend(command)

    return result


# ============================================================
# 7. STAGE DEFINITIONS
# ============================================================


def get_stage_definitions(
    run_dir: Path,
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []

    for name, script in PYSPARK_JOBS:
        code_path = ROOT / "services" / "pyspark" / script

        definitions.append(
            {
                "number": len(definitions) + 1,
                "name": name,
                "service": "pyspark",
                "command": ["python", script],
                "environment_kind": "pyspark",
                "code_paths": [code_path],
                "outputs": STAGE_OUTPUTS[len(definitions) + 1],
            }
        )

    # Stage 11: check that every split has both classes.
    distribution_script = r"""
import json
import sys
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("PipelineSplitDistributionCheck")
    .getOrCreate()
)


try:
    root = "/app/data/processed"
    result = {}
    warnings = []

    for split in ("train", "validation", "test"):
        path = f"{root}/{split}"

        rows = (
            spark.read.parquet(path)
            .groupBy("fault_label")
            .count()
            .collect()
        )

        counts = {
            int(row["fault_label"]): int(row["count"])
            for row in rows
        }

        normal = counts.get(0, 0)
        fault = counts.get(1, 0)

        result[split] = {
            "rows": normal + fault,
            "normal": normal,
            "fault": fault,
            "status": (
                "passed"
                if normal > 0 and fault > 0
                else "warning"
            ),
        }

        print(
            f"{split}: normal={normal}, fault={fault}",
            flush=True,
        )

        if normal == 0 and fault == 0:
            warnings.append(
                f"{split} is empty. Check the dataset."
            )
        elif normal == 0 or fault == 0:
            warnings.append(
                f"{split} contains only one class: "
                f"normal={normal}, fault={fault}. "
                "Some training or evaluation tasks may be unavailable."
            )

    with open(sys.argv[1], "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    if warnings:
        print("\n========== CLASS DISTRIBUTION WARNINGS ==========")
        for warning in warnings:
            print(f"WARNING: {warning}", flush=True)

    print(
        "\nClass-distribution validation completed. "
        "Review any warnings above.",
        flush=True,
    )

finally:
    spark.stop()

"""

    definitions.append(
        {
            "number": 11,
            "name": "Fault-label distribution gate",
            "service": "pyspark",
            "command": [
                "python",
                "-c",
                distribution_script,
                container_data_path(run_dir / "split_distribution.json"),
            ],
            "environment_kind": "pyspark",
            "code_paths": [],
            "outputs": [],
        }
    )

    for name, command in ML_STAGES:
        module = command[-1]

        if command[1].endswith(".py"):
            code_path = ROOT / "services" / "ml-training" / command[1]
        else:
            code_path = (
                ROOT
                / "services"
                / "ml-training"
                / Path(*module.split(".")).with_suffix(".py")
            )

        definitions.append(
            {
                "number": len(definitions) + 1,
                "name": name,
                "service": "ml-training",
                "command": command,
                "environment_kind": "ml",
                "code_paths": [code_path],
                "outputs": STAGE_OUTPUTS[len(definitions) + 1],
            }
        )

    return definitions


def stage_output_paths(stage: dict[str, Any]) -> list[Path]:
    return [DATA_DIR / item for item in stage["outputs"]]


def outputs_exist(stage: dict[str, Any]) -> bool:
    for path in stage_output_paths(stage):
        if not path.exists():
            return False

        if path.is_file() and path.stat().st_size == 0:
            return False

        if path.is_dir() and not any(path.iterdir()):
            return False

    return True


def can_reuse_stage(
    checkpoint: dict[str, Any],
    stage: dict[str, Any],
) -> bool:
    number = stage["number"]
    saved = checkpoint.get("stages", {}).get(str(number), {})

    if saved.get("status") != "SUCCESS":
        return False

    current_fingerprint = code_fingerprint(stage["code_paths"])
    saved_fingerprint = saved.get("code_sha256")

    # Stages with source files must have a matching fingerprint.
    if stage["code_paths"]:
        if current_fingerprint is None:
            return False
        if current_fingerprint != saved_fingerprint:
            return False

    # Do not trust a cached stage if mapped outputs have disappeared.
    if not outputs_exist(stage):
        return False

    # Stages with no mapped outputs are reusable only on recorded success.
    return True


# ============================================================
# 8. REPORTS
# ============================================================


def save_report(
    report_path: Path,
    run_id: str,
    checkpoint: dict[str, Any],
    started_at: str,
    invocation_seconds: float,
    failure: str | None = None,
) -> None:
    report = {
        "run_id": run_id,
        "status": "FAILED" if failure else checkpoint["status"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds_this_invocation": round(
            time.perf_counter() - invocation_seconds,
            2,
        ),
        "last_completed_stage": checkpoint.get("last_completed_stage", 0),
        "next_stage": checkpoint.get("next_stage"),
        "stages": checkpoint.get("stages", {}),
        "input": checkpoint.get("input", {}),
        "failure": failure or checkpoint.get("failure"),
    }

    save_json(report_path, report)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================


def main() -> int:
    args = parse_args()
    invocation_seconds = time.perf_counter()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "models" / "onnx").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "mlflow-data").mkdir(parents=True, exist_ok=True)

    if args.resume:
        run_id = args.resume
        run_dir = RUNS_DIR / run_id

        if not run_dir.is_dir():
            print(
                f"ERROR: Run directory not found: {run_dir}",
                file=sys.stderr,
            )
            return 2

        checkpoint_path = run_dir / "checkpoint.json"
        report_path = run_dir / "run_report.json"
        override_path = run_dir / "docker-compose.sample.yml"

        try:
            checkpoint = load_json(checkpoint_path)
        except (OSError, json.JSONDecodeError) as error:
            print(f"ERROR: Cannot load checkpoint: {error}", file=sys.stderr)
            return 2

        if checkpoint.get("run_id") != run_id:
            print("ERROR: Run ID does not match checkpoint.", file=sys.stderr)
            return 2

        staged_input = Path(checkpoint["input"]["path"])

        if not staged_input.is_file():
            print(
                f"ERROR: Saved input is missing: {staged_input}",
                file=sys.stderr,
            )
            return 2

        if sha256_file(staged_input) != checkpoint["input"]["sha256"]:
            print(
                "ERROR: Saved input changed since the run started.",
                file=sys.stderr,
            )
            return 2

        if not override_path.is_file():
            print(
                f"ERROR: Compose override is missing: {override_path}",
                file=sys.stderr,
            )
            return 2

        # For Excel runs, reuse the staged CSV if it was already created.
        csv_input = (
            run_dir / "input.csv"
            if staged_input.suffix.lower() in {".xlsx", ".xlsm"}
            else staged_input
        )

        started_at = checkpoint.get("started_at_utc", utc_now())

    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        staged_input, converted_csv = prepare_input(args.input, run_dir)
        csv_input = converted_csv or staged_input

        checkpoint_path = run_dir / "checkpoint.json"
        report_path = run_dir / "run_report.json"
        override_path = write_compose_override(run_dir, run_id)

        checkpoint = new_checkpoint(run_id, staged_input)
        checkpoint["input"]["csv_path"] = str(csv_input.resolve())
        checkpoint["input"]["excel_sheet"] = args.sheet
        checkpoint["input"]["csv_sha256"] = (
            sha256_file(csv_input) if csv_input.is_file() else None
        )

        started_at = checkpoint["started_at_utc"]
        save_checkpoint(checkpoint_path, checkpoint)

    compose_args = compose_prefix(override_path, run_id)
    stages = get_stage_definitions(run_dir)

    print("=" * 65)
    print(" VEHICLE FAULT PREDICTION PIPELINE")
    print("=" * 65)
    print(f"Run ID:     {run_id}")
    print(f"Run folder: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    try:
        # Start the containers under this run's unique Compose project.
        code, output = run_command(
            [
                *compose_args,
                "up",
                "-d",
                "pyspark",
                "ml-training",
                "mlflow",
            ]
        )

        if code != 0:
            raise RuntimeError(f"Could not start pipeline services:\n{output}")

        # Convert Excel only if no converted CSV has been saved yet.
        if staged_input.suffix.lower() in {".xlsx", ".xlsm"}:
            if not csv_input.is_file() or csv_input.stat().st_size == 0:
                sheet = args.sheet or checkpoint["input"].get("excel_sheet") or "0"

                conversion_code = (
                    "import pandas as pd, sys; "
                    "sheet = int(sys.argv[3]) if sys.argv[3].isdigit() "
                    "else sys.argv[3]; "
                    "df = pd.read_excel("
                    "sys.argv[1], sheet_name=sheet, engine='openpyxl'); "
                    "df.to_csv(sys.argv[2], index=False); "
                    "print(f'Converted {len(df)} rows and "
                    "{len(df.columns)} columns')"
                )

                command = docker_exec_command(
                    compose_args,
                    "ml-training",
                    [
                        "python",
                        "-c",
                        conversion_code,
                        container_data_path(staged_input),
                        container_data_path(csv_input),
                        str(sheet),
                    ],
                )

                code, output = run_command(command)

                if code != 0:
                    raise RuntimeError(f"Excel-to-CSV conversion failed:\n{output}")

            # Save the converted CSV fingerprint for resume validation.
            current_csv_hash = sha256_file(csv_input)
            saved_csv_hash = checkpoint["input"].get("csv_sha256")

            if saved_csv_hash and current_csv_hash != saved_csv_hash:
                raise ValueError(
                    "Converted CSV fingerprint differs from the " "previous checkpoint."
                )

            checkpoint["input"]["csv_sha256"] = current_csv_hash
            save_checkpoint(checkpoint_path, checkpoint)

        if not csv_input.is_file() or csv_input.stat().st_size == 0:
            raise FileNotFoundError(f"Pipeline CSV is missing or empty: {csv_input}")

        pyspark_environment = {
            "CSV_PATH": container_data_path(csv_input),
            "PARQUET_PATH": CONTAINER_PARQUET_PATH,
            "DATA_ROOT": CONTAINER_DATA_ROOT,
        }

        # Find the earliest stage that is not safely reusable.
        earliest_invalid = TOTAL_STAGES + 1

        for stage in stages:
            if args.resume and can_reuse_stage(checkpoint, stage):
                print(
                    f"[{stage['number']}/{TOTAL_STAGES}] "
                    f"{stage['name']} ... SKIPPED (cached SUCCESS)"
                )
                continue

            earliest_invalid = stage["number"]
            break

        if args.from_step:
            if args.from_step > earliest_invalid:
                raise ValueError(
                    f"Stage {earliest_invalid} is the earliest stage "
                    "that is not safely reusable. Cannot skip it."
                )

            start_stage = args.from_step
        else:
            start_stage = earliest_invalid

        if start_stage > TOTAL_STAGES:
            print("\nAll stages are already complete and reusable.")
        else:
            print(
                f"\nResuming from stage {start_stage}: "
                f"{stages[start_stage - 1]['name']}"
            )

        for stage in stages:
            number = stage["number"]

            if number < start_stage:
                continue

            # A later stage is not allowed to run while an earlier
            # stage is incomplete or has invalid cached outputs.
            fingerprint = code_fingerprint(stage["code_paths"])

            mark_running(
                checkpoint_path,
                checkpoint,
                number,
                stage["name"],
                fingerprint,
            )

            print(
                f"\n[{number}/{TOTAL_STAGES}] " f"{stage['name']} ... RUNNING",
                flush=True,
            )

            stage_clock = time.perf_counter()

            environment = (
                pyspark_environment
                if stage["environment_kind"] == "pyspark"
                else {"PYTHONDONTWRITEBYTECODE": "1"}
            )

            command = docker_exec_command(
                compose_args,
                stage["service"],
                stage["command"],
                environment,
            )

            code, output = run_command(command)

            error = None

            if code != 0:
                error = f"Command exited with code {code}.\n{output[-5000:]}"

            if stage["name"] == "MLflow tracking" and "Could not log " in output:
                error = "MLflow tracking reported a logging failure."

            # Stage 11 must leave its distribution report behind.
            if number == 11:
                distribution_path = run_dir / "split_distribution.json"

                if not distribution_path.is_file():
                    error = "Fault-label distribution report was not created."
                elif distribution_path.stat().st_size == 0:
                    error = "Fault-label distribution report is empty."

            # Validate mapped outputs after a successful command.
            if error is None and not outputs_exist(stage):
                error = (
                    "Stage exited successfully but expected output(s) "
                    f"are missing: {stage['outputs']}"
                )

            success = error is None

            elapsed = round(time.perf_counter() - stage_clock, 2)

            mark_finished(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                number=number,
                name=stage["name"],
                success=success,
                fingerprint=fingerprint,
                error=error,
                output_paths=[str(path) for path in stage_output_paths(stage)],
            )

            checkpoint["stages"][str(number)]["elapsed_seconds"] = elapsed
            save_checkpoint(checkpoint_path, checkpoint)

            save_report(
                report_path,
                run_id,
                checkpoint,
                started_at,
                invocation_seconds,
                error,
            )

            print(
                f"Stage {number}: "
                f"{'SUCCESS' if success else 'FAILED'} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

            if not success:
                print(
                    "\nPipeline stopped. Completed stages have been "
                    "saved to checkpoint.json."
                )
                print("Resume with:\n" f"python .\\run_pipeline.py --resume {run_id}")
                return 1

        checkpoint["status"] = "SUCCESS"
        checkpoint["failure"] = None
        checkpoint["next_stage"] = None
        checkpoint["finished_at_utc"] = utc_now()
        save_checkpoint(checkpoint_path, checkpoint)

        if args.start_api:
            code, output = run_command(
                [
                    *compose_args,
                    "up",
                    "-d",
                    "api",
                ]
            )

            if code != 0:
                checkpoint["status"] = "FAILED"
                checkpoint["failure"] = {
                    "stage": "API startup",
                    "message": output[-5000:],
                }
                save_checkpoint(checkpoint_path, checkpoint)

                save_report(
                    report_path,
                    run_id,
                    checkpoint,
                    started_at,
                    invocation_seconds,
                    "Could not start API service.",
                )
                return 1

        save_report(
            report_path,
            run_id,
            checkpoint,
            started_at,
            invocation_seconds,
        )

        print("\n" + "=" * 65)
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Report:     {report_path}")

        return 0

    except Exception as error:
        checkpoint["status"] = "FAILED"
        checkpoint["failure"] = {
            "message": str(error),
            "time_utc": utc_now(),
        }

        if checkpoint.get("next_stage") is None:
            checkpoint["next_stage"] = checkpoint.get("last_completed_stage", 0) + 1

        save_checkpoint(checkpoint_path, checkpoint)

        save_report(
            report_path,
            run_id,
            checkpoint,
            started_at,
            invocation_seconds,
            str(error),
        )

        print(f"\nERROR: {error}", file=sys.stderr)
        print(
            "\nTo retry this run, execute:\n"
            f"python .\\run_pipeline.py --resume {run_id}",
            file=sys.stderr,
        )

        return 1
    finally:
        cleanup_compose_project(compose_args)


if __name__ == "__main__":
    raise SystemExit(main())


# """Run the existing PySpark and ML-training pipeline in Docker Compose."""

# from __future__ import annotations

# import argparse
# from datetime import datetime, timezone
# import json
# import shutil
# import subprocess
# import sys
# import time
# from pathlib import Path
# from typing import Any

# ROOT = Path(__file__).resolve().parent
# DATA_DIR = ROOT / "data_Sample"
# RUNS_DIR = DATA_DIR / "pipeline_runs"
# METRICS_DIR = DATA_DIR / "metrics"
# DATA_ROOT_IN_CONTAINER = "/app/data"
# PARQUET_PATH_IN_CONTAINER = f"{DATA_ROOT_IN_CONTAINER}/processed/metropt"

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
#     ("Baseline model", "ml-training", ["python", "src/models/baseline_11.py"]),
#     (
#         "Random Forest model",
#         "ml-training",
#         ["python", "src/models/random_forest_12.py"],
#     ),
#     (
#         "Deep-learning model",
#         "ml-training",
#         ["python", "-m", "src.models.deep_learning_13"],
#     ),
#     ("Model evaluation", "ml-training", ["python", "-m", "src.evaluation.evaluate_14"]),
#     ("Random Forest tuning", "ml-training", ["python", "-m", "src.models.tuning_15"]),
#     (
#         "MLflow tracking",
#         "ml-training",
#         ["python", "-m", "src.experiments.mlflow_tracking_16"],
#     ),
#     ("ONNX export", "ml-training", ["python", "-m", "src.export.export_onnx_17"]),
#     (
#         "ONNX benchmark",
#         "ml-training",
#         ["python", "-m", "src.benchmarking.benchmark_onnx_18"],
#     ),
# ]

# EXPECTED_OUTPUTS = [
#     "processed/metropt",
#     "processed/metropt_timeseries",
#     "processed/metropt_labeled",
#     "processed/metropt_features",
#     "processed/train",
#     "processed/validation",
#     "processed/test",
#     "metrics/baseline_logistic_regression.json",
#     "metrics/random_forest.json",
#     "metrics/deep_learning.json",
#     "metrics/evaluation_14.json",
#     "metrics/random_forest_tuned.json",
#     "metrics/onnx_benchmark_18.json",
#     "models/baseline_logistic_regression.joblib",
#     "models/random_forest.joblib",
#     "models/random_forest_tuned.joblib",
#     "models/deep_learning.pt",
#     "models/onnx/random_forest.onnx",
#     "models/onnx/random_forest_metadata.json",
# ]

# METRIC_FILES = [
#     "baseline_logistic_regression.json",
#     "random_forest.json",
#     "deep_learning.json",
#     "evaluation_14.json",
#     "random_forest_tuned.json",
#     "onnx_benchmark_18.json",
# ]


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Run the existing PySpark and ML-training jobs in sequence. "
#             "Accepts CSV, XLSX, or XLSM input."
#         )
#     )
#     parser.add_argument("input", type=Path, help="Input CSV, XLSX, or XLSM file")
#     parser.add_argument(
#         "--sheet",
#         help="Excel sheet name or zero-based index (default: first sheet)",
#     )
#     parser.add_argument(
#         "--start-api",
#         action="store_true",
#         help="Start the API service after a successful pipeline run",
#     )
#     return parser.parse_args()


# def container_data_path(host_path: Path) -> str:
#     relative_path = host_path.resolve().relative_to(DATA_DIR.resolve())
#     return f"{DATA_ROOT_IN_CONTAINER}/{relative_path.as_posix()}"


# def prepare_input(source: Path, run_dir: Path) -> tuple[Path, Path | None]:
#     source = source.expanduser().resolve()
#     if not source.is_file():
#         raise FileNotFoundError(f"Input file does not exist: {source}")

#     extension = source.suffix.lower()
#     if extension not in {".csv", ".xlsx", ".xlsm"}:
#         raise ValueError("Supported input formats are .csv, .xlsx, and .xlsm.")

#     with source.open("rb") as input_file:
#         signature = input_file.read(4)
#     is_excel_archive = signature.startswith(b"PK\x03\x04")
#     if extension == ".csv" and is_excel_archive:
#         extension = ".xlsx"
#     elif extension in {".xlsx", ".xlsm"} and not is_excel_archive:
#         raise ValueError(
#             f"The file extension {extension} does not match an Excel workbook."
#         )

#     DATA_DIR.mkdir(parents=True, exist_ok=True)
#     (DATA_DIR / "models" / "onnx").mkdir(parents=True, exist_ok=True)
#     (DATA_DIR / "mlflow-data").mkdir(parents=True, exist_ok=True)
#     run_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         source.resolve().relative_to(DATA_DIR.resolve())
#         if source.suffix.lower() == extension:
#             staged_source = source
#         else:
#             staged_source = run_dir / f"input{extension}"
#             shutil.copy2(source, staged_source)
#     except ValueError:
#         staged_source = run_dir / f"input{extension}"
#         shutil.copy2(source, staged_source)

#     if extension == ".csv":
#         return staged_source, None

#     return staged_source, run_dir / "input.csv"


# def docker_exec_command(
#     service: str,
#     command: list[str],
#     compose_prefix: list[str],
#     environment: dict[str, str] | None = None,
# ) -> list[str]:
#     docker_command = [*compose_prefix, "exec", "-T"]
#     for name, value in (environment or {}).items():
#         docker_command.extend(["-e", f"{name}={value}"])
#     docker_command.append(service)
#     docker_command.extend(command)
#     return docker_command


# def write_sample_compose_override(run_dir: Path, run_id: str) -> Path:
#     suffix = run_id.lower()
#     override_path = run_dir / "docker-compose.sample.yml"
#     override = f"""services:
#     ml-training:
#         container_name: vehicle-fault-training-sample-{suffix}
#         environment:
#             PYTHONDONTWRITEBYTECODE: "1"
#         volumes:
#             - ./data_Sample:/app/data
#             - ./data_Sample/models:/app/models
#     api:
#         container_name: vehicle-fault-api-sample-{suffix}
#         ports: !override
#             - "8001:8000"
#         volumes:
#             - ./data_Sample/models/onnx:/app/models/onnx:ro
#     pyspark:
#         container_name: vehicle-fault-pyspark-sample-{suffix}
#         environment:
#             PYTHONDONTWRITEBYTECODE: "1"
#         volumes:
#             - ./data_Sample:/app/data
#     mlflow:
#         container_name: vehicle-fault-mlflow-sample-{suffix}
#         ports: !override []
#         volumes:
#             - ./data_Sample/mlflow-data:/mlruns
# """
#     override_path.write_text(override, encoding="utf-8")
#     return override_path


# def compose_prefix(override_path: Path, run_id: str) -> list[str]:
#     return [
#         "docker",
#         "compose",
#         "--project-name",
#         f"vehicle-fault-prediction-sample-{run_id.lower()}",
#         "-f",
#         "docker-compose.yml",
#         "-f",
#         override_path.resolve().relative_to(ROOT).as_posix(),
#     ]


# def run_streaming(command: list[str], *, cwd: Path = ROOT) -> tuple[int, str]:
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

#     captured_output: list[str] = []
#     assert process.stdout is not None
#     for line in process.stdout:
#         print(line, end="", flush=True)
#         captured_output.append(line)

#     return_code = process.wait()
#     return return_code, "".join(captured_output)


# def validate_split_distribution(report_path: str) -> list[str]:
#     script = """
# import json
# import sys
# from pyspark.sql import SparkSession

# spark = SparkSession.builder.appName("PipelineSplitDistributionCheck").getOrCreate()
# try:
#     root = "/app/data/processed"
#     distribution = {}
#     invalid_splits = []
#     for split in ("train", "validation", "test"):
#         counts = {
#             int(row["fault_label"]): int(row["count"])
#             for row in spark.read.parquet(f"{root}/{split}")
#             .groupBy("fault_label").count().collect()
#         }
#         normal_count = counts.get(0, 0)
#         fault_count = counts.get(1, 0)
#         distribution[split] = {
#             "rows": normal_count + fault_count,
#             "normal": normal_count,
#             "fault": fault_count,
#         }
#         print(f"{split}: normal={normal_count}, fault={fault_count}")
#         if normal_count == 0 or fault_count == 0:
#             invalid_splits.append(
#                 f"{split} split must contain both normal and fault examples; "
#                 f"found normal={normal_count}, fault={fault_count}"
#             )
#     with open(sys.argv[1], "w", encoding="utf-8") as report_file:
#         json.dump(distribution, report_file, indent=2)
#     if invalid_splits:
#         raise ValueError("; ".join(invalid_splits))
# finally:
#     spark.stop()
# """
#     return ["python", "-c", script, report_path]


# def record_stage(
#     records: list[dict[str, Any]],
#     name: str,
#     service: str,
#     command: list[str],
#     compose_args: list[str],
#     *,
#     environment: dict[str, str] | None = None,
#     failure_markers: tuple[str, ...] = (),
# ) -> bool:
#     index = len(records) + 1
#     total = 19
#     print(f"\n[{index}/{total}] {name} ... RUNNING", flush=True)
#     started = time.perf_counter()
#     full_command = docker_exec_command(service, command, compose_args, environment)
#     return_code, output = run_streaming(full_command)
#     marker_failure = next(
#         (marker for marker in failure_markers if marker in output), None
#     )
#     succeeded = return_code == 0 and marker_failure is None
#     elapsed = time.perf_counter() - started
#     status = "SUCCESS" if succeeded else "FAILED"
#     print(f"      STATUS: {status} ({elapsed:.1f}s)", flush=True)

#     records.append(
#         {
#             "name": name,
#             "service": service,
#             "command": command,
#             "exit_code": return_code,
#             "status": status,
#             "elapsed_seconds": round(elapsed, 3),
#             "failure_marker": marker_failure,
#         }
#     )
#     return succeeded


# def collect_metrics(previous_mtimes: dict[str, int]) -> dict[str, Any]:
#     results: dict[str, Any] = {}
#     for filename in METRIC_FILES:
#         path = METRICS_DIR / filename
#         if not path.is_file():
#             continue
#         try:
#             results[filename] = {
#                 "generated_this_run": path.stat().st_mtime_ns
#                 != previous_mtimes.get(filename),
#                 "values": json.loads(path.read_text(encoding="utf-8")),
#             }
#         except (OSError, json.JSONDecodeError) as error:
#             results[filename] = {"read_error": str(error)}
#     return results


# def write_report(report_path: Path, report: dict[str, Any]) -> None:
#     report_path.parent.mkdir(parents=True, exist_ok=True)
#     temporary_path = report_path.with_suffix(".tmp")
#     temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
#     temporary_path.replace(report_path)


# def main() -> int:
#     args = parse_args()
#     run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
#     run_dir = RUNS_DIR / run_id
#     report_path = run_dir / "run_report.json"
#     started_at = datetime.now(timezone.utc)
#     started_clock = time.perf_counter()
#     previous_metric_mtimes = (
#         {
#             path.name: path.stat().st_mtime_ns
#             for path in METRICS_DIR.glob("*.json")
#             if path.is_file()
#         }
#         if METRICS_DIR.exists()
#         else {}
#     )
#     stages: list[dict[str, Any]] = []
#     failure: str | None = None
#     staged_input: Path | None = None
#     csv_input: Path | None = None
#     split_distribution_path = run_dir / "split_distribution.json"
#     split_distribution: dict[str, Any] | None = None
#     compose_args: list[str] | None = None

#     print("=" * 60)
#     print(" VEHICLE FAULT PREDICTION - PIPELINE RUN")
#     print("=" * 60)

#     try:
#         staged_input, converted_csv = prepare_input(args.input, run_dir)
#         csv_input = converted_csv or staged_input
#         override_path = write_sample_compose_override(run_dir, run_id)
#         compose_args = compose_prefix(override_path, run_id)

#         print(f"Input: {args.input.expanduser().resolve()}")
#         print(f"Isolated output root: {DATA_DIR}")
#         print(f"Run ID: {run_id}")

#         up_command = [*compose_args, "up", "-d", "pyspark", "ml-training", "mlflow"]
#         print("\nStarting PySpark, ML-training, and MLflow services...", flush=True)
#         return_code, output = run_streaming(up_command)
#         if return_code != 0:
#             raise RuntimeError(
#                 f"Could not start pipeline services (exit code {return_code}).\n{output}"
#             )

#         if converted_csv is not None:
#             source_in_container = container_data_path(staged_input)
#             csv_in_container = container_data_path(converted_csv)
#             sheet_value = args.sheet or "0"
#             excel_conversion = (
#                 "import pandas as pd, sys; "
#                 "sheet = int(sys.argv[3]) if sys.argv[3].isdigit() else sys.argv[3]; "
#                 "frame = pd.read_excel(sys.argv[1], sheet_name=sheet, engine='openpyxl'); "
#                 "frame.to_csv(sys.argv[2], index=False); "
#                 "print(f'Converted {len(frame)} rows and {len(frame.columns)} columns to CSV')"
#             )
#             conversion_command = docker_exec_command(
#                 "ml-training",
#                 [
#                     "python",
#                     "-c",
#                     excel_conversion,
#                     source_in_container,
#                     csv_in_container,
#                     sheet_value,
#                 ],
#                 compose_args,
#             )
#             print(
#                 "\nConverting Excel input to CSV in the ML-training container...",
#                 flush=True,
#             )
#             return_code, output = run_streaming(conversion_command)
#             if return_code != 0:
#                 raise RuntimeError(
#                     f"Excel conversion failed (exit code {return_code}).\n{output}"
#                 )

#         assert csv_input is not None
#         pyspark_environment = {
#             "CSV_PATH": container_data_path(csv_input),
#             "PARQUET_PATH": PARQUET_PATH_IN_CONTAINER,
#             "DATA_ROOT": DATA_ROOT_IN_CONTAINER,
#         }

#         for name, script_path in PYSPARK_JOBS:
#             if not record_stage(
#                 stages,
#                 name,
#                 "pyspark",
#                 ["python", script_path],
#                 compose_args,
#                 environment=pyspark_environment,
#             ):
#                 raise RuntimeError(f"Stage failed: {name}")

#         if not record_stage(
#             stages,
#             "Fault-label distribution gate",
#             "pyspark",
#             validate_split_distribution(container_data_path(split_distribution_path)),
#             compose_args,
#             environment=pyspark_environment,
#         ):
#             raise RuntimeError("Fault-label distribution check failed.")
#         split_distribution = json.loads(
#             split_distribution_path.read_text(encoding="utf-8")
#         )

#         for name, service, command in ML_STAGES:
#             failure_markers = ("Could not log ",) if name == "MLflow tracking" else ()
#             if not record_stage(
#                 stages,
#                 name,
#                 service,
#                 command,
#                 compose_args,
#                 environment={"PYTHONDONTWRITEBYTECODE": "1"},
#                 failure_markers=failure_markers,
#             ):
#                 raise RuntimeError(f"Stage failed: {name}")

#         if args.start_api:
#             print("\nStarting API service...", flush=True)
#             return_code, output = run_streaming([*compose_args, "up", "-d", "api"])
#             if return_code != 0:
#                 raise RuntimeError(
#                     f"Could not start API service (exit code {return_code}).\n{output}"
#                 )

#     except Exception as error:
#         failure = str(error)

#     elapsed = time.perf_counter() - started_clock
#     success = failure is None
#     available_outputs = {
#         output: (DATA_DIR / output).exists() for output in EXPECTED_OUTPUTS
#     }
#     report = {
#         "run_id": run_id,
#         "status": "SUCCESS" if success else "FAILED",
#         "started_at_utc": started_at.isoformat(),
#         "finished_at_utc": datetime.now(timezone.utc).isoformat(),
#         "elapsed_seconds": round(elapsed, 3),
#         "input": {
#             "provided_path": str(args.input.expanduser().resolve()),
#             "staged_path": str(staged_input) if staged_input else None,
#             "csv_path": str(csv_input) if csv_input else None,
#             "excel_sheet": (args.sheet if converted_csv is not None else None),
#         },
#         "stages": stages,
#         "split_distribution": split_distribution,
#         "failure": failure,
#         "outputs_present": available_outputs,
#         "metrics": collect_metrics(previous_metric_mtimes),
#         "api_started": bool(args.start_api and success),
#     }

#     try:
#         write_report(report_path, report)
#     except OSError as error:
#         print(f"Could not write run report: {error}", file=sys.stderr)
#         success = False

#     print("\n" + "=" * 60)
#     print(" PIPELINE COMPLETED" if success else " PIPELINE FAILED")
#     print("=" * 60)
#     print(f"Report: {report_path}")
#     if failure:
#         print(f"Failure: {failure}")
#     if success:
#         print(
#             "Output: processed datasets, metrics, models, ONNX export, and run report"
#         )
#     return 0 if success else 1


# if __name__ == "__main__":
#     raise SystemExit(main())
