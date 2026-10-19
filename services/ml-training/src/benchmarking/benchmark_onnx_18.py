"""
STEP 18: ONNX Model Benchmarking

Measures:
- Prediction consistency against the original sklearn model, if found
- ONNX inference latency
- Throughput
- ONNX model size
- Benchmark report saved as JSON
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_ROOT = Path("/app/data")
MODEL_ROOTS = [
    Path("/app/data/models"),
    Path("/app/models"),
]

ONNX_DIR = Path("/app/data/models/onnx")
METADATA_PATH = ONNX_DIR / "random_forest_metadata.json"
REPORT_PATH = Path("/app/data/metrics/onnx_benchmark_18.json")

MAX_ROWS = 5000
BATCH_SIZE = 256
WARMUP_RUNS = 3
BENCHMARK_RUNS = 20


def find_onnx_model():
    """Find the exported Random Forest ONNX model."""

    candidates = [
        ONNX_DIR / "random_forest.onnx",
        ONNX_DIR / "random_forest_model.onnx",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(ONNX_DIR.glob("*.onnx")) if ONNX_DIR.exists() else []

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No ONNX model found in {ONNX_DIR}. " "Check the output path from Step 17."
    )


def find_test_data():
    """Find a parquet file or directory containing test data."""

    candidates = []

    for path in DATA_ROOT.rglob("*"):
        if path.name.lower() in {"test", "test.parquet", "test.parquet.gzip"}:
            if path.is_file() and path.suffix == ".parquet":
                candidates.append(path)
            elif path.is_dir() and any(path.rglob("*.parquet")):
                candidates.append(path)

    # Also accept parquet filenames containing 'test'.
    for path in DATA_ROOT.rglob("*test*.parquet"):
        if path.is_file() and path not in candidates:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find test parquet data under {DATA_ROOT}. "
            "Check the output of Step 10."
        )

    # Prefer the most specific test directory/file.
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    return candidates[0]


def load_metadata():
    """Load feature information produced by Step 17."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_feature_columns(metadata, dataframe, session):
    """Resolve feature names from metadata or ONNX input shape."""

    possible_keys = [
        "feature_columns",
        "features",
        "input_features",
        "feature_names",
    ]

    for key in possible_keys:
        value = metadata.get(key)
        if isinstance(value, list) and value:
            missing = [col for col in value if col not in dataframe.columns]
            if missing:
                raise ValueError(
                    f"Metadata lists features missing from test data: "
                    f"{missing[:10]}"
                )
            return value

    # Fall back to the ONNX model's declared input dimension.
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    expected_features = shape[-1] if shape else None

    excluded = {
        "fault_label",
        "label",
        "target",
        "failure_id",
        "timestamp",
        "failure",
    }

    columns = [col for col in dataframe.columns if col.lower() not in excluded]

    columns = [col for col in columns if pd.api.types.is_numeric_dtype(dataframe[col])]

    if isinstance(expected_features, int):
        if len(columns) != expected_features:
            raise ValueError(
                f"Found {len(columns)} candidate features, but ONNX "
                f"expects {expected_features}. Add the exact feature list "
                "to random_forest_metadata.json to avoid incorrect ordering."
            )
        return columns

    raise ValueError(
        "Could not determine feature columns. Add 'feature_columns' "
        "to random_forest_metadata.json."
    )


def find_sklearn_model():
    """Find an optional saved sklearn model for consistency checks."""

    extensions = ("*.joblib", "*.pkl", "*.pickle")

    for root in MODEL_ROOTS:
        if not root.exists():
            continue

        for extension in extensions:
            for path in root.rglob(extension):
                if "random_forest" in path.name.lower():
                    return path

    return None


def load_sklearn_model(path):
    """Load a saved sklearn model."""

    import joblib

    return joblib.load(path)


def predict_onnx(session, features):
    """Run ONNX inference on float32 input."""

    input_name = session.get_inputs()[0].name
    outputs = session.run(
        None,
        {input_name: features.astype(np.float32, copy=False)},
    )
    return outputs


def main():
    print("\n========== STEP 18: ONNX BENCHMARKING ==========")

    onnx_path = find_onnx_model()
    metadata = load_metadata()
    test_path = find_test_data()

    print(f"ONNX model: {onnx_path}")
    print(f"Test data:  {test_path}")

    dataframe = pd.read_parquet(test_path)
    dataframe = dataframe.head(MAX_ROWS).copy()

    if dataframe.empty:
        raise ValueError("Test dataset is empty.")

    # Create the ONNX Runtime CPU session.
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    feature_columns = get_feature_columns(metadata, dataframe, session)

    X = dataframe[feature_columns].apply(pd.to_numeric, errors="raise")

    if X.isnull().any().any():
        raise ValueError(
            "Test features contain missing values. Apply the same "
            "preprocessing used during training before benchmarking."
        )

    X = X.to_numpy(dtype=np.float32)

    print(f"Rows benchmarked: {len(X)}")
    print(f"Features:         {len(feature_columns)}")
    print(f"ONNX inputs:      {[item.name for item in session.get_inputs()]}")
    print(f"ONNX outputs:     {[item.name for item in session.get_outputs()]}")

    # ----------------------------------------------
    # 1. Warm-up
    # ----------------------------------------------

    sample = X[: min(BATCH_SIZE, len(X))]

    for _ in range(WARMUP_RUNS):
        predict_onnx(session, sample)

    # ----------------------------------------------
    # 2. ONNX latency and throughput
    # ----------------------------------------------

    batch = X[: min(BATCH_SIZE, len(X))]

    durations = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        predict_onnx(session, batch)
        durations.append(time.perf_counter() - start)

    average_batch_seconds = float(np.mean(durations))
    p95_batch_seconds = float(np.percentile(durations, 95))

    # Full-dataset inference in batches.
    start = time.perf_counter()

    for offset in range(0, len(X), BATCH_SIZE):
        predict_onnx(session, X[offset : offset + BATCH_SIZE])

    full_inference_seconds = time.perf_counter() - start

    throughput = len(X) / full_inference_seconds if full_inference_seconds > 0 else None

    # ----------------------------------------------
    # 3. Optional sklearn vs ONNX consistency
    # ----------------------------------------------

    sklearn_path = find_sklearn_model()
    consistency = {
        "sklearn_model_found": sklearn_path is not None,
        "checked": False,
        "prediction_match_rate": None,
    }

    if sklearn_path is not None:
        try:
            sklearn_model = load_sklearn_model(sklearn_path)

            if hasattr(sklearn_model, "predict"):
                sklearn_predictions = np.asarray(
                    sklearn_model.predict(pd.DataFrame(X, columns=feature_columns))
                ).reshape(-1)

                onnx_outputs = predict_onnx(session, X)
                onnx_predictions = np.asarray(onnx_outputs[0]).reshape(-1)

                if len(onnx_predictions) == len(sklearn_predictions):
                    match_rate = float(np.mean(onnx_predictions == sklearn_predictions))

                    consistency = {
                        "sklearn_model_found": True,
                        "sklearn_model_path": str(sklearn_path),
                        "checked": True,
                        "prediction_match_rate": match_rate,
                    }

        except Exception as exc:
            consistency["error"] = str(exc)

    # ----------------------------------------------
    # 4. Model size
    # ----------------------------------------------

    model_size_bytes = onnx_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    # ----------------------------------------------
    # 5. Save report
    # ----------------------------------------------

    report = {
        "step": 18,
        "model": onnx_path.name,
        "test_data": str(test_path),
        "rows_benchmarked": int(len(X)),
        "feature_count": int(len(feature_columns)),
        "batch_size": int(len(batch)),
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "onnx_cpu": {
            "average_batch_latency_ms": average_batch_seconds * 1000,
            "p95_batch_latency_ms": p95_batch_seconds * 1000,
            "average_latency_per_row_ms": (average_batch_seconds / len(batch) * 1000),
            "full_dataset_inference_seconds": full_inference_seconds,
            "throughput_rows_per_second": throughput,
        },
        "model_size": {
            "bytes": int(model_size_bytes),
            "megabytes": model_size_mb,
        },
        "prediction_consistency": consistency,
        "runtime_providers": session.get_providers(),
        "feature_columns": feature_columns,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Model size:           {model_size_mb:.2f} MB")
    print(f"Avg batch latency:    {average_batch_seconds * 1000:.3f} ms")
    print(f"P95 batch latency:    {p95_batch_seconds * 1000:.3f} ms")
    print(f"Throughput:           {throughput:.2f} rows/sec")

    if consistency["checked"]:
        print(
            "Prediction match:     "
            f"{consistency['prediction_match_rate'] * 100:.2f}%"
        )
    else:
        print("Prediction match:     Not checked (sklearn model unavailable)")

    print(f"\nReport saved: {REPORT_PATH}")
    print("========== STEP 18 COMPLETED ==========")


if __name__ == "__main__":
    main()
