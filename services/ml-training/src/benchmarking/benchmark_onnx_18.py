import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

DATA_ROOT = Path("/app/data")
MODEL_ROOTS = [
    Path("/app/data/models"),
    Path("/app/models"),
]

ONNX_DIR = DATA_ROOT / "models" / "onnx"
METADATA_PATH = ONNX_DIR / "random_forest_metadata.json"
REPORT_PATH = DATA_ROOT / "metrics" / "onnx_benchmark_18.json"

MAX_ROWS = 5000
BATCH_SIZE = 256
WARMUP_RUNS = 3
BENCHMARK_RUNS = 20


def find_onnx_model(onnx_dir):
    candidates = [
        onnx_dir / "random_forest.onnx",
        onnx_dir / "random_forest_model.onnx",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else []

    if matches:
        return matches[0]

    raise FileNotFoundError(f"No ONNX model found in {onnx_dir}. Check Step 17.")


def find_test_data(data_root):
    candidates = []

    for path in data_root.rglob("*"):
        if path.name.lower() in {"test", "test.parquet", "test.parquet.gzip"}:
            if path.is_file() and path.suffix == ".parquet":
                candidates.append(path)
            elif path.is_dir() and any(path.rglob("*.parquet")):
                candidates.append(path)

    for path in data_root.rglob("*test*.parquet"):
        if path.is_file() and path not in candidates:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"Could not find test parquet data under {data_root}.")

    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    return candidates[0]


def get_feature_columns(metadata, dataframe, session):
    for key in [
        "feature_columns",
        "features",
        "input_features",
        "feature_names",
    ]:
        value = metadata.get(key)

        if isinstance(value, list) and value:
            missing = [col for col in value if col not in dataframe.columns]

            if missing:
                raise ValueError(
                    f"Metadata lists features missing from test data: {missing[:10]}"
                )

            return value

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

    columns = [
        col
        for col in dataframe.columns
        if col.lower() not in excluded and pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    if isinstance(expected_features, int):
        if len(columns) != expected_features:
            raise ValueError(
                f"Found {len(columns)} candidate features, but ONNX "
                f"expects {expected_features}. Add the exact feature list "
                "to random_forest_metadata.json."
            )
        return columns

    raise ValueError(
        "Could not determine feature columns. Add feature_columns "
        "to random_forest_metadata.json."
    )


def find_sklearn_model(model_roots):
    for root in model_roots:
        if not root.exists():
            continue

        for extension in ("*.joblib", "*.pkl", "*.pickle"):
            for path in root.rglob(extension):
                if "random_forest" in path.name.lower():
                    return path

    return None


def predict_onnx(session, features):
    input_name = session.get_inputs()[0].name

    return session.run(
        None,
        {input_name: features.astype(np.float32, copy=False)},
    )


def benchmark_onnx(config=None, test_df=None):

    config = config or {}

    data_root = Path(config.get("data_root", DATA_ROOT))
    model_roots = [Path(path) for path in config.get("model_roots", MODEL_ROOTS)]
    onnx_dir = Path(config.get("onnx_dir", ONNX_DIR))
    metadata_path = Path(config.get("metadata_path", METADATA_PATH))
    report_path = Path(config.get("report_path", REPORT_PATH))

    max_rows = config.get("max_rows", MAX_ROWS)
    batch_size = config.get("batch_size", BATCH_SIZE)
    warmup_runs = config.get("warmup_runs", WARMUP_RUNS)
    benchmark_runs = config.get("benchmark_runs", BENCHMARK_RUNS)

    print("\n========== STEP 18: ONNX BENCHMARKING ==========")

    onnx_path = find_onnx_model(onnx_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if test_df is None:
        test_path = find_test_data(data_root)
        dataframe = pd.read_parquet(test_path)
        print(f"Test data: {test_path}")
    else:
        dataframe = test_df.copy()
        print("Using externally supplied test DataFrame.")
        test_path = "externally_supplied_dataframe"

    print(f"ONNX model: {onnx_path}")

    dataframe = dataframe.head(max_rows).copy()

    if dataframe.empty:
        raise ValueError("Test dataset is empty.")

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

    # 1. Warm-up
    batch = X[: min(batch_size, len(X))]

    for _ in range(warmup_runs):
        predict_onnx(session, batch)

    # 2. ONNX latency and throughput
    durations = []

    for _ in range(benchmark_runs):
        start = time.perf_counter()
        predict_onnx(session, batch)
        durations.append(time.perf_counter() - start)

    average_batch_seconds = float(np.mean(durations))
    p95_batch_seconds = float(np.percentile(durations, 95))

    start = time.perf_counter()

    for offset in range(0, len(X), batch_size):
        predict_onnx(session, X[offset : offset + batch_size])

    full_inference_seconds = time.perf_counter() - start

    throughput = len(X) / full_inference_seconds if full_inference_seconds > 0 else None

    # 3. Optional sklearn vs ONNX consistency
    sklearn_path = find_sklearn_model(model_roots)

    consistency = {
        "sklearn_model_found": sklearn_path is not None,
        "checked": False,
        "prediction_match_rate": None,
    }

    if sklearn_path is not None:
        try:
            import joblib

            sklearn_model = joblib.load(sklearn_path)

            if hasattr(sklearn_model, "predict"):
                sklearn_predictions = np.asarray(
                    sklearn_model.predict(pd.DataFrame(X, columns=feature_columns))
                ).reshape(-1)

                onnx_outputs = predict_onnx(session, X)
                onnx_predictions = np.asarray(onnx_outputs[0]).reshape(-1)

                if len(onnx_predictions) == len(sklearn_predictions):
                    consistency = {
                        "sklearn_model_found": True,
                        "sklearn_model_path": str(sklearn_path),
                        "checked": True,
                        "prediction_match_rate": float(
                            np.mean(onnx_predictions == sklearn_predictions)
                        ),
                    }

        except Exception as exc:
            consistency["error"] = str(exc)

    # 4. Model size
    model_size_bytes = onnx_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    # 5. Save report
    report = {
        "step": 18,
        "model": onnx_path.name,
        "test_data": str(test_path),
        "rows_benchmarked": int(len(X)),
        "feature_count": int(len(feature_columns)),
        "batch_size": int(len(batch)),
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
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

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
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
        print("Prediction match:     Not checked")

    print(f"\nReport saved: {report_path}")
    print("========== STEP 18 COMPLETED ==========")

    return report


def main():
    benchmark_onnx()


if __name__ == "__main__":
    main()
