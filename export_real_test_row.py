import json
import math
from pathlib import Path

import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent

METADATA_PATH = ROOT / "data" / "models" / "onnx" / "random_forest_metadata.json"
TEST_DIR = ROOT / "data" / "processed" / "test"

OUTPUT_DIR = ROOT / "artifacts" / "api_validation"
REQUEST_PATH = OUTPUT_DIR / "swagger_request.json"
LABEL_PATH = OUTPUT_DIR / "actual_label.txt"


def main():
    # 1. Load the exact feature order expected by the ONNX model
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    if len(feature_columns) != metadata["feature_count"]:
        raise ValueError("Feature count does not match model metadata.")

    # 2. Read the real held-out test dataset
    parquet_files = sorted(TEST_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet file found in {TEST_DIR}")

    if len(parquet_files) != 1:
        raise ValueError(f"Expected one Parquet file, found {len(parquet_files)}.")

    df = pd.read_parquet(parquet_files[0])

    if "fault_label" not in df.columns:
        raise ValueError("fault_label is missing from the test dataset.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    # 3. Prefer a real fault row for this first validation
    fault_rows = df.index[df["fault_label"] == 1]

    if len(fault_rows) > 0:
        row_index = fault_rows[0]
    else:
        row_index = df.index[0]
        print("No fault-labelled row found; using the first test row.")

    row = df.loc[row_index]
    actual_label = int(row["fault_label"])

    # 4. Build a request containing ONLY the 61 model features
    features = {}

    for column in feature_columns:
        value = float(row[column])

        if not math.isfinite(value):
            raise ValueError(f"Feature {column} has a non-finite value: {value}")

        features[column] = value

    request = {"features": features}

    # 5. Save the request and the true label separately
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with REQUEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(request, file, indent=2, allow_nan=False)

    with LABEL_PATH.open("w", encoding="utf-8") as file:
        file.write(f"Actual fault_label: {actual_label}\n")
        file.write(f"Test row index: {row_index}\n")

    print("Real test row prepared successfully.")
    print(f"Dataset: {parquet_files[0]}")
    print(f"Test row index: {row_index}")
    print(f"Actual fault_label: {actual_label}")
    print(f"Features included: {len(features)}")
    print(f"Swagger request: {REQUEST_PATH}")
    print(f"Actual label saved separately: {LABEL_PATH}")


if __name__ == "__main__":
    main()
