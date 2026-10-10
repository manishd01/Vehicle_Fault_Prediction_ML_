# export_onnx_17.py
import json
from pathlib import Path

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

DATA_ROOT = Path("/app/data")
MODEL_DIR = DATA_ROOT / "models"
METRICS_DIR = DATA_ROOT / "metrics"
EXPORT_DIR = MODEL_DIR / "onnx"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n========== STEP 17: ONNX EXPORT ==========")

    tuned_model_path = MODEL_DIR / "random_forest_tuned.joblib"
    base_model_path = MODEL_DIR / "random_forest.joblib"

    if tuned_model_path.exists():
        model_path = tuned_model_path
        metrics_path = METRICS_DIR / "random_forest_tuned.json"
    elif base_model_path.exists():
        model_path = base_model_path
        metrics_path = METRICS_DIR / "random_forest.json"
    else:
        raise FileNotFoundError(
            "No Random Forest model found. Run Step 12 or Step 15 first."
        )

    if not metrics_path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {metrics_path}")

    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    with open(metrics_path, encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata.get("feature_columns") or metadata.get("features")

    if not feature_columns:
        raise ValueError("The metrics JSON does not contain the ordered feature list.")

    feature_count = len(feature_columns)

    print(f"Input feature count: {feature_count}")

    # The exported ONNX model accepts float32 feature matrices.
    initial_types = [("float_input", FloatTensorType([None, feature_count]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=17,
    )

    output_path = EXPORT_DIR / "random_forest.onnx"

    with open(output_path, "wb") as file:
        file.write(onnx_model.SerializeToString())

    # Verify that the exported ONNX graph is structurally valid.
    loaded_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(loaded_onnx)

    export_metadata = {
        "source_model": str(model_path),
        "onnx_model": str(output_path),
        "feature_count": feature_count,
        "feature_columns": feature_columns,
        "input_name": "float_input",
        "input_type": "float32",
        "output_format": "converted scikit-learn model outputs",
    }

    metadata_path = EXPORT_DIR / "random_forest_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(export_metadata, file, indent=4)

    print(f"ONNX model saved: {output_path}")
    print(f"Export metadata saved: {metadata_path}")
    print("ONNX structural validation passed.")
    print("========== STEP 17 COMPLETED ==========")


if __name__ == "__main__":
    main()
