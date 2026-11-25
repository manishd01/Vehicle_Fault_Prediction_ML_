import json
from pathlib import Path

import joblib
import onnx

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

DATA_ROOT = Path("/app/data")
MODEL_DIR = DATA_ROOT / "models"
METRICS_DIR = DATA_ROOT / "metrics"
EXPORT_DIR = MODEL_DIR / "onnx"


def contains_dummy_classifier(estimator):
    if isinstance(estimator, DummyClassifier):
        return True

    if isinstance(estimator, Pipeline):
        return any(contains_dummy_classifier(step) for _, step in estimator.steps)

    return False


def export_random_forest_onnx(config=None):
    config = config or {}

    model_dir = Path(config.get("model_dir", MODEL_DIR))
    metrics_dir = Path(config.get("metrics_dir", METRICS_DIR))
    export_dir = Path(config.get("export_dir", EXPORT_DIR))
    target_opset = config.get("target_opset", 17)

    export_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== STEP 17: ONNX EXPORT ==========")

    tuned_model_path = model_dir / "random_forest_tuned.joblib"
    base_model_path = model_dir / "random_forest.joblib"

    if tuned_model_path.exists():
        model_path = tuned_model_path
        metrics_path = metrics_dir / "random_forest_tuned.json"
    elif base_model_path.exists():
        model_path = base_model_path
        metrics_path = metrics_dir / "random_forest.json"
    else:
        raise FileNotFoundError(
            "No Random Forest model found. Run Step 12 or Step 15 first."
        )

    if not metrics_path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {metrics_path}")

    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    if contains_dummy_classifier(model):
        print(
            "WARNING: ONNX export skipped because the saved model "
            "contains DummyClassifier."
        )
        print(
            "The training data must contain both normal (0) and "
            "fault (1) examples before ONNX export can run."
        )
        return None

    with open(metrics_path, encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata.get("feature_columns") or metadata.get("features")

    if not feature_columns:
        raise ValueError("The metrics JSON does not contain the ordered feature list.")

    feature_count = len(feature_columns)

    print(f"Input feature count: {feature_count}")

    initial_types = [("float_input", FloatTensorType([None, feature_count]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=target_opset,
    )

    output_path = export_dir / "random_forest.onnx"

    with open(output_path, "wb") as file:
        file.write(onnx_model.SerializeToString())

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

    metadata_path = export_dir / "random_forest_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(export_metadata, file, indent=4)

    print(f"ONNX model saved: {output_path}")
    print(f"Export metadata saved: {metadata_path}")
    print("ONNX structural validation passed.")
    print("========== STEP 17 COMPLETED ==========")

    return {
        "onnx_path": str(output_path),
        "metadata_path": str(metadata_path),
    }


def main():
    export_random_forest_onnx()


if __name__ == "__main__":
    main()
