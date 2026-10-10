import json
from pathlib import Path

import joblib
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data_utils import (
    DATA_ROOT,
    TARGET_COLUMN,
    load_split,
    get_feature_columns,
    prepare_features,
)
from src.models.deep_learning_13 import FaultPredictionNetwork

MODEL_DIR = DATA_ROOT / "models"
METRICS_DIR = DATA_ROOT / "metrics"
OUTPUT_PATH = METRICS_DIR / "evaluation_14.json"


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    print("\n========== STEP 14: MODEL EVALUATION ==========")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    validation_df = load_split("validation")
    y_true = validation_df[TARGET_COLUMN].astype(int).to_numpy()

    results = {}

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    rf_path = MODEL_DIR / "random_forest.joblib"
    rf_metrics_path = METRICS_DIR / "random_forest.json"

    if rf_path.exists():
        print("\nEvaluating Random Forest...")

        model = joblib.load(rf_path)

        saved_metrics = {}
        if rf_metrics_path.exists():
            with open(rf_metrics_path, encoding="utf-8") as file:
                saved_metrics = json.load(file)

        features = (
            saved_metrics.get("feature_columns")
            or saved_metrics.get("features")
            or getattr(model, "feature_names_in_", None)
        )

        if features is None:
            raise ValueError(
                "Cannot identify the Random Forest feature order. "
                "Check random_forest.json for its saved feature list."
            )

        X_df, _ = prepare_features(validation_df, list(features))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_df)[:, 1]
        else:
            raise TypeError("Random Forest model has no predict_proba method.")

        results["random_forest"] = calculate_metrics(y_true, probabilities)

    # --------------------------------------------------------
    # Deep Learning
    # --------------------------------------------------------
    dl_path = MODEL_DIR / "deep_learning.pt"

    if dl_path.exists():
        print("\nEvaluating PyTorch neural network...")

        checkpoint = torch.load(dl_path, map_location="cpu", weights_only=True)

        features = checkpoint["feature_columns"]
        X_df, medians = prepare_features(
            validation_df,
            features,
            # Use the medians learned from training.
            __import__("pandas").Series(checkpoint["medians"]),
        )

        mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        scale[scale == 0] = 1.0

        X = (X_df.to_numpy(dtype=np.float32) - mean) / scale
        X = np.nan_to_num(X, copy=False)

        model = FaultPredictionNetwork(input_features=checkpoint["input_features"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        probability_parts = []

        with torch.no_grad():
            for start in range(0, len(X), 2048):
                batch = torch.from_numpy(X[start : start + 2048])
                probabilities = torch.sigmoid(model(batch))
                probability_parts.append(probabilities.numpy())

        probabilities = np.concatenate(probability_parts)
        threshold = float(checkpoint.get("decision_threshold", 0.5))

        results["deep_learning"] = calculate_metrics(y_true, probabilities, threshold)

    if not results:
        raise FileNotFoundError(
            "No supported model files found under /app/data/models."
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n========== VALIDATION COMPARISON ==========")
    for name, metrics in results.items():
        print(
            f"{name}: "
            f"F1={metrics['f1']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    print(f"\nEvaluation saved: {OUTPUT_PATH}")
    print("========== STEP 14 COMPLETED ==========")


if __name__ == "__main__":
    main()
