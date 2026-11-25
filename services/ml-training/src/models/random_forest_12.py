import json
import os
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# ============================================================
# 1. CONFIGURATION
# ============================================================


def get_default_config():
    data_root = Path(os.environ.get("DATA_ROOT", "/app/data"))

    return {
        "train_path": data_root / "processed" / "train",
        "validation_path": data_root / "processed" / "validation",
        "model_path": data_root / "models" / "random_forest.joblib",
        "metrics_path": data_root / "metrics" / "random_forest.json",
        "max_train_rows": int(os.environ.get("MAX_TRAIN_ROWS", "100000")),
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "n_jobs": -1,
    }


# ============================================================
# 2. LOAD DATA
# ============================================================


def load_data(train_path, validation_path):
    print("\n========== LOADING DATA ==========")

    train_df = pd.read_parquet(train_path)
    validation_df = pd.read_parquet(validation_path)

    print("Original training shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)

    return train_df, validation_df


# ============================================================
# 3. VALIDATE DATA
# ============================================================


def validate_data(train_df, validation_df):

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    if "fault_label" not in train_df.columns:
        raise ValueError("Training dataset is missing fault_label.")

    if "fault_label" not in validation_df.columns:
        raise ValueError("Validation dataset is missing fault_label.")

    train_df = train_df.dropna(subset=["fault_label"]).copy()
    validation_df = validation_df.dropna(subset=["fault_label"]).copy()

    if train_df.empty:
        raise ValueError("Training dataset has no valid fault_label values.")

    if validation_df.empty:
        raise ValueError("Validation dataset has no valid fault_label values.")

    train_df["fault_label"] = train_df["fault_label"].astype(int)
    validation_df["fault_label"] = validation_df["fault_label"].astype(int)

    train_classes = sorted(train_df["fault_label"].unique().tolist())

    if not set(train_classes).issubset({0, 1}):
        raise ValueError(f"Unexpected training labels: {train_classes}")

    validation_classes = sorted(validation_df["fault_label"].unique().tolist())

    if not set(validation_classes).issubset({0, 1}):
        raise ValueError(f"Unexpected validation labels: {validation_classes}")

    print("\nTraining target distribution:")
    print(train_df["fault_label"].value_counts().sort_index())

    print("\nValidation target distribution:")
    print(validation_df["fault_label"].value_counts().sort_index())

    print("\nTraining classes:", train_classes)

    if len(train_classes) == 1:
        print(
            f"WARNING: Only class {train_classes[0]} exists. "
            "DummyClassifier fallback will be used."
        )

    return train_df, validation_df, train_classes


# ============================================================
# 4. PREPARE TRAINING DATA
# ============================================================


def prepare_training_data(
    train_df,
    validation_df,
    max_train_rows=100000,
    random_state=42,
):

    print("\n========== PREPARING TRAINING SAMPLE ==========")

    drop_columns = [
        "_c0",
        "timestamp",
        "failure_id",
        "fault_label",
    ]

    feature_columns = [
        column
        for column in train_df.columns
        if column not in drop_columns
        and pd.api.types.is_numeric_dtype(train_df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric model features were found.")

    missing_columns = set(feature_columns) - set(validation_df.columns)

    if missing_columns:
        raise ValueError(f"Validation data is missing features: {missing_columns}")

    print("Number of features:", len(feature_columns))

    original_training_rows = len(train_df)

    positive_df = train_df[train_df["fault_label"] == 1]
    negative_df = train_df[train_df["fault_label"] == 0]

    if len(train_df) > max_train_rows:

        positive_count = min(
            len(positive_df),
            max_train_rows // 3,
        )

        negative_count = min(
            len(negative_df),
            max_train_rows - positive_count,
        )

        sampled_parts = []

        if positive_count > 0:
            sampled_parts.append(
                positive_df.sample(
                    n=positive_count,
                    random_state=random_state,
                )
            )

        if negative_count > 0:
            sampled_parts.append(
                negative_df.sample(
                    n=negative_count,
                    random_state=random_state,
                )
            )

        train_df = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        train_df = train_df.sample(
            frac=1.0,
            random_state=random_state,
        ).reset_index(drop=True)

    X_train = train_df[feature_columns]
    y_train = train_df["fault_label"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["fault_label"]

    print("Original training rows:", original_training_rows)
    print("Training rows used:", len(X_train))

    print("\nSample target distribution:")
    print(y_train.value_counts().sort_index())

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "feature_columns": feature_columns,
        "original_training_rows": original_training_rows,
    }


# ============================================================
# 5. BUILD RANDOM FOREST MODEL
# ============================================================


def build_random_forest_model(config, train_classes):

    if len(train_classes) == 1:

        only_class = int(train_classes[0])

        model = DummyClassifier(
            strategy="constant",
            constant=only_class,
        )

        print(f"Using DummyClassifier: " f"every prediction will be {only_class}.")

    else:

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=config["random_state"],
            n_jobs=config["n_jobs"],
        )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# 6. TRAIN MODEL
# ============================================================


def train_model(pipeline, X_train, y_train):

    print("\n========== TRAINING RANDOM FOREST ==========")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    return pipeline


# ============================================================
# 7. EVALUATE MODEL
# ============================================================


def evaluate_model(
    pipeline,
    X_validation,
    y_validation,
    feature_columns,
    y_train,
    train_classes,
    original_training_rows,
):

    print("\n========== VALIDATING MODEL ==========")

    y_pred = pipeline.predict(X_validation)

    model_classes = pipeline.named_steps["model"].classes_

    if 1 in model_classes:
        fault_index = list(model_classes).index(1)

        y_probability = pipeline.predict_proba(X_validation)[:, fault_index]

    else:
        y_probability = np.zeros(
            len(X_validation),
            dtype=float,
        )

    # --------------------------------------------------------
    # Prediction diagnostics
    # --------------------------------------------------------

    print("\n========== PREDICTION DIAGNOSTICS ==========")

    print("Actual validation labels:", Counter(y_validation))
    print("Predicted labels:", Counter(y_pred))

    print("Probability minimum:", float(y_probability.min()))
    print("Probability median:", float(np.median(y_probability)))
    print("Probability maximum:", float(y_probability.max()))

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

    for threshold in thresholds:
        predicted_faults = int((y_probability >= threshold).sum())

        print(
            f"Threshold={threshold:.2f} | " f"Predicted fault cases={predicted_faults}"
        )

    # --------------------------------------------------------
    # Threshold comparison
    # --------------------------------------------------------

    print("\n========== THRESHOLD COMPARISON ==========")

    threshold_results = []

    for threshold in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:

        threshold_pred = (y_probability >= threshold).astype(int)

        precision = precision_score(
            y_validation,
            threshold_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            threshold_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            threshold_pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            threshold_pred,
            labels=[0, 1],
        ).ravel()

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"TP={tp} | FP={fp} | FN={fn}"
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
            }
        )

    # --------------------------------------------------------
    # Standard validation metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_validation,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        y_pred,
        zero_division=0,
    )

    if y_validation.nunique() == 2:

        roc_auc = roc_auc_score(
            y_validation,
            y_probability,
        )

        pr_auc = average_precision_score(
            y_validation,
            y_probability,
        )

    else:
        roc_auc = None
        pr_auc = None

    metrics = {
        "model": "random_forest",
        "status": (
            "completed_with_warning" if len(train_classes) == 1 else "completed"
        ),
        "single_class_fallback": len(train_classes) == 1,
        "training_classes": [int(c) for c in train_classes],
        "training_class_distribution": {
            str(int(label)): int(count)
            for label, count in y_train.value_counts().items()
        },
        "validation_class_distribution": {
            str(int(label)): int(count)
            for label, count in y_validation.value_counts().items()
        },
        "training_rows_original": int(original_training_rows),
        "training_rows_used": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (float(roc_auc) if roc_auc is not None else None),
        "pr_auc": (float(pr_auc) if pr_auc is not None else None),
        "threshold_comparison": threshold_results,
    }

    print("\n========== VALIDATION METRICS ==========")

    for name in [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        print(f"{name}: {metrics[name]}")

    return metrics


# ============================================================
# 8. SAVE MODEL AND METRICS
# ============================================================


def save_artifacts(pipeline, metrics, model_path, metrics_path):

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(pipeline, model_path)

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)


# ============================================================
# 9. REUSABLE TRAINING ENTRY POINT
# ============================================================


def train_random_forest(
    config=None,
    train_df=None,
    validation_df=None,
):

    default_config = get_default_config()

    if config is None:
        config = {}

    config = {
        **default_config,
        **config,
    }

    if (train_df is None) != (validation_df is None):
        raise ValueError("Both train_df and validation_df must be provided together.")

    # Load local Parquet files when DataFrames aren't supplied.
    if train_df is None:

        train_df, validation_df = load_data(
            config["train_path"],
            config["validation_path"],
        )

    else:
        print("\nUsing externally supplied training DataFrames.")

    train_df = train_df.copy()
    validation_df = validation_df.copy()

    # Validate datasets.
    train_df, validation_df, train_classes = validate_data(
        train_df,
        validation_df,
    )

    # Prepare features and labels.
    data = prepare_training_data(
        train_df=train_df,
        validation_df=validation_df,
        max_train_rows=config["max_train_rows"],
        random_state=config["random_state"],
    )

    # Build model.
    pipeline = build_random_forest_model(
        config,
        train_classes,
    )

    # Train.
    pipeline = train_model(
        pipeline,
        data["X_train"],
        data["y_train"],
    )

    # Evaluate.
    metrics = evaluate_model(
        pipeline=pipeline,
        X_validation=data["X_validation"],
        y_validation=data["y_validation"],
        feature_columns=data["feature_columns"],
        y_train=data["y_train"],
        train_classes=train_classes,
        original_training_rows=data["original_training_rows"],
    )

    # Save artifacts.
    save_artifacts(
        pipeline,
        metrics,
        config["model_path"],
        config["metrics_path"],
    )

    print("\n========== STEP 12 COMPLETED ==========")

    return {
        "model": pipeline,
        "metrics": metrics,
        "model_path": str(config["model_path"]),
        "metrics_path": str(config["metrics_path"]),
    }


# ============================================================
# 10. DIRECT SCRIPT EXECUTION
# ============================================================


def main():
    """Preserve direct script execution."""

    train_random_forest()


if __name__ == "__main__":
    main()
