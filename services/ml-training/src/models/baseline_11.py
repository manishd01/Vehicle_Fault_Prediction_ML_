import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. DEFAULT CONFIGURATION
# ============================================================


def get_default_config():
    """Build default configuration for local execution."""

    data_root = Path(os.environ.get("DATA_ROOT", "/app/data"))

    return {
        "train_path": data_root / "processed" / "train",
        "validation_path": data_root / "processed" / "validation",
        "model_dir": data_root / "models",
        "metrics_dir": data_root / "metrics",
        "max_train_rows": int(os.environ.get("MAX_TRAIN_ROWS", "300000")),
        "random_state": 42,
    }


# ============================================================
# 2. DATA LOADING
# ============================================================


def load_data(train_path, validation_path):
    """Load local training and validation Parquet datasets."""

    print("\n========== LOADING TRAIN DATA ==========")
    print("Train path:", train_path)

    train_df = pd.read_parquet(train_path)

    print("Original train shape:", train_df.shape)

    print("\n========== LOADING VALIDATION DATA ==========")
    print("Validation path:", validation_path)

    validation_df = pd.read_parquet(validation_path)

    print("Validation shape:", validation_df.shape)

    return train_df, validation_df


# ============================================================
# 3. DATA VALIDATION
# ============================================================


def validate_data(train_df, validation_df):
    """Validate required columns and dataset contents."""

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    if "fault_label" not in train_df.columns:
        raise ValueError("Training data is missing fault_label.")

    if "fault_label" not in validation_df.columns:
        raise ValueError("Validation data is missing fault_label.")

    train_df = train_df.dropna(subset=["fault_label"]).copy()
    validation_df = validation_df.dropna(subset=["fault_label"]).copy()

    if train_df.empty:
        raise ValueError("Training dataset has no valid fault labels.")

    if validation_df.empty:
        raise ValueError("Validation dataset has no valid fault labels.")

    train_df["fault_label"] = train_df["fault_label"].astype(int)
    validation_df["fault_label"] = validation_df["fault_label"].astype(int)

    for name, df in [
        ("Training", train_df),
        ("Validation", validation_df),
    ]:
        labels = set(df["fault_label"].unique())

        if not labels.issubset({0, 1}):
            raise ValueError(f"{name} dataset contains unexpected labels: {labels}")

    return train_df, validation_df


# ============================================================
# 4. FEATURE SELECTION AND TRAINING SAMPLE
# ============================================================


def prepare_training_data(
    train_df,
    validation_df,
    max_train_rows=300000,
    random_state=42,
):
    """Select features, sample training records and prepare X/y."""

    if max_train_rows <= 0:
        raise ValueError("max_train_rows must be greater than zero.")

    print("\n========== ORIGINAL TRAIN TARGET ==========")
    print(train_df["fault_label"].value_counts().sort_index())

    print("\n========== VALIDATION TARGET ==========")
    print(validation_df["fault_label"].value_counts().sort_index())

    drop_columns = [
        "_c0",
        "timestamp",
        "failure_id",
        "fault_label",
    ]

    feature_columns = [
        column for column in train_df.columns if column not in drop_columns
    ]

    if not feature_columns:
        raise ValueError("No model features were found.")

    missing_columns = set(feature_columns) - set(validation_df.columns)

    if missing_columns:
        raise ValueError(f"Validation data is missing features: {missing_columns}")

    print("\n========== FEATURES ==========")
    print("Number of features:", len(feature_columns))

    for feature in feature_columns:
        print(" -", feature)

    print("\n========== PREPARING TRAINING SAMPLE ==========")

    positive_df = train_df[train_df["fault_label"] == 1]
    negative_df = train_df[train_df["fault_label"] == 0]

    print("Positive rows:", len(positive_df))
    print("Negative rows:", len(negative_df))

    if len(train_df) > max_train_rows:

        if positive_df.empty or negative_df.empty:

            train_sample = train_df.sample(
                n=max_train_rows,
                random_state=random_state,
            )

        else:

            max_positive = min(
                len(positive_df),
                max_train_rows // 3,
            )

            remaining_rows = max_train_rows - max_positive

            negative_count = min(
                len(negative_df),
                remaining_rows,
            )

            positive_sample = positive_df.sample(
                n=max_positive,
                random_state=random_state,
            )

            negative_sample = negative_df.sample(
                n=negative_count,
                random_state=random_state,
            )

            train_sample = pd.concat(
                [positive_sample, negative_sample],
                ignore_index=True,
            )

    else:
        train_sample = train_df.copy()

    train_sample = train_sample.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    print("Final training sample shape:", train_sample.shape)

    print("\nTraining sample target distribution:")
    print(train_sample["fault_label"].value_counts().sort_index())

    X_train = train_sample[feature_columns]
    y_train = train_sample["fault_label"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["fault_label"]

    print("\nX_train shape:", X_train.shape)
    print("X_validation shape:", X_validation.shape)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "feature_columns": feature_columns,
        "train_sample": train_sample,
        "train_classes": sorted(y_train.unique().tolist()),
    }


# ============================================================
# 5. BUILD BASELINE MODEL
# ============================================================


def build_baseline_model(feature_columns, train_classes):
    """Build Logistic Regression or single-class fallback."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            )
        ],
        remainder="drop",
    )

    if len(train_classes) == 0:
        raise ValueError("Training data contains no valid classes.")

    if not set(train_classes).issubset({0, 1}):
        raise ValueError(f"Unexpected fault_label values: {train_classes}")

    if len(train_classes) == 1:

        only_class = int(train_classes[0])

        print(
            f"WARNING: Training contains only class {only_class}. "
            "Using a constant-prediction fallback model."
        )

        model = DummyClassifier(
            strategy="constant",
            constant=only_class,
        )

    else:

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# 6. MODEL TRAINING
# ============================================================


def train_model(pipeline, X_train, y_train):
    """Fit the baseline model."""

    print("\n========== TRAINING BASELINE MODEL ==========")

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    return pipeline


# ============================================================
# 7. MODEL EVALUATION
# ============================================================


def evaluate_model(pipeline, X_validation, y_validation):
    """Generate predictions and calculate validation metrics."""

    print("\n========== VALIDATION PREDICTION ==========")

    if len(X_validation) == 0:
        raise ValueError("Validation dataset is empty.")

    y_pred = pipeline.predict(X_validation)

    model_classes = pipeline.named_steps["model"].classes_

    if 1 in model_classes:

        positive_index = list(model_classes).index(1)

        y_probability = pipeline.predict_proba(X_validation)[:, positive_index]

    else:

        y_probability = [0.0] * len(X_validation)

    print("\n========== VALIDATION METRICS ==========")

    precision = precision_score(
        y_validation,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        y_pred,
        pos_label=1,
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

        print(
            "WARNING: Validation contains only one class. "
            "ROC-AUC and PR-AUC are not reported."
        )

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (float(roc_auc) if roc_auc is not None else None),
        "pr_auc": (float(pr_auc) if pr_auc is not None else None),
    }

    for name, value in metrics.items():
        print(f"{name}: {value if value is not None else 'N/A'}")

    return metrics


# ============================================================
# 8. SAVE MODEL AND METRICS
# ============================================================


def save_artifacts(
    pipeline,
    metrics,
    model_path,
    metrics_path,
):
    """Save trained model and evaluation metrics."""

    print("\n========== SAVING MODEL ==========")

    joblib.dump(pipeline, model_path)

    print("Model saved to:", model_path)

    print("\n========== SAVING METRICS ==========")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved to:", metrics_path)


# ============================================================
# 9. MAIN TRAINING ENTRY POINT
# ============================================================


def train_baseline(
    config=None,
    train_df=None,
    validation_df=None,
):
    """
    Execute baseline training.

    Local Docker:
        Reads Parquet files when DataFrames are not supplied.

    Databricks:
        Receives training and validation DataFrames directly.
    """

    print("\n========== STEP 11: BASELINE MODEL ==========")

    if config is None:
        config = get_default_config()

    config = {
        **get_default_config(),
        **config,
    }

    # Either both DataFrames must be supplied or neither.
    if (train_df is None) != (validation_df is None):
        raise ValueError("Provide both train_df and validation_df, or neither.")

    # Local execution loads Parquet.
    if train_df is None:

        train_df, validation_df = load_data(
            config["train_path"],
            config["validation_path"],
        )

    # Avoid modifying DataFrames supplied by the caller.
    train_df = train_df.copy()
    validation_df = validation_df.copy()

    train_df, validation_df = validate_data(
        train_df,
        validation_df,
    )

    prepared = prepare_training_data(
        train_df=train_df,
        validation_df=validation_df,
        max_train_rows=config["max_train_rows"],
        random_state=config["random_state"],
    )

    pipeline = build_baseline_model(
        feature_columns=prepared["feature_columns"],
        train_classes=prepared["train_classes"],
    )

    pipeline = train_model(
        pipeline=pipeline,
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
    )

    metrics = evaluate_model(
        pipeline=pipeline,
        X_validation=prepared["X_validation"],
        y_validation=prepared["y_validation"],
    )

    metrics.update(
        {
            "model": "logistic_regression",
            "dataset": "MetroPT3 Air Compressor",
            "status": (
                "completed_with_warning"
                if len(prepared["train_classes"]) == 1
                else "completed"
            ),
            "single_class_fallback": (len(prepared["train_classes"]) == 1),
            "training_classes": prepared["train_classes"],
            "training_rows_original": int(len(train_df)),
            "training_rows_used": int(len(prepared["train_sample"])),
            "validation_rows": int(len(validation_df)),
            "feature_count": len(prepared["feature_columns"]),
            "features": prepared["feature_columns"],
            "training_class_distribution": {
                str(int(label)): int(count)
                for label, count in prepared["y_train"].value_counts().items()
            },
            "validation_class_distribution": {
                str(int(label)): int(count)
                for label, count in prepared["y_validation"].value_counts().items()
            },
        }
    )

    model_dir = Path(config["model_dir"])
    metrics_dir = Path(config["metrics_dir"])

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline_logistic_regression.joblib"
    metrics_path = metrics_dir / "baseline_logistic_regression.json"

    save_artifacts(
        pipeline=pipeline,
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    print("\n========== STEP 11 COMPLETED ==========")

    return {
        "model": pipeline,
        "metrics": metrics,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


# ============================================================
# 10. STANDALONE EXECUTION
# ============================================================


def main():
    """Preserve direct script execution."""

    train_baseline()


if __name__ == "__main__":
    main()
