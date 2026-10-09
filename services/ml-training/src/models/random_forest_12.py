import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/app/data"))

TRAIN_PATH = DATA_ROOT / "processed" / "train"
VALIDATION_PATH = DATA_ROOT / "processed" / "validation"

MODEL_DIR = DATA_ROOT / "models"
METRICS_DIR = DATA_ROOT / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METRICS_PATH = METRICS_DIR / "random_forest.json"

MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "100000"))


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

train_df = pd.read_parquet(TRAIN_PATH)
validation_df = pd.read_parquet(VALIDATION_PATH)

print("Original training shape:", train_df.shape)
print("Validation shape:", validation_df.shape)

train_df = train_df.dropna(subset=["fault_label"]).copy()
validation_df = validation_df.dropna(subset=["fault_label"]).copy()

train_df["fault_label"] = train_df["fault_label"].astype(int)
validation_df["fault_label"] = validation_df["fault_label"].astype(int)

print("\nTraining target distribution:")
print(train_df["fault_label"].value_counts().sort_index())

print("\nValidation target distribution:")
print(validation_df["fault_label"].value_counts().sort_index())

if train_df["fault_label"].nunique() != 2:
    raise ValueError("Training data must contain both normal and fault examples.")

if validation_df.empty:
    raise ValueError("Validation dataset is empty.")


# ============================================================
# 3. SELECT FEATURES
# ============================================================

DROP_COLUMNS = [
    "_c0",
    "timestamp",
    "failure_id",
    "fault_label",
]

feature_columns = [
    column
    for column in train_df.columns
    if column not in DROP_COLUMNS and pd.api.types.is_numeric_dtype(train_df[column])
]

if not feature_columns:
    raise ValueError("No numeric model features were found.")

missing_columns = set(feature_columns) - set(validation_df.columns)

if missing_columns:
    raise ValueError(f"Validation data is missing features: {missing_columns}")

print("\nNumber of features:", len(feature_columns))


# ============================================================
# 4. SAMPLE TRAINING DATA
# ============================================================

print("\n========== PREPARING TRAINING SAMPLE ==========")

positive_df = train_df[train_df["fault_label"] == 1]
negative_df = train_df[train_df["fault_label"] == 0]

if len(train_df) > MAX_TRAIN_ROWS:

    # Keep up to one-third of the sample for fault examples.
    positive_count = min(
        len(positive_df),
        MAX_TRAIN_ROWS // 3,
    )

    negative_count = min(
        len(negative_df),
        MAX_TRAIN_ROWS - positive_count,
    )

    train_df = pd.concat(
        [
            positive_df.sample(
                n=positive_count,
                random_state=42,
            ),
            negative_df.sample(
                n=negative_count,
                random_state=42,
            ),
        ],
        ignore_index=True,
    )

    train_df = train_df.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

print("Training rows used:", len(train_df))
print("Sample target distribution:")
print(train_df["fault_label"].value_counts().sort_index())


# ============================================================
# 5. PREPARE FEATURES AND LABELS
# ============================================================

X_train = train_df[feature_columns]
y_train = train_df["fault_label"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["fault_label"]


# ============================================================
# 6. BUILD RANDOM FOREST PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\n========== TRAINING RANDOM FOREST ==========")

pipeline.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 8. VALIDATION
# ============================================================

print("\n========== VALIDATING MODEL ==========")

y_pred = pipeline.predict(X_validation)

y_probability = pipeline.predict_proba(X_validation)[:, 1]

from collections import Counter

print("\n========== PREDICTION DIAGNOSTICS ==========")

print("Actual validation labels:", Counter(y_validation))
print("Predicted labels:", Counter(y_pred))

print("Probability minimum:", float(y_probability.min()))
print("Probability median:", float(__import__("numpy").median(y_probability)))
print("Probability maximum:", float(y_probability.max()))

for threshold in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    predicted_faults = int((y_probability >= threshold).sum())
    print(f"Threshold={threshold:.2f} | " f"Predicted fault cases={predicted_faults}")
# ----------------------

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
    threshold_pred = (y_probability >= threshold).astype(int)

    precision = precision_score(y_validation, threshold_pred, zero_division=0)
    recall = recall_score(y_validation, threshold_pred, zero_division=0)
    f1 = f1_score(y_validation, threshold_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_validation, threshold_pred, labels=[0, 1]
    ).ravel()

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"TP={tp} | FP={fp} | FN={fn}"
    )
# /////////////

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
    "training_rows_used": int(len(X_train)),
    "validation_rows": int(len(X_validation)),
    "feature_count": len(feature_columns),
    "features": feature_columns,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": (float(roc_auc) if roc_auc is not None else None),
    "pr_auc": (float(pr_auc) if pr_auc is not None else None),
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


# ============================================================
# 9. SAVE MODEL AND METRICS
# ============================================================

joblib.dump(pipeline, MODEL_PATH)

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)

print("\n========== STEP 12 COMPLETED ==========")
