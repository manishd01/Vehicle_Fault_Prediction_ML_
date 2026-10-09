import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
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
# CONFIGURATION
# ============================================================

DATA_ROOT = os.environ.get("DATA_ROOT", "/app/data")

TRAIN_PATH = Path(DATA_ROOT) / "processed" / "train"
VALIDATION_PATH = Path(DATA_ROOT) / "processed" / "validation"

MODEL_DIR = Path(DATA_ROOT) / "models"
METRICS_DIR = Path(DATA_ROOT) / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "baseline_logistic_regression.joblib"
METRICS_PATH = METRICS_DIR / "baseline_logistic_regression.json"

# Maximum number of training rows used by the baseline.
# Validation data is NOT sampled.
MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "300000"))


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== LOADING TRAIN DATA ==========")
print("Train path:", TRAIN_PATH)

train_df = pd.read_parquet(TRAIN_PATH)

print("Original train shape:", train_df.shape)

print("\n========== LOADING VALIDATION DATA ==========")
print("Validation path:", VALIDATION_PATH)

validation_df = pd.read_parquet(VALIDATION_PATH)

print("Validation shape:", validation_df.shape)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n========== ORIGINAL TRAIN TARGET ==========")

print(train_df["fault_label"].value_counts().sort_index())


print("\n========== VALIDATION TARGET ==========")

print(validation_df["fault_label"].value_counts().sort_index())


# ============================================================
# REMOVE INVALID / IDENTIFIER COLUMNS
# ============================================================

DROP_COLUMNS = [
    "_c0",
    "timestamp",
    "failure_id",
    "fault_label",
]

feature_columns = [column for column in train_df.columns if column not in DROP_COLUMNS]

print("\n========== FEATURES ==========")
print("Number of features:", len(feature_columns))
print("Features:")

for feature in feature_columns:
    print(" -", feature)


# ============================================================
# PREPARE TARGET
# ============================================================

train_df["fault_label"] = train_df["fault_label"].astype(int)
validation_df["fault_label"] = validation_df["fault_label"].astype(int)


# ============================================================
# BALANCED TRAINING SAMPLE
# ============================================================
#
# The dataset is highly imbalanced.
#
# Instead of taking the first 300,000 chronological rows,
# we keep fault examples and sample normal examples.
#
# Validation remains untouched.
# ============================================================

print("\n========== PREPARING TRAINING SAMPLE ==========")

positive_df = train_df[train_df["fault_label"] == 1]
negative_df = train_df[train_df["fault_label"] == 0]

print("Positive rows:", len(positive_df))
print("Negative rows:", len(negative_df))

if len(train_df) > MAX_TRAIN_ROWS:

    # Keep all positive examples if possible.
    max_positive = min(
        len(positive_df),
        MAX_TRAIN_ROWS // 3,
    )

    positive_sample = positive_df.sample(
        n=max_positive,
        random_state=42,
    )

    remaining_rows = MAX_TRAIN_ROWS - len(positive_sample)

    negative_sample = negative_df.sample(
        n=remaining_rows,
        random_state=42,
    )

    train_sample = pd.concat(
        [
            positive_sample,
            negative_sample,
        ],
        ignore_index=True,
    )

else:

    train_sample = train_df.copy()


# Shuffle the resulting training sample.

train_sample = train_sample.sample(
    frac=1.0,
    random_state=42,
).reset_index(drop=True)


print("Final training sample shape:", train_sample.shape)

print("\nTraining sample target distribution:")

print(train_sample["fault_label"].value_counts().sort_index())


# ============================================================
# CREATE X / y
# ============================================================

X_train = train_sample[feature_columns]
y_train = train_sample["fault_label"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["fault_label"]


print("\nX_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
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


# ============================================================
# BASELINE MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n========== TRAINING BASELINE MODEL ==========")

pipeline.fit(
    X_train,
    y_train,
)

print("Training completed.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n========== VALIDATION PREDICTION ==========")

y_pred = pipeline.predict(X_validation)

y_probability = pipeline.predict_proba(X_validation)[:, 1]


# ============================================================
# METRICS
# ============================================================

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

roc_auc = roc_auc_score(
    y_validation,
    y_probability,
)

pr_auc = average_precision_score(
    y_validation,
    y_probability,
)


print("\n========== VALIDATION METRICS ==========")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

print("\n========== SAVING MODEL ==========")

joblib.dump(
    pipeline,
    MODEL_PATH,
)

print("Model saved to:")
print(MODEL_PATH)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "logistic_regression",
    "dataset": "MetroPT3 Air Compressor",
    "training_rows_original": int(len(train_df)),
    "training_rows_used": int(len(train_sample)),
    "validation_rows": int(len(validation_df)),
    "feature_count": int(len(feature_columns)),
    "features": feature_columns,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


print("\nMetrics saved to:")
print(METRICS_PATH)

print("\n========== STEP 11 COMPLETED ==========")
