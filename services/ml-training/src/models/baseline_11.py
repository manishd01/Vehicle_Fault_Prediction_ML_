import json
import os
from pathlib import Path
from sklearn.dummy import DummyClassifier

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

    # NEW: Sample only classes that actually exist.
    positive_count = len(positive_df)
    negative_count = len(negative_df)

    if positive_count == 0 or negative_count == 0:
        # Single-class data: sample available records safely.
        train_sample = train_df.sample(
            n=MAX_TRAIN_ROWS,
            random_state=42,
        )

    else:
        # Both classes exist: preserve a balanced sample.
        max_positive = min(
            positive_count,
            MAX_TRAIN_ROWS // 3,
        )

        remaining_rows = MAX_TRAIN_ROWS - max_positive

        negative_count_to_sample = min(
            negative_count,
            remaining_rows,
        )

        positive_sample = positive_df.sample(
            n=max_positive,
            random_state=42,
        )

        negative_sample = negative_df.sample(
            n=negative_count_to_sample,
            random_state=42,
        )

        train_sample = pd.concat(
            [positive_sample, negative_sample],
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

# NEW: Use Logistic Regression when both classes exist.
# Use DummyClassifier when the training data has only one class.
train_classes = sorted(y_train.dropna().unique().tolist())

if len(train_classes) == 0:
    raise ValueError("Training data contains no valid fault_label values.")

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

if len(X_train) == 0:
    raise ValueError("Training dataset is empty.")

if len(X_validation) == 0:
    raise ValueError("Validation dataset is empty.")

# Train either the normal model or the single-class fallback.
pipeline.fit(X_train, y_train)
print("Training completed.")

# ========================================================
# VALIDATION PREDICTIONS
# ========================================================

print("\n========== VALIDATION PREDICTION ==========")

y_pred = pipeline.predict(X_validation)

# DummyClassifier may have only one probability column.
# Always select the probability column for class 1 if it exists.
model_classes = pipeline.named_steps["model"].classes_

if 1 in model_classes:
    positive_index = list(model_classes).index(1)
    y_probability = pipeline.predict_proba(X_validation)[:, positive_index]
else:
    # No fault examples were available during training.
    y_probability = [0.0] * len(X_validation)

# ========================================================
# METRICS
# ========================================================

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
    roc_auc = roc_auc_score(y_validation, y_probability)
    pr_auc = average_precision_score(y_validation, y_probability)
else:
    roc_auc = None
    pr_auc = None
    print(
        "WARNING: Validation contains only one class. "
        "ROC-AUC and PR-AUC are not reported."
    )

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc if roc_auc is not None else 'N/A'}")
print(f"PR-AUC    : {pr_auc if pr_auc is not None else 'N/A'}")

# ========================================================
# SAVE MODEL
# ========================================================

print("\n========== SAVING MODEL ==========")

joblib.dump(pipeline, MODEL_PATH)
print("Model saved to:", MODEL_PATH)

# ========================================================
# SAVE METRICS
# ========================================================

metrics = {
    "model": "logistic_regression",
    "dataset": "MetroPT3 Air Compressor",
    "status": ("completed_with_warning" if len(train_classes) == 1 else "completed"),
    "single_class_fallback": len(train_classes) == 1,
    "training_classes": train_classes,
    "training_rows_original": int(len(train_df)),
    "training_rows_used": int(len(train_sample)),
    "validation_rows": int(len(validation_df)),
    "feature_count": int(len(feature_columns)),
    "features": feature_columns,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "pr_auc": float(pr_auc) if pr_auc is not None else None,
    "training_class_distribution": {
        str(int(label)): int(count) for label, count in y_train.value_counts().items()
    },
    "validation_class_distribution": {
        str(int(label)): int(count)
        for label, count in y_validation.value_counts().items()
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved to:", METRICS_PATH)
print("\n========== STEP 11 COMPLETED ==========")


# # ============================================================
# # TRAIN
# # ============================================================

# print("\n========== TRAINING BASELINE MODEL ==========")

# pipeline.fit(
#     X_train,
#     y_train,
# )

# print("Training completed.")


# # ============================================================
# # VALIDATION PREDICTIONS
# # ============================================================

# print("\n========== VALIDATION PREDICTION ==========")

# y_pred = pipeline.predict(X_validation)

# y_probability = pipeline.predict_proba(X_validation)[:, 1]


# # ============================================================
# # METRICS
# # ============================================================

# precision = precision_score(
#     y_validation,
#     y_pred,
#     zero_division=0,
# )

# recall = recall_score(
#     y_validation,
#     y_pred,
#     zero_division=0,
# )

# f1 = f1_score(
#     y_validation,
#     y_pred,
#     zero_division=0,
# )

# roc_auc = roc_auc_score(
#     y_validation,
#     y_probability,
# )

# pr_auc = average_precision_score(
#     y_validation,
#     y_probability,
# )


# print("\n========== VALIDATION METRICS ==========")

# print(f"Precision : {precision:.4f}")
# print(f"Recall    : {recall:.4f}")
# print(f"F1 Score  : {f1:.4f}")
# print(f"ROC-AUC   : {roc_auc:.4f}")
# print(f"PR-AUC    : {pr_auc:.4f}")


# # ============================================================
# # SAVE MODEL
# # ============================================================

# print("\n========== SAVING MODEL ==========")

# joblib.dump(
#     pipeline,
#     MODEL_PATH,
# )

# print("Model saved to:")
# print(MODEL_PATH)


# # ============================================================
# # SAVE METRICS
# # ============================================================

# metrics = {
#     "model": "logistic_regression",
#     "dataset": "MetroPT3 Air Compressor",
#     "training_rows_original": int(len(train_df)),
#     "training_rows_used": int(len(train_sample)),
#     "validation_rows": int(len(validation_df)),
#     "feature_count": int(len(feature_columns)),
#     "features": feature_columns,
#     "precision": float(precision),
#     "recall": float(recall),
#     "f1": float(f1),
#     "roc_auc": float(roc_auc),
#     "pr_auc": float(pr_auc),
# }

# with open(
#     METRICS_PATH,
#     "w",
#     encoding="utf-8",
# ) as file:

#     json.dump(
#         metrics,
#         file,
#         indent=4,
#     )


# print("\nMetrics saved to:")
# print(METRICS_PATH)

# print("\n========== STEP 11 COMPLETED ==========")
