import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

MODEL_PATH = MODEL_DIR / "pytorch_mlp.pt"
PREPROCESSOR_PATH = MODEL_DIR / "pytorch_preprocessing.joblib"
METRICS_PATH = METRICS_DIR / "pytorch_mlp.json"

MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "100000"))

EPOCHS = 10
BATCH_SIZE = 2048

torch.manual_seed(42)
np.random.seed(42)


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
# 3. SELECT NUMERIC FEATURES
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
    raise ValueError("No numeric features were found.")

missing_columns = set(feature_columns) - set(validation_df.columns)

if missing_columns:
    raise ValueError(f"Validation features missing: {missing_columns}")

print("\nNumber of features:", len(feature_columns))


# ============================================================
# 4. SAMPLE TRAINING DATA
# ============================================================

print("\n========== PREPARING TRAINING SAMPLE ==========")

if len(train_df) > MAX_TRAIN_ROWS:

    positive_df = train_df[train_df["fault_label"] == 1]
    negative_df = train_df[train_df["fault_label"] == 0]

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
print("Training target distribution:")
print(train_df["fault_label"].value_counts().sort_index())


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

X_train_df = train_df[feature_columns]
y_train = train_df["fault_label"].to_numpy(dtype=np.float32)

X_val_df = validation_df[feature_columns]
y_val = validation_df["fault_label"].to_numpy(dtype=np.int64)

# Fit preprocessing ONLY on training data.
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train_df)
X_val = imputer.transform(X_val_df)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

# Prevent infinite or invalid values from reaching the network.
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

train_x = torch.from_numpy(X_train)
train_y = torch.from_numpy(y_train).reshape(-1, 1)
val_x = torch.from_numpy(X_val)

dataset = TensorDataset(train_x, train_y)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ============================================================
# 6. DEFINE NEURAL NETWORK
# ============================================================


class FaultMLP(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


model = FaultMLP(len(feature_columns))

positive_count = max(
    float((y_train == 1).sum()),
    1.0,
)

negative_count = max(
    float((y_train == 0).sum()),
    1.0,
)

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32,
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n========== TRAINING PYTORCH MODEL ==========")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Positive class weight:", float(pos_weight.item()))

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_x)

    average_loss = total_loss / len(dataset)

    print(f"Epoch {epoch + 1}/{EPOCHS} " f"- loss: {average_loss:.4f}")

print("Training completed.")


# ============================================================
# 8. VALIDATION
# ============================================================

print("\n========== VALIDATING MODEL ==========")

model.eval()

with torch.no_grad():
    logits = model(val_x)
    probabilities = torch.sigmoid(logits).numpy().ravel()

# Default threshold. We will inspect its behaviour first.
threshold = 0.5
predictions = (probabilities >= threshold).astype(int)

precision = precision_score(
    y_val,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0,
)

if len(np.unique(y_val)) == 2:

    roc_auc = roc_auc_score(y_val, probabilities)

    pr_auc = average_precision_score(y_val, probabilities)

else:
    roc_auc = None
    pr_auc = None

print("\n========== VALIDATION METRICS ==========")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc if roc_auc is not None else 'N/A'}")
print(f"PR-AUC:    {pr_auc if pr_auc is not None else 'N/A'}")

print("\n========== PREDICTION DIAGNOSTICS ==========")

print("Classification threshold:", threshold)
print("Actual fault rows:", int((y_val == 1).sum()))
print("Predicted fault rows:", int(predictions.sum()))
print("Minimum probability:", float(probabilities.min()))
print("Maximum probability:", float(probabilities.max()))
print("Mean probability:", float(probabilities.mean()))


# ============================================================
# 9. SAVE MODEL AND PREPROCESSING
# ============================================================

torch.save(
    {
        "state_dict": model.state_dict(),
        "input_size": len(feature_columns),
        "features": feature_columns,
    },
    MODEL_PATH,
)

joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "features": feature_columns,
    },
    PREPROCESSOR_PATH,
)

metrics = {
    "model": "pytorch_mlp",
    "training_rows_used": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "feature_count": len(feature_columns),
    "threshold": threshold,
    "actual_fault_rows": int((y_val == 1).sum()),
    "predicted_fault_rows": int(predictions.sum()),
    "min_probability": float(probabilities.min()),
    "max_probability": float(probabilities.max()),
    "mean_probability": float(probabilities.mean()),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": (float(roc_auc) if roc_auc is not None else None),
    "pr_auc": (float(pr_auc) if pr_auc is not None else None),
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved:", MODEL_PATH)
print("Preprocessing saved:", PREPROCESSOR_PATH)
print("Metrics saved:", METRICS_PATH)

print("\n========== STEP 13 COMPLETED ==========")
