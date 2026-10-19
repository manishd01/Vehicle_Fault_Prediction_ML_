# services/ml-training/src/models/deep_learning_13.py

from pathlib import Path
from collections import Counter
import json
import random
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
TARGET_COLUMN = "fault_label"

MAX_TRAIN_ROWS = 100_000
BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 0.001
DECISION_THRESHOLD = 0.50

DATA_ROOT = Path("/app/data")
MODEL_DIR = DATA_ROOT / "models"
METRICS_DIR = DATA_ROOT / "metrics"

MODEL_PATH = MODEL_DIR / "deep_learning.pt"
METRICS_PATH = METRICS_DIR / "deep_learning.json"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================


class FaultPredictionNetwork(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(1)


# ============================================================
# DATASET DISCOVERY
# ============================================================


def find_parquet_dataset(split_name: str) -> Path:
    """
    Find a training or validation Parquet file/dataset.

    Searches under /app/data. A dataset may be either a .parquet
    file or a directory containing Parquet files.
    """
    candidates = []

    for path in DATA_ROOT.rglob("*"):
        if not (path.is_file() or path.is_dir()):
            continue

        lower_name = path.name.lower()
        lower_path = str(path).lower()

        # Avoid accidentally reading saved models or MLflow artifacts.
        if any(
            part in lower_path
            for part in (
                "/models/",
                "/mlruns/",
                "/metrics/",
                "/.git/",
            )
        ):
            continue

        valid_parquet = path.suffix.lower() == ".parquet" or (
            path.is_dir() and any(path.glob("*.parquet"))
        )

        if not valid_parquet:
            continue

        if split_name == "train":
            matches = (
                lower_name
                in {
                    "train",
                    "train.parquet",
                    "training",
                    "training.parquet",
                }
                or "train" in lower_name
            )
        else:
            matches = (
                lower_name
                in {
                    "validation",
                    "validation.parquet",
                    "val",
                    "val.parquet",
                }
                or "validation" in lower_name
                or lower_name.startswith("val_")
            )

        if matches:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find the {split_name} Parquet dataset under "
            f"{DATA_ROOT}. Check the output paths from "
            "10_train_val_test_split.py."
        )

    # Prefer an explicitly named split directory/file over a nested
    # individual Parquet part file.
    candidates.sort(
        key=lambda path: (
            (
                0
                if path.name.lower()
                in {
                    split_name,
                    f"{split_name}.parquet",
                    "validation" if split_name == "validation" else "train",
                }
                else 1
            ),
            len(path.parts),
        )
    )

    selected = candidates[0]
    print(f"{split_name.title()} dataset found: {selected}")
    return selected


def load_parquet_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading dataset: {path}")
    dataframe = pd.read_parquet(path)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in {path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe


# ============================================================
# PREPARE TRAINING SAMPLE
# ============================================================


def create_training_sample(
    dataframe: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """Sample safely when both classes or only one class exists."""

    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).copy()
    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN])

    if dataframe.empty:
        raise ValueError("Training dataset contains no valid labels.")

    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    if not set(dataframe[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("fault_label must contain only 0 and 1.")

    if len(dataframe) <= max_rows:
        sample = dataframe.copy()
    else:
        normal_rows = dataframe[dataframe[TARGET_COLUMN] == 0]
        fault_rows = dataframe[dataframe[TARGET_COLUMN] == 1]

        if normal_rows.empty or fault_rows.empty:
            sample = dataframe.sample(
                n=max_rows,
                random_state=SEED,
            )
        else:
            fault_count = min(len(fault_rows), max_rows // 3)
            normal_count = min(
                len(normal_rows),
                max_rows - fault_count,
            )

            sample = pd.concat(
                [
                    fault_rows.sample(n=fault_count, random_state=SEED),
                    normal_rows.sample(n=normal_count, random_state=SEED),
                ],
                ignore_index=True,
            )

    sample = sample.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    print(f"Training rows used: {len(sample):,}")
    print("Training sample label counts:", Counter(sample[TARGET_COLUMN]))
    return sample


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================


def main():
    print("\n========== STEP 13: DEEP LEARNING ==========")
    print(f"Device: {DEVICE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load existing chronological split datasets.
    print("\n========== LOADING DATA ==========")

    train_path = find_parquet_dataset("train")
    validation_path = find_parquet_dataset("validation")

    train_df = load_parquet_dataset(train_path)
    validation_df = load_parquet_dataset(validation_path)

    validation_df = validation_df.dropna(subset=[TARGET_COLUMN]).copy()
    validation_df[TARGET_COLUMN] = pd.to_numeric(
        validation_df[TARGET_COLUMN],
        errors="coerce",
    )
    validation_df = validation_df.dropna(subset=[TARGET_COLUMN])

    if validation_df.empty:
        raise ValueError("Validation dataset has no valid target labels.")

    validation_df[TARGET_COLUMN] = validation_df[TARGET_COLUMN].astype(int)

    if not set(validation_df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Validation fault_label must contain only 0 and 1.")

    print(f"Original training shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    print("\nTraining target distribution:")
    print(train_df[TARGET_COLUMN].value_counts(dropna=False))

    print("\nValidation target distribution:")
    print(validation_df[TARGET_COLUMN].value_counts(dropna=False))

    # # 2. Create a manageable training sample.
    # print("\n========== PREPARING TRAINING SAMPLE ==========")

    # train_df = create_training_sample(
    #     train_df,
    #     MAX_TRAIN_ROWS,
    # )
    # ;;;better way:\
    # 2. Check class distribution before creating a training sample.
    print("\n========== PREPARING TRAINING SAMPLE ==========")

    train_df = train_df.dropna(subset=[TARGET_COLUMN]).copy()
    train_df[TARGET_COLUMN] = pd.to_numeric(
        train_df[TARGET_COLUMN],
        errors="coerce",
    )
    train_df = train_df.dropna(subset=[TARGET_COLUMN])
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)

    normal_count = int((train_df[TARGET_COLUMN] == 0).sum())
    fault_count = int((train_df[TARGET_COLUMN] == 1).sum())

    if train_df.empty:
        raise ValueError("Training dataset is empty after label cleaning.")

    train_classes = sorted(train_df[TARGET_COLUMN].unique().tolist())

    if not set(train_classes).issubset({0, 1}):
        raise ValueError(f"Unexpected training labels: {train_classes}")

    single_class_fallback = len(train_classes) == 1
    fallback_class = int(train_classes[0]) if single_class_fallback else None

    if single_class_fallback:
        print(
            f"WARNING: Training contains only class {fallback_class}. "
            "A constant-prediction fallback will be used."
        )
    else:
        print("Both normal and fault classes are present.")

    train_df = create_training_sample(train_df, MAX_TRAIN_ROWS)

    train_df = create_training_sample(
        train_df,
        MAX_TRAIN_ROWS,
    )

    # Ensure labels are binary integers.
    validation_df = validation_df.dropna(subset=[TARGET_COLUMN]).copy()

    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)
    validation_df[TARGET_COLUMN] = validation_df[TARGET_COLUMN].astype(int)

    # 3. Use numeric engineered features only.
    excluded_columns = {
        TARGET_COLUMN,
        "timestamp",
        "datetime",
        "date",
    }

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(train_df[column])
        and column in validation_df.columns
    ]

    if not feature_columns:
        raise ValueError(
            "No common numeric feature columns found between "
            "training and validation datasets."
        )

    print(f"\nNumber of features: {len(feature_columns)}")

    X_train_df = train_df[feature_columns].copy()
    X_validation_df = validation_df[feature_columns].copy()

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_validation = validation_df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    # Replace infinite values and impute missing values using
    # training-set medians only.
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
    X_validation_df = X_validation_df.replace([np.inf, -np.inf], np.nan)

    medians = X_train_df.median().fillna(0)

    X_train_df = X_train_df.fillna(medians)
    X_validation_df = X_validation_df.fillna(medians)

    # 4. Scale numeric features.
    print("\n========== SCALING FEATURES ==========")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_df).astype(np.float32)

    X_validation = scaler.transform(X_validation_df).astype(np.float32)

    # Protect against any remaining non-finite values.
    X_train = np.nan_to_num(X_train, copy=False)
    X_validation = np.nan_to_num(X_validation, copy=False)

    # 5. Convert to PyTorch tensors.
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_tensor = torch.from_numpy(X_validation).to(DEVICE)

    # 6. Initialize neural network.
    print("\n========== BUILDING NEURAL NETWORK ==========")

    model = FaultPredictionNetwork(input_features=len(feature_columns)).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.BCEWithLogitsLoss()

    # 7. Train.
    print("\n========== TRAINING ==========")

    # 7. Train normally or configure a constant-prediction fallback.
    print("\n========== TRAINING ==========")

    if single_class_fallback:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

            model.network[-1].bias.fill_(20.0 if fallback_class == 1 else -20.0)

        print(
            f"Fallback configured for class {fallback_class}. "
            "No neural-network training was performed."
        )
    else:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            rows_seen = 0

            for features, labels in train_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits = model(features)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                batch_rows = len(labels)
                total_loss += loss.item() * batch_rows
                rows_seen += batch_rows

            average_loss = total_loss / max(rows_seen, 1)
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} "
                f"| Training loss: {average_loss:.6f}"
            )

    # 8. Predict on the original, imbalanced validation data.
    print("\n========== VALIDATING MODEL ==========")

    model.eval()
    probability_batches = []

    with torch.no_grad():
        for start in range(0, len(validation_tensor), BATCH_SIZE):
            batch = validation_tensor[start : start + BATCH_SIZE]

            logits = model(batch)
            probabilities = torch.sigmoid(logits)

            probability_batches.append(probabilities.cpu().numpy())

    y_probability = np.concatenate(probability_batches).astype(np.float64)

    y_pred = (y_probability >= DECISION_THRESHOLD).astype(np.int64)

    # 9. Evaluate.
    precision = precision_score(y_validation, y_pred, zero_division=0)
    recall = recall_score(y_validation, y_pred, zero_division=0)
    f1 = f1_score(y_validation, y_pred, zero_division=0)
    accuracy = accuracy_score(y_validation, y_pred)

    if len(np.unique(y_validation)) == 2:
        roc_auc = float(roc_auc_score(y_validation, y_probability))
        pr_auc = float(average_precision_score(y_validation, y_probability))
    else:
        roc_auc = None
        pr_auc = None
        print(
            "WARNING: Validation contains only one class. "
            "ROC-AUC and PR-AUC are not reported."
        )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        y_pred,
        labels=[0, 1],
    ).ravel()

    print("\n========== VALIDATION METRICS ==========")
    print(f"Decision threshold: {DECISION_THRESHOLD}")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"ROC-AUC:   {roc_auc}")
    print(f"PR-AUC:    {pr_auc}")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    # 10. Save model, preprocessing details and metrics.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_features": len(feature_columns),
        "feature_columns": feature_columns,
        "medians": medians.to_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden_layers": [128, 64],
        "dropout": 0.20,
        "decision_threshold": DECISION_THRESHOLD,
        "target_column": TARGET_COLUMN,
        "seed": SEED,
        "status": ("completed_with_warning" if single_class_fallback else "completed"),
        "single_class_fallback": single_class_fallback,
        "fallback_class": fallback_class,
        "training_class_distribution": {
            str(int(label)): int(count)
            for label, count in Counter(y_train.tolist()).items()
        },
        "validation_class_distribution": {
            str(int(label)): int(count)
            for label, count in Counter(y_validation.tolist()).items()
        },
    }

    torch.save(checkpoint, MODEL_PATH)

    metrics = {
        "model": "pytorch_feedforward_neural_network",
        "training_rows_used": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "feature_count": int(len(feature_columns)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "decision_threshold": DECISION_THRESHOLD,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (float(roc_auc) if roc_auc is not None else None),
        "pr_auc": (float(pr_auc) if pr_auc is not None else None),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "feature_columns": feature_columns,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print("\n========== STEP 13 COMPLETED ==========")


if __name__ == "__main__":
    main()
