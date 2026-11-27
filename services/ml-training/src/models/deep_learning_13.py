from pathlib import Path
from collections import Counter
import json
import os
import random

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
# 1. MODEL
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
# 2. CONFIGURATION
# ============================================================


def get_default_config():

    data_root = Path(os.environ.get("DATA_ROOT", "/app/data"))

    return {
        "data_root": data_root,
        "train_path": data_root / "processed" / "train",
        "validation_path": data_root / "processed" / "validation",
        "model_path": data_root / "models" / "deep_learning.pt",
        "metrics_path": data_root / "metrics" / "deep_learning.json",
        "target_column": "fault_label",
        "max_train_rows": 100000,
        "batch_size": 2048,
        "epochs": 10,
        "learning_rate": 0.001,
        "decision_threshold": 0.50,
        "seed": 42,
        "device": "cpu",
    }


# ============================================================
# 3. DATASET DISCOVERY AND LOADING
# ============================================================


def find_parquet_dataset(data_root, split_name):

    data_root = Path(data_root)
    candidates = []

    for path in data_root.rglob("*"):

        if not (path.is_file() or path.is_dir()):
            continue

        lower_name = path.name.lower()
        lower_path = str(path).lower()

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
            f"Could not find the {split_name} Parquet dataset " f"under {data_root}."
        )

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


def load_parquet_dataset(path, target_column):

    print(f"Loading dataset: {path}")

    dataframe = pd.read_parquet(path)

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe


def load_data(config):

    print("\n========== LOADING DATA ==========")

    train_path = Path(config["train_path"])
    validation_path = Path(config["validation_path"])

    if not train_path.exists():
        train_path = find_parquet_dataset(
            config["data_root"],
            "train",
        )

    if not validation_path.exists():
        validation_path = find_parquet_dataset(
            config["data_root"],
            "validation",
        )

    train_df = load_parquet_dataset(
        train_path,
        config["target_column"],
    )

    validation_df = load_parquet_dataset(
        validation_path,
        config["target_column"],
    )

    return train_df, validation_df


# ============================================================
# 4. VALIDATE DATA
# ============================================================


def validate_data(train_df, validation_df, target_column):

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    for name, dataframe in [
        ("Training", train_df),
        ("Validation", validation_df),
    ]:

        if target_column not in dataframe.columns:
            raise ValueError(f"{name} dataset is missing {target_column}.")

    train_df = train_df.copy()
    validation_df = validation_df.copy()

    for dataframe in [train_df, validation_df]:

        dataframe[target_column] = pd.to_numeric(
            dataframe[target_column],
            errors="coerce",
        )

        dataframe.dropna(
            subset=[target_column],
            inplace=True,
        )

        dataframe[target_column] = dataframe[target_column].astype(int)

        if not set(dataframe[target_column].unique()).issubset({0, 1}):

            raise ValueError("fault_label must contain only 0 and 1.")

    if train_df.empty or validation_df.empty:
        raise ValueError("No valid training or validation labels remain.")

    train_classes = sorted(train_df[target_column].unique().tolist())

    print("\nTraining target distribution:")
    print(train_df[target_column].value_counts().sort_index())

    print("\nValidation target distribution:")
    print(validation_df[target_column].value_counts().sort_index())

    return train_df, validation_df, train_classes


# ============================================================
# 5. PREPARE TRAINING SAMPLE
# ============================================================


def create_training_sample(
    dataframe,
    max_rows,
    random_state=42,
    target_column="fault_label",
):

    dataframe = dataframe.copy()

    if len(dataframe) <= max_rows:
        sample = dataframe.copy()

    else:

        normal_rows = dataframe[dataframe[target_column] == 0]

        fault_rows = dataframe[dataframe[target_column] == 1]

        if normal_rows.empty or fault_rows.empty:

            sample = dataframe.sample(
                n=max_rows,
                random_state=random_state,
            )

        else:

            fault_count = min(
                len(fault_rows),
                max_rows // 3,
            )

            normal_count = min(
                len(normal_rows),
                max_rows - fault_count,
            )

            sample = pd.concat(
                [
                    fault_rows.sample(
                        n=fault_count,
                        random_state=random_state,
                    ),
                    normal_rows.sample(
                        n=normal_count,
                        random_state=random_state,
                    ),
                ],
                ignore_index=True,
            )

    sample = sample.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    print(f"Training rows used: {len(sample):,}")
    print(
        "Training sample label counts:",
        Counter(sample[target_column]),
    )

    return sample


# ============================================================
# 6. PREPARE FEATURES
# ============================================================


def prepare_features(
    train_df,
    validation_df,
    target_column,
):

    excluded_columns = {
        target_column,
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
        and pd.api.types.is_numeric_dtype(validation_df[column])
    ]

    if not feature_columns:
        raise ValueError("No common numeric feature columns found.")

    print("\nNumber of features:", len(feature_columns))

    X_train_df = train_df[feature_columns].copy()
    X_validation_df = validation_df[feature_columns].copy()

    X_train_df = X_train_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_validation_df = X_validation_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    medians = X_train_df.median().fillna(0)

    X_train_df = X_train_df.fillna(medians)
    X_validation_df = X_validation_df.fillna(medians)

    return X_train_df, X_validation_df, medians, feature_columns


# ============================================================
# 7. TRAIN MODEL
# ============================================================


def train_model(
    X_train,
    y_train,
    train_classes,
    config,
):

    device = torch.device(config["device"])

    model = FaultPredictionNetwork(input_features=X_train.shape[1]).to(device)

    if len(train_classes) == 1:

        fallback_class = int(train_classes[0])

        with torch.no_grad():

            for parameter in model.parameters():
                parameter.zero_()

            model.network[-1].bias.fill_(20.0 if fallback_class == 1 else -20.0)

        print(f"Single-class fallback configured for " f"class {fallback_class}.")

        return model, True, fallback_class

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    loss_function = nn.BCEWithLogitsLoss()

    print("\n========== TRAINING ==========")

    for epoch in range(config["epochs"]):

        model.train()

        total_loss = 0.0
        rows_seen = 0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

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
            f"Epoch {epoch + 1:02d}/{config['epochs']} "
            f"| Training loss: {average_loss:.6f}"
        )

    return model, False, None


# ============================================================
# 8. EVALUATE MODEL
# ============================================================


def evaluate_model(
    model,
    X_validation,
    y_validation,
    config,
):

    print("\n========== VALIDATING MODEL ==========")

    device = torch.device(config["device"])

    model.eval()

    validation_tensor = torch.from_numpy(X_validation).to(device)

    probability_parts = []

    with torch.no_grad():

        for start in range(
            0,
            len(validation_tensor),
            config["batch_size"],
        ):

            batch = validation_tensor[start : start + config["batch_size"]]

            logits = model(batch)

            probabilities = torch.sigmoid(logits)

            probability_parts.append(probabilities.cpu().numpy())

    y_probability = np.concatenate(probability_parts).astype(np.float64)

    threshold = config["decision_threshold"]

    y_pred = (y_probability >= threshold).astype(np.int64)

    accuracy = accuracy_score(y_validation, y_pred)

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

    if len(np.unique(y_validation)) == 2:

        roc_auc = float(roc_auc_score(y_validation, y_probability))

        pr_auc = float(average_precision_score(y_validation, y_probability))

    else:
        roc_auc = None
        pr_auc = None

        print(
            "WARNING: Validation contains one class. "
            "ROC-AUC and PR-AUC are unavailable."
        )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "decision_threshold": float(threshold),
    }

    print("\n========== VALIDATION METRICS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics


# ============================================================
# 9. SAVE MODEL AND METRICS
# ============================================================


def save_artifacts(
    model,
    metrics,
    medians,
    scaler,
    feature_columns,
    train_classes,
    train_df,
    validation_df,
    single_class_fallback,
    fallback_class,
    config,
):

    model_path = Path(config["model_path"])
    metrics_path = Path(config["metrics_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_features": len(feature_columns),
        "feature_columns": feature_columns,
        "medians": medians.to_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden_layers": [128, 64],
        "dropout": 0.20,
        "decision_threshold": config["decision_threshold"],
        "target_column": config["target_column"],
        "seed": config["seed"],
        "status": ("completed_with_warning" if single_class_fallback else "completed"),
        "single_class_fallback": single_class_fallback,
        "fallback_class": fallback_class,
        "training_class_distribution": {
            str(int(label)): int(count)
            for label, count in Counter(
                train_df[config["target_column"]].tolist()
            ).items()
        },
        "validation_class_distribution": {
            str(int(label)): int(count)
            for label, count in Counter(
                validation_df[config["target_column"]].tolist()
            ).items()
        },
    }

    torch.save(checkpoint, model_path)

    metrics.update(
        {
            "model": "pytorch_feedforward_neural_network",
            "status": (
                "completed_with_warning" if single_class_fallback else "completed"
            ),
            "single_class_fallback": single_class_fallback,
            "fallback_class": fallback_class,
            "training_classes": [int(value) for value in train_classes],
            "training_rows_used": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
        }
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")


# ============================================================
# 10. REUSABLE TRAINING ENTRY POINT
# ============================================================


def train_deep_learning(
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

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if (train_df is None) != (validation_df is None):
        raise ValueError("Both train_df and validation_df must be provided together.")

    print("\n========== STEP 13: DEEP LEARNING ==========")
    print(f"Device: {config['device']}")

    if train_df is None:
        train_df, validation_df = load_data(config)
    else:
        print("\nUsing externally supplied training DataFrames.")

    train_df = train_df.copy()
    validation_df = validation_df.copy()

    target_column = config["target_column"]

    train_df, validation_df, train_classes = validate_data(
        train_df,
        validation_df,
        target_column,
    )

    single_class_fallback = len(train_classes) == 1

    fallback_class = int(train_classes[0]) if single_class_fallback else None

    train_df = create_training_sample(
        train_df,
        config["max_train_rows"],
        config["seed"],
        target_column,
    )

    X_train_df, X_validation_df, medians, feature_columns = prepare_features(
        train_df,
        validation_df,
        target_column,
    )

    y_train = train_df[target_column].to_numpy(dtype=np.int64)
    y_validation = validation_df[target_column].to_numpy(dtype=np.int64)

    print("\n========== SCALING FEATURES ==========")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_df).astype(np.float32)

    X_validation = scaler.transform(X_validation_df).astype(np.float32)

    X_train = np.nan_to_num(X_train, copy=False)
    X_validation = np.nan_to_num(X_validation, copy=False)

    model, single_class_fallback, fallback_class = train_model(
        X_train,
        y_train,
        train_classes,
        config,
    )

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        config,
    )

    save_artifacts(
        model=model,
        metrics=metrics,
        medians=medians,
        scaler=scaler,
        feature_columns=feature_columns,
        train_classes=train_classes,
        train_df=train_df,
        validation_df=validation_df,
        single_class_fallback=single_class_fallback,
        fallback_class=fallback_class,
        config=config,
    )

    print("\n========== STEP 13 COMPLETED ==========")

    return {
        "model": model,
        "metrics": metrics,
        "model_path": str(config["model_path"]),
        "metrics_path": str(config["metrics_path"]),
    }


# ============================================================
# 11. DIRECT SCRIPT EXECUTION
# ============================================================


def main():
    """Preserve direct script execution."""

    train_deep_learning()


if __name__ == "__main__":
    main()
