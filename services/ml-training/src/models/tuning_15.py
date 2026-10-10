import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_utils import (
    DATA_ROOT,
    TARGET_COLUMN,
    load_split,
    get_feature_columns,
)

MODEL_DIR = DATA_ROOT / "models"
METRICS_DIR = DATA_ROOT / "metrics"

MAX_TRAIN_ROWS = 100_000
SEED = 42


def balanced_sample(dataframe):
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).copy()
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    faults = dataframe[dataframe[TARGET_COLUMN] == 1]
    normal = dataframe[dataframe[TARGET_COLUMN] == 0]

    if len(faults) == 0:
        raise ValueError("No fault examples found in training data.")

    if len(dataframe) <= MAX_TRAIN_ROWS:
        return dataframe.sample(frac=1, random_state=SEED)

    if len(faults) >= MAX_TRAIN_ROWS:
        return faults.sample(n=MAX_TRAIN_ROWS, random_state=SEED)

    normal_count = MAX_TRAIN_ROWS - len(faults)
    sampled_normal = normal.sample(
        n=min(normal_count, len(normal)),
        random_state=SEED,
    )

    import pandas as pd

    return pd.concat([faults, sampled_normal]).sample(frac=1, random_state=SEED)


def main():
    print("\n========== STEP 15: HYPERPARAMETER TUNING ==========")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    validation_df = load_split("validation")

    train_df = balanced_sample(train_df)

    feature_columns = get_feature_columns(train_df, validation_df)

    X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_validation = validation_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    candidates = [
        {"n_estimators": 80, "max_depth": 16, "min_samples_leaf": 1},
        {"n_estimators": 80, "max_depth": 24, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2},
    ]

    best_model = None
    best_metrics = None
    best_score = -1.0

    for index, parameters in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}: {parameters}")

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        **parameters,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_validation)[:, 1]
        pr_auc = average_precision_score(y_validation, probabilities)
        roc_auc = roc_auc_score(y_validation, probabilities)

        print(f"Validation PR-AUC: {pr_auc:.6f}")
        print(f"Validation ROC-AUC: {roc_auc:.6f}")

        if pr_auc > best_score:
            best_score = pr_auc
            best_model = model
            best_metrics = {
                "model": "tuned_random_forest",
                "parameters": parameters,
                "feature_columns": feature_columns,
                "training_rows_used": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "selection_metric": "validation_pr_auc",
            }

    model_path = MODEL_DIR / "random_forest_tuned.joblib"
    metrics_path = METRICS_DIR / "random_forest_tuned.json"

    joblib.dump(best_model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=4)

    print("\n========== BEST MODEL ==========")
    print(json.dumps(best_metrics, indent=4))
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("========== STEP 15 COMPLETED ==========")


if __name__ == "__main__":
    main()
