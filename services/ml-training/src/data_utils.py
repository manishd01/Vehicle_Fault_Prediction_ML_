from pathlib import Path

import pandas as pd

DATA_ROOT = Path("/app/data")
TARGET_COLUMN = "fault_label"

SPLIT_ALIASES = {
    "train": ["train", "training"],
    "validation": ["validation", "val", "valid"],
    "test": ["test", "testing"],
}


def find_split_path(split_name: str) -> Path:
    aliases = SPLIT_ALIASES[split_name]
    candidates = []

    for path in DATA_ROOT.rglob("*"):
        if any(
            part.lower() in {"models", "metrics", "mlruns", "mlflow-data"}
            for part in path.parts
        ):
            continue

        if path.is_file():
            if path.suffix.lower() != ".parquet":
                continue
        elif path.is_dir():
            if not any(path.rglob("*.parquet")):
                continue
        else:
            continue

        name = path.stem.lower() if path.is_file() else path.name.lower()

        # Avoid selecting a parent directory containing all splits.
        if name in {"train_val_test", "train-validation-test", "splits"}:
            continue

        if name in aliases or any(name == f"{alias}_data" for alias in aliases):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find the {split_name} Parquet dataset under "
            f"{DATA_ROOT}. Check the output paths from "
            "10_train_val_test_split.py."
        )

    def rank(path: Path):
        name = path.stem.lower() if path.is_file() else path.name.lower()
        exact = name in aliases
        return (0 if exact else 1, len(path.parts))

    candidates.sort(key=rank)
    selected = candidates[0]

    print(f"{split_name.title()} dataset: {selected}")
    return selected


def load_split(split_name: str) -> pd.DataFrame:
    path = find_split_path(split_name)
    dataframe = pd.read_parquet(path)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"{TARGET_COLUMN} is missing from {path}")

    return dataframe


def get_feature_columns(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        "timestamp",
        "datetime",
        "date",
        "failure_id",
        "_c0",
    }

    return [
        column
        for column in train_df.columns
        if column in evaluation_df.columns
        and column not in excluded
        and pd.api.types.is_numeric_dtype(train_df[column])
        and pd.api.types.is_numeric_dtype(evaluation_df[column])
    ]


def prepare_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    medians: pd.Series | None = None,
):
    features = dataframe[feature_columns].copy()
    features = features.replace([float("inf"), -float("inf")], float("nan"))

    if medians is None:
        medians = features.median().fillna(0)

    features = features.fillna(medians).fillna(0)

    return features, medians
