import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# The Compose configuration must mount the ONNX folder here.
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models/onnx"))
MODEL_PATH = MODEL_DIR / "random_forest.onnx"
METADATA_PATH = MODEL_DIR / "random_forest_metadata.json"

session = None
feature_columns = []
input_name = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, feature_columns, input_name

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"ONNX model not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Model metadata not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    if len(feature_columns) != metadata["feature_count"]:
        raise RuntimeError("Feature count does not match model metadata.")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = metadata.get("input_name") or session.get_inputs()[0].name

    actual_input = session.get_inputs()[0]
    if actual_input.name != input_name:
        raise RuntimeError(
            f"Input mismatch: metadata says '{input_name}', "
            f"but model expects '{actual_input.name}'."
        )

    if len(actual_input.shape) != 2 or actual_input.shape[1] != len(feature_columns):
        raise RuntimeError(
            f"Model input shape {actual_input.shape} does not match "
            f"{len(feature_columns)} features."
        )

    print(f"ONNX model loaded: {MODEL_PATH}")
    print(f"Expected feature count: {len(feature_columns)}")

    yield

    session = None


app = FastAPI(
    title="Vehicle Fault Prediction API",
    description="Predicts vehicle faults using an exported ONNX model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="All 61 engineered model features, keyed by feature name."
    )


def extract_prediction(outputs):
    """Handle common scikit-learn ONNX classifier output formats."""
    predicted_class = None
    fault_probability = None

    for output in outputs:
        # Some converters return labels as a one-dimensional array.
        if isinstance(output, np.ndarray):
            if output.size == 1 and predicted_class is None:
                predicted_class = int(output.reshape(-1)[0])

            # Other converters return probabilities as an N x classes array.
            elif output.ndim == 2 and output.shape[0] == 1:
                if output.shape[1] >= 2:
                    fault_probability = float(output[0, 1])

        # ZipMap-based converters may return a list of dictionaries.
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            first = output[0]

            if isinstance(first, dict):
                probabilities = {int(key): float(value) for key, value in first.items()}
                if 1 in probabilities:
                    fault_probability = probabilities[1]

    if predicted_class is None:
        raise RuntimeError("Could not interpret the ONNX prediction output.")

    return predicted_class, fault_probability


@app.get("/")
def root():
    return {
        "service": "Vehicle Fault Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if session is not None else "not_ready",
        "model_loaded": session is not None,
        "feature_count": len(feature_columns),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    supplied = request.features
    expected = set(feature_columns)
    received = set(supplied)

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature names do not match the model metadata.",
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    # Always arrange values in the exact feature order used during training.
    try:
        values = np.asarray(
            [[float(supplied[name]) for name in feature_columns]],
            dtype=np.float32,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail="All feature values must be valid numbers.",
        ) from exc

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="Feature values must be finite; NaN and infinity are not allowed.",
        )

    try:
        outputs = session.run(None, {input_name: values})
        predicted_class, fault_probability = extract_prediction(outputs)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {type(exc).__name__}",
        ) from exc

    return {
        "prediction": predicted_class,
        "fault_detected": predicted_class == 1,
        "fault_probability": fault_probability,
        "feature_count": len(feature_columns),
        "model": MODEL_PATH.name,
    }
