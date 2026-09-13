"""Load the trained gesture model and make landmark-feature predictions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.recognition.features import validate_feature_vector


@dataclass(frozen=True)
class Prediction:
    """One model prediction, independent of commands or device actions."""

    label: str
    confidence: float


class GestureClassifier:
    """Thin recognition-layer wrapper around the saved scikit-learn pipeline."""

    def __init__(self, model_path: Path | str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run 'python -m src.recognition.train_model' first."
            )
        try:
            with path.open("rb") as model_file:
                self._model = pickle.load(model_file)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as error:
            raise RuntimeError(
                f"Could not load model at {path}. Retrain it with 'python -m src.recognition.train_model'."
            ) from error

        if not hasattr(self._model, "predict") or not hasattr(self._model, "predict_proba"):
            raise RuntimeError(f"Model at {path} does not support prediction with confidence.")

    def predict(self, features: np.ndarray) -> Prediction:
        """Return the most likely gesture label and its model confidence."""
        vector = validate_feature_vector(features)

        feature_batch = vector.reshape(1, -1)
        label = str(self._model.predict(feature_batch)[0])
        probabilities = self._model.predict_proba(feature_batch)[0]
        return Prediction(label=label, confidence=float(np.max(probabilities)))
