"""Train and evaluate the static hand-gesture classifier from CSV data."""

from __future__ import annotations

import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import DATASET_PATH, MODEL_PATH
from src.recognition.features import (
    FEATURE_COUNT,
    canonicalize_features,
    feature_column_names,
    mirror_features,
)


REQUIRED_LABELS = {"open_palm", "fist", "thumbs_up", "peace_sign"}


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate labeled feature rows from the collection CSV."""
    expected_columns = ["label", *feature_column_names()]
    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != expected_columns:
            raise ValueError("Dataset CSV header does not match the expected label and 63 feature columns.")
        rows = list(reader)

    if not rows:
        raise ValueError("The dataset is empty. Collect gesture samples before training.")

    malformed_rows = [index + 2 for index, row in enumerate(rows) if set(row) != set(expected_columns)]
    if malformed_rows:
        raise ValueError(f"Dataset has malformed row(s): {malformed_rows[:5]}.")

    labels = np.array([row["label"] for row in rows])
    missing_labels = REQUIRED_LABELS - set(labels)
    if missing_labels:
        raise ValueError(f"Dataset is missing labels: {', '.join(sorted(missing_labels))}.")
    unknown_labels = set(labels) - REQUIRED_LABELS
    if unknown_labels:
        raise ValueError(f"Dataset contains unknown labels: {', '.join(sorted(unknown_labels))}.")

    feature_names = feature_column_names()
    features = np.array(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32
    )
    if features.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features per row.")
    if not np.isfinite(features).all():
        raise ValueError("Dataset contains non-finite feature values.")

    counts = Counter(labels)
    too_small = [label for label, count in counts.items() if count < 5]
    if too_small:
        raise ValueError("Each gesture needs at least 5 samples for stratified splitting.")
    return np.array([canonicalize_features(row) for row in features]), labels


def main() -> None:
    features, labels = load_dataset(DATASET_PATH)

    # 60% training, 20% validation, 20% held-out test data.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=42
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=42,
    )

    # Both hand sides represent the same gesture. Add a mirrored counterpart
    # only to training rows; validation and test rows stay genuinely unseen.
    x_train = np.vstack([x_train, np.array([mirror_features(row) for row in x_train])])
    y_train = np.concatenate([y_train, y_train])

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Samples by gesture: {dict(sorted(Counter(labels).items()))}")
    print(
        "Split sizes: "
        f"train={len(y_train) // 2} source rows ({len(y_train)} with mirroring), "
        f"validation={len(y_validation)}, test={len(y_test)}"
    )
    print(f"Validation accuracy: {validation_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print("\nHeld-out test classification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))
    labels_in_order = sorted(REQUIRED_LABELS)
    print("Held-out test confusion matrix (rows=true, columns=predicted):")
    print(f"Labels: {labels_in_order}")
    print(confusion_matrix(y_test, test_predictions, labels=labels_in_order))
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
