"""Convert MediaPipe hand landmarks into classifier-ready numeric features."""

from __future__ import annotations

from typing import Any

import numpy as np


LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3
FEATURE_COUNT = LANDMARK_COUNT * COORDINATES_PER_LANDMARK


def validate_feature_vector(features: np.ndarray) -> np.ndarray:
    """Return one finite 63-value feature vector or raise a clear error."""
    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (FEATURE_COUNT,):
        raise ValueError(f"Expected a feature vector of shape ({FEATURE_COUNT},).")
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains non-finite values.")
    return vector


def feature_column_names() -> list[str]:
    """Return the stable CSV column names for one landmark feature vector."""
    return [f"feature_{index}" for index in range(FEATURE_COUNT)]


def extract_features(hand_landmarks: Any) -> np.ndarray:
    """Return translation-, scale-, and rotation-normalized landmark features.

    The wrist is the origin and the wrist-to-middle-finger-base direction is
    aligned vertically. This makes a horizontal or tilted presentation of
    the same static gesture comparable with an upright presentation.
    """
    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    )
    if points.shape != (LANDMARK_COUNT, COORDINATES_PER_LANDMARK):
        raise ValueError(f"Expected {LANDMARK_COUNT} hand landmarks, got {points.shape[0]}.")

    return canonicalize_features((points - points[0]).reshape(FEATURE_COUNT))


def canonicalize_features(features: np.ndarray) -> np.ndarray:
    """Canonicalize an existing 63-value feature vector for model input."""
    vector = validate_feature_vector(features)

    relative_points = vector.reshape(LANDMARK_COUNT, COORDINATES_PER_LANDMARK).copy()
    # Landmark 9 is the middle-finger MCP joint. Align wrist -> landmark 9
    # to the upward vertical axis so in-plane hand rotation is normalized.
    reference = relative_points[9, :2]
    if float(np.linalg.norm(reference)) <= 1e-6:
        raise ValueError("Hand landmarks have an invalid orientation reference.")
    angle = -np.pi / 2 - float(np.arctan2(reference[1], reference[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    relative_points[:, :2] = relative_points[:, :2] @ rotation.T

    scale = float(np.max(np.linalg.norm(relative_points[:, :2], axis=1)))
    if scale <= 1e-6:
        raise ValueError("Hand landmarks have an invalid scale.")

    return (relative_points / scale).reshape(FEATURE_COUNT)


def mirror_features(features: np.ndarray) -> np.ndarray:
    """Return a left/right mirrored canonical feature vector for augmentation."""
    mirrored = validate_feature_vector(features).reshape(
        LANDMARK_COUNT, COORDINATES_PER_LANDMARK
    ).copy()
    mirrored[:, 0] *= -1
    return mirrored.reshape(FEATURE_COUNT)
