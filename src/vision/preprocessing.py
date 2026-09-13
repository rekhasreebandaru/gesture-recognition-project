"""Vision-only frame preprocessing used before hand detection."""

from __future__ import annotations

import cv2
import numpy as np

from config import LOW_LIGHT_GAMMA, LOW_LIGHT_LUMINANCE_THRESHOLD


def enhance_low_light_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE and conditional gamma brightening to a BGR camera frame."""
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)

    if float(lightness.mean()) < LOW_LIGHT_LUMINANCE_THRESHOLD:
        normalized_lightness = enhanced_lightness.astype(np.float32) / 255.0
        enhanced_lightness = np.uint8(
            np.clip(np.power(normalized_lightness, LOW_LIGHT_GAMMA) * 255.0, 0, 255)
        )

    enhanced_lab_frame = cv2.merge((enhanced_lightness, green_red, blue_yellow))
    return cv2.cvtColor(enhanced_lab_frame, cv2.COLOR_LAB2BGR)
