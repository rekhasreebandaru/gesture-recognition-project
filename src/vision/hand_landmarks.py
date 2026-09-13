"""MediaPipe-based hand landmark detection and drawing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# MediaPipe imports Matplotlib internally.  Use a project-local cache because
# the Windows user cache directory may be unavailable in restricted setups.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import cv2
import mediapipe as mp
import numpy as np


class HandLandmarkDetector:
    """Detect up to ``max_num_hands`` hands in BGR OpenCV frames.

    This class only detects and visualizes landmarks. It deliberately does
    not extract ML features, classify gestures, or issue commands.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles

    def detect(self, frame: np.ndarray) -> Any:
        """Return MediaPipe's result for a single BGR camera frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """Draw detected landmarks and connections onto ``frame`` in place."""
        if not results.multi_hand_landmarks:
            return frame

        for hand_landmarks in results.multi_hand_landmarks:
            self._drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._drawing_styles.get_default_hand_landmarks_style(),
                self._drawing_styles.get_default_hand_connections_style(),
            )
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
