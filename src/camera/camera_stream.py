"""Small wrapper around OpenCV's webcam capture object."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Open, read from, and release one webcam device."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Check that the webcam is connected and not in use by another app."
            )
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            raise RuntimeError("CameraStream.start() must be called before read().")

        success, frame = self._capture.read()
        return frame if success else None

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
