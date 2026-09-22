"""Central runtime settings for the gesture-recognition application."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "gestures.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.pkl"

CAMERA_INDEX = 0
MAX_NUM_HANDS = 1
# Use MediaPipe's video tracker after initial palm acquisition.  This keeps
# landmarks more stable while a detected hand changes between static poses.
DETECT_HANDS_EVERY_FRAME = False
MIN_DETECTION_CONFIDENCE = 0.35
MIN_TRACKING_CONFIDENCE = 0.35
ENABLE_LOW_LIGHT_ENHANCEMENT = True
LOW_LIGHT_LUMINANCE_THRESHOLD = 90.0
LOW_LIGHT_GAMMA = 0.60
MIN_PREDICTION_CONFIDENCE = 0.60
REQUIRED_CONSECUTIVE_FRAMES = 5

# Device selection: set DEVICE_MODE to "virtual" (default) or "arduino".
DEVICE_MODE = os.getenv("DEVICE_MODE", "virtual").strip().lower()
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM3").strip()
ARDUINO_BAUD_RATE = 9600
