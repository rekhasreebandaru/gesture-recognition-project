"""Collect additional labeled landmark samples without replacing the live app."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DATASET_PATH, CAMERA_INDEX, MAX_NUM_HANDS
from src.camera.camera_stream import CameraStream
from src.recognition.features import extract_features, feature_column_names, validate_feature_vector
from src.vision.hand_landmarks import HandLandmarkDetector


GESTURE_LABELS = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "thumbs_up",
    ord("4"): "peace_sign",
}


def validate_dataset_schema(dataset_path: Path = DATASET_PATH) -> None:
    """Ensure collection appends only to the expected landmark CSV schema."""
    expected_columns = ["label", *feature_column_names()]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file), None)
    if header != expected_columns:
        raise ValueError("Dataset CSV header does not match the expected label and 63 feature columns.")


def append_sample(
    label: str, features: list[float] | np.ndarray, dataset_path: Path = DATASET_PATH
) -> None:
    """Append one labeled feature row to the existing CSV dataset."""
    if label not in GESTURE_LABELS.values():
        raise ValueError(f"Unknown gesture label: {label}.")
    vector = validate_feature_vector(np.asarray(features, dtype=np.float32))
    with dataset_path.open("a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow([label, *vector.tolist()])


def main() -> int:
    """Run the data collector and return a process-style status code."""
    camera: CameraStream | None = None
    detector: HandLandmarkDetector | None = None

    try:
        validate_dataset_schema()
        camera = CameraStream(camera_index=CAMERA_INDEX)
        detector = HandLandmarkDetector(max_num_hands=MAX_NUM_HANDS)
        selected_label = "open_palm"
        message = "Use BOTH hands and rotate slightly. Press 1-4, then S to save."
        camera.start()
        while True:
            frame = camera.read()
            if frame is None:
                print("Unable to read a frame from the camera. Stopping.")
                return 1

            frame = cv2.flip(frame, 1)
            results = detector.detect(frame)
            detector.draw_landmarks(frame, results)
            hand_detected = bool(results.multi_hand_landmarks)

            cv2.putText(frame, f"Selected: {selected_label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, message, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.putText(
                frame,
                "1: Palm  2: Fist  3: Thumbs Up  4: Peace  S: Save  Q: Quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Gesture Recognition - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in GESTURE_LABELS:
                selected_label = GESTURE_LABELS[key]
                message = f"Selected {selected_label}. Use left hand, right hand, and slight rotations."
            elif key in (ord("s"), ord("S")):
                if not hand_detected:
                    message = "No sample saved: no hand detected."
                else:
                    try:
                        append_sample(
                            selected_label, extract_features(results.multi_hand_landmarks[0])
                        )
                    except (OSError, ValueError) as error:
                        message = f"Sample not saved: {error}"
                    else:
                        message = f"Saved {selected_label}. Change hand/angle before the next sample."
            elif key in (ord("q"), ord("Q")):
                return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Data collection error: {error}", file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
