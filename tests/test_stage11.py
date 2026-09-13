"""Stage 11 checks for the non-UI gesture-recognition pipeline."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_stream import CameraStream
from src.commands.command_mapper import Command, map_gesture_to_command
from src.devices.virtual_device import VirtualDevice
from src.recognition.classifier import GestureClassifier
from src.recognition.collect_data import append_sample, validate_dataset_schema
from src.recognition.features import (
    FEATURE_COUNT,
    canonicalize_features,
    feature_column_names,
    mirror_features,
    validate_feature_vector,
)
from src.recognition.stability import ConsecutiveFrameStabilizer
from src.recognition.train_model import load_dataset


class GesturePipelineTests(unittest.TestCase):
    def test_rotation_canonicalization(self) -> None:
        """The same landmark set rotated in the camera plane stays equivalent."""
        points = np.array([[index * 0.02, -index * 0.03, index * 0.01] for index in range(21)])
        rotated = points.copy()
        rotated[:, :2] = points[:, :2] @ np.array([[0.0, -1.0], [1.0, 0.0]]).T
        np.testing.assert_allclose(
            canonicalize_features(points.reshape(FEATURE_COUNT)),
            canonicalize_features(rotated.reshape(FEATURE_COUNT)),
            atol=1e-5,
        )

    def test_feature_dimension_finiteness_and_mirroring(self) -> None:
        features = np.arange(FEATURE_COUNT, dtype=np.float32)
        mirrored = mirror_features(features)
        self.assertEqual(validate_feature_vector(features).shape, (FEATURE_COUNT,))
        self.assertTrue(np.array_equal(mirrored[1::3], features[1::3]))
        self.assertTrue(np.array_equal(mirrored[2::3], features[2::3]))
        self.assertTrue(np.array_equal(mirrored[0::3], -features[0::3]))
        with self.assertRaises(ValueError):
            validate_feature_vector(np.zeros(FEATURE_COUNT - 1))
        with self.assertRaises(ValueError):
            validate_feature_vector(np.full(FEATURE_COUNT, np.nan))

    def test_saved_model_predicts_a_dataset_row(self) -> None:
        """The saved model loads and produces a valid labeled prediction."""
        with (PROJECT_ROOT / "data" / "gestures.csv").open(newline="", encoding="utf-8") as csv_file:
            row = next(csv.DictReader(csv_file))
        raw_features = np.array([float(row[f"feature_{index}"]) for index in range(FEATURE_COUNT)])
        prediction = GestureClassifier(PROJECT_ROOT / "models" / "gesture_model.pkl").predict(
            canonicalize_features(raw_features)
        )
        self.assertIn(prediction.label, {"open_palm", "fist", "thumbs_up", "peace_sign"})
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)

    def test_stability_requires_consecutive_frames_and_resets(self) -> None:
        stabilizer = ConsecutiveFrameStabilizer(required_consecutive_frames=3)
        self.assertIsNone(stabilizer.update("fist").confirmed_label)
        self.assertIsNone(stabilizer.update("fist").confirmed_label)
        self.assertEqual(stabilizer.update("fist").confirmed_label, "fist")
        self.assertEqual(stabilizer.update("open_palm").confirmed_label, "fist")
        self.assertIsNone(stabilizer.reset().confirmed_label)

    def test_all_gesture_command_mappings(self) -> None:
        expected = {
            "open_palm": Command.LIGHT_ON,
            "fist": Command.LIGHT_OFF,
            "thumbs_up": Command.FAN_ON,
            "peace_sign": Command.FAN_OFF,
        }
        self.assertEqual({label: map_gesture_to_command(label) for label in expected}, expected)
        self.assertIsNone(map_gesture_to_command(None))
        with self.assertRaises(ValueError):
            map_gesture_to_command("unknown")

    def test_virtual_device_state_transitions(self) -> None:
        device = VirtualDevice()
        self.assertFalse(device.get_state().light_on)
        self.assertFalse(device.get_state().fan_on)
        with self.assertRaises(ValueError):
            device.apply("LIGHT_ON")  # type: ignore[arg-type]
        device.apply(Command.LIGHT_ON)
        device.apply(Command.FAN_ON)
        self.assertTrue(device.get_state().light_on)
        self.assertTrue(device.get_state().fan_on)
        device.apply(Command.LIGHT_OFF)
        device.apply(Command.FAN_OFF)
        self.assertFalse(device.get_state().light_on)
        self.assertFalse(device.get_state().fan_on)

    def test_invalid_stability_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConsecutiveFrameStabilizer().update("")

    def test_missing_and_invalid_model_are_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            GestureClassifier(PROJECT_ROOT / "models" / "missing.pkl")
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_model = Path(temporary_directory) / "invalid.pkl"
            invalid_model.write_text("not a pickle", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Could not load model"):
                GestureClassifier(invalid_model)

    def test_application_handles_missing_model_gracefully(self) -> None:
        import app

        missing_model = PROJECT_ROOT / "models" / "missing.pkl"
        with patch.object(app, "MODEL_PATH", missing_model), redirect_stderr(StringIO()):
            self.assertEqual(app.main(), 1)

    def test_collection_and_training_reject_invalid_csv_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "gestures.csv"
            expected_header = ["label", *feature_column_names()]
            with dataset.open("w", newline="", encoding="utf-8") as csv_file:
                csv.writer(csv_file).writerow(expected_header)
            validate_dataset_schema(dataset)
            append_sample("fist", np.zeros(FEATURE_COUNT), dataset)
            with self.assertRaises(ValueError):
                append_sample("invalid", np.zeros(FEATURE_COUNT), dataset)

            malformed = Path(temporary_directory) / "malformed.csv"
            malformed.write_text("label,wrong_feature\nfist,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(malformed)

    @patch("src.camera.camera_stream.cv2.VideoCapture")
    def test_camera_failure_raises_clear_error(self, video_capture: MagicMock) -> None:
        capture = video_capture.return_value
        capture.isOpened.return_value = False
        camera = CameraStream()
        with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
            camera.start()
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
