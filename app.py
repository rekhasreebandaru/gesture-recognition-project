"""Live static-gesture recognition with virtual and Arduino device control."""

from __future__ import annotations

import sys

import cv2

from config import (
    CAMERA_INDEX,
    ARDUINO_BAUD_RATE,
    ARDUINO_PORT,
    DETECT_HANDS_EVERY_FRAME,
    DEVICE_MODE,
    ENABLE_LOW_LIGHT_ENHANCEMENT,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_PREDICTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    REQUIRED_CONSECUTIVE_FRAMES,
)
from src.camera.camera_stream import CameraStream
from src.commands.command_mapper import Command, map_gesture_to_command
from src.devices.virtual_device import VirtualDevice
from src.devices.arduino_device import ArduinoDevice, ArduinoDeviceError
from src.devices.virtual_device import DeviceInterface
from src.recognition.classifier import GestureClassifier
from src.recognition.features import extract_features
from src.recognition.stability import ConsecutiveFrameStabilizer
from src.vision.hand_landmarks import HandLandmarkDetector
from src.vision.preprocessing import enhance_low_light_frame


def draw_device_panel(frame, x: int, y: int, name: str, is_on: bool) -> None:
    """Draw a high-visibility virtual appliance state panel."""
    color = (0, 150, 0) if is_on else (50, 50, 50)
    state = "ON" if is_on else "OFF"
    cv2.rectangle(frame, (x, y), (x + 190, y + 62), color, thickness=-1)
    cv2.rectangle(frame, (x, y), (x + 190, y + 62), (255, 255, 255), thickness=1)
    cv2.putText(frame, name, (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, state, (x + 10, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def create_device() -> DeviceInterface:
    """Create the selected device without allowing serial failures to stop the app."""
    if DEVICE_MODE == "virtual":
        return VirtualDevice()
    if DEVICE_MODE == "arduino":
        try:
            device = ArduinoDevice(port=ARDUINO_PORT, baudrate=ARDUINO_BAUD_RATE)
            device.connect()
        except (ArduinoDeviceError, ValueError) as error:
            print(f"Arduino unavailable; commands will not reach hardware: {error}", file=sys.stderr)
            return VirtualDevice()
        return device
    raise ValueError("DEVICE_MODE must be either 'virtual' or 'arduino'.")


def main() -> int:
    """Run the webcam application and return a process-style status code."""
    camera: CameraStream | None = None
    detector: HandLandmarkDetector | None = None

    try:
        classifier = GestureClassifier(MODEL_PATH)
        camera = CameraStream(camera_index=CAMERA_INDEX)
        detector = HandLandmarkDetector(
            static_image_mode=DETECT_HANDS_EVERY_FRAME,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        stabilizer = ConsecutiveFrameStabilizer(
            required_consecutive_frames=REQUIRED_CONSECUTIVE_FRAMES
        )
        device = create_device()
        last_applied_gesture: str | None = None
        last_command: Command | None = None
        camera.start()

        while True:
            frame = camera.read()
            if frame is None:
                print("Unable to read a frame from the camera. Stopping.")
                return 1

            frame = cv2.flip(frame, 1)
            detection_frame = (
                enhance_low_light_frame(frame) if ENABLE_LOW_LIGHT_ENHANCEMENT else frame
            )
            results = detector.detect(detection_frame)
            detector.draw_landmarks(frame, results)

            if results.multi_hand_landmarks:
                try:
                    features = extract_features(results.multi_hand_landmarks[0])
                    prediction = classifier.predict(features)
                except (ValueError, TypeError) as error:
                    stabilizer.reset()
                    status = "Recognition input error"
                    confirmation = str(error)
                    status_color = (0, 0, 255)
                else:
                    status = f"Raw: {prediction.label} ({prediction.confidence:.1%})"
                    if prediction.confidence < MIN_PREDICTION_CONFIDENCE:
                        stabilizer.reset()
                        confirmation = (
                            f"Uncertain prediction: need {MIN_PREDICTION_CONFIDENCE:.0%} confidence"
                        )
                        status_color = (0, 165, 255)
                    else:
                        stability = stabilizer.update(prediction.label)
                        confirmed = stability.confirmed_label or "Waiting for confirmation"
                        confirmation = (
                            f"Confirmed: {confirmed} "
                            f"({stability.consecutive_frames}/{REQUIRED_CONSECUTIVE_FRAMES})"
                        )
                        if stability.confirmed_label != last_applied_gesture:
                            command = map_gesture_to_command(stability.confirmed_label)
                            if command is not None:
                                last_applied_gesture = stability.confirmed_label
                                try:
                                    device.apply(command)
                                except ArduinoDeviceError as error:
                                    print(f"Arduino command failed: {error}", file=sys.stderr)
                                    confirmation = f"Arduino error: {error}"
                                    status_color = (0, 165, 255)
                                else:
                                    last_command = command
                        if stability.confirmed_label:
                            status_color = (
                                (0, 165, 255)
                                if confirmation.startswith("Arduino error:")
                                else (0, 255, 0)
                            )
                        else:
                            status_color = (0, 255, 255)
            else:
                stabilizer.reset()
                status = "No hand detected"
                confirmation = "Hold a gesture steady to confirm it."
                status_color = (0, 0, 255)

            state = device.get_state()
            command_text = last_command.value if last_command else "--"
            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, confirmation, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Last command: {command_text}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            draw_device_panel(frame, 20, 130, "VIRTUAL LIGHT", state.light_on)
            draw_device_panel(frame, 220, 130, "VIRTUAL FAN", state.fan_on)
            cv2.putText(frame, "Press Q to quit", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Gesture Recognition - Virtual & Arduino IoT Control", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Application error: {error}", file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.close()
        if isinstance(locals().get("device"), ArduinoDevice):
            device.disconnect()
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
