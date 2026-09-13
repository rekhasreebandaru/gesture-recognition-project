# Project Review Summary — Gesture Recognition

## Project overview

This is a Windows/Python static hand-gesture recognition application. It uses a laptop webcam to detect one hand, extracts MediaPipe hand landmarks, classifies one of four gestures, confirms the result across consecutive frames, maps the confirmed gesture to a command, and updates a software-only virtual light/fan device display.

Implemented gesture-to-command mappings:

| Gesture | Command | Virtual effect |
|---|---|---|
| `open_palm` | `LIGHT_ON` | Light ON |
| `fist` | `LIGHT_OFF` | Light OFF |
| `thumbs_up` | `FAN_ON` | Fan ON |
| `peace_sign` | `FAN_OFF` | Fan OFF |

## Architecture and workflow

`Webcam → CameraStream → MediaPipe Hands → normalized landmark features → scikit-learn classifier → consecutive-frame stabilizer → command mapper → VirtualDevice → OpenCV display`

- `CameraStream` opens, reads, and releases the webcam.
- MediaPipe Hands returns up to one hand with 21 landmarks.
- Landmark coordinates are translated to the wrist, rotation-normalized using the wrist-to-middle-finger-base direction, and scale-normalized. Each sample has 63 values: 21 landmarks × x/y/z.
- A trained scikit-learn pipeline predicts a gesture and confidence.
- Five matching consecutive predictions are required before a gesture is confirmed.
- Only confirmed gestures are mapped to commands and applied to the virtual device.

## Technologies actually used

- Python 3.11.2 (reviewed virtual environment)
- OpenCV (`opencv-contrib-python`) for webcam access and the on-screen UI
- MediaPipe Hands for hand landmark detection
- NumPy for landmark feature processing
- scikit-learn for preprocessing, classification, calibration, data splitting, and metrics
- CSV for collected labeled landmark data
- `pickle` for the saved trained model
- Python `unittest` for automated checks

## Gesture-recognition approach

The project uses landmark-based classical machine learning, not image-based deep learning and not deterministic gesture rules.

- Input: MediaPipe's 21 3D-normalized hand landmarks.
- Features: 63 landmark coordinate values after translation, in-plane rotation, and scale normalization.
- Training augmentation: a mirrored version of each training sample is added so the model is less dependent on hand side. Mirroring is applied only to training rows; validation/test rows are kept unaugmented.
- Model: `StandardScaler` followed by an RBF-kernel `SVC`, wrapped in `CalibratedClassifierCV` for probability/confidence output.
- Current dataset: 219 labeled rows: fist 38, open palm 80, peace sign 52, thumbs up 49.

## IoT-related functionality

No physical IoT hardware, ESP32, MQTT, cloud service, or database is implemented.

`DeviceInterface` defines the device contract, and `VirtualDevice` is an in-memory implementation with separate `light_on` and `fan_on` state. This provides a software simulation and a clear extension point for a future hardware-backed device implementation.

## How to run

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Press `Q` in the OpenCV camera window to quit.

Optional project commands:

```powershell
# Collect additional labeled data
python -m src.recognition.collect_data

# Retrain and overwrite the saved model
python -m src.recognition.train_model

# Run automated checks
python -m unittest discover -s tests -v
```

## Main features

- Live laptop-webcam input with failure handling for unavailable camera/frame reads.
- Real-time one-hand landmark display.
- Four-class static gesture recognition with confidence display.
- Landmark position, scale, and in-plane rotation normalization.
- Mirrored training augmentation for hand-side robustness.
- Five-frame confirmation to reduce rapid output changes.
- Gesture-to-command separation from recognition logic.
- Virtual light/fan status panels (green ON, gray OFF).
- Automated non-camera tests for feature normalization, saved-model inference, stability behavior, command mappings, virtual device transitions, and camera-open failure handling.

## Review results and known limitations

- `README.md` is missing. This review summary is the only project-level documentation currently present.
- `requirements.txt` lists the direct runtime packages, but `scikit-learn` is not version-pinned and the file does not state the required Python version.
- The reviewed `.venv` has an installed-package conflict: it contains both `opencv-contrib-python 4.11.0.86` and leftover `opencv-python 5.0.0.93`; `pip check` reports that the latter expects NumPy 2 or newer while the project pins NumPy 1.26.4. The reviewed automated tests still pass, but a clean review/install environment should not retain that extra OpenCV 5 package.
- `pip check` also reports that the current environment lacks MediaPipe optional/transitive packages `jax`, `jaxlib`, and `sentencepiece`. The current tested application imports and runs its MediaPipe Hands usage, but the environment is not dependency-clean according to `pip check`.
- `app.py` uses a relative model path (`models/gesture_model.pkl`) and camera index `0`; it is intended to be started from the project root and assumes the default webcam.
- There is no configuration file for camera index, thresholds, model path, or five-frame confirmation count.
- The live window title and module docstring retain earlier Stage 6/7 labels even though the code includes full Stage 10 integration. This is a naming/documentation inconsistency only.
- The model is trained from a small, user-collected dataset. Its measured metrics are specific to that dataset and should not be presented as a general real-world accuracy claim.
- The system supports one detected hand at a time and static gestures only. Fast pose transitions, occlusion, lighting changes, and poses where the palm is hidden can reduce MediaPipe acquisition/tracking reliability.
- The model artifact is a Python pickle, so it must be treated as trusted local content and should not be loaded from untrusted sources.

## Possible improvements

- Add a README with installation, run, training, data-collection, testing, and troubleshooting instructions.
- Clean and lock the virtual-environment dependencies; pin the scikit-learn version and record Python 3.11 as the target runtime.
- Move runtime constants (camera index, MediaPipe thresholds, model path, stability-frame count) into a configuration module/file.
- Expand and balance the labeled data across users, both hands, lighting conditions, distances, and gesture orientations; then remeasure on an independently collected test set.
- Add tests for CSV validation edge cases, unknown command labels, and the full application orchestration logic.
- Add optional state persistence, UI polish, or a real device implementation only if the project scope later requires them.

## Important files

| File | Purpose |
|---|---|
| `app.py` | Main application loop and on-screen integration of all layers. |
| `requirements.txt` | Direct Python runtime dependencies. |
| `data/gestures.csv` | Collected labeled gesture feature dataset. |
| `models/gesture_model.pkl` | Saved trained scikit-learn pipeline; binary file, not source code. |
| `src/camera/camera_stream.py` | OpenCV webcam wrapper and camera failure handling. |
| `src/vision/hand_landmarks.py` | MediaPipe Hands detection and landmark drawing. |
| `src/recognition/features.py` | 63-value feature extraction, rotation/scale canonicalization, and mirroring. |
| `src/recognition/collect_data.py` | Interactive webcam CSV data-collection tool. |
| `src/recognition/train_model.py` | Dataset validation, train/validation/test split, model training, evaluation, and model saving. |
| `src/recognition/classifier.py` | Saved-model loading and gesture/confidence prediction wrapper. |
| `src/recognition/stability.py` | Consecutive-frame gesture confirmation. |
| `src/commands/command_mapper.py` | Gesture-to-command mapping. |
| `src/devices/virtual_device.py` | Device interface and in-memory virtual light/fan implementation. |
| `tests/test_stage11.py` | Automated non-camera pipeline tests. |
