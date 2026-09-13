# Gesture Recognition with Virtual IoT Control

This Windows/Python application recognizes four static hand gestures from a laptop webcam. MediaPipe Hands detects the hand landmarks, a trained scikit-learn classifier recognizes the gesture, and a stability layer confirms it before the gesture is mapped to a command for a software-only virtual light and fan.

## Gestures and commands

| Gesture | Command | Virtual device effect |
| --- | --- | --- |
| Open Palm | `LIGHT_ON` | Turns the virtual light on |
| Fist | `LIGHT_OFF` | Turns the virtual light off |
| Thumbs Up | `FAN_ON` | Turns the virtual fan on |
| Peace Sign | `FAN_OFF` | Turns the virtual fan off |

## Setup

Use Python 3.11 from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current terminal and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run the application

```powershell
python app.py
```

Press `Q` while the OpenCV camera window is focused to quit.

## Collect more training data

```powershell
python -m src.recognition.collect_data
```

Collect real samples for the requested gesture in varied lighting, distances, orientations, and with both hands where possible.

## Retrain the model

```powershellq
python -m src.recognition.train_model
```

This evaluates the dataset and overwrites `models/gesture_model.pkl` with the newly trained model.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Known limitations

- The system recognizes one hand and four static gestures only.
- Low-light hand detection was empirically tested. Before MediaPipe detection, the application can apply CLAHE to the LAB lightness channel and conditional gamma brightening for dark frames; it is toggleable with `ENABLE_LOW_LIGHT_ENHANCEMENT` in `config.py`. Very dark conditions may still reduce detection reliability.
- Performance can be reduced by occlusion, rapid motion, or unusual hand orientations.
- The model is only as representative as its user-collected landmark dataset; collect and evaluate additional independent samples before making broad accuracy claims.
- The virtual light and fan demonstrate the command/device layer only. No physical IoT hardware, MQTT, or cloud service is included.
- The saved model is a trusted local Python pickle; do not replace it with a model file from an untrusted source.
