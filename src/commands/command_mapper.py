"""Map confirmed gesture labels to application commands.

This layer deliberately knows nothing about MediaPipe, classifiers, cameras,
or device implementations.
"""

from __future__ import annotations

from enum import Enum


class Command(str, Enum):
    LIGHT_ON = "LIGHT_ON"
    LIGHT_OFF = "LIGHT_OFF"
    FAN_ON = "FAN_ON"
    FAN_OFF = "FAN_OFF"


GESTURE_TO_COMMAND: dict[str, Command] = {
    "open_palm": Command.LIGHT_ON,
    "fist": Command.LIGHT_OFF,
    "thumbs_up": Command.FAN_ON,
    "peace_sign": Command.FAN_OFF,
}


def map_gesture_to_command(confirmed_gesture: str | None) -> Command | None:
    """Return a command only for a known, confirmed gesture.

    ``None`` means no gesture has been confirmed yet.  Unknown labels are
    rejected so an unexpected recognition output cannot trigger an action.
    """
    if confirmed_gesture is None:
        return None
    try:
        return GESTURE_TO_COMMAND[confirmed_gesture]
    except KeyError as error:
        raise ValueError(f"No command mapping exists for gesture '{confirmed_gesture}'.") from error
