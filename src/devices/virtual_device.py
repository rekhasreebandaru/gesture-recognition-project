"""Virtual device implementation for the gesture-recognition demonstration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.commands.command_mapper import Command


@dataclass(frozen=True)
class DeviceState:
    """A read-only snapshot of controllable virtual device state."""

    light_on: bool
    fan_on: bool


class DeviceInterface(ABC):
    """Contract a future hardware-backed device must also satisfy."""

    @abstractmethod
    def apply(self, command: Command) -> DeviceState:
        """Apply one command and return the resulting state."""

    @abstractmethod
    def get_state(self) -> DeviceState:
        """Return the current device state without changing it."""


class VirtualDevice(DeviceInterface):
    """In-memory light/fan device used until real hardware is added."""

    def __init__(self) -> None:
        self._light_on = False
        self._fan_on = False

    def apply(self, command: Command) -> DeviceState:
        """Apply a mapped command to the appropriate virtual appliance."""
        if command is Command.LIGHT_ON:
            self._light_on = True
        elif command is Command.LIGHT_OFF:
            self._light_on = False
        elif command is Command.FAN_ON:
            self._fan_on = True
        elif command is Command.FAN_OFF:
            self._fan_on = False
        else:
            raise ValueError(f"Unsupported command: {command}")
        return self.get_state()

    def get_state(self) -> DeviceState:
        return DeviceState(light_on=self._light_on, fan_on=self._fan_on)
