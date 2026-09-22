"""USB-serial Arduino implementation of the device interface."""

from __future__ import annotations

from typing import Any, Callable

import serial

from src.commands.command_mapper import Command
from src.devices.virtual_device import DeviceInterface, DeviceState


class ArduinoDeviceError(RuntimeError):
    """Base error for recoverable Arduino communication failures."""


class ArduinoNotConnectedError(ArduinoDeviceError):
    """Raised when a command is sent without an open serial connection."""


class ArduinoConnectionError(ArduinoDeviceError):
    """Raised when the configured serial port cannot be opened."""


class ArduinoWriteError(ArduinoDeviceError):
    """Raised when an open serial connection cannot send a command."""


class ArduinoDevice(DeviceInterface):
    """Send mapped device commands to an Arduino over USB serial.

    The class accepts only the project's ``Command`` values and sends each as
    a UTF-8 line such as ``LIGHT_ON\\n``. It keeps a local state snapshot for
    the existing application display; that snapshot is not confirmation that
    a physical relay changed state.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 1.0,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not port or not port.strip():
            raise ValueError("Arduino serial port must be a non-empty value, such as COM3.")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial_factory = serial_factory or serial.Serial
        self._serial: Any | None = None
        self._light_on = False
        self._fan_on = False

    @property
    def is_connected(self) -> bool:
        """Return whether the serial connection is currently open."""
        return bool(self._serial is not None and getattr(self._serial, "is_open", False))

    def connect(self) -> None:
        """Open the configured Arduino USB-serial connection."""
        if self.is_connected:
            return
        try:
            connection = self._serial_factory(self.port, self.baudrate, timeout=self.timeout)
        except (serial.SerialException, OSError, ValueError) as error:
            raise ArduinoConnectionError(
                f"Could not connect to Arduino on {self.port} at {self.baudrate} baud: {error}"
            ) from error
        if not getattr(connection, "is_open", True):
            raise ArduinoConnectionError(f"Arduino serial port {self.port} did not open.")
        self._serial = connection

    def disconnect(self) -> None:
        """Close the serial connection when the application exits."""
        if self._serial is None:
            return
        try:
            if getattr(self._serial, "is_open", False):
                self._serial.close()
        finally:
            self._serial = None

    @staticmethod
    def _validate_command(command: Command | str) -> Command:
        try:
            return command if isinstance(command, Command) else Command(command)
        except ValueError as error:
            raise ValueError(f"Unsupported Arduino command: {command}") from error

    def send_command(self, command: Command | str) -> None:
        """Write one supported command line to the connected Arduino."""
        normalized_command = self._validate_command(command)
        if not self.is_connected:
            raise ArduinoNotConnectedError(
                f"Arduino is not connected. Check port {self.port} and call connect()."
            )
        try:
            self._serial.write(f"{normalized_command.value}\n".encode("utf-8"))
            self._serial.flush()
        except (serial.SerialException, OSError) as error:
            raise ArduinoWriteError(
                f"Could not send {normalized_command.value} to Arduino on {self.port}: {error}"
            ) from error

    def execute(self, command: Command | str) -> DeviceState:
        """Send one command and update the local display state after a write."""
        normalized_command = self._validate_command(command)
        self.send_command(normalized_command)
        if normalized_command is Command.LIGHT_ON:
            self._light_on = True
        elif normalized_command is Command.LIGHT_OFF:
            self._light_on = False
        elif normalized_command is Command.FAN_ON:
            self._fan_on = True
        elif normalized_command is Command.FAN_OFF:
            self._fan_on = False
        return self.get_state()

    def apply(self, command: Command) -> DeviceState:
        """Implement the shared device interface used by the application."""
        return self.execute(command)

    def get_state(self) -> DeviceState:
        """Return the last successfully sent virtual display state."""
        return DeviceState(light_on=self._light_on, fan_on=self._fan_on)
