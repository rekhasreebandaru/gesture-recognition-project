"""Mocked tests for the local Arduino USB-serial device layer."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import serial

from src.commands.command_mapper import Command
from src.devices.arduino_device import (
    ArduinoConnectionError,
    ArduinoDevice,
    ArduinoNotConnectedError,
    ArduinoWriteError,
)


class ArduinoDeviceTests(unittest.TestCase):
    def make_connected_device(self) -> tuple[ArduinoDevice, MagicMock, MagicMock]:
        connection = MagicMock()
        connection.is_open = True
        serial_factory = MagicMock(return_value=connection)
        device = ArduinoDevice(port="COM9", serial_factory=serial_factory)
        device.connect()
        return device, connection, serial_factory

    def test_supported_commands_are_sent_as_utf8_lines(self) -> None:
        for command in Command:
            with self.subTest(command=command.value):
                device, connection, _ = self.make_connected_device()
                device.execute(command)
                connection.write.assert_called_once_with(f"{command.value}\n".encode("utf-8"))
                connection.flush.assert_called_once_with()

    def test_invalid_command_is_rejected(self) -> None:
        device = ArduinoDevice(port="COM9")
        with self.assertRaisesRegex(ValueError, "Unsupported Arduino command"):
            device.send_command("UNKNOWN")

    def test_connection_failure_is_reported(self) -> None:
        serial_factory = MagicMock(side_effect=serial.SerialException("port unavailable"))
        device = ArduinoDevice(port="COM404", serial_factory=serial_factory)
        with self.assertRaisesRegex(ArduinoConnectionError, "Could not connect"):
            device.connect()

    def test_not_connected_and_write_failures_are_reported(self) -> None:
        device = ArduinoDevice(port="COM9")
        with self.assertRaises(ArduinoNotConnectedError):
            device.send_command(Command.LIGHT_ON)

        device, connection, _ = self.make_connected_device()
        connection.write.side_effect = serial.SerialException("disconnected")
        with self.assertRaisesRegex(ArduinoWriteError, "Could not send"):
            device.send_command(Command.LIGHT_ON)

    def test_disconnect_closes_open_connection(self) -> None:
        device, connection, _ = self.make_connected_device()
        device.disconnect()
        connection.close.assert_called_once_with()
        self.assertFalse(device.is_connected)


if __name__ == "__main__":
    unittest.main()
