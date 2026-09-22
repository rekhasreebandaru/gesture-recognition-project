"""Manually test the Arduino relay controller without the AI pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.commands.command_mapper import Command
from src.devices.arduino_device import ArduinoDevice, ArduinoDeviceError


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one relay command to an Arduino.")
    parser.add_argument(
        "command",
        choices=[command.value for command in Command],
        help="Relay command to send.",
    )
    parser.add_argument(
        "--port",
        default=os.getenv("ARDUINO_PORT", "COM3"),
        help="Arduino serial port (default: ARDUINO_PORT or COM3).",
    )
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate (default: 9600).")
    args = parser.parse_args()

    device = ArduinoDevice(port=args.port, baudrate=args.baud)
    try:
        print(f"Connecting to Arduino on {args.port} at {args.baud} baud...")
        device.connect()
        device.execute(args.command)
        print(f"Sent {args.command} successfully.")
        return 0
    except (ArduinoDeviceError, ValueError) as error:
        print(f"Arduino test failed: {error}", file=sys.stderr)
        return 1
    finally:
        device.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
