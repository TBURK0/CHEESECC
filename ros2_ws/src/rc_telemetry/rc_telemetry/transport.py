"""Telemetry transport: push one JSON frame toward the phone.

RC dashboards usually take BLE. The simplest robust path is newline-delimited JSON over a
serial link carried by a BLE-UART module (HM-10 / nRF52 / ESP32) or plain USB — the phone
sees a BLE device and we don't run a GATT server on the compute box.

  StdoutTransport     : print frames (bench testing, no hardware)
  SerialLineTransport : write `<json>\n` to a serial port (pyserial)

For a native BLE GATT peripheral on Linux (BlueZ), see DASHBOARD_SIGNALS.md.
"""


class StdoutTransport:
    def __init__(self, logger=None):
        self.logger = logger

    def send(self, line: str):
        print(line, flush=True)


class SerialLineTransport:
    def __init__(self, port: str, baud: int, logger=None):
        import serial  # pyserial; imported lazily so the sim/tests don't need it
        self.ser = serial.Serial(port, baud, timeout=0)
        self.logger = logger

    def send(self, line: str):
        try:
            self.ser.write((line + "\n").encode("ascii"))
        except Exception as e:  # noqa: BLE001 - a dropped link must not kill the node
            if self.logger:
                self.logger.warn(f"telemetry serial write failed: {e}")


def make_transport(kind: str, port: str, baud: int, logger=None):
    if kind == "serial":
        return SerialLineTransport(port, baud, logger)
    return StdoutTransport(logger)
