"""Byte transport to the UNO Q: a serial port, or `sim:<path>` to run the host simulator.

The link never sends anything on its own (in particular it never ARMs after a reconnect).
"""
import os
import queue
import shlex
import subprocess
import threading


class Link:
    def __init__(self, port, baud=460800):
        self.port = port
        self.baud = baud
        self._rx = queue.Queue()
        self._dev = None
        self._proc = None
        self._broken = True
        self.last_error = None

    @property
    def connected(self):
        return not self._broken

    def open(self):
        """(Re)open. Returns True on success; failure is recorded in last_error."""
        self.close()
        try:
            if self.port.startswith('sim:'):
                self._proc = subprocess.Popen(shlex.split(self.port[4:]), stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, bufsize=0)
                reader = self._read_proc
            else:
                import serial  # pyserial; imported lazily so sim/tests don't need it
                self._dev = serial.Serial(self.port, self.baud, timeout=0.05)
                reader = self._read_serial
        except Exception as e:  # noqa: BLE001 - any failure means "not connected"
            self.last_error = str(e)
            return False
        self._rx = queue.Queue()
        self._broken = False
        threading.Thread(target=reader, daemon=True).start()
        return True

    def _read_proc(self):
        proc = self._proc
        try:
            while True:
                data = os.read(proc.stdout.fileno(), 4096)
                if not data:
                    raise EOFError('simulator exited')
                self._rx.put(data)
        except Exception as e:  # noqa: BLE001
            if proc is self._proc:
                self._fail(e)

    def _read_serial(self):
        dev = self._dev
        try:
            while True:
                data = dev.read(dev.in_waiting or 1)
                if data:
                    self._rx.put(data)
        except Exception as e:  # noqa: BLE001
            if dev is self._dev:
                self._fail(e)

    def _fail(self, e):
        self.last_error = str(e)
        self._broken = True

    def read(self):
        """All bytes received since the last call (non-blocking)."""
        chunks = []
        while True:
            try:
                chunks.append(self._rx.get_nowait())
            except queue.Empty:
                return b''.join(chunks)

    def write(self, data: bytes):
        if self._broken:
            return False
        try:
            if self._proc:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            else:
                self._dev.write(data)
            return True
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return False

    def close(self):
        self._broken = True
        proc, dev = self._proc, self._dev
        self._proc = self._dev = None
        if proc:
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        if dev:
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass
