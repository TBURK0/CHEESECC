"""Map MCU micros() timestamps onto host time.

offset = min over a sliding window of (host_rx_time - mcu_time). The minimum is the sample
with the least transport delay, so stamps = mcu_time + offset carry only that minimal delay.
Assumption: MCU/host clock drift is negligible over the window (200 samples at ~200 frames/s
is ~1 s). ponytail: no drift model; add a linear fit if HIL shows stamps drifting.
"""
from collections import deque

WRAP = 1 << 32


class McuClock:
    def __init__(self, window=200):
        self._last_raw = None
        self._epoch = 0
        self._offsets = deque(maxlen=window)

    def unwrap(self, t_us):
        # Frames are not strictly ordered (a US frame carries its trigger time), so a slightly
        # older stamp can arrive just after a wrap: treat a jump of > half the range as a wrap.
        last = self._last_raw
        if last is not None:
            if last - t_us > WRAP // 2:
                self._epoch += WRAP
            elif t_us - last > WRAP // 2:
                return self._epoch - WRAP + t_us   # late frame from before the wrap
            elif t_us < last:
                return self._epoch + t_us          # slightly older frame; keep the newest as reference
        self._last_raw = t_us
        return self._epoch + t_us

    def observe(self, t_us, host_time_s):
        """Feed one frame; returns its MCU time mapped to host seconds."""
        mcu_s = self.unwrap(t_us) * 1e-6
        self._offsets.append(host_time_s - mcu_s)
        return mcu_s + min(self._offsets)
