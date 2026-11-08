"""Gyro bias from N stationary samples. ponytail: accel bias not implemented (needs the car
level and a known gravity axis); add if the EKF needs it."""
import statistics


class GyroBiasCalibrator:
    def __init__(self, samples=200, max_std=0.02):
        self.samples = samples
        self.max_std = max_std
        self.bias = None          # (bx, by, bz) rad/s once calibrated
        self.message = 'calibrating'
        self._buf = []

    @property
    def calibrated(self):
        return self.bias is not None

    def restart(self):
        self.bias = None
        self.message = 'calibrating'
        self._buf = []

    def add(self, gyro):
        """Feed one rad/s sample. A rejected attempt starts over automatically."""
        if self.calibrated:
            return
        self._buf.append(gyro)
        if len(self._buf) < self.samples:
            return
        axes = list(zip(*self._buf))
        stds = [statistics.pstdev(a) for a in axes]
        self._buf = []
        if max(stds) > self.max_std:
            self.message = (f'vehicle moving (gyro std {max(stds):.3f} > {self.max_std} rad/s), '
                            'retrying; keep the car still')
            return
        self.bias = tuple(statistics.fmean(a) for a in axes)
        self.message = 'calibrated'

    def correct(self, gyro):
        return tuple(g - b for g, b in zip(gyro, self.bias))
