"""Write a short synthetic log so the Step 0 tools can be tried before the
hardware adapter exists. NOT a model of your car: just plausible-looking
streams, including a 'tap' event with a spike in the IMU and a VO result
that arrives late.

    python -m tools.make_demo_log logs/
"""
import sys

import numpy as np

from fusion.io import RunLogger
from fusion.messages import Event, ImuSample, UltrasonicReading, VoPose, WheelSpeed


def make(directory="logs", seconds=6.0, seed=0):
    rng = np.random.default_rng(seed)
    with RunLogger(directory, notes="synthetic demo, not real data", name="demo") as log:
        tap_t = 3.0
        for k in range(int(seconds * 500)):
            t = k / 500
            spike = 6.0 * np.exp(-((t - tap_t) / 0.01) ** 2)
            log.write(ImuSample(t=t,
                                gyro=tuple(rng.normal(0, 0.005, 3) + [0.0, spike * 0.1, 0.0]),
                                accel=tuple(rng.normal(0, 0.08, 3) + [spike, 0.0, 9.81])))
            if k % 25 == 0:
                log.write(WheelSpeed(t=t, ticks=0))
            if k % 33 == 0:   # VO result logged ~80 ms after its frame
                if t >= 0.08:
                    log.write(VoPose(t=t - 0.08, x=0.0, y=0.0, yaw=0.0, t_result=t))
            if k % 33 == 5:
                for name, r in (("front_center", 1.50), ("front_left", None)):
                    log.write(UltrasonicReading(t=t, sensor=name,
                                                range_m=None if r is None else r + rng.normal(0, 0.01)))
            if abs(t - tap_t) < 1e-9:
                log.write(Event(t=t, text="tap"))
    return log.path


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else "logs"))
