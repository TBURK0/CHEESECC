"""Host-side command policy (no ROS). The MCU is the safety authority; this layer only
adds host-side limits and turns host-visible faults into explicit safe-stop requests."""
import math

from . import protocol


class BridgeLogic:
    def __init__(self, allow_arm=False, cmd_rate_hz=50.0, cmd_input_timeout_s=0.2,
                 telemetry_timeout_s=0.5, max_throttle=0.25, max_brake=1.0, max_steering=1.0):
        self.allow_arm = allow_arm
        self.cmd_period = 1.0 / cmd_rate_hz
        self.cmd_input_timeout_s = cmd_input_timeout_s
        self.telemetry_timeout_s = telemetry_timeout_s
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_steering = max_steering

        self.status = None          # last protocol.Status
        self.link_ok = False
        self._seq = 0
        self._input = None          # (throttle, brake, steering, t)
        self._status_t = None
        self._last_cmd_t = None
        self._stop_sent = False     # one DISARM per armed episode
        self._needs_rearm = False   # set on link loss; cleared only by an explicit arm request

    def _next(self, kind):
        self._seq += 1
        return protocol.simple_body(kind, self._seq)

    @property
    def mcu_armed(self):
        return self.link_ok and self.status is not None and self.status.state == 'ARMED'

    # ---- inputs ----

    def on_input(self, throttle, brake, steering, now):
        """Latest autonomy request. Returns an error string if rejected (and drops it,
        so a non-finite command is treated like a stale one: safe stop)."""
        if not all(math.isfinite(v) for v in (throttle, brake, steering)):
            self._input = None
            return 'non-finite command rejected'
        self._input = (min(max(throttle, 0.0), self.max_throttle),
                       min(max(brake, 0.0), self.max_brake),
                       min(max(steering, -self.max_steering), self.max_steering), now)
        return None

    def on_status(self, status, now):
        self.status = status
        self._status_t = now
        self.link_ok = True
        if status.state != 'ARMED':
            self._stop_sent = False

    def request(self, kind):
        """Operator request. Returns (accepted, reason, [bodies to send])."""
        if kind == 'ARM':
            if not self.allow_arm:
                return False, 'arming disabled (allow_arm is false)', []
            if not self.link_ok:
                return False, 'no telemetry from MCU', []
            self._needs_rearm = False
            self._stop_sent = False
        return True, '', [self._next(kind)]

    # ---- periodic ----

    def tick(self, now):
        out = []
        stale_link = self._status_t is None or now - self._status_t > self.telemetry_timeout_s
        if stale_link:
            if self.link_ok:  # transition: link just lost
                self.link_ok = False
                self._needs_rearm = True
                out.append(self._next('DISARM'))
            return out
        if self.status.state != 'ARMED':
            return out

        fresh = self._input is not None and now - self._input[3] <= self.cmd_input_timeout_s
        if self._needs_rearm or not fresh:
            if not self._stop_sent:
                self._stop_sent = True
                out.append(self._next('DISARM'))
            return out

        if self._last_cmd_t is None or now - self._last_cmd_t >= self.cmd_period:
            t, b, s, _ = self._input
            if b > 0.0:
                t = 0.0
            self._seq += 1
            out.append(protocol.cmd_body(self._seq, t, b, s))
            self._last_cmd_t = now
        return out
