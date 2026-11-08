// High-level actuator values (per-mille) and their mapping to RC PWM pulse widths.
// The safety layer only ever produces Outputs; only Hal::write_outputs touches hardware.
#pragma once
#include <stdint.h>

struct Outputs {
  int16_t thr_pm = 0;  // 0..1000 forward
  int16_t brk_pm = 0;  // 0..1000
  int16_t str_pm = 0;  // -1000..1000, positive = left
  bool operator==(const Outputs& o) const {
    return thr_pm == o.thr_pm && brk_pm == o.brk_pm && str_pm == o.str_pm;
  }
};

struct PwmUs {
  uint16_t thr, brk, str;
};

// Linear map to the calibrated µs endpoints in config.h, clamped to those endpoints.
PwmUs to_pwm_us(const Outputs& o);
