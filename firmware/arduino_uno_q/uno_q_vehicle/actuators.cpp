#include "actuators.h"
#include "config.h"

static int32_t clampi(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// from + pm/1000 * (to - from), clamped between from and to.
static uint16_t lerp_us(int32_t pm, int32_t from, int32_t to) {
  pm = clampi(pm, 0, 1000);
  int32_t us = from + pm * (to - from) / 1000;
  return (uint16_t)clampi(us, from < to ? from : to, from < to ? to : from);
}

PwmUs to_pwm_us(const Outputs& o) {
  PwmUs p;
  p.thr = lerp_us(o.thr_pm, cfg::THROTTLE_NEUTRAL_US, cfg::THROTTLE_MAX_US);
  p.brk = lerp_us(o.brk_pm, cfg::BRAKE_RELEASE_US, cfg::BRAKE_FULL_US);
  p.str = o.str_pm >= 0 ? lerp_us(o.str_pm, cfg::STEER_CENTER_US, cfg::STEER_LEFT_US)
                        : lerp_us(-(int32_t)o.str_pm, cfg::STEER_CENTER_US, cfg::STEER_RIGHT_US);
  return p;
}
