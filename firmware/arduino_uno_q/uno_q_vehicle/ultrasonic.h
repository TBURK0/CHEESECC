// Four trigger/echo ultrasonics (HC-SR04-style, UNVERIFIED model), fired one at a time.
#pragma once
#include <stdint.h>
#include "hal.h"

enum class UsStatus : uint8_t { OK, TOO_CLOSE, TOO_FAR, TIMEOUT, NO_ECHO, STUCK_HIGH, UNASSIGNED };
const char* us_status_name(UsStatus s);

struct UsReading {
  uint32_t t_us;    // trigger time
  uint8_t id;       // 0=FL 1=FR 2=RL 3=RR
  int32_t dist_mm;  // -1 = not measured ("-" on the wire)
  UsStatus status;
};

class Ultrasonic {
 public:
  // Called from the echo-pin ISR (or the simulator). ISR-safe: only writes volatile fields.
  void on_echo_edge(uint8_t id, bool high, uint32_t t_us);
  // Run every loop. Returns true when a slot finished and `out` holds its reading.
  bool step(Hal& hal, uint32_t now_us, UsReading& out);
  // US_FAULT: a sensor's last status was NO_ECHO, STUCK_HIGH or UNASSIGNED.
  bool any_fault() const;

 private:
  static UsReading classify(uint8_t id, uint32_t t_trig, uint32_t echo_us);

  volatile uint8_t active_id_ = 255;
  volatile uint32_t rise_us_ = 0, fall_us_ = 0;
  volatile bool rise_seen_ = false, fall_seen_ = false;

  bool active_ = false, started_ = false;
  uint8_t idx_ = 0;
  uint32_t slot_start_us_ = 0, trig_us_ = 0;
  UsStatus last_[4] = {UsStatus::OK, UsStatus::OK, UsStatus::OK, UsStatus::OK};
};
