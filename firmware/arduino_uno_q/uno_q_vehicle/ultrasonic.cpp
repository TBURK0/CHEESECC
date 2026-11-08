#include "ultrasonic.h"
#include "config.h"

const char* us_status_name(UsStatus s) {
  switch (s) {
    case UsStatus::OK: return "OK";
    case UsStatus::TOO_CLOSE: return "TOO_CLOSE";
    case UsStatus::TOO_FAR: return "TOO_FAR";
    case UsStatus::TIMEOUT: return "TIMEOUT";
    case UsStatus::NO_ECHO: return "NO_ECHO";
    case UsStatus::STUCK_HIGH: return "STUCK_HIGH";
    default: return "UNASSIGNED";
  }
}

void Ultrasonic::on_echo_edge(uint8_t id, bool high, uint32_t t_us) {
  if (id != active_id_) return;  // edges from a sensor that isn't being measured are ignored
  if (high) {
    if (!rise_seen_) {
      rise_us_ = t_us;
      rise_seen_ = true;
    }
  } else if (rise_seen_ && !fall_seen_) {
    fall_us_ = t_us;
    fall_seen_ = true;
  }
}

UsReading Ultrasonic::classify(uint8_t id, uint32_t t_trig, uint32_t echo_us) {
  UsReading r{t_trig, id, -1, UsStatus::TIMEOUT};
  if (echo_us > cfg::US_ECHO_MAX_US) return r;
  r.dist_mm = (int32_t)(echo_us * 343u / 2000u);  // echo_us <= 30000 -> no overflow
  r.status = r.dist_mm < cfg::US_MIN_MM ? UsStatus::TOO_CLOSE
           : r.dist_mm > cfg::US_MAX_MM ? UsStatus::TOO_FAR
                                        : UsStatus::OK;
  return r;
}

// ponytail: one sensor per 60 ms slot = ~4.2 Hz each; firing diagonal pairs (FL+RR, FR+RL) together is the upgrade path if that's too slow.
bool Ultrasonic::step(Hal& hal, uint32_t now_us, UsReading& out) {
  if (active_) {
    uint8_t id = active_id_;
    bool done = true;
    if (fall_seen_) {
      out = classify(id, trig_us_, fall_us_ - rise_us_);  // uint32 subtraction is wrap-safe
    } else if (!rise_seen_ && (uint32_t)(now_us - trig_us_) > cfg::US_ECHO_START_TIMEOUT_US) {
      out = UsReading{trig_us_, id, -1, UsStatus::NO_ECHO};
    } else if (rise_seen_ && (uint32_t)(now_us - rise_us_) > cfg::US_ECHO_MAX_US) {
      out = UsReading{trig_us_, id, -1, UsStatus::TIMEOUT};
    } else {
      done = false;
    }
    if (!done) return false;
    active_ = false;
    active_id_ = 255;
    last_[id] = out.status;
    return true;
  }

  if (started_ && (uint32_t)(now_us - slot_start_us_) < cfg::US_SLOT_MS * 1000u) return false;
  started_ = true;
  slot_start_us_ = now_us;
  uint8_t id = cfg::US_ORDER[idx_];
  idx_ = (uint8_t)((idx_ + 1) % 4);

  out = UsReading{now_us, id, -1, UsStatus::UNASSIGNED};
  if (!hal.us_assigned(id)) {
    last_[id] = out.status;
    return true;
  }
  if (hal.us_echo_level(id)) {
    out.status = UsStatus::STUCK_HIGH;
    last_[id] = out.status;
    return true;
  }
  active_id_ = 255;  // block the ISR while resetting
  rise_seen_ = fall_seen_ = false;
  active_id_ = id;
  trig_us_ = now_us;
  active_ = true;
  hal.us_trigger(id);
  return false;
}

bool Ultrasonic::any_fault() const {
  for (UsStatus s : last_)
    if (s == UsStatus::NO_ECHO || s == UsStatus::STUCK_HIGH || s == UsStatus::UNASSIGNED) return true;
  return false;
}
