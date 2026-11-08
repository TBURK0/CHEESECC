#include "safety.h"
#include "config.h"

const char* state_name(State s) {
  switch (s) {
    case State::ARMED: return "ARMED";
    case State::ESTOP: return "ESTOP";
    default: return "DISARMED";
  }
}

const char* ack_name(Ack a) {
  switch (a) {
    case Ack::OK: return "-";
    case Ack::ESTOP_ACTIVE: return "ESTOP_ACTIVE";
    case Ack::ESTOP_LATCHED: return "ESTOP_LATCHED";
    case Ack::ESTOP_PIN_UNASSIGNED: return "ESTOP_PIN_UNASSIGNED";
    case Ack::NOT_ARMED: return "NOT_ARMED";
    case Ack::ALREADY_ARMED: return "ALREADY_ARMED";
    case Ack::BAD_ARGS: return "BAD_ARGS";
    default: return "UNKNOWN_TYPE";
  }
}

// Rule 2: throttle 0, brake SAFE_STOP_BRAKE_PM, steering centred.
Outputs safe_outputs() {
  Outputs o;
  o.thr_pm = 0;
  o.brk_pm = (int16_t)cfg::SAFE_STOP_BRAKE_PM;
  o.str_pm = 0;
  return o;
}

static int32_t clampi(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

void SafetySupervisor::safe_stop(uint16_t sticky_bit) {
  sticky_ |= sticky_bit;
  if (state_ == State::ARMED) state_ = State::DISARMED;
  out_ = safe_outputs();
}

void SafetySupervisor::enter_estop() {
  state_ = State::ESTOP;
  out_ = safe_outputs();
}

// Rule 8: any active sample, any state, no debounce.
void SafetySupervisor::sample_estop(bool active) {
  estop_active_ = active;
  if (active && state_ != State::ESTOP) enter_estop();
}

// Rule 6: no accepted CMD for CMD_TIMEOUT_MS while ARMED -> safe stop.
void SafetySupervisor::step(uint32_t now_ms) {
  if (state_ == State::ARMED && (uint32_t)(now_ms - timer_ms_) >= cfg::CMD_TIMEOUT_MS) safe_stop(fault::CMD_TIMEOUT);
}

// Rule 3.
Ack SafetySupervisor::on_arm(uint32_t now_ms) {
  if (state_ == State::ARMED) return Ack::ALREADY_ARMED;
  if (state_ == State::ESTOP) return Ack::ESTOP_LATCHED;
  if (estop_active_) return Ack::ESTOP_ACTIVE;
  if (!estop_assigned_) return Ack::ESTOP_PIN_UNASSIGNED;
  state_ = State::ARMED;
  sticky_ = 0;
  timer_ms_ = now_ms;     // as if a command had just arrived
  out_ = safe_outputs();  // arming never moves anything by itself
  return Ack::OK;
}

// Rules 4 and 5.
Ack SafetySupervisor::on_cmd(uint32_t seq, int32_t thr, int32_t brk, int32_t str, uint32_t now_ms) {
  if (state_ != State::ARMED) return Ack::NOT_ARMED;
  bool in_range = thr >= 0 && thr <= 1000 && brk >= 0 && brk <= 1000 && str >= -1000 && str <= 1000;
  if (!in_range) {
    safe_stop(fault::INVALID_CMD);
    return Ack::BAD_ARGS;
  }
  Outputs o;
  o.brk_pm = (int16_t)clampi(brk, 0, cfg::MAX_BRAKE_PM);
  o.thr_pm = o.brk_pm > 0 ? 0 : (int16_t)clampi(thr, 0, cfg::MAX_THROTTLE_PM);
  o.str_pm = (int16_t)clampi(str, -cfg::MAX_STEER_PM, cfg::MAX_STEER_PM);
  out_ = o;
  timer_ms_ = last_cmd_ms_ = now_ms;
  last_cmd_seq_ = seq;
  has_cmd_ = true;
  return Ack::OK;
}

Ack SafetySupervisor::on_disarm() {
  if (state_ == State::ARMED) safe_stop(0);
  return Ack::OK;
}

Ack SafetySupervisor::on_estop_msg() {
  enter_estop();
  return Ack::OK;
}

// Rule 9: only out of ESTOP, only with the input released, and only to DISARMED.
Ack SafetySupervisor::on_reset() {
  if (state_ != State::ESTOP) return Ack::OK;
  if (estop_active_) return Ack::ESTOP_ACTIVE;
  state_ = State::DISARMED;
  out_ = safe_outputs();
  return Ack::OK;
}

// Rule 7.
void SafetySupervisor::on_bad_frame() {
  if (state_ == State::ARMED) safe_stop(fault::BAD_FRAME);
}

uint16_t SafetySupervisor::faults() const {
  uint16_t f = sticky_;
  if (estop_active_) f |= fault::ESTOP_INPUT;
  if (state_ == State::ESTOP) f |= fault::ESTOP_LATCHED;
  if (!estop_assigned_) f |= fault::ESTOP_PIN_UNASSIGNED;
  return f;
}

uint16_t SafetySupervisor::cmd_age_ms(uint32_t now_ms) const {
  if (!has_cmd_) return 65535;
  uint32_t d = now_ms - last_cmd_ms_;
  return d > 65535 ? 65535 : (uint16_t)d;
}
