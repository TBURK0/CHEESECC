// SafetySupervisor: the MCU-side safety authority (docs/serial_protocol.md -> Safety rules).
// Pure state machine: no hardware, time passed in, so every rule is unit-tested on the host.
#pragma once
#include <stdint.h>
#include "actuators.h"

enum class State : uint8_t { DISARMED, ARMED, ESTOP };
enum class Ack : uint8_t {
  OK, ESTOP_ACTIVE, ESTOP_LATCHED, ESTOP_PIN_UNASSIGNED, NOT_ARMED, ALREADY_ARMED, BAD_ARGS, UNKNOWN_TYPE
};
const char* state_name(State s);
const char* ack_name(Ack a);  // "-" for OK

namespace fault {
constexpr uint16_t ESTOP_INPUT = 0x0001;
constexpr uint16_t ESTOP_LATCHED = 0x0002;
constexpr uint16_t CMD_TIMEOUT = 0x0004;
constexpr uint16_t BAD_FRAME = 0x0008;
constexpr uint16_t INVALID_CMD = 0x0010;
constexpr uint16_t ESTOP_PIN_UNASSIGNED = 0x0020;
constexpr uint16_t OUTPUTS_DISABLED_BUILD = 0x0040;
constexpr uint16_t HW_WATCHDOG_UNVERIFIED = 0x0080;
constexpr uint16_t IMU_FAULT = 0x0100;
constexpr uint16_t US_FAULT = 0x0200;
}  // namespace fault

Outputs safe_outputs();

class SafetySupervisor {
 public:
  void set_estop_assigned(bool assigned) { estop_assigned_ = assigned; }
  void sample_estop(bool active);       // call every loop, before handling messages
  void step(uint32_t now_ms);           // command expiry
  Ack on_arm(uint32_t now_ms);
  Ack on_cmd(uint32_t seq, int32_t thr, int32_t brk, int32_t str, uint32_t now_ms);
  Ack on_disarm();
  Ack on_estop_msg();
  Ack on_reset();
  void on_bad_frame();

  State state() const { return state_; }
  const Outputs& outputs() const { return out_; }
  uint16_t faults() const;              // live + sticky bits owned by the supervisor
  bool has_cmd() const { return has_cmd_; }
  uint32_t last_cmd_seq() const { return last_cmd_seq_; }
  uint16_t cmd_age_ms(uint32_t now_ms) const;

 private:
  void safe_stop(uint16_t sticky_bit);
  void enter_estop();

  State state_ = State::DISARMED;
  Outputs out_ = safe_outputs();
  bool estop_assigned_ = false;
  bool estop_active_ = false;
  uint16_t sticky_ = 0;
  uint32_t timer_ms_ = 0;       // expiry reference (last CMD or ARM)
  uint32_t last_cmd_ms_ = 0;    // last accepted CMD, for cmd_age
  uint32_t last_cmd_seq_ = 0;
  bool has_cmd_ = false;
};
