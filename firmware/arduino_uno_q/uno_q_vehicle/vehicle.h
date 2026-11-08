// Vehicle: ties link, safety, sensors and outputs together. One loop_once() per main-loop pass.
#pragma once
#include "hal.h"
#include "imu.h"
#include "protocol.h"
#include "safety.h"
#include "ultrasonic.h"

class Vehicle {
 public:
  // Host-simulator hook for "SIM,..." bodies. Real firmware passes nullptr, so SIM is UNKNOWN_TYPE.
  using SimHook = Ack (*)(char** fields, int n);

  void begin(Hal& hal, SimHook sim_hook = nullptr);
  void loop_once();

  Ultrasonic ultra;  // public so the echo ISRs / simulator can feed edges
  const SafetySupervisor& safety() const { return sup_; }
  uint16_t faults() const;

 private:
  void dispatch(char* body);
  void bad_frame();
  void send(const char* body);
  void ack(uint32_t seq, Ack a);
  void send_status();
  void send_imu(const char* src, const ImuReading& r);
  void scan();
  uint32_t now_ms() const { return (uint32_t)(now_us64_ / 1000u); }

  Hal* hal_ = nullptr;
  SimHook sim_hook_ = nullptr;
  SafetySupervisor sup_;
  LineAssembler rx_;
  Mpu6050 mpu_;
  Itg3200 itg_;
  uint64_t now_us64_ = 0;  // unwrapped micros(), so ms timers survive the 71-minute wrap
  uint32_t last_micros_ = 0;
  uint32_t last_status_ms_ = 0;
  uint32_t last_imu_ms_ = 0;
  State last_state_ = State::DISARMED;
  uint32_t bad_frames_ = 0;
  bool imu_fault_ = false;
};
