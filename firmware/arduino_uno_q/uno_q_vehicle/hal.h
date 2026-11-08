// Hardware boundary. The core (vehicle/safety/ultrasonic/imu) only talks to this;
// uno_q_vehicle.ino implements it on the board, host/ implements it for sim and tests.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "actuators.h"

struct I2cBus {
  virtual bool write_reg(uint8_t addr, uint8_t reg, uint8_t val) = 0;
  virtual bool read_regs(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t n) = 0;
  virtual bool probe(uint8_t addr) = 0;
  virtual ~I2cBus() {}
};

struct Hal {
  virtual uint32_t micros() = 0;
  virtual void us_trigger(uint8_t id) = 0;      // 10 µs trigger pulse
  virtual bool us_echo_level(uint8_t id) = 0;   // current echo pin level
  virtual bool us_assigned(uint8_t id) = 0;
  virtual bool estop_assigned() = 0;
  virtual bool estop_active() = 0;              // true = stop requested (or wire broken)
  virtual I2cBus& i2c() = 0;
  virtual void write_outputs(const Outputs& o) = 0;
  virtual void watchdog_kick() = 0;
  virtual bool hw_watchdog_verified() = 0;
  virtual bool outputs_enabled() = 0;
  virtual size_t link_read(char* buf, size_t cap) = 0;  // non-blocking, returns bytes read
  virtual void link_write(const char* buf, size_t n) = 0;
  virtual ~Hal() {}
};
