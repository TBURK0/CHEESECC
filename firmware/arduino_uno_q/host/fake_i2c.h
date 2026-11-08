// Register-level fake I2C bus shared by the simulator and the tests, so the real
// imu.cpp drivers and SCAN code run against it unchanged.
#pragma once
#include <functional>
#include "hal.h"

struct FakeI2c : I2cBus {
  struct Dev {
    bool present = false;
    bool fail = false;  // all transactions NACK while true
    uint8_t regs[256] = {};
  };
  Dev dev[128];
  std::function<void(uint8_t addr, uint8_t reg)> on_read;  // lets the sim refresh data registers

  bool ok(uint8_t a) const { return a < 128 && dev[a].present && !dev[a].fail; }
  bool write_reg(uint8_t a, uint8_t r, uint8_t v) override {
    if (!ok(a)) return false;
    dev[a].regs[r] = v;
    return true;
  }
  bool read_regs(uint8_t a, uint8_t r, uint8_t* buf, uint8_t n) override {
    if (!ok(a)) return false;
    if (on_read) on_read(a, r);
    for (uint8_t i = 0; i < n; ++i) buf[i] = dev[a].regs[(uint8_t)(r + i)];
    return true;
  }
  bool probe(uint8_t a) override { return ok(a); }

  void put16(uint8_t a, uint8_t reg, int16_t v) {
    dev[a].regs[reg] = (uint8_t)((uint16_t)v >> 8);
    dev[a].regs[(uint8_t)(reg + 1)] = (uint8_t)(v & 0xFF);
  }
  Dev& add(uint8_t a) {
    dev[a] = Dev();
    dev[a].present = true;
    return dev[a];
  }
  void add_mpu6050(uint8_t a) { add(a).regs[0x75] = 0x68; }
  void add_itg3200(uint8_t a) { add(a).regs[0x00] = 0x68; }  // WHO_AM_I ID bits; reg 0x75 reads 0
};
