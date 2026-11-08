// MPU-6050 (accel+gyro) and ITG-3200 (gyro) drivers, plus I2C scan identification.
// Models are UNVERIFIED: SCAN exists so the real boards can be identified on the bus.
// Output is in wire units, in the chip's own axes (mapping happens on the Jetson).
#pragma once
#include <stdint.h>
#include "config.h"
#include "hal.h"

enum class ImuStatus : uint8_t { OK, READ_FAIL, NOT_FOUND };
const char* imu_status_name(ImuStatus s);

struct ImuReading {
  uint32_t t_us = 0;  // start of the I2C burst read
  ImuStatus status = ImuStatus::NOT_FOUND;
  bool has_accel = false;
  int32_t accel[3] = {0, 0, 0};  // mm/s^2
  int32_t gyro[3] = {0, 0, 0};   // µrad/s
  int32_t temp_cC = 0;           // centi-°C
};

// Raw-count conversions (exposed for tests).
int32_t mpu_accel_mm_s2(int16_t raw);  // ±4 g, 8192 LSB/g
int32_t mpu_gyro_urad_s(int16_t raw);  // ±500 °/s, 65.5 LSB/(°/s)
int32_t mpu_temp_cC(int16_t raw);      // raw/340 + 36.53 °C
int32_t itg_gyro_urad_s(int16_t raw);  // ±2000 °/s, 14.375 LSB/(°/s)
int32_t itg_temp_cC(int16_t raw);      // 35 + (raw+13200)/280 °C

class Mpu6050 {
 public:
  explicit Mpu6050(uint8_t addr = cfg::MPU6050_ADDR) : addr_(addr) {}
  bool init(I2cBus& bus);  // false -> NOT_FOUND forever (no retry)
  ImuReading read(I2cBus& bus, uint32_t t_us);
  bool found() const { return found_; }

 private:
  uint8_t addr_;
  bool found_ = false;
};

class Itg3200 {
 public:
  explicit Itg3200(uint8_t addr = cfg::ITG3200_ADDR) : addr_(addr) {}
  bool init(I2cBus& bus);
  ImuReading read(I2cBus& bus, uint32_t t_us);
  bool found() const { return found_; }

 private:
  uint8_t addr_;
  bool found_ = false;
};

// Identifies whatever answers at addr (assumes probe already succeeded).
const char* i2c_ident(I2cBus& bus, uint8_t addr);
