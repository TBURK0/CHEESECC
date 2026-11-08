#include "imu.h"

const char* imu_status_name(ImuStatus s) {
  switch (s) {
    case ImuStatus::OK: return "OK";
    case ImuStatus::READ_FAIL: return "READ_FAIL";
    default: return "NOT_FOUND";
  }
}

static int32_t div_round(int64_t n, int64_t d) {
  return (int32_t)(n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d));
}
static int16_t be16(const uint8_t* p) { return (int16_t)(((uint16_t)p[0] << 8) | p[1]); }

// 1 °/s = 17453.2925 µrad/s  ->  ×174532925 / (LSB_per_dps × 10000)
int32_t mpu_accel_mm_s2(int16_t raw) { return div_round((int64_t)raw * 980665, 819200); }  // 9806.65 mm/s² / 8192
int32_t mpu_gyro_urad_s(int16_t raw) { return div_round((int64_t)raw * 174532925, 655000); }
int32_t mpu_temp_cC(int16_t raw) { return div_round((int64_t)raw * 100, 340) + 3653; }
int32_t itg_gyro_urad_s(int16_t raw) { return div_round((int64_t)raw * 174532925, 143750); }
int32_t itg_temp_cC(int16_t raw) { return div_round(((int64_t)raw + 13200) * 100, 280) + 3500; }

static bool read_u8(I2cBus& bus, uint8_t addr, uint8_t reg, uint8_t& v) { return bus.read_regs(addr, reg, &v, 1); }

static bool is_mpu_family(I2cBus& bus, uint8_t addr) {
  uint8_t v;
  return read_u8(bus, addr, 0x75, v) && (v == 0x68 || v == 0x70 || v == 0x71);
}

bool Mpu6050::init(I2cBus& bus) {
  uint8_t who;
  found_ = read_u8(bus, addr_, 0x75, who) && who == 0x68 &&
           bus.write_reg(addr_, 0x6B, 0x01) &&  // wake, PLL on gyro X
           bus.write_reg(addr_, 0x19, 9) &&     // 1 kHz / (1+9) = 100 Hz
           bus.write_reg(addr_, 0x1A, 0x03) &&  // DLPF 44 Hz
           bus.write_reg(addr_, 0x1B, 0x08) &&  // ±500 °/s
           bus.write_reg(addr_, 0x1C, 0x08);    // ±4 g
  return found_;
}

// ponytail: no re-init after READ_FAIL; reads just keep retrying each period. Add re-init if a bus glitch leaves the chip asleep.
ImuReading Mpu6050::read(I2cBus& bus, uint32_t t_us) {
  ImuReading r;
  r.t_us = t_us;
  r.has_accel = true;
  if (!found_) return r;
  uint8_t b[14];
  if (!bus.read_regs(addr_, 0x3B, b, 14)) {
    r.status = ImuStatus::READ_FAIL;
    return r;
  }
  for (int i = 0; i < 3; ++i) {
    r.accel[i] = mpu_accel_mm_s2(be16(b + 2 * i));
    r.gyro[i] = mpu_gyro_urad_s(be16(b + 8 + 2 * i));
  }
  r.temp_cC = mpu_temp_cC(be16(b + 6));
  r.status = ImuStatus::OK;
  return r;
}

bool Itg3200::init(I2cBus& bus) {
  uint8_t who;
  found_ = !is_mpu_family(bus, addr_) &&  // don't mistake an MPU at this address for the ITG
           read_u8(bus, addr_, 0x00, who) && (who & 0x7E) == 0x68 &&
           bus.write_reg(addr_, 0x3E, 0x01) &&  // PLL on gyro X
           bus.write_reg(addr_, 0x15, 9) &&     // 1 kHz / (1+9) = 100 Hz
           bus.write_reg(addr_, 0x16, 0x1B);    // ±2000 °/s, DLPF 42 Hz
  return found_;
}

ImuReading Itg3200::read(I2cBus& bus, uint32_t t_us) {
  ImuReading r;
  r.t_us = t_us;
  if (!found_) return r;
  uint8_t b[8];
  if (!bus.read_regs(addr_, 0x1B, b, 8)) {
    r.status = ImuStatus::READ_FAIL;
    return r;
  }
  r.temp_cC = itg_temp_cC(be16(b));
  for (int i = 0; i < 3; ++i) r.gyro[i] = itg_gyro_urad_s(be16(b + 2 + 2 * i));
  r.status = ImuStatus::OK;
  return r;
}

const char* i2c_ident(I2cBus& bus, uint8_t addr) {
  uint8_t v;
  if (read_u8(bus, addr, 0x75, v)) {
    if (v == 0x68) return "MPU6050";
    if (v == 0x70) return "MPU6500";
    if (v == 0x71) return "MPU9250";
  }
  if ((addr == 0x68 || addr == 0x69) && read_u8(bus, addr, 0x00, v) && (v & 0x7E) == 0x68) return "ITG3200";
  if ((addr == 0x53 || addr == 0x1D) && read_u8(bus, addr, 0x00, v) && v == 0xE5) return "ADXL345";
  if (addr == 0x1E) {
    uint8_t id[3];
    if (bus.read_regs(addr, 0x0A, id, 3) && id[0] == 'H' && id[1] == '4' && id[2] == '3') return "HMC5883L";
  }
  if (addr == 0x0D && read_u8(bus, addr, 0x0D, v) && v == 0xFF) return "QMC5883L";
  if (addr == 0x77 && read_u8(bus, addr, 0xD0, v) && v == 0x55) return "BMP180";
  return "UNKNOWN";
}
