#include "vehicle.h"
#include <stdio.h>
#include <string.h>
#include "config.h"

void Vehicle::begin(Hal& hal, SimHook sim_hook) {
  hal_ = &hal;
  sim_hook_ = sim_hook;
  sup_.set_estop_assigned(hal.estop_assigned());
  last_micros_ = hal.micros();
  mpu_.init(hal.i2c());
  itg_.init(hal.i2c());
  hal.write_outputs(sup_.outputs());  // rule 1: boot DISARMED with safe outputs
  char b[64];
  snprintf(b, sizeof b, "BOOT,%s,%s,%s", FW_VERSION, hal.outputs_enabled() ? "OUTPUTS_ENABLED" : "OUTPUTS_DISABLED",
           hal.hw_watchdog_verified() ? "HWWDT_ON" : "HWWDT_UNVERIFIED");
  send(b);
  send_status();
}

uint16_t Vehicle::faults() const {
  uint16_t f = sup_.faults();
  if (!hal_->outputs_enabled()) f |= fault::OUTPUTS_DISABLED_BUILD;
  if (!hal_->hw_watchdog_verified()) f |= fault::HW_WATCHDOG_UNVERIFIED;
  if (imu_fault_) f |= fault::IMU_FAULT;
  if (ultra.any_fault()) f |= fault::US_FAULT;
  return f;
}

void Vehicle::loop_once() {
  Hal& hal = *hal_;
  uint32_t m = hal.micros();
  now_us64_ += (uint32_t)(m - last_micros_);
  last_micros_ = m;
  uint32_t now = now_ms();

  sup_.sample_estop(hal.estop_assigned() && hal.estop_active());  // before messages, so ARM sees it

  char in[64];
  size_t n;
  while ((n = hal.link_read(in, sizeof in)) > 0) {
    for (size_t i = 0; i < n; ++i) {
      LineAssembler::Result r = rx_.feed(in[i]);
      if (r == LineAssembler::FRAME) {
        char body[MAX_LINE + 1];
        strncpy(body, rx_.body(), sizeof body - 1);
        body[sizeof body - 1] = '\0';
        dispatch(body);
      } else if (r == LineAssembler::BAD) {
        bad_frame();
      }
    }
  }

  sup_.step(now);

  UsReading us;
  if (ultra.step(hal, m, us)) {
    char b[64], d[12] = "-";
    if (us.dist_mm >= 0) snprintf(d, sizeof d, "%ld", (long)us.dist_mm);
    snprintf(b, sizeof b, "US,%lu,%u,%s,%s", (unsigned long)us.t_us, (unsigned)us.id, d, us_status_name(us.status));
    send(b);
  }

  if ((uint32_t)(now - last_imu_ms_) >= cfg::IMU_PERIOD_MS) {
    last_imu_ms_ = now;
    ImuReading a = mpu_.read(hal.i2c(), hal.micros());
    ImuReading g = itg_.read(hal.i2c(), hal.micros());
    imu_fault_ = a.status != ImuStatus::OK || g.status != ImuStatus::OK;
    send_imu("MPU6050", a);
    send_imu("ITG3200", g);
  }

  if (sup_.state() != last_state_ || (uint32_t)(now - last_status_ms_) >= cfg::STATUS_PERIOD_MS) send_status();

  hal.write_outputs(sup_.outputs());
  hal.watchdog_kick();  // rule 10: only after a complete iteration
}

void Vehicle::dispatch(char* body) {
  char* f[12];
  int n = split_fields(body, f, 12);
  if (n < 1) {
    bad_frame();
    return;
  }
  const char* t = f[0];
  if (sim_hook_ && strcmp(t, "SIM") == 0) {
    ack(0, sim_hook_(f, n));
    return;
  }
  uint32_t seq = 0;
  bool seq_ok = n >= 2 && parse_u32(f[1], seq);
  if (!seq_ok) seq = 0;
  uint32_t now = now_ms();

  if (strcmp(t, "CMD") == 0) {
    int32_t thr, brk, str;
    if (n != 5 || !seq_ok || !parse_i32(f[2], thr) || !parse_i32(f[3], brk) || !parse_i32(f[4], str)) {
      bad_frame();
      ack(seq, Ack::BAD_ARGS);
      return;
    }
    Ack a = sup_.on_cmd(seq, thr, brk, str, now);
    if (a != Ack::OK) ack(seq, a);  // CMD is acked only on error
    return;
  }

  static const char* const simple[] = {"ARM", "DISARM", "ESTOP", "RESET", "SCAN"};
  int k = -1;
  for (int i = 0; i < 5; ++i)
    if (strcmp(t, simple[i]) == 0) k = i;
  if (k < 0) {
    bad_frame();
    ack(seq, Ack::UNKNOWN_TYPE);
    return;
  }
  if (n != 2 || !seq_ok) {
    bad_frame();
    ack(seq, Ack::BAD_ARGS);
    return;
  }
  switch (k) {
    case 0: ack(seq, sup_.on_arm(now)); break;
    case 1: ack(seq, sup_.on_disarm()); break;
    case 2: ack(seq, sup_.on_estop_msg()); break;
    case 3: ack(seq, sup_.on_reset()); break;
    default:
      scan();
      ack(seq, Ack::OK);
  }
}

void Vehicle::bad_frame() {
  ++bad_frames_;
  sup_.on_bad_frame();
}

void Vehicle::send(const char* body) {
  char out[MAX_LINE + 8];
  size_t n = format_frame(body, out, sizeof out);
  if (n) hal_->link_write(out, n);
}

void Vehicle::ack(uint32_t seq, Ack a) {
  char b[64];
  snprintf(b, sizeof b, "ACK,%lu,%s,%s", (unsigned long)seq, a == Ack::OK ? "OK" : "ERR", ack_name(a));
  send(b);
}

void Vehicle::send_status() {
  uint32_t now = now_ms();
  last_status_ms_ = now;
  last_state_ = sup_.state();
  const Outputs& o = sup_.outputs();
  char seq[12] = "-";
  if (sup_.has_cmd()) snprintf(seq, sizeof seq, "%lu", (unsigned long)sup_.last_cmd_seq());
  char b[MAX_BODY + 1];
  snprintf(b, sizeof b, "STA,%lu,%s,%04X,%s,%u,%d,%d,%d,%lu", (unsigned long)last_micros_, state_name(sup_.state()),
           (unsigned)faults(), seq, (unsigned)sup_.cmd_age_ms(now), (int)o.thr_pm, (int)o.brk_pm, (int)o.str_pm,
           (unsigned long)bad_frames_);
  send(b);
}

void Vehicle::send_imu(const char* src, const ImuReading& r) {
  char b[MAX_BODY + 1];
  if (r.status != ImuStatus::OK) {
    snprintf(b, sizeof b, "IMU,%lu,%s,-,-,-,-,-,-,-,%s", (unsigned long)r.t_us, src, imu_status_name(r.status));
  } else if (r.has_accel) {
    snprintf(b, sizeof b, "IMU,%lu,%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,OK", (unsigned long)r.t_us, src, (long)r.accel[0],
             (long)r.accel[1], (long)r.accel[2], (long)r.gyro[0], (long)r.gyro[1], (long)r.gyro[2], (long)r.temp_cC);
  } else {
    snprintf(b, sizeof b, "IMU,%lu,%s,-,-,-,%ld,%ld,%ld,%ld,OK", (unsigned long)r.t_us, src, (long)r.gyro[0],
             (long)r.gyro[1], (long)r.gyro[2], (long)r.temp_cC);
  }
  send(b);
}

void Vehicle::scan() {
  I2cBus& bus = hal_->i2c();
  unsigned count = 0;
  char b[32];
  for (uint8_t a = 0x08; a <= 0x77; ++a) {
    if (!bus.probe(a)) continue;
    snprintf(b, sizeof b, "I2C,%02X,%s", (unsigned)a, i2c_ident(bus, a));
    send(b);
    ++count;
  }
  snprintf(b, sizeof b, "I2C,END,%u", count);
  send(b);
}
