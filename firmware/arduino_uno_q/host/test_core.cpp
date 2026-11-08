// Host unit tests for the firmware core. No framework: run ./test_core, nonzero exit on failure.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "config.h"
#include "fake_i2c.h"
#include "vehicle.h"

static int failures = 0;
#define CHECK(c)                                                       \
  do {                                                                 \
    if (!(c)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

static std::string frame(const char* body) {
  char out[160];
  size_t n = format_frame(body, out, sizeof out);
  return std::string(out, n);
}

// Feeds a whole string; returns the last non-NONE result and the last body.
static LineAssembler::Result feed_all(LineAssembler& a, const std::string& s, std::string* body = nullptr) {
  LineAssembler::Result last = LineAssembler::NONE;
  for (char c : s) {
    LineAssembler::Result r = a.feed(c);
    if (r != LineAssembler::NONE) {
      last = r;
      if (r == LineAssembler::FRAME && body) *body = a.body();
    }
  }
  return last;
}

static bool safe(const Outputs& o) { return o == safe_outputs(); }

// ---------------- protocol ----------------
static void test_protocol() {
  CHECK(crc16_ccitt_false("123456789", 9) == 0x29B1);

  LineAssembler a;
  std::string body;
  CHECK(feed_all(a, "garbage" + frame("ARM,1"), &body) == LineAssembler::FRAME);
  CHECK(body == "ARM,1");
  std::string f = frame("ARM,1");
  CHECK(f.size() == 12 && f[0] == '$' && f.back() == '\n');
  CHECK(feed_all(a, f.substr(0, f.size() - 1) + "\r\n", &body) == LineAssembler::FRAME);  // \r ignored

  std::string bad = f;
  bad[bad.size() - 2] = bad[bad.size() - 2] == '0' ? '1' : '0';
  CHECK(feed_all(a, bad) == LineAssembler::BAD);  // CRC mismatch

  std::string lower = frame("DISARM,7");  // CRC 2 hex letters? force lowercase if any letter
  bool has_letter = false;
  for (size_t i = lower.size() - 5; i < lower.size() - 1; ++i)
    if (lower[i] >= 'A' && lower[i] <= 'F') { lower[i] = (char)(lower[i] - 'A' + 'a'); has_letter = true; }
  if (has_letter) CHECK(feed_all(a, lower) == LineAssembler::BAD);

  CHECK(feed_all(a, "$ARM,1\n") == LineAssembler::BAD);  // no '*'
  std::string longline = "$" + std::string(130, 'A') + "*0000\n";
  int bads = 0;
  for (char c : longline) bads += a.feed(c) == LineAssembler::BAD;
  CHECK(bads == 1);  // overlength -> exactly one bad frame
  CHECK(feed_all(a, frame("ARM,2"), &body) == LineAssembler::FRAME && body == "ARM,2");  // recovers
  CHECK(feed_all(a, "$ARM,1*" + frame("ARM,3")) == LineAssembler::FRAME);  // truncated then good
  char big[150];
  std::memset(big, 'A', 121);
  big[121] = '\0';
  char out[200];
  CHECK(format_frame(big, out, sizeof out) == 0);  // body > 120 refused

  uint32_t u;
  int32_t i;
  CHECK(!parse_u32("12x", u) && !parse_u32("", u) && !parse_u32("-1", u) && !parse_u32("4294967296", u));
  CHECK(parse_u32("4294967295", u) && u == 4294967295u);
  CHECK(parse_i32("-2147483648", i) && i == INT32_MIN);
  CHECK(!parse_i32("2147483648", i) && !parse_i32("-", i) && !parse_i32("+5", i) && !parse_i32("1.5", i));
  CHECK(parse_i32("-1000", i) && i == -1000);

  char s[] = "CMD,1,2,3,4";
  char* fl[12];
  CHECK(split_fields(s, fl, 12) == 5 && std::strcmp(fl[4], "4") == 0);
  char s2[] = "a,b,c";
  CHECK(split_fields(s2, fl, 2) == -1);
}

// ---------------- safety ----------------
static void test_safety() {
  SafetySupervisor s;
  CHECK(s.state() == State::DISARMED && safe(s.outputs()));
  CHECK(s.on_arm(0) == Ack::ESTOP_PIN_UNASSIGNED && s.state() == State::DISARMED);
  CHECK(s.faults() & fault::ESTOP_PIN_UNASSIGNED);

  s.set_estop_assigned(true);
  s.sample_estop(true);
  CHECK(s.state() == State::ESTOP && s.on_arm(0) != Ack::OK);

  // estop -> reset refused while active; release -> still ESTOP; DISARM stays ESTOP; RESET -> DISARMED
  CHECK(s.on_reset() == Ack::ESTOP_ACTIVE && s.state() == State::ESTOP);
  s.sample_estop(false);
  CHECK(s.state() == State::ESTOP && (s.faults() & fault::ESTOP_LATCHED) && !(s.faults() & fault::ESTOP_INPUT));
  CHECK(s.on_disarm() == Ack::OK && s.state() == State::ESTOP);
  CHECK(s.on_cmd(1, 100, 0, 0, 0) == Ack::NOT_ARMED && safe(s.outputs()));
  CHECK(s.on_reset() == Ack::OK && s.state() == State::DISARMED);

  // not armed: CMD ignored even if in range
  CHECK(s.on_cmd(1, 100, 0, 0, 0) == Ack::NOT_ARMED && safe(s.outputs()));
  CHECK(s.on_cmd(1, 5000, 0, 0, 0) == Ack::NOT_ARMED && s.state() == State::DISARMED);

  // arm never moves anything by itself
  CHECK(s.on_arm(1000) == Ack::OK && s.state() == State::ARMED && safe(s.outputs()));
  CHECK(s.on_arm(1000) == Ack::ALREADY_ARMED);
  // ARM starts the timer: 199 ms ok, 200 ms -> timeout
  s.step(1199);
  CHECK(s.state() == State::ARMED);
  s.step(1200);
  CHECK(s.state() == State::DISARMED && (s.faults() & fault::CMD_TIMEOUT) && safe(s.outputs()));

  // sticky cleared on ARM; clamping
  CHECK(s.on_arm(2000) == Ack::OK && !(s.faults() & fault::CMD_TIMEOUT));
  CHECK(s.on_cmd(5, 1000, 0, 1000, 2010) == Ack::OK);
  CHECK(s.outputs().thr_pm == cfg::MAX_THROTTLE_PM && s.outputs().brk_pm == 0 && s.outputs().str_pm == 1000);
  CHECK(s.last_cmd_seq() == 5 && s.cmd_age_ms(2015) == 5);
  CHECK(s.on_cmd(6, 500, 300, -1000, 2020) == Ack::OK);  // brake forces throttle 0
  CHECK(s.outputs().thr_pm == 0 && s.outputs().brk_pm == 300 && s.outputs().str_pm == -1000);
  s.step(2219);
  CHECK(s.state() == State::ARMED);
  s.step(2220);
  CHECK(s.state() == State::DISARMED && (s.faults() & fault::CMD_TIMEOUT));

  // out of wire range while armed -> INVALID_CMD safe stop
  s.on_arm(3000);
  s.on_cmd(7, 200, 0, 0, 3001);
  CHECK(s.on_cmd(8, 1001, 0, 0, 3002) == Ack::BAD_ARGS);
  CHECK(s.state() == State::DISARMED && (s.faults() & fault::INVALID_CMD) && safe(s.outputs()));
  s.on_arm(3100);
  CHECK(s.on_cmd(9, 0, 0, -1001, 3101) == Ack::BAD_ARGS && s.state() == State::DISARMED);
  s.on_arm(3200);
  CHECK(s.on_cmd(9, 0, -1, 0, 3201) == Ack::BAD_ARGS && s.state() == State::DISARMED);

  // bad frame: disarms when armed, harmless otherwise
  s.on_bad_frame();
  CHECK(s.state() == State::DISARMED);
  s.on_arm(4000);
  s.on_cmd(10, 200, 0, 0, 4001);
  s.on_bad_frame();
  CHECK(s.state() == State::DISARMED && (s.faults() & fault::BAD_FRAME) && safe(s.outputs()));

  // estop input from ARMED; software ESTOP
  s.on_arm(5000);
  s.on_cmd(11, 200, 0, 300, 5001);
  s.sample_estop(true);
  CHECK(s.state() == State::ESTOP && safe(s.outputs()));
  s.sample_estop(false);
  s.on_reset();
  s.on_arm(6000);
  CHECK(s.on_estop_msg() == Ack::OK && s.state() == State::ESTOP);
  CHECK(s.on_reset() == Ack::OK && s.state() == State::DISARMED);  // never straight to ARMED

  // DISARM while armed
  s.on_arm(7000);
  s.on_cmd(12, 200, 0, 0, 7001);
  CHECK(s.on_disarm() == Ack::OK && s.state() == State::DISARMED && safe(s.outputs()));

  // randomized sweep: outputs always within limits, safe whenever not ARMED
  SafetySupervisor r;
  r.set_estop_assigned(true);
  uint32_t seed = 12345, t = 0;
  auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 8) & 0xFFFFFF; };
  for (int k = 0; k < 200000; ++k) {
    t += rnd() % 60;
    switch (rnd() % 10) {
      case 0: r.on_arm(t); break;
      case 1: r.sample_estop(rnd() % 50 == 0); break;
      case 2: r.on_reset(); break;
      case 3: if (rnd() % 20 == 0) r.on_bad_frame(); break;
      case 4: if (rnd() % 20 == 0) r.on_disarm(); break;
      default: r.on_cmd(k, (int32_t)(rnd() % 2400) - 200, (int32_t)(rnd() % 2400) - 1200, (int32_t)(rnd() % 2400) - 1200, t);
    }
    r.step(t);
    const Outputs& o = r.outputs();
    bool ok = o.thr_pm >= 0 && o.thr_pm <= cfg::MAX_THROTTLE_PM && o.brk_pm >= 0 && o.brk_pm <= cfg::MAX_BRAKE_PM &&
              o.str_pm >= -cfg::MAX_STEER_PM && o.str_pm <= cfg::MAX_STEER_PM && !(o.brk_pm > 0 && o.thr_pm > 0) &&
              (r.state() == State::ARMED || safe(o));
    if (!ok) {
      CHECK(ok);
      break;
    }
  }
}

// ---------------- actuators ----------------
static void test_actuators() {
  Outputs o;
  PwmUs p = to_pwm_us(o);
  CHECK(p.thr == 1500 && p.brk == 1000 && p.str == 1500);
  o.thr_pm = 250;
  o.brk_pm = 1000;
  o.str_pm = 1000;
  p = to_pwm_us(o);
  CHECK(p.thr == 1625 && p.brk == 2000 && p.str == 2000);
  o.str_pm = -1000;
  CHECK(to_pwm_us(o).str == 1000);
  o.thr_pm = 5000;
  o.str_pm = -5000;
  p = to_pwm_us(o);
  CHECK(p.thr == 2000 && p.str == 1000);  // clamped to calibrated endpoints
}

// ---------------- fake HAL ----------------
struct FakeHal : Hal {
  uint32_t t = 0;
  bool assigned[4] = {true, true, true, true};
  bool level[4] = {};
  std::vector<uint8_t> triggers;
  bool estop_pin = true, estop = false;
  FakeI2c bus;
  Outputs last;
  std::string in, out;

  uint32_t micros() override { return t; }
  void us_trigger(uint8_t id) override { triggers.push_back(id); }
  bool us_echo_level(uint8_t id) override { return level[id]; }
  bool us_assigned(uint8_t id) override { return assigned[id]; }
  bool estop_assigned() override { return estop_pin; }
  bool estop_active() override { return estop; }
  I2cBus& i2c() override { return bus; }
  void write_outputs(const Outputs& o) override { last = o; }
  void watchdog_kick() override {}
  bool hw_watchdog_verified() override { return false; }
  bool outputs_enabled() override { return false; }
  size_t link_read(char* buf, size_t cap) override {
    size_t n = in.size() < cap ? in.size() : cap;
    std::memcpy(buf, in.data(), n);
    in.erase(0, n);
    return n;
  }
  void link_write(const char* buf, size_t n) override { out.append(buf, n); }

  // All CRC-valid bodies written so far; any invalid output line is a failure.
  std::vector<std::string> bodies() {
    std::vector<std::string> v;
    LineAssembler a;
    for (char c : out) {
      LineAssembler::Result r = a.feed(c);
      CHECK(r != LineAssembler::BAD);
      if (r == LineAssembler::FRAME) v.push_back(a.body());
    }
    out.clear();
    return v;
  }
};

// Starts one slot at t0, optional rise/fall offsets (µs after trigger), runs until a reading pops.
static UsReading run_slot(FakeHal& h, Ultrasonic& u, uint32_t t0, long rise, long fall) {
  UsReading r{};
  h.t = t0;
  if (u.step(h, h.t, r)) return r;  // UNASSIGNED / STUCK_HIGH finish immediately
  uint8_t id = h.triggers.back();
  for (uint32_t dt = 1; dt < 100000; ++dt) {
    h.t = t0 + dt;
    if (rise >= 0 && dt == (uint32_t)rise) u.on_echo_edge(id, true, h.t);
    if (fall >= 0 && dt == (uint32_t)fall) u.on_echo_edge(id, false, h.t);
    if (u.step(h, h.t, r)) return r;
  }
  CHECK(false);
  return r;
}

static void test_ultrasonic() {
  FakeHal h;
  {
    Ultrasonic u;
    UsReading r;
    for (int k = 0; k < 4; ++k) {
      h.t = (uint32_t)k * 60000u;
      u.step(h, h.t, r);
      h.t += 5001;
      u.step(h, h.t, r);  // NO_ECHO closes the slot
    }
    CHECK(h.triggers.size() == 4 && h.triggers[0] == 0 && h.triggers[1] == 3 && h.triggers[2] == 1 &&
          h.triggers[3] == 2);
    CHECK(u.any_fault());
    h.t = 3 * 60000u + 30000;  // slot not over yet: no new trigger
    CHECK(!u.step(h, h.t, r) && h.triggers.size() == 4);
  }
  Ultrasonic u1;
  UsReading r = run_slot(h, u1, 1000, 450, 450 + 5831);
  CHECK(r.status == UsStatus::OK && r.dist_mm == 1000 && r.id == 0 && r.t_us == 1000);
  Ultrasonic u2;
  r = run_slot(h, u2, 0, 450, 550);  // 100 µs -> 17 mm
  CHECK(r.status == UsStatus::TOO_CLOSE && r.dist_mm == 17);
  Ultrasonic u3;
  r = run_slot(h, u3, 0, 450, 450 + 25000);  // 4287 mm
  CHECK(r.status == UsStatus::TOO_FAR && r.dist_mm == 4287);
  Ultrasonic u4;
  r = run_slot(h, u4, 0, 450, -1);
  CHECK(r.status == UsStatus::TIMEOUT && r.dist_mm == -1 && h.t == 450 + 30001);
  Ultrasonic u5;
  r = run_slot(h, u5, 0, -1, -1);
  CHECK(r.status == UsStatus::NO_ECHO && h.t == 5001);
  Ultrasonic u6;
  h.level[0] = true;
  size_t ntrig = h.triggers.size();
  r = run_slot(h, u6, 0, -1, -1);
  CHECK(r.status == UsStatus::STUCK_HIGH && h.triggers.size() == ntrig);  // never triggered
  h.level[0] = false;
  Ultrasonic u7;
  h.assigned[0] = false;
  r = run_slot(h, u7, 0, -1, -1);
  CHECK(r.status == UsStatus::UNASSIGNED && r.dist_mm == -1 && u7.any_fault());
  h.assigned[0] = true;
  Ultrasonic u8;  // micros() wraps between trigger and echo
  r = run_slot(h, u8, 0xFFFFFF00u, 450, 450 + 5831);
  CHECK(r.status == UsStatus::OK && r.dist_mm == 1000 && !u8.any_fault());
  Ultrasonic u9;  // edges from another sensor are ignored
  h.t = 0;
  u9.step(h, 0, r);
  u9.on_echo_edge(2, true, 100);
  u9.on_echo_edge(2, false, 200);
  h.t = 5001;
  CHECK(u9.step(h, h.t, r) && r.status == UsStatus::NO_ECHO);
}

// ---------------- IMU ----------------
static void test_imu() {
  CHECK(mpu_accel_mm_s2(8192) == 9807 && mpu_accel_mm_s2(-8192) == -9807);
  CHECK(mpu_gyro_urad_s(655) == 174533);  // 10 °/s
  CHECK(mpu_temp_cC(-3920) == 2500);
  CHECK(itg_gyro_urad_s(2875) == 3490659);  // 200 °/s
  CHECK(itg_temp_cC(-16000) == 2500);

  FakeI2c bus;
  bus.add_mpu6050(0x68);
  bus.add_itg3200(0x69);
  Mpu6050 mpu;
  Itg3200 itg;
  CHECK(mpu.init(bus) && itg.init(bus));
  auto& m = bus.dev[0x68].regs;
  CHECK(m[0x6B] == 0x01 && m[0x19] == 9 && m[0x1A] == 0x03 && m[0x1B] == 0x08 && m[0x1C] == 0x08);
  auto& g = bus.dev[0x69].regs;
  CHECK(g[0x3E] == 0x01 && g[0x15] == 9 && g[0x16] == 0x1B);

  bus.put16(0x68, 0x3B, 0);
  bus.put16(0x68, 0x3D, -8192);
  bus.put16(0x68, 0x3F, 8192);
  bus.put16(0x68, 0x41, -3920);
  bus.put16(0x68, 0x43, 655);
  bus.put16(0x68, 0x45, -655);
  bus.put16(0x68, 0x47, 0);
  ImuReading a = mpu.read(bus, 42);
  CHECK(a.status == ImuStatus::OK && a.has_accel && a.t_us == 42);
  CHECK(a.accel[0] == 0 && a.accel[1] == -9807 && a.accel[2] == 9807 && a.temp_cC == 2500);
  CHECK(a.gyro[0] == 174533 && a.gyro[1] == -174533 && a.gyro[2] == 0);

  bus.put16(0x69, 0x1B, -16000);
  bus.put16(0x69, 0x1D, 2875);
  bus.put16(0x69, 0x1F, -2875);
  bus.put16(0x69, 0x21, 0);
  ImuReading b = itg.read(bus, 7);
  CHECK(b.status == ImuStatus::OK && !b.has_accel && b.gyro[0] == 3490659 && b.gyro[1] == -3490659 && b.temp_cC == 2500);

  bus.dev[0x68].fail = true;
  CHECK(mpu.read(bus, 0).status == ImuStatus::READ_FAIL);
  bus.dev[0x68].fail = false;
  CHECK(mpu.read(bus, 0).status == ImuStatus::OK);  // recovers

  FakeI2c bad;
  bad.add_mpu6050(0x68);
  bad.dev[0x68].regs[0x75] = 0x12;
  Mpu6050 m2;
  CHECK(!m2.init(bad) && m2.read(bad, 0).status == ImuStatus::NOT_FOUND);
  Itg3200 i2;
  CHECK(!i2.init(bad) && i2.read(bad, 0).status == ImuStatus::NOT_FOUND);  // nothing at 0x69
  FakeI2c both;  // an MPU at 0x69 must not be taken for the ITG
  both.add_mpu6050(0x69);
  both.dev[0x69].regs[0x00] = 0x68;
  Itg3200 i3;
  CHECK(!i3.init(both));

  FakeI2c s;
  s.add_mpu6050(0x68);
  s.add_itg3200(0x69);
  s.add(0x53).regs[0x00] = 0xE5;
  FakeI2c::Dev& hmc = s.add(0x1E);
  hmc.regs[0x0A] = 'H';
  hmc.regs[0x0B] = '4';
  hmc.regs[0x0C] = '3';
  s.add(0x0D).regs[0x0D] = 0xFF;
  s.add(0x77).regs[0xD0] = 0x55;
  s.add(0x40);
  s.add(0x6A).regs[0x75] = 0x70;
  CHECK(!std::strcmp(i2c_ident(s, 0x68), "MPU6050"));
  CHECK(!std::strcmp(i2c_ident(s, 0x69), "ITG3200"));
  CHECK(!std::strcmp(i2c_ident(s, 0x53), "ADXL345"));
  CHECK(!std::strcmp(i2c_ident(s, 0x1E), "HMC5883L"));
  CHECK(!std::strcmp(i2c_ident(s, 0x0D), "QMC5883L"));
  CHECK(!std::strcmp(i2c_ident(s, 0x77), "BMP180"));
  CHECK(!std::strcmp(i2c_ident(s, 0x40), "UNKNOWN"));
  CHECK(!std::strcmp(i2c_ident(s, 0x6A), "MPU6500"));
}

// ---------------- vehicle (core end to end) ----------------
static std::vector<std::string> with_prefix(const std::vector<std::string>& v, const char* p) {
  std::vector<std::string> r;
  for (auto& s : v)
    if (s.rfind(p, 0) == 0) r.push_back(s);
  return r;
}
static std::string field(const std::string& s, int idx) {
  size_t start = 0;
  for (int i = 0; i < idx; ++i) start = s.find(',', start) + 1;
  size_t end = s.find(',', start);
  return s.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

static void run_ms(FakeHal& h, Vehicle& v, uint32_t ms) {
  for (uint32_t i = 0; i < ms; ++i) {
    h.t += 1000;
    v.loop_once();
  }
}

static void test_vehicle() {
  FakeHal h;
  h.bus.add_mpu6050(0x68);
  for (bool& a : h.assigned) a = false;
  Vehicle v;
  v.begin(h);
  auto b = h.bodies();
  CHECK(b.size() >= 2 && b[0] == "BOOT,0.1.0,OUTPUTS_DISABLED,HWWDT_UNVERIFIED");
  CHECK(b[1].rfind("STA,", 0) == 0 && field(b[1], 2) == "DISARMED" && field(b[1], 4) == "-" && field(b[1], 5) == "65535");
  CHECK(safe(h.last));

  run_ms(h, v, 120);  // > one STATUS_PERIOD_MS
  b = h.bodies();
  auto imu = with_prefix(b, "IMU,");
  CHECK(!imu.empty() && field(imu[0], 2) == "MPU6050" && field(imu[0], 10) == "OK");
  bool itg_missing = false;
  for (auto& s : imu)
    if (field(s, 2) == "ITG3200") itg_missing = s.find(",-,-,-,-,-,-,-,NOT_FOUND") != std::string::npos;
  CHECK(itg_missing);
  auto us = with_prefix(b, "US,");
  CHECK(!us.empty() && field(us[0], 3) == "-" && field(us[0], 4) == "UNASSIGNED");
  auto sta = with_prefix(b, "STA,");
  CHECK(!sta.empty());
  unsigned f = sta.empty() ? 0 : std::stoul(field(sta.back(), 3), nullptr, 16);
  CHECK((f & fault::OUTPUTS_DISABLED_BUILD) && (f & fault::HW_WATCHDOG_UNVERIFIED) && (f & fault::IMU_FAULT) &&
        (f & fault::US_FAULT) && !(f & fault::ESTOP_PIN_UNASSIGNED));

  h.in = frame("ARM,1");
  run_ms(h, v, 1);
  b = h.bodies();
  CHECK(!with_prefix(b, "ACK,1,OK,-").empty());
  sta = with_prefix(b, "STA,");
  CHECK(!sta.empty() && field(sta[0], 2) == "ARMED");  // immediate STA on state change

  h.in = frame("CMD,2,1000,0,-400");
  run_ms(h, v, 1);
  CHECK(h.last.thr_pm == cfg::MAX_THROTTLE_PM && h.last.str_pm == -400);
  CHECK(with_prefix(h.bodies(), "ACK,").empty());  // successful CMD is not acked
  run_ms(h, v, 100);
  sta = with_prefix(h.bodies(), "STA,");
  CHECK(!sta.empty() && field(sta.back(), 4) == "2" && field(sta.back(), 6) == "250" && field(sta.back(), 8) == "-400");

  run_ms(h, v, 150);  // no CMD for >200 ms total -> timeout
  CHECK(v.safety().state() == State::DISARMED && safe(h.last));
  sta = with_prefix(h.bodies(), "STA,");
  CHECK(!sta.empty() && field(sta.back(), 2) == "DISARMED" &&
        (std::stoul(field(sta.back(), 3), nullptr, 16) & fault::CMD_TIMEOUT));

  h.in = frame("CMD,3,100,0,0");
  run_ms(h, v, 1);
  CHECK(!with_prefix(h.bodies(), "ACK,3,ERR,NOT_ARMED").empty());

  // bad frames: garbage, unknown type, SIM on real firmware, malformed CMD
  h.in = frame("ARM,4");
  run_ms(h, v, 1);
  h.in = "$ARM,5*0000\n";
  run_ms(h, v, 1);
  CHECK(v.safety().state() == State::DISARMED && (v.faults() & fault::BAD_FRAME));
  h.in = frame("SIM,ESTOP,1") + frame("FOO,9") + frame("CMD,10,1,2") + frame("ARM");
  run_ms(h, v, 1);
  b = h.bodies();
  CHECK(!with_prefix(b, "ACK,0,ERR,UNKNOWN_TYPE").empty() && !with_prefix(b, "ACK,9,ERR,UNKNOWN_TYPE").empty());
  CHECK(!with_prefix(b, "ACK,10,ERR,BAD_ARGS").empty() && !with_prefix(b, "ACK,0,ERR,BAD_ARGS").empty());
  run_ms(h, v, 100);
  sta = with_prefix(h.bodies(), "STA,");
  CHECK(!sta.empty() && field(sta.back(), 9) == "5");

  // physical e-stop
  h.in = frame("ARM,11");
  run_ms(h, v, 1);
  h.estop = true;
  run_ms(h, v, 1);
  CHECK(v.safety().state() == State::ESTOP && safe(h.last));
  h.in = frame("RESET,12");
  run_ms(h, v, 1);
  CHECK(!with_prefix(h.bodies(), "ACK,12,ERR,ESTOP_ACTIVE").empty());
  h.estop = false;
  h.in = frame("RESET,13") + frame("SCAN,14");
  run_ms(h, v, 1);
  b = h.bodies();
  CHECK(v.safety().state() == State::DISARMED && !with_prefix(b, "ACK,13,OK,-").empty());
  CHECK(!with_prefix(b, "I2C,68,MPU6050").empty() && !with_prefix(b, "I2C,END,1").empty() &&
        !with_prefix(b, "ACK,14,OK,-").empty());

  // no e-stop pin -> arming refused
  FakeHal h2;
  h2.estop_pin = false;
  Vehicle v2;
  v2.begin(h2);
  h2.in = frame("ARM,1");
  run_ms(h2, v2, 1);
  CHECK(!with_prefix(h2.bodies(), "ACK,1,ERR,ESTOP_PIN_UNASSIGNED").empty() && v2.safety().state() == State::DISARMED);
}

int main() {
  test_protocol();
  test_safety();
  test_actuators();
  test_ultrasonic();
  test_imu();
  test_vehicle();
  if (failures) {
    std::fprintf(stderr, "%d CHECK(S) FAILED\n", failures);
    return 1;
  }
  std::printf("ALL TESTS PASSED\n");
  return 0;
}
