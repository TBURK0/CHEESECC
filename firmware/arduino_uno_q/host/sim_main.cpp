// uno_q_sim: the real firmware core running on Linux with simulated sensors/actuators.
// Speaks docs/serial_protocol.md on stdin/stdout, plus the SIM,* messages. Exits on stdin EOF.
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "config.h"
#include "fake_i2c.h"
#include "vehicle.h"

using Clock = std::chrono::steady_clock;

struct Edge {
  uint32_t t;
  uint8_t id;
  bool level;
};

struct SimHal : Hal {
  Clock::time_point t0 = Clock::now();
  FakeI2c bus;
  int32_t dist_mm[4] = {1500, 1500, 1500, 1500};  // 0 = nothing in range, -1 = disconnected
  bool level[4] = {};
  std::vector<Edge> pending;
  bool estop = false;
  bool eof = false;
  Outputs last;

  uint32_t micros() override {
    return (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
  }
  void us_trigger(uint8_t id) override {
    int32_t d = dist_mm[id];
    if (d < 0) return;  // disconnected: echo never rises
    uint32_t t = micros() + 450;  // HC-SR04-ish delay before the echo pulse starts
    // nothing in range: HC-SR04 holds echo high ~38 ms; otherwise round trip, rounded up
    uint32_t echo = d == 0 ? 38000u : (uint32_t)(((int64_t)d * 2000 + 342) / 343);
    pending.push_back({t, id, true});
    pending.push_back({t + echo, id, false});
  }
  bool us_echo_level(uint8_t id) override { return level[id]; }
  bool us_assigned(uint8_t) override { return true; }
  bool estop_assigned() override { return true; }  // the sim has a (simulated) e-stop input
  bool estop_active() override { return estop; }
  I2cBus& i2c() override { return bus; }
  void write_outputs(const Outputs& o) override { last = o; }
  void watchdog_kick() override {}
  bool hw_watchdog_verified() override { return false; }
  bool outputs_enabled() override { return true; }
  size_t link_read(char* buf, size_t cap) override {
    ssize_t r = ::read(0, buf, cap);
    if (r == 0) eof = true;
    return r > 0 ? (size_t)r : 0;
  }
  void link_write(const char* buf, size_t n) override {
    while (n > 0) {
      ssize_t w = ::write(1, buf, n);
      if (w <= 0) return;
      buf += w;
      n -= (size_t)w;
    }
  }
  void deliver_edges(Ultrasonic& u) {
    uint32_t now = micros();
    for (size_t i = 0; i < pending.size();) {
      if ((int32_t)(now - pending[i].t) >= 0) {
        level[pending[i].id] = pending[i].level;
        u.on_echo_edge(pending[i].id, pending[i].level, pending[i].t);  // exact edge time
        pending.erase(pending.begin() + (long)i);
      } else {
        ++i;
      }
    }
  }
};

static SimHal* g_sim = nullptr;

static int noise() { return std::rand() % 5 - 2; }  // ±2 counts

// Vehicle at rest and level: MPU sees +1 g on z and a +0.5 °/s x-gyro bias; ITG has its own biases.
static void refresh(uint8_t a, uint8_t r) {
  FakeI2c& b = g_sim->bus;
  if (a == cfg::MPU6050_ADDR && r == 0x3B) {
    b.put16(a, 0x3B, (int16_t)noise());
    b.put16(a, 0x3D, (int16_t)noise());
    b.put16(a, 0x3F, (int16_t)(8192 + noise()));
    b.put16(a, 0x41, (int16_t)-3920);  // 25 °C
    b.put16(a, 0x43, (int16_t)(33 + noise()));
    b.put16(a, 0x45, (int16_t)noise());
    b.put16(a, 0x47, (int16_t)noise());
  } else if (a == cfg::ITG3200_ADDR && r == 0x1B) {
    b.put16(a, 0x1B, (int16_t)-16000);  // 25 °C
    b.put16(a, 0x1D, (int16_t)(4 + noise()));
    b.put16(a, 0x1F, (int16_t)(-6 + noise()));
    b.put16(a, 0x21, (int16_t)noise());
  }
}

static Ack sim_hook(char** f, int n) {
  int32_t a, b;
  if (n == 4 && strcmp(f[1], "US") == 0 && parse_i32(f[2], a) && parse_i32(f[3], b) && a >= 0 && a < 4 && b >= -1) {
    g_sim->dist_mm[a] = b;
    return Ack::OK;
  }
  if (n == 3 && strcmp(f[1], "ESTOP") == 0 && parse_i32(f[2], a) && (a == 0 || a == 1)) {
    g_sim->estop = a == 1;
    return Ack::OK;
  }
  if (n == 4 && strcmp(f[1], "IMUFAIL") == 0 && parse_i32(f[3], b) && (b == 0 || b == 1)) {
    uint8_t addr = strcmp(f[2], "MPU6050") == 0 ? cfg::MPU6050_ADDR
                 : strcmp(f[2], "ITG3200") == 0 ? cfg::ITG3200_ADDR
                                                : 0;
    if (!addr) return Ack::BAD_ARGS;
    g_sim->bus.dev[addr].fail = b == 1;
    return Ack::OK;
  }
  return Ack::BAD_ARGS;
}

int main() {
  fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
  std::srand(1);
  SimHal hal;
  g_sim = &hal;
  hal.bus.add_mpu6050(cfg::MPU6050_ADDR);
  hal.bus.add_itg3200(cfg::ITG3200_ADDR);
  hal.bus.on_read = refresh;

  Vehicle v;
  v.begin(hal, sim_hook);
  while (!hal.eof) {
    hal.deliver_edges(v.ultra);
    v.loop_once();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return 0;
}
