// Arduino UNO Q — STM32 MCU sketch: board glue only. All logic lives in the .cpp files
// next to this one and is unit-tested on the host (../host).
//
// NOT COMPILED FOR THE TARGET YET: the UNO Q Arduino core was not installed when this was
// written. It uses only plain Arduino APIs (micros, digitalWrite, attachInterrupt, Wire,
// Servo, HardwareSerial). Confirm each exists on the UNO Q core before trusting it:
//   arduino-cli compile --fqbn <UNO Q MCU fqbn> firmware/arduino_uno_q/uno_q_vehicle
#include <Arduino.h>
#include <Servo.h>
#include <Wire.h>

#include "config.h"
#include "hal.h"
#include "vehicle.h"

// TODO(UNVERIFIED): the UNO Q core's hardware (independent) watchdog API has not been
// verified, so these are no-ops and STA reports HW_WATCHDOG_UNVERIFIED. Wire them to the
// core's real watchdog once confirmed (do not guess devicetree aliases), then set to 1.
#define HW_WATCHDOG_VERIFIED 0
static void watchdog_start() {}
static void watchdog_kick() {}

static Vehicle g_vehicle;

static bool assigned(uint8_t pin) { return pin != cfg::PIN_UNASSIGNED; }

struct WireBus : I2cBus {
  bool write_reg(uint8_t addr, uint8_t reg, uint8_t val) override {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.write(val);
    return Wire.endTransmission() == 0;
  }
  bool read_regs(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t n) override {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((int)addr, (int)n) != (int)n) return false;
    for (uint8_t i = 0; i < n; ++i) buf[i] = (uint8_t)Wire.read();
    return true;
  }
  bool probe(uint8_t addr) override {
    Wire.beginTransmission(addr);
    return Wire.endTransmission() == 0;
  }
};

struct TargetHal : Hal {
  WireBus bus;
#if ENABLE_ACTUATOR_OUTPUT
  Servo thr, brk, str;
#endif

  uint32_t micros() override { return ::micros(); }
  void us_trigger(uint8_t id) override {
    uint8_t p = cfg::US_TRIG_PIN[id];
    digitalWrite(p, HIGH);
    delayMicroseconds(10);
    digitalWrite(p, LOW);
  }
  bool us_echo_level(uint8_t id) override { return digitalRead(cfg::US_ECHO_PIN[id]) == HIGH; }
  bool us_assigned(uint8_t id) override { return assigned(cfg::US_TRIG_PIN[id]) && assigned(cfg::US_ECHO_PIN[id]); }
  bool estop_assigned() override { return assigned(cfg::ESTOP_PIN); }
  bool estop_active() override { return digitalRead(cfg::ESTOP_PIN) == LOW; }  // active-low, pull-up
  I2cBus& i2c() override { return bus; }
  void write_outputs(const Outputs& o) override {
#if ENABLE_ACTUATOR_OUTPUT
    PwmUs p = to_pwm_us(o);
    if (thr.attached()) thr.writeMicroseconds(p.thr);
    if (brk.attached()) brk.writeMicroseconds(p.brk);
    if (str.attached()) str.writeMicroseconds(p.str);
#else
    (void)o;  // outputs disabled at build time: pins are never attached
#endif
  }
  void watchdog_kick() override { ::watchdog_kick(); }
  bool hw_watchdog_verified() override { return HW_WATCHDOG_VERIFIED; }
  bool outputs_enabled() override { return ENABLE_ACTUATOR_OUTPUT; }
  size_t link_read(char* buf, size_t cap) override {
    size_t n = 0;
    while (n < cap && LINK_SERIAL.available() > 0) {
      int c = LINK_SERIAL.read();
      if (c < 0) break;
      buf[n++] = (char)c;
    }
    return n;
  }
  // Blocks only if the TX buffer is full; at 460800 baud the ~15 kB/s budget leaves headroom.
  void link_write(const char* buf, size_t n) override { LINK_SERIAL.write((const uint8_t*)buf, n); }
};

static TargetHal g_hal;

static void echo_isr(uint8_t id) {
  g_vehicle.ultra.on_echo_edge(id, digitalRead(cfg::US_ECHO_PIN[id]) == HIGH, ::micros());
}
static void echo_isr0() { echo_isr(0); }
static void echo_isr1() { echo_isr(1); }
static void echo_isr2() { echo_isr(2); }
static void echo_isr3() { echo_isr(3); }

void setup() {
  LINK_SERIAL.begin(LINK_BAUD);
  Wire.begin();
  Wire.setClock(cfg::I2C_CLOCK_HZ);

  if (assigned(cfg::ESTOP_PIN)) pinMode(cfg::ESTOP_PIN, INPUT_PULLUP);

  void (*const isrs[4])() = {echo_isr0, echo_isr1, echo_isr2, echo_isr3};
  for (uint8_t i = 0; i < 4; ++i) {
    if (!g_hal.us_assigned(i)) continue;
    pinMode(cfg::US_TRIG_PIN[i], OUTPUT);
    digitalWrite(cfg::US_TRIG_PIN[i], LOW);
    pinMode(cfg::US_ECHO_PIN[i], INPUT);
    attachInterrupt(digitalPinToInterrupt(cfg::US_ECHO_PIN[i]), isrs[i], CHANGE);
  }

#if ENABLE_ACTUATOR_OUTPUT
  // Attach at the safe pulse width so the first signal the ESC/servos ever see is "stopped".
  PwmUs safe = to_pwm_us(safe_outputs());
  if (assigned(cfg::THROTTLE_PWM_PIN)) { g_hal.thr.attach(cfg::THROTTLE_PWM_PIN); g_hal.thr.writeMicroseconds(safe.thr); }
  if (assigned(cfg::BRAKE_PWM_PIN)) { g_hal.brk.attach(cfg::BRAKE_PWM_PIN); g_hal.brk.writeMicroseconds(safe.brk); }
  if (assigned(cfg::STEER_PWM_PIN)) { g_hal.str.attach(cfg::STEER_PWM_PIN); g_hal.str.writeMicroseconds(safe.str); }
#endif

  watchdog_start();
  g_vehicle.begin(g_hal);
}

void loop() { g_vehicle.loop_once(); }
