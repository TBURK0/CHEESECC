// Build-time configuration for the UNO Q MCU firmware.
// Every pin is PIN_UNASSIGNED until someone reads the real wiring: unassigned
// ultrasonics report UNASSIGNED, and an unassigned e-stop pin makes arming impossible.
#pragma once
#include <stdint.h>

#define FW_VERSION "0.1.0"

// OFF by default: PWM pins are never attached, so the ESC/servos receive no signal at all.
// Set to 1 by hand only after the hardware verification checklist passes.
#ifndef ENABLE_ACTUATOR_OUTPUT
#define ENABLE_ACTUATOR_OUTPUT 0
#endif

// UNVERIFIED: which Serial object is an MCU UART on the UNO Q header pins. Confirm at bring-up.
#define LINK_SERIAL Serial1
#define LINK_BAUD 460800

namespace cfg {

constexpr uint8_t PIN_UNASSIGNED = 255;

// Ultrasonic pins, index = sensor id: 0=FL, 1=FR, 2=RL, 3=RR.
// UNKNOWN: set from real wiring. HC-SR04-style echo is 5 V: needs a divider/level shifter
// to the 3.3 V MCU pin (UNVERIFIED whether UNO Q pins are 5 V tolerant; assume not).
constexpr uint8_t US_TRIG_PIN[4] = {PIN_UNASSIGNED, PIN_UNASSIGNED, PIN_UNASSIGNED, PIN_UNASSIGNED};
constexpr uint8_t US_ECHO_PIN[4] = {PIN_UNASSIGNED, PIN_UNASSIGNED, PIN_UNASSIGNED, PIN_UNASSIGNED};

// E-stop: INPUT_PULLUP, active LOW, so a broken wire reads as "stop". UNKNOWN: set from real wiring.
constexpr uint8_t ESTOP_PIN = PIN_UNASSIGNED;

// PWM outputs. UNKNOWN: set from real wiring (how gas/brake/steering are actually driven is unverified).
constexpr uint8_t THROTTLE_PWM_PIN = PIN_UNASSIGNED;
constexpr uint8_t BRAKE_PWM_PIN = PIN_UNASSIGNED;
constexpr uint8_t STEER_PWM_PIN = PIN_UNASSIGNED;

// PWM calibration in microseconds. UNVERIFIED: typical RC values, measure on the real ESC/servos.
constexpr uint16_t THROTTLE_NEUTRAL_US = 1500;
constexpr uint16_t THROTTLE_MAX_US = 2000;
constexpr uint16_t BRAKE_RELEASE_US = 1000;
constexpr uint16_t BRAKE_FULL_US = 2000;
constexpr uint16_t STEER_RIGHT_US = 1000;
constexpr uint16_t STEER_CENTER_US = 1500;
constexpr uint16_t STEER_LEFT_US = 2000;

// Safety limits and timing (docs/serial_protocol.md -> Defaults).
constexpr uint32_t CMD_TIMEOUT_MS = 200;
constexpr uint32_t STATUS_PERIOD_MS = 100;
constexpr int32_t MAX_THROTTLE_PM = 250;   // conservative bring-up limit, not a tuned value
constexpr int32_t MAX_BRAKE_PM = 1000;
constexpr int32_t MAX_STEER_PM = 1000;
constexpr int32_t SAFE_STOP_BRAKE_PM = 1000;  // tune on hardware
static_assert(SAFE_STOP_BRAKE_PM <= MAX_BRAKE_PM, "safe-stop brake must be within limits");

// Ultrasonics. Range numbers are HC-SR04 nominal, UNVERIFIED for the actual sensors.
constexpr uint32_t US_SLOT_MS = 60;
constexpr uint8_t US_ORDER[4] = {0, 3, 1, 2};  // FL, RR, FR, RL: diagonal alternation
constexpr uint32_t US_ECHO_START_TIMEOUT_US = 5000;
constexpr uint32_t US_ECHO_MAX_US = 30000;
constexpr int32_t US_MIN_MM = 20;
constexpr int32_t US_MAX_MM = 4000;

// IMUs. UNVERIFIED models/addresses. Both parts default to 0x68; the ITG-3200 is assumed
// strapped to 0x69 (AD0 high). If both answer at 0x68 they conflict: re-strap one.
constexpr uint32_t IMU_PERIOD_MS = 10;
constexpr uint8_t MPU6050_ADDR = 0x68;
constexpr uint8_t ITG3200_ADDR = 0x69;
constexpr uint32_t I2C_CLOCK_HZ = 400000;  // UNVERIFIED: depends on pull-ups / wire length

}  // namespace cfg
