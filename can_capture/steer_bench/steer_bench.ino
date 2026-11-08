// Suspended-car STEERING BENCH controller — UNO Q + Inland/MCP2515 shield.
//
// SCOPE: car on a lift, wheels off the ground, steering only. NOT for a car that
// can move or touch the ground with this firmware. This is the actuator bring-up
// bench from docs/hardware.md, with a real MDPS as the actuator.
//
// SAFETY ENVELOPE (software) sits UNDER the physical layer (E-stop contactor that
// removes MDPS power, double-pole CAN kill, ARMED light). The physical layer is the
// real guarantee; this just keeps the software well-behaved.
//
// THIS BUILD CANNOT MOVE ANYTHING:
//   * ENABLE_STEER_OUTPUT is 0 -> the CAN controller stays in LISTEN-ONLY and
//     physically cannot transmit.
//   * build_steer_frame() is a STUB and returns false until we fill it in from the
//     decoded LKAS/MDPS message (your captured logs).
// To ever actuate: decode the frame, implement build_steer_frame(), verify the
// physical safety layer with a meter, set ENABLE_STEER_OUTPUT 1, then start at the
// SMALLEST torque/angle with everyone's hands clear of the wheels and linkage.

#include <SPI.h>
#include <mcp_can.h>

// ---------------- config (all UNVERIFIED until bench bring-up) ----------------
#define ENABLE_STEER_OUTPUT 0        // keep 0 until the whole safety layer is proven

const int  SPI_CS_PIN   = 9;         // same shield as your logger; try 10 if init fails
const int  PIN_ESTOP    = 2;         // reads the NC safety loop: LOW/open = STOP (active-low, pull-up)
const int  PIN_ARMED_LED = 7;        // ARMED indicator (informational)

const uint32_t CMD_TIMEOUT_MS = 150; // no fresh command -> release steering
const uint32_t TX_PERIOD_MS   = 10;  // LKAS is ~100 Hz; confirm from your logs
const int16_t  MAX_CMD        = 150;  // start TINY (per-mille of full). Raise only after bench proof.
const int16_t  MAX_RATE_PER_TX = 5;   // max change per TX tick -> no snapping to lock
// ------------------------------------------------------------------------------

MCP_CAN CAN(SPI_CS_PIN);

enum State { DISARMED, ARMED, ESTOP };
State    state = DISARMED;
int16_t  cmd_target = 0;    // requested steering, -MAX_CMD..MAX_CMD (per-mille)
int16_t  cmd_out = 0;       // rate-limited value actually heading to the actuator
uint32_t last_cmd_ms = 0;
uint32_t last_tx_ms = 0;

bool estop_active() { return digitalRead(PIN_ESTOP) == LOW; }   // open loop reads LOW

void safe_release() { cmd_target = 0; cmd_out = 0; }

// STUB: fill from the decoded Hyundai steering message (ID, byte layout, rolling
// counter, checksum). Returns false => nothing is sent. Do NOT guess these bytes.
bool build_steer_frame(int16_t /*value_pm*/, unsigned long& /*id*/,
                       unsigned char& /*len*/, unsigned char* /*buf*/) {
  return false;   // not implemented until logs are decoded
}

void setup() {
  Serial.begin(1000000);
  pinMode(PIN_ESTOP, INPUT_PULLUP);
  pinMode(PIN_ARMED_LED, OUTPUT);
  digitalWrite(PIN_ARMED_LED, LOW);

  if (CAN.begin(MCP_ANY, CAN_500KBPS, MCP_16MHZ) != CAN_OK &&
      CAN.begin(MCP_ANY, CAN_500KBPS, MCP_8MHZ)  != CAN_OK) {
    Serial.println(F("# CAN INIT FAIL"));
    while (1);
  }
#if ENABLE_STEER_OUTPUT
  CAN.setMode(MCP_NORMAL);           // can transmit — only in an explicitly-enabled build
  Serial.println(F("# steer_bench: OUTPUT ENABLED (bench only, wheels up, hands clear)"));
#else
  CAN.setMode(MCP_LISTENONLY);       // default: cannot transmit at all
  Serial.println(F("# steer_bench: OUTPUT DISABLED (listen-only). Safe."));
#endif
}

// Commands from the Jetson/camera loop, newline-terminated:
//   C,<value>   desired steering, per-mille (clamped to MAX_CMD)
//   ARM / DISARM / ESTOP
void handle_line(char* s) {
  if (!strcmp(s, "ARM")) {
    if (state == DISARMED && !estop_active()) { state = ARMED; last_cmd_ms = millis(); }
  } else if (!strcmp(s, "DISARM")) {
    if (state == ARMED) { state = DISARMED; safe_release(); }
  } else if (!strcmp(s, "ESTOP")) {
    state = ESTOP; safe_release();
  } else if (s[0] == 'C' && s[1] == ',') {
    long v = atol(s + 2);
    cmd_target = (int16_t)constrain(v, -MAX_CMD, MAX_CMD);
    last_cmd_ms = millis();
  }
}

char buf[32];
uint8_t blen = 0;

void loop() {
  // --- read serial commands ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { buf[blen] = 0; if (blen) handle_line(buf); blen = 0; }
    else if (blen < sizeof(buf) - 1) buf[blen++] = c;
  }

  // --- safety supervisor ---
  uint32_t now = millis();
  if (estop_active() && state != ESTOP) { state = ESTOP; safe_release(); }     // latches
  if (state == ARMED && (uint32_t)(now - last_cmd_ms) >= CMD_TIMEOUT_MS) {      // command expiry
    state = DISARMED; safe_release();
  }
  digitalWrite(PIN_ARMED_LED, state == ARMED ? HIGH : LOW);

  // --- rate-limit toward target (only while armed) ---
  int16_t want = (state == ARMED) ? cmd_target : 0;
  if (cmd_out < want) cmd_out = min(want, (int16_t)(cmd_out + MAX_RATE_PER_TX));
  else if (cmd_out > want) cmd_out = max(want, (int16_t)(cmd_out - MAX_RATE_PER_TX));

  // --- transmit (inert until build_steer_frame is implemented AND output enabled) ---
  if (state == ARMED && (uint32_t)(now - last_tx_ms) >= TX_PERIOD_MS) {
    last_tx_ms = now;
    unsigned long id; unsigned char len, frame[8];
    if (build_steer_frame(cmd_out, id, len, frame)) {
#if ENABLE_STEER_OUTPUT
      CAN.sendMsgBuf(id, 0, len, frame);
#endif
    }
  }
}
