// Passive CAN logger for the Inland/MCP2515 shield (same board as your desk test).
// LISTEN-ONLY: the controller is put in a mode where it physically cannot transmit,
// not even a CAN ACK. It can only read. It cannot disturb or inject onto the car bus.
//
// Output is CSV to serial at 1 Mbps:   millis,ID(hex),len,BB BB BB ...
// Capture it to a file with can_capture/capture.py (see that file).

#include <SPI.h>
#include <mcp_can.h>

const int SPI_CS_PIN = 9;        // same as your test; if init fails, try 10
MCP_CAN CAN(SPI_CS_PIN);

void setup() {
  Serial.begin(1000000);         // 115200 is too slow for a busy 500k bus; set the capture tool to 1000000 too
  while (!Serial);

  // 500 kbps is standard for Hyundai C-CAN. Try 16 MHz crystal, then 8 MHz.
  if (CAN.begin(MCP_ANY, CAN_500KBPS, MCP_16MHZ) != CAN_OK &&
      CAN.begin(MCP_ANY, CAN_500KBPS, MCP_8MHZ)  != CAN_OK) {
    Serial.println(F("# CAN INIT FAIL - try CS pin 10, check wiring"));
    while (1);
  }

  CAN.setMode(MCP_LISTENONLY);   // read-only. The device cannot write to the bus.
  Serial.println(F("# ms,id,len,data   listen-only"));
}

unsigned long rxId;
unsigned char len;
unsigned char buf[8];
char line[64];

void loop() {
  // Drain every waiting frame each pass, no delay, so we keep up with the bus.
  // ponytail: MCP2515 has only 2 RX buffers; on a very busy bus some frames drop.
  //   Fine here - LKAS frames are periodic (~100 Hz), so a diff still catches them.
  //   Upgrade path if drops matter: hardware RX filters, or a native-CAN MCU.
  while (CAN.checkReceive() == CAN_MSGAVAIL) {
    CAN.readMsgBuf(&rxId, &len, buf);
    int n = sprintf(line, "%lu,%lX,%u,", millis(), rxId & 0x1FFFFFFF, len);
    for (unsigned char i = 0; i < len; i++) n += sprintf(line + n, "%02X ", buf[i]);
    Serial.println(line);
  }
}
