// Wire framing: $<body>*<CRC16 hex>\n  (docs/serial_protocol.md -> Framing)
#pragma once
#include <stddef.h>
#include <stdint.h>

constexpr size_t MAX_BODY = 120;
constexpr size_t MAX_LINE = 128;  // '$' + body + '*' + 4 hex + '\n'

uint16_t crc16_ccitt_false(const char* data, size_t n);

// Writes "$body*XXXX\n" into out (NUL-terminated). Returns length, or 0 if it doesn't fit / body too long.
size_t format_frame(const char* body, char* out, size_t cap);

// Strict decimal parsing: whole string must be digits (optional leading '-' for i32), in range.
bool parse_u32(const char* s, uint32_t& out);
bool parse_i32(const char* s, int32_t& out);

// Splits body in place on ','. Returns field count, or -1 if more than max fields.
int split_fields(char* body, char** fields, int max);

class LineAssembler {
 public:
  enum Result { NONE, FRAME, BAD };
  Result feed(char c);                 // on FRAME, body() holds the CRC-checked body
  const char* body() const { return buf_; }

 private:
  Result finish();
  char buf_[MAX_LINE + 1] = {};
  size_t len_ = 0;
  bool in_frame_ = false;
  bool overflow_ = false;
};
