#include "protocol.h"
#include <stdio.h>
#include <string.h>

uint16_t crc16_ccitt_false(const char* data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; ++i) {
    crc ^= (uint16_t)((uint8_t)data[i]) << 8;
    for (int b = 0; b < 8; ++b) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

size_t format_frame(const char* body, char* out, size_t cap) {
  size_t n = strlen(body);
  if (n > MAX_BODY || n + 8 > cap) return 0;
  int w = snprintf(out, cap, "$%s*%04X\n", body, (unsigned)crc16_ccitt_false(body, n));
  return (w > 0 && (size_t)w < cap) ? (size_t)w : 0;
}

bool parse_u32(const char* s, uint32_t& out) {
  if (!s || !*s) return false;
  uint64_t v = 0;
  for (const char* p = s; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
    v = v * 10 + (uint64_t)(*p - '0');
    if (v > 0xFFFFFFFFull) return false;
  }
  out = (uint32_t)v;
  return true;
}

bool parse_i32(const char* s, int32_t& out) {
  if (!s) return false;
  bool neg = *s == '-';
  uint32_t mag;
  if (!parse_u32(neg ? s + 1 : s, mag)) return false;
  if (neg ? mag > 2147483648u : mag > 2147483647u) return false;
  out = neg ? (int32_t)(-(int64_t)mag) : (int32_t)mag;
  return true;
}

int split_fields(char* body, char** fields, int max) {
  int n = 0;
  char* p = body;
  while (true) {
    if (n >= max) return -1;
    fields[n++] = p;
    char* c = strchr(p, ',');
    if (!c) return n;
    *c = '\0';
    p = c + 1;
  }
}

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;  // lowercase is rejected: contract says uppercase
}

LineAssembler::Result LineAssembler::feed(char c) {
  if (c == '$') {
    // A '$' inside a frame means the previous frame was cut short.
    bool was_in = in_frame_;
    in_frame_ = true;
    len_ = 0;
    overflow_ = false;
    return was_in ? BAD : NONE;
  }
  if (!in_frame_) return NONE;  // bytes before '$' are discarded
  if (c == '\r') return NONE;
  if (c == '\n') {
    in_frame_ = false;
    return finish();
  }
  if (len_ + 3 > MAX_LINE) {  // '$' + stored chars (incl. this one) + '\n' would exceed MAX_LINE
    overflow_ = true;
    return NONE;
  }
  buf_[len_++] = c;
  return NONE;
}

LineAssembler::Result LineAssembler::finish() {
  if (overflow_ || len_ < 5) return BAD;
  size_t star = len_ - 5;
  if (buf_[star] != '*') return BAD;
  if (star > MAX_BODY) return BAD;
  uint16_t want = 0;
  for (size_t i = star + 1; i < len_; ++i) {
    int h = hexval(buf_[i]);
    if (h < 0) return BAD;
    want = (uint16_t)((want << 4) | h);
  }
  for (size_t i = 0; i < star; ++i) {
    char ch = buf_[i];
    if (ch < 0x20 || ch > 0x7E || ch == '*') return BAD;
  }
  if (crc16_ccitt_false(buf_, star) != want) return BAD;
  buf_[star] = '\0';
  return FRAME;
}
