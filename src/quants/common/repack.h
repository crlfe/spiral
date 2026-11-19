#pragma once

#include <stdint.h>
#include <string.h>

// Host-side repack helpers shared by every format. This header is also
// included by callers that compile the repackers inline (tests and
// llama.cpp), so keep it free of device types and external dependencies.
//
// A caller may override QUANTS_REPACK_WARN to route the one-line warning
// through its own logger, or define it to nothing to suppress the warning.
#ifndef QUANTS_REPACK_WARN
#include <stdio.h>
#define QUANTS_REPACK_WARN(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace spiral::quants {

// Canonical GGUF blocks are little-endian.
static inline uint16_t load_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

static inline void store_u16(uint8_t* p, uint16_t value) {
  p[0] = static_cast<uint8_t>(value & 0xffu);
  p[1] = static_cast<uint8_t>(value >> 8);
}

namespace detail {

// Exact IEEE fp16 -> f32 conversion, including subnormals. Subnormal fp16 is
// mantissa * 2^-24, which is exactly representable in f32.
static inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x03ffu;
  if (exp == 0) {
    const float value = static_cast<float>(mant) * 0x1p-24f;
    return (sign != 0) ? -value : value;
  }
  uint32_t bits;
  if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

// Round f32 to bf16 with round-to-nearest-even.
static inline uint16_t f32_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static inline float bf16_to_f32(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

}  // namespace detail

}  // namespace spiral::quants
