#pragma once

#include <stdint.h>

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

// Subnormal scales are flushed to zero in host repack. The device fp16
// conversion treats exp == 0 as zero, so the repacked stream must not rely on
// subnormal values.
static inline bool fp16_is_subnormal(uint16_t h) {
  return (h & 0x7c00u) == 0 && (h & 0x03ffu) != 0;
}

namespace detail {

struct RepackWarnings {
  bool subnormal_scale = false;
  bool min_int8 = false;

  // Emit at most one line per repack call, regardless of how many values
  // triggered the warning.
  void warn(const char* format) const {
    (void)format;
    if (!subnormal_scale && !min_int8) {
      return;
    }
    if (subnormal_scale && min_int8) {
      QUANTS_REPACK_WARN(
          "quants: %s repack flushed subnormal fp16 scales to zero and found "
          "-128 int8 weights\n",
          format);
    } else if (subnormal_scale) {
      QUANTS_REPACK_WARN(
          "quants: %s repack flushed subnormal fp16 scales to zero\n", format);
    } else {
      QUANTS_REPACK_WARN("quants: %s repack found -128 int8 weights\n", format);
    }
  }
};

}  // namespace detail

}  // namespace spiral::quants
