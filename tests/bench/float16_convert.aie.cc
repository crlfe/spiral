// Conversion kernels for `float16_convert.py`.
//
// Exactly one probe is compiled per translation unit, selected by
// `-DBENCH_PROBE_<NAME>`.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "float16_bench.aie.h"

using namespace bench_float16;

namespace {

constexpr int kIters = 32;

}  // namespace

// Supported fallback: raw float16 bits -> f32 with the mask/rebias decode.
#if defined(BENCH_PROBE_FLOAT16_BITS_F32_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto f = load_float16x8(in + i * 16);
    aie::store_v(reinterpret_cast<float*>(out) + i * 8, f);
  }
  event1();
}

// Shift-based float16 -> f32 decode.
#elif defined(BENCH_PROBE_FLOAT16_SHIFT_F32_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto f = load_float16x8_shift(in + i * 16);
    aie::store_v(reinterpret_cast<float*>(out) + i * 8, f);
  }
  event1();
}

#else
#error "no BENCH_PROBE_* selected"
#endif
