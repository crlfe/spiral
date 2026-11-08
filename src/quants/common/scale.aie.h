#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace spiral::quants {

// Convert eight contiguous fp16 values to f32 with a packed bit trick:
//   f32 = sign | ((exp16 + 112) << 23) | (mantissa << 13)
// Zero and subnormal fp16 values (exp16 == 0) are returned as zero. Host
// repack flushes subnormal scales, so the repacked stream never needs a
// subnormal value here.
static inline __attribute__((always_inline)) aie::vector<float, 8> load_fp16x8(
    const uint8_t* p) {
  const auto h = aie::load_v<8>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto exp = aie::logical_downshift(aie::bit_and(0x7c00u, u), 10);
  const auto exponent = aie::upshift(aie::add(exp, 112u), 23);
  const auto mantissa = aie::upshift(aie::bit_and(0x03ffu, u), 13);
  const auto bits = aie::bit_or(sign, aie::bit_or(exponent, mantissa));
  const auto normal = aie::neq(static_cast<uint32_t>(0), exp);
  return aie::select(aie::zeros<uint32_t, 8>(), bits, normal)
      .template cast_to<float>();
}

// Split f32 scales into bf16 hi/lo components. hi + lo retains fp16 scale
// precision while keeping the per-weight multiply in bf16.
static inline __attribute__((always_inline)) void split_bf16_hi_lo(
    const aie::vector<float, 8>& scales,
    aie::vector<bfloat16, 8>& hi,
    aie::vector<bfloat16, 8>& lo) {
  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(scales));
  const auto hi_f = aie::accum<accfloat, 8>(hi).template to_vector<float>();
  lo =
      aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(aie::sub(scales, hi_f)));
}

}  // namespace spiral::quants
