// Shared device helpers for the float16 conversion and scale probes.
//
// These mirror the production helpers in `src/quants/common/` so the bench
// stays standalone, but they are only compiled into the probes that use them.

#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace bench_float16 {

// Convert eight contiguous float16 values to f32 from the raw bits. This
// mirrors the production `load_fp16x8` helper; zero and subnormal inputs are
// returned as zero.
static inline __attribute__((always_inline)) aie::vector<float, 8>
load_float16x8(const uint8_t* p) {
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

// Shift-based float16 -> f32 decode: zero-extend, shift the magnitude into the
// f32 mantissa field, restore the sign, then add the exponent bias. This is
// `((h & 0x7fff) << 13) | sign | 0x38000000`, with zero/subnormal flushed.
static inline __attribute__((always_inline)) aie::vector<float, 8>
load_float16x8_shift(const uint8_t* p) {
  const auto h = aie::load_v<8>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto magnitude = aie::logical_downshift(aie::upshift(u, 17), 4);
  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto bits =
      aie::add(aie::bit_or(magnitude, sign), static_cast<uint32_t>(0x38000000));
  const auto normal = aie::neq(static_cast<uint32_t>(0),
                               aie::bit_and(static_cast<uint32_t>(0x7c00), u));
  return aie::select(aie::zeros<uint32_t, 8>(), bits, normal)
      .template cast_to<float>();
}

// Shift-based decode for host-pre-biased values. The host stores d * 2^10, so
// the f32 exponent bias is 102 instead of 112 and canonical subnormals never
// reach the device. Zero still needs the exponent select.
static inline __attribute__((always_inline)) aie::vector<float, 8>
load_float16x8_biased_shift(const uint8_t* p) {
  const auto h = aie::load_v<8>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto magnitude = aie::logical_downshift(aie::upshift(u, 17), 4);
  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto bits =
      aie::add(aie::bit_or(magnitude, sign), static_cast<uint32_t>(0x33000000));
  const auto normal = aie::neq(static_cast<uint32_t>(0),
                               aie::bit_and(static_cast<uint32_t>(0x7c00), u));
  return aie::select(aie::zeros<uint32_t, 8>(), bits, normal)
      .template cast_to<float>();
}

static inline __attribute__((always_inline)) aie::vector<float, 32>
load_float16x32(const uint8_t* p) {
  const auto h = aie::load_v<32>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto exp = aie::logical_downshift(aie::bit_and(0x7c00u, u), 10);
  const auto exponent = aie::upshift(aie::add(exp, 112u), 23);
  const auto mantissa = aie::upshift(aie::bit_and(0x03ffu, u), 13);
  const auto bits = aie::bit_or(sign, aie::bit_or(exponent, mantissa));
  const auto normal = aie::neq(static_cast<uint32_t>(0), exp);
  return aie::select(aie::zeros<uint32_t, 32>(), bits, normal)
      .template cast_to<float>();
}

static inline __attribute__((always_inline)) void split_float16x32_direct(
    const uint8_t* p,
    aie::vector<bfloat16, 32>& hi,
    aie::vector<bfloat16, 32>& lo) {
  const auto h = aie::load_v<32>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto sign = aie::bit_and(0x8000u, u);
  const auto e = aie::logical_downshift(aie::bit_and(0x7c00u, u), 10);
  const auto m = aie::bit_and(0x03ffu, u);

  const auto hi_bits =
      aie::bit_or(sign, aie::bit_or(aie::upshift(aie::add(e, 112u), 7),
                                    aie::logical_downshift(m, 3)));

  const auto r = aie::bit_and(0x7u, m);
  const auto r_bf16 =
      aie::to_float<bfloat16>(r.template pack_sign<uint16>(false));
  const auto factor_bits =
      aie::bit_or(sign, aie::upshift(aie::add(e, 102u), 7));
  const auto factor = factor_bits.template pack_sign<uint16>(false)
                          .template cast_to<bfloat16>();
  const auto lo_value = aie::mul(r_bf16, factor).template to_vector<bfloat16>();

  const auto normal = aie::neq(static_cast<uint32_t>(0), e);
  hi = aie::select(aie::zeros<uint32_t, 32>(), hi_bits, normal)
           .template pack_sign<uint16>(false)
           .template cast_to<bfloat16>();
  lo = aie::select(aie::zeros<bfloat16, 32>(), lo_value, normal);
}

// 32-value versions used to fill the native 512-bit bf16 width.
static inline __attribute__((always_inline)) aie::vector<float, 32>
load_float16x32_shift(const uint8_t* p) {
  const auto h = aie::load_v<32>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto magnitude = aie::logical_downshift(aie::upshift(u, 17), 4);
  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto bits =
      aie::add(aie::bit_or(magnitude, sign), static_cast<uint32_t>(0x38000000));
  const auto normal = aie::neq(static_cast<uint32_t>(0),
                               aie::bit_and(static_cast<uint32_t>(0x7c00), u));
  return aie::select(aie::zeros<uint32_t, 32>(), bits, normal)
      .template cast_to<float>();
}

static inline __attribute__((always_inline)) aie::vector<float, 32>
load_float16x32_biased_shift(const uint8_t* p) {
  const auto h = aie::load_v<32>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto magnitude = aie::logical_downshift(aie::upshift(u, 17), 4);
  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto bits =
      aie::add(aie::bit_or(magnitude, sign), static_cast<uint32_t>(0x33000000));
  const auto normal = aie::neq(static_cast<uint32_t>(0),
                               aie::bit_and(static_cast<uint32_t>(0x7c00), u));
  return aie::select(aie::zeros<uint32_t, 32>(), bits, normal)
      .template cast_to<float>();
}

static inline __attribute__((always_inline)) void split_bf16_hi_lo32(
    const aie::vector<float, 32>& scales,
    aie::vector<bfloat16, 32>& hi,
    aie::vector<bfloat16, 32>& lo) {
  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, 32>(scales));
  const auto hi_f = aie::accum<accfloat, 32>(hi).template to_vector<float>();
  lo =
      aie::to_vector<bfloat16>(aie::accum<accfloat, 32>(aie::sub(scales, hi_f)));
}

// Direct float16 -> bf16 hi/lo split, no f32 intermediate. hi is the truncated
// top eight significand bits; lo is the low three bits scaled by the same
// power-of-two exponent. Both are exact.
static inline __attribute__((always_inline)) void split_float16x8_direct(
    const uint8_t* p,
    aie::vector<bfloat16, 8>& hi,
    aie::vector<bfloat16, 8>& lo) {
  const auto h = aie::load_v<8>(reinterpret_cast<const uint16_t*>(p));
  const auto u = h.unpack();

  const auto sign = aie::bit_and(0x8000u, u);
  const auto e = aie::logical_downshift(aie::bit_and(0x7c00u, u), 10);
  const auto m = aie::bit_and(0x03ffu, u);

  const auto hi_bits =
      aie::bit_or(sign, aie::bit_or(aie::upshift(aie::add(e, 112u), 7),
                                    aie::logical_downshift(m, 3)));

  const auto r = aie::bit_and(0x7u, m);
  const auto r_bf16 =
      aie::to_float<bfloat16>(r.template pack_sign<uint16>(false));
  const auto factor_bits =
      aie::bit_or(sign, aie::upshift(aie::add(e, 102u), 7));
  const auto factor = factor_bits.template pack_sign<uint16>(false)
                          .template cast_to<bfloat16>();
  const auto lo_value = aie::mul(r_bf16, factor).template to_vector<bfloat16>();

  const auto normal = aie::neq(static_cast<uint32_t>(0), e);
  hi = aie::select(aie::zeros<uint32_t, 8>(), hi_bits, normal)
           .template pack_sign<uint16>(false)
           .template cast_to<bfloat16>();
  lo = aie::select(aie::zeros<bfloat16, 8>(), lo_value, normal);
}

// Split f32 scales into bf16 hi/lo components. Mirrors
// `spiral::quants::split_bf16_hi_lo`.
static inline __attribute__((always_inline)) void split_bf16_hi_lo(
    const aie::vector<float, 8>& scales,
    aie::vector<bfloat16, 8>& hi,
    aie::vector<bfloat16, 8>& lo) {
  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(scales));
  const auto hi_f = aie::accum<accfloat, 8>(hi).template to_vector<float>();
  lo =
      aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(aie::sub(scales, hi_f)));
}

// Q6_K effective scales need three bf16 terms. Mirrors
// `spiral::quants::q6_k_int8::detail::load_chunk_scales32`.
static inline __attribute__((always_inline)) void split3(
    const aie::vector<float, 8>& product,
    aie::vector<bfloat16, 8>& hi,
    aie::vector<bfloat16, 8>& mid,
    aie::vector<bfloat16, 8>& lo) {
  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(product));
  const auto hi_f = aie::accum<accfloat, 8>(hi).template to_vector<float>();
  const auto r1 = aie::sub(product, hi_f);
  mid = aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(r1));
  const auto mid_f = aie::accum<accfloat, 8>(mid).template to_vector<float>();
  const auto r2 = aie::sub(r1, mid_f);
  lo = aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(r2));
}

// 32-lane three-term split for Q6_K chunk scales.
static inline __attribute__((always_inline)) void split3_32(
    const aie::vector<float, 32>& product,
    aie::vector<bfloat16, 32>& hi,
    aie::vector<bfloat16, 32>& mid,
    aie::vector<bfloat16, 32>& lo) {
  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, 32>(product));
  const auto hi_f = aie::accum<accfloat, 32>(hi).template to_vector<float>();
  const auto r1 = aie::sub(product, hi_f);
  mid = aie::to_vector<bfloat16>(aie::accum<accfloat, 32>(r1));
  const auto mid_f = aie::accum<accfloat, 32>(mid).template to_vector<float>();
  const auto r2 = aie::sub(r1, mid_f);
  lo = aie::to_vector<bfloat16>(aie::accum<accfloat, 32>(r2));
}

// Decode eight float16 scales to integer mantissas (11 bits) and bf16
// power-of-two factors. The mantissas are positive; the factor carries sign.
static inline __attribute__((always_inline)) void decode_float16x8(
    const uint8_t* p,
    aie::vector<int16, 8>& mantissa,
    aie::vector<bfloat16, 8>& factor) {
  const auto h = aie::load_v<8>(reinterpret_cast<const uint16_t*>(p));
  const auto mant =
      aie::bit_or(aie::bit_and(h, aie::broadcast<uint16_t, 8>(0x03ff)),
                  aie::broadcast<uint16_t, 8>(0x0400));
  const auto exp = aie::logical_downshift(
      aie::bit_and(h, aie::broadcast<uint16_t, 8>(0x7c00)), 10);
  const auto sign = aie::bit_and(h, aie::broadcast<uint16_t, 8>(0x8000));
  const auto factor_bits = aie::bit_or(
      aie::upshift(aie::add(exp, aie::broadcast<uint16_t, 8>(102)), 7), sign);
  mantissa = mant.template cast_to<int16>();
  factor = factor_bits.template cast_to<bfloat16>();
}

}  // namespace bench_float16
