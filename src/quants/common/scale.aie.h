#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace spiral::quants {

// Exact two-term split of canonical fp16 bytes into bf16 hi/lo terms. The
// low byte is converted to bf16 as an integer and its exponent field is
// adjusted by (E - 25); the high byte supplies the coarse term. Denormals use
// hi = top2 * 2^-16 and a -24 exponent delta for lo, so no subnormal branch is
// needed in the caller.
template <unsigned Elems>
static inline __attribute__((always_inline)) void split_fp16_exact(
    const aie::vector<uint8_t, 2 * Elems>& bytes,
    aie::vector<bfloat16, Elems>& hi,
    aie::vector<bfloat16, Elems>& lo) {
  const auto low8 = aie::filter_even(bytes);
  const auto high8 = aie::filter_odd(bytes);
  const auto low_bits =
      aie::to_float<bfloat16>(low8).template cast_to<uint16_t>();
  const auto high = aie::unpack(high8);

  const auto zero = aie::zeros<uint16_t, Elems>();
  const auto one = aie::broadcast<uint16_t, Elems>(1);
  const auto sign_mask = aie::broadcast<uint16_t, Elems>(0x80u);
  const auto exponent_mask = aie::broadcast<uint16_t, Elems>(0x7cu);
  const auto top_mask = aie::broadcast<uint16_t, Elems>(3u);
  const auto two_mask = aie::broadcast<uint16_t, Elems>(2u);
  const auto hi_bias = aie::broadcast<uint16_t, Elems>(112u << 7);
  const auto sub_bias = aie::broadcast<uint16_t, Elems>(0x3780u);
  const auto lo_bias = aie::broadcast<uint16_t, Elems>(25u << 7);

  const auto sign = aie::upshift(aie::bit_and(sign_mask, high), 8);
  const auto exponent = aie::bit_and(exponent_mask, high);
  const auto top = aie::bit_and(top_mask, high);
  const auto denorm = aie::eq(static_cast<uint16_t>(0), exponent);

  // (E << 7) is shared by the hi and lo exponents.
  const auto exponent7 = aie::upshift(exponent, 5);

  // sign | ((E + 112) << 7) | (top << 5); the mantissa byte fields do not
  // overlap, so one masked shift plus the bias covers both.
  auto hi_normal = aie::bit_or(
      sign, aie::add(aie::upshift(aie::bit_or(exponent, top), 5), hi_bias));

  auto hi_sub = aie::add(
      aie::add(sub_bias, aie::upshift(aie::bit_and(two_mask, top), 6)),
      aie::upshift(
          aie::select(zero, one, aie::eq(static_cast<uint16_t>(3), top)), 6));
  hi_sub = aie::bit_or(hi_sub, sign);
  hi_sub = aie::select(zero, hi_sub,
                       aie::neq(static_cast<uint16_t>(0), top));
  hi = aie::select(hi_normal, hi_sub, denorm).template cast_to<bfloat16>();

  // Denormal lo uses one exponent step more than the normal delta.
  const auto delta = aie::add(aie::sub(exponent7, lo_bias),
                              aie::select(zero, aie::upshift(one, 7), denorm));
  auto lo_bits = aie::add(low_bits, delta);
  lo_bits = aie::bit_or(lo_bits, sign);
  lo = aie::select(zero, lo_bits,
                   aie::neq(static_cast<uint8_t>(0), low8))
           .template cast_to<bfloat16>();
}

static inline __attribute__((always_inline)) void split_fp16x16_exact(
    const uint8_t* p,
    aie::vector<bfloat16, 16>& hi,
    aie::vector<bfloat16, 16>& lo) {
  split_fp16_exact<16>(aie::load_v<32>(p), hi, lo);
}

static inline __attribute__((always_inline)) void split_fp16x32_exact(
    const uint8_t* p0,
    const uint8_t* p1,
    aie::vector<bfloat16, 32>& hi,
    aie::vector<bfloat16, 32>& lo) {
  split_fp16_exact<32>(
      aie::concat(aie::load_v<32>(p0), aie::load_v<32>(p1)), hi, lo);
}

}  // namespace spiral::quants
