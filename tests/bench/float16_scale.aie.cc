// Scale split and application kernels for `float16_scale.py`.
//
// Exactly one probe is compiled per translation unit, selected by
// `-DBENCH_PROBE_<NAME>`.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "float16_bench.aie.h"

using namespace bench_float16;

namespace {

constexpr int kIters = 32;
constexpr int kInt8Chunk = 64;
// One Q6_K probe record: 8 float16 d, 8 int8 subgroup scales, 64 int8 weights.
constexpr int kQ6KStride = 128;

}  // namespace

// Supported fallback: raw float16 bits -> exact bf16 hi/lo pair.
#if defined(BENCH_PROBE_FLOAT16_BITS_SPLIT_BF16_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_bf16_hi_lo(load_float16x8(in + i * 16), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
  }
  event1();
}

// Shift-based float16 decode followed by the bf16 hi/lo split.
#elif defined(BENCH_PROBE_FLOAT16_SHIFT_SPLIT_BF16_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_bf16_hi_lo(load_float16x8_shift(in + i * 16), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
  }
  event1();
}

// Host-pre-biased float16 decode followed by the bf16 hi/lo split.
#elif defined(BENCH_PROBE_FLOAT16_BIASED_SHIFT_SPLIT_BF16_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_bf16_hi_lo(load_float16x8_biased_shift(in + i * 16), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
  }
  event1();
}


// 32-value mask/rebias split.
#elif defined(BENCH_PROBE_FLOAT16_BITS_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 32> hi;
    aie::vector<bfloat16, 32> lo;
    split_bf16_hi_lo32(load_float16x32(in + i * 64), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64, hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64 + 32, lo);
  }
  event1();
}

// 32-value direct split.
#elif defined(BENCH_PROBE_FLOAT16_DIRECT_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 32> hi;
    aie::vector<bfloat16, 32> lo;
    split_float16x32_direct(in + i * 64, hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64, hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64 + 32, lo);
  }
  event1();
}

// 32-value shift split, filling one native bf16 vector for each term.
#elif defined(BENCH_PROBE_FLOAT16_SHIFT_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 32> hi;
    aie::vector<bfloat16, 32> lo;
    split_bf16_hi_lo32(load_float16x32_shift(in + i * 64), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64, hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64 + 32, lo);
  }
  event1();
}

#elif defined(BENCH_PROBE_FLOAT16_BIASED_SHIFT_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 32> hi;
    aie::vector<bfloat16, 32> lo;
    split_bf16_hi_lo32(load_float16x32_biased_shift(in + i * 64), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64, hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64 + 32, lo);
  }
  event1();
}

// Direct float16 -> bf16 hi/lo split, no f32 intermediate.
#elif defined(BENCH_PROBE_FLOAT16_DIRECT_SPLIT_BF16_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_float16x8_direct(in + i * 16, hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
  }
  event1();
}

// Host-packed 16-value two-term split. The host normalizes each float16 to a
// bf16 exponent (biased by 102) plus a nine-bit fraction and packs the top
// seven fraction bits and a two-bit residual code into the canonical two
// bytes. The device produces the hi and lo bf16 vectors with integer bit ops
// only; the residual code already carries lo's exponent delta and mantissa
// bit, so there is no float16 decode, float conversion, or normalization in the
// kernel. Normal and subnormal inputs share the same path.
#elif defined(BENCH_PROBE_FLOAT16_PACKED16_SPLIT_BF16_X16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto magnitude_mask = aie::broadcast<uint16_t, 16>(0x7fffu);
  const auto fraction_mask = aie::broadcast<uint16_t, 16>(0x7f80u);
  const auto sign_mask = aie::broadcast<uint16_t, 16>(0x8000u);
  const auto hi_bias = aie::broadcast<uint16_t, 16>(102u << 7);
  const auto lo_bias = aie::broadcast<uint16_t, 16>((102u - 9u) << 7);
  const auto zero = aie::zeros<uint16_t, 16>();
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto packed =
        aie::load_v<16>(reinterpret_cast<const uint16_t*>(in) + i * 16);
    const auto sign = aie::bit_and(packed, sign_mask);
    const auto fraction = aie::logical_downshift(
        aie::bit_and(packed, magnitude_mask), 2);
    const auto code = aie::bit_and(static_cast<uint16_t>(3), packed);

    auto hi = aie::bit_or(aie::add(fraction, hi_bias), sign);
    auto lo = aie::add(aie::add(aie::bit_and(fraction, fraction_mask), lo_bias),
                       aie::upshift(code, 6));
    lo = aie::bit_or(lo, sign);

    const auto exponent =
        aie::bit_and(static_cast<uint16_t>(0x7e00), packed);
    const auto nonzero = aie::neq(static_cast<uint16_t>(0), exponent);
    const auto code_nonzero = aie::neq(static_cast<uint16_t>(0), code);
    hi = aie::select(zero, hi, nonzero);
    lo = aie::select(zero, lo, code_nonzero);

    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 32, hi);
    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 32 + 16, lo);
  }
  event1();
}


// 32-value packed16 split.
#elif defined(BENCH_PROBE_FLOAT16_PACKED16_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto magnitude_mask = aie::broadcast<uint16_t, 32>(0x7fffu);
  const auto fraction_mask = aie::broadcast<uint16_t, 32>(0x7f80u);
  const auto sign_mask = aie::broadcast<uint16_t, 32>(0x8000u);
  const auto hi_bias = aie::broadcast<uint16_t, 32>(102u << 7);
  const auto lo_bias = aie::broadcast<uint16_t, 32>((102u - 9u) << 7);
  const auto zero = aie::zeros<uint16_t, 32>();
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto packed =
        aie::load_v<32>(reinterpret_cast<const uint16_t*>(in) + i * 32);
    const auto sign = aie::bit_and(packed, sign_mask);
    const auto fraction = aie::logical_downshift(
        aie::bit_and(packed, magnitude_mask), 2);
    const auto code = aie::bit_and(static_cast<uint16_t>(3), packed);

    auto hi = aie::bit_or(aie::add(fraction, hi_bias), sign);
    auto lo = aie::add(aie::add(aie::bit_and(fraction, fraction_mask), lo_bias),
                       aie::upshift(code, 6));
    lo = aie::bit_or(lo, sign);

    const auto exponent =
        aie::bit_and(static_cast<uint16_t>(0x7e00), packed);
    const auto nonzero = aie::neq(static_cast<uint16_t>(0), exponent);
    const auto code_nonzero = aie::neq(static_cast<uint16_t>(0), code);
    hi = aie::select(zero, hi, nonzero);
    lo = aie::select(zero, lo, code_nonzero);

    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 64, hi);
    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 64 + 32, lo);
  }
  event1();
}

// Exponent-add split of canonical float16 bytes. The low byte becomes bf16(L)
// through the native uint8 -> bf16 conversion; its exponent field is then
// adjusted by (E - 25) with an integer add, and the high byte supplies hi.
// This avoids both a lookup table and a bf16 multiply. The flush variant drops
// E == 0; the denorm variant normalizes the top two mantissa bits of a denorm
// into hi and scales lo by 2^-24 instead of 2^(E-25).
#elif defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_FLUSH_X16) || \
    defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_DENORM_X16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto zero = aie::zeros<uint16_t, 16>();
  const auto one = aie::broadcast<uint16_t, 16>(1);
  const auto sign_mask = aie::broadcast<uint16_t, 16>(0x80u);
  const auto exponent_mask = aie::broadcast<uint16_t, 16>(0x7cu);
  const auto top_mask = aie::broadcast<uint16_t, 16>(3u);
  const auto two_mask = aie::broadcast<uint16_t, 16>(2u);
  const auto hi_bias = aie::broadcast<uint16_t, 16>(112u << 7);
  const auto sub_bias = aie::broadcast<uint16_t, 16>(0x3780u);
  const auto shift_25 = aie::broadcast<uint16_t, 16>(25u);
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto bytes =
        aie::load_v<32>(reinterpret_cast<const uint8_t*>(in) + i * 32);
    const auto low8 = aie::filter_even(bytes);
    const auto high8 = aie::filter_odd(bytes);
    const auto low_bits =
        aie::to_float<bfloat16>(low8).template cast_to<uint16_t>();
    const auto high = aie::unpack(high8);

    const auto sign =
        aie::upshift(aie::bit_and(sign_mask, high), 8);
    const auto exponent =
        aie::logical_downshift(aie::bit_and(exponent_mask, high), 2);
    const auto top = aie::bit_and(top_mask, high);

    auto hi = aie::bit_or(
        sign, aie::add(aie::upshift(exponent, 7), hi_bias));
    hi = aie::bit_or(hi, aie::upshift(top, 5));

    auto delta = aie::sub(exponent, shift_25);
    const auto denorm = aie::eq(static_cast<uint16_t>(0), exponent);
#if defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_DENORM_X16)
    delta = aie::add(delta, aie::select(zero, one, denorm));

    auto hi_sub = aie::add(
        aie::add(sub_bias, aie::upshift(aie::bit_and(two_mask, top), 6)),
        aie::upshift(aie::select(zero, one,
                                 aie::eq(static_cast<uint16_t>(3), top)),
                     6));
    hi_sub = aie::bit_or(hi_sub, sign);
    hi_sub = aie::select(zero, hi_sub,
                         aie::neq(static_cast<uint16_t>(0), top));
    hi = aie::select(hi, hi_sub, denorm);
#else
    hi = aie::select(zero, hi, aie::neq(static_cast<uint16_t>(0), exponent));
#endif

    auto lo = aie::add(low_bits, aie::upshift(delta, 7));
    lo = aie::bit_or(lo, sign);
    lo = aie::select(zero, lo, aie::neq(static_cast<uint8_t>(0), low8));
#if !defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_DENORM_X16)
    lo = aie::select(zero, lo, aie::neq(static_cast<uint16_t>(0), exponent));
#endif

    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 32, hi);
    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 32 + 16, lo);
  }
  event1();
}


// 32-value exponent-add split.
#elif defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_FLUSH) || \
    defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_DENORM)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto zero = aie::zeros<uint16_t, 32>();
  const auto one = aie::broadcast<uint16_t, 32>(1);
  const auto sign_mask = aie::broadcast<uint16_t, 32>(0x80u);
  const auto exponent_mask = aie::broadcast<uint16_t, 32>(0x7cu);
  const auto top_mask = aie::broadcast<uint16_t, 32>(3u);
  const auto two_mask = aie::broadcast<uint16_t, 32>(2u);
  const auto hi_bias = aie::broadcast<uint16_t, 32>(112u << 7);
  const auto sub_bias = aie::broadcast<uint16_t, 32>(0x3780u);
  const auto shift_25 = aie::broadcast<uint16_t, 32>(25u);
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto bytes =
        aie::load_v<64>(reinterpret_cast<const uint8_t*>(in) + i * 64);
    const auto low8 = aie::filter_even(bytes);
    const auto high8 = aie::filter_odd(bytes);
    const auto low_bits =
        aie::to_float<bfloat16>(low8).template cast_to<uint16_t>();
    const auto high = aie::unpack(high8);

    const auto sign = aie::upshift(aie::bit_and(sign_mask, high), 8);
    const auto exponent =
        aie::logical_downshift(aie::bit_and(exponent_mask, high), 2);
    const auto top = aie::bit_and(top_mask, high);

    auto hi = aie::bit_or(
        sign, aie::add(aie::upshift(exponent, 7), hi_bias));
    hi = aie::bit_or(hi, aie::upshift(top, 5));

    auto delta = aie::sub(exponent, shift_25);
    const auto denorm = aie::eq(static_cast<uint16_t>(0), exponent);
#if defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_DENORM)
    delta = aie::add(delta, aie::select(zero, one, denorm));

    auto hi_sub = aie::add(
        aie::add(sub_bias, aie::upshift(aie::bit_and(two_mask, top), 6)),
        aie::upshift(aie::select(zero, one,
                                 aie::eq(static_cast<uint16_t>(3), top)),
                     6));
    hi_sub = aie::bit_or(hi_sub, sign);
    hi_sub = aie::select(zero, hi_sub,
                         aie::neq(static_cast<uint16_t>(0), top));
    hi = aie::select(hi, hi_sub, denorm);
#else
    hi = aie::select(zero, hi, aie::neq(static_cast<uint16_t>(0), exponent));
#endif

    auto lo = aie::add(low_bits, aie::upshift(delta, 7));
    lo = aie::bit_or(lo, sign);
    lo = aie::select(zero, lo, aie::neq(static_cast<uint8_t>(0), low8));
#if !defined(BENCH_PROBE_FLOAT16_EXPADD_SPLIT_BF16_DENORM)
    lo = aie::select(zero, lo, aie::neq(static_cast<uint16_t>(0), exponent));
#endif

    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 64, hi);
    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 64 + 32, lo);
  }
  event1();
}

// Supported emulated path: int8 -> bf16 -> f32 multiply with a f32 scale.
#elif defined(BENCH_PROBE_INT8_SCALE_F32)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  const auto scale =
      aie::broadcast<float, kInt8Chunk>(*reinterpret_cast<const float*>(in));
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::to_float<bfloat16>(aie::load_v<64>(
        reinterpret_cast<const int8*>(in + 64) + i * kInt8Chunk));
    const auto qf =
        aie::accum<accfloat, kInt8Chunk>(q).template to_vector<float>();
    const auto r = aie::mul(qf, scale).template to_vector<float>();
    aie::store_v(reinterpret_cast<float*>(out) + i * kInt8Chunk, r);
  }
  event1();
}

// Supported path: int8 -> bf16 multiply with a single bf16 scale.
#elif defined(BENCH_PROBE_INT8_SCALE_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  const auto scale = aie::broadcast<bfloat16, kInt8Chunk>(
      *reinterpret_cast<const bfloat16*>(in));
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::to_float<bfloat16>(aie::load_v<64>(
        reinterpret_cast<const int8*>(in + 64) + i * kInt8Chunk));
    const auto r = aie::mul(q, scale).template to_vector<bfloat16>();
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk, r);
  }
  event1();
}

// Same two-term scale application through aie::accumulate. The coefficient
// vector is scalar-per-data-vector, so this only works for a uniform scale.
#elif defined(BENCH_PROBE_INT8_SCALE_BF16_ACCUM)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::vector<bfloat16, 8> hi8;
  aie::vector<bfloat16, 8> lo8;
  split_bf16_hi_lo(load_float16x8(in), hi8, lo8);
  auto coeff = aie::zeros<bfloat16, 8>();
  coeff.set(hi8.get(0), 0);
  coeff.set(lo8.get(0), 1);
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::to_float<bfloat16>(aie::load_v<64>(
        reinterpret_cast<const int8*>(in + 64) + i * kInt8Chunk));
    const auto acc = aie::accumulate<kInt8Chunk>(coeff, 0, q, q);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk,
                 acc.template to_vector<bfloat16>());
  }
  event1();
}

// Exact baseline: float16 scale split into bf16 hi/lo, two bf16 multiplies.
#elif defined(BENCH_PROBE_INT8_SCALE_BF16_2TERM)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::vector<bfloat16, 8> hi8;
  aie::vector<bfloat16, 8> lo8;
  split_bf16_hi_lo(load_float16x8(in), hi8, lo8);
  const auto hi = aie::broadcast<bfloat16, kInt8Chunk>(hi8.get(0));
  const auto lo = aie::broadcast<bfloat16, kInt8Chunk>(lo8.get(0));
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::to_float<bfloat16>(aie::load_v<64>(
        reinterpret_cast<const int8*>(in + 64) + i * kInt8Chunk));
    auto acc = aie::mul(q, hi);
    acc = aie::mac(acc, q, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk,
                 acc.template to_vector<bfloat16>());
  }
  event1();
}

// Alternate exact decoder: float16 = int16 mantissa * bf16 power-of-two. The
// int8 x int16 product is exact in int32, and rounding to bf16 commutes with
// the power-of-two factor, so this matches the two-term bf16 split.
#elif defined(BENCH_PROBE_INT8_SCALE_INT_SPLIT) || \
    defined(BENCH_PROBE_INT8_SCALE_INT_MUL) ||     \
    defined(BENCH_PROBE_INT8_SCALE_INT_ROUND)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto bits = *reinterpret_cast<const uint16_t*>(in);
  const int16_t mantissa = static_cast<int16_t>((bits & 0x03ffu) | 0x0400u);
  const uint16_t factor_bits =
      static_cast<uint16_t>(((bits & 0x7c00u) >> 3) + 0x3300u) |
      static_cast<uint16_t>(bits & 0x8000u);
  const auto factor = aie::broadcast<bfloat16, kInt8Chunk>(
      *reinterpret_cast<const bfloat16*>(&factor_bits));
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::load_v<64>(reinterpret_cast<const int8*>(in + 64) +
                                   i * kInt8Chunk);
    const auto scaled = aie::mul(q, mantissa);
#if defined(BENCH_PROBE_INT8_SCALE_INT_MUL)
    aie::store_v(reinterpret_cast<int32*>(out) + i * kInt8Chunk,
                 scaled.template to_vector<int32>());
#else
    const auto rounded =
        aie::to_float<bfloat16>(scaled.template to_vector<int32>());
#if defined(BENCH_PROBE_INT8_SCALE_INT_ROUND)
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk, rounded);
#else
    const auto r = aie::mul(rounded, factor).template to_vector<bfloat16>();
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk, r);
#endif
#endif
  }
  event1();
}

// Q6_K three-term scale preparation only. The 32-wide default builds one
// product lane per subgroup scale; the x8 variant keeps the old eight-column
// group shape.
#elif defined(BENCH_PROBE_Q6K_3TERM_PREP)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const uint8_t* rec = in + i * 64;
    const auto d8 = load_float16x8(rec);
    const auto d32 = aie::concat(aie::concat(d8, d8), aie::concat(d8, d8));
    const auto s32 = aie::to_float<float>(
        aie::load_v<32>(reinterpret_cast<const int8*>(rec + 32)));
    const auto product = aie::mul(d32, s32).template to_vector<float>();
    aie::vector<bfloat16, 32> hi;
    aie::vector<bfloat16, 32> mid;
    aie::vector<bfloat16, 32> lo;
    split3_32(product, hi, mid, lo);
    auto* dst = reinterpret_cast<bfloat16*>(out) + i * 96;
    aie::store_v(dst, hi);
    aie::store_v(dst + 32, mid);
    aie::store_v(dst + 64, lo);
  }
  event1();
}

#elif defined(BENCH_PROBE_Q6K_3TERM_PREP_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const uint8_t* rec = in + i * 32;
    const auto d8 = load_float16x8(rec);
    const auto s8 = aie::to_float<float>(
                      aie::load_v<16>(reinterpret_cast<const int8*>(rec + 16)))
                      .template extract<8>(0);
    const auto product = aie::mul(d8, s8).template to_vector<float>();
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> mid;
    aie::vector<bfloat16, 8> lo;
    split3(product, hi, mid, lo);
    auto* dst = reinterpret_cast<bfloat16*>(out) + i * 24;
    aie::store_v(dst, hi);
    aie::store_v(dst + 8, mid);
    aie::store_v(dst + 16, lo);
  }
  event1();
}

// Q6_K baseline: float16 d * int8 subgroup scale, split into three bf16 terms.
// Scales are prepared once and reused for all payloads.
#elif defined(BENCH_PROBE_Q6K_3TERM_APPLY_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto d8 = load_float16x8(in);
  const auto s8 = aie::to_float<float>(
                      aie::load_v<16>(reinterpret_cast<const int8*>(in + 16)))
                      .template extract<8>(0);
  const auto product = aie::mul(d8, s8).template to_vector<float>();
  aie::vector<bfloat16, 8> hi8;
  aie::vector<bfloat16, 8> mid8;
  aie::vector<bfloat16, 8> lo8;
  split3(product, hi8, mid8, lo8);
  const auto hi = aie::broadcast_vector<8, bfloat16, kInt8Chunk>(hi8, 0);
  const auto mid = aie::broadcast_vector<8, bfloat16, kInt8Chunk>(mid8, 0);
  const auto lo = aie::broadcast_vector<8, bfloat16, kInt8Chunk>(lo8, 0);
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::to_float<bfloat16>(aie::load_v<64>(
        reinterpret_cast<const int8*>(in + 64 + i * kQ6KStride)));
    auto acc = aie::mul(q, hi);
    acc = aie::mac(acc, q, mid);
    acc = aie::mac(acc, q, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk,
                 acc.template to_vector<bfloat16>());
  }
  event1();
}

// Q6_K baseline with scale preparation inside the loop.
#elif defined(BENCH_PROBE_Q6K_3TERM_FULL_X8)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const uint8_t* rec = in + i * kQ6KStride;
    const auto d8 = load_float16x8(rec);
    const auto s8 =
        aie::to_float<float>(
            aie::load_v<16>(reinterpret_cast<const int8*>(rec + 16)))
            .template extract<8>(0);
    const auto product = aie::mul(d8, s8).template to_vector<float>();
    aie::vector<bfloat16, 8> hi8;
    aie::vector<bfloat16, 8> mid8;
    aie::vector<bfloat16, 8> lo8;
    split3(product, hi8, mid8, lo8);
    const auto hi = aie::broadcast_vector<8, bfloat16, kInt8Chunk>(hi8, 0);
    const auto mid = aie::broadcast_vector<8, bfloat16, kInt8Chunk>(mid8, 0);
    const auto lo = aie::broadcast_vector<8, bfloat16, kInt8Chunk>(lo8, 0);
    const auto q = aie::to_float<bfloat16>(
        aie::load_v<64>(reinterpret_cast<const int8*>(rec + 64)));
    auto acc = aie::mul(q, hi);
    acc = aie::mac(acc, q, mid);
    acc = aie::mac(acc, q, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk,
                 acc.template to_vector<bfloat16>());
  }
  event1();
}

// Q6_K integer scale: M_eff = M_d * int8_scale (19 bits), pre-split on the
// host into two int16 terms. Scales are prepared once and reused.
#elif defined(BENCH_PROBE_Q6K_INT_APPLY)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto mhi8 = aie::load_v<8>(reinterpret_cast<const int16*>(in));
  const auto mlo8 = aie::load_v<8>(reinterpret_cast<const int16*>(in + 16));
  const auto factor8 = aie::load_v<8>(reinterpret_cast<const uint16*>(in + 32))
                           .template cast_to<bfloat16>();
  const auto mhi = aie::broadcast_vector<8, int16, kInt8Chunk>(mhi8, 0);
  const auto mlo = aie::broadcast_vector<8, int16, kInt8Chunk>(mlo8, 0);
  const auto factor =
      aie::broadcast_vector<8, bfloat16, kInt8Chunk>(factor8, 0);
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto q = aie::load_v<64>(
        reinterpret_cast<const int8*>(in + 64 + i * kQ6KStride));
    const auto phi = aie::mul(q, mhi).template to_vector<int32>();
    const auto plo = aie::mul(q, mlo).template to_vector<int32>();
    const auto p = aie::add(aie::upshift(phi, 8), plo);
    const auto rounded = aie::to_float<bfloat16>(p);
    const auto r = aie::mul(rounded, factor).template to_vector<bfloat16>();
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk, r);
  }
  event1();
}

// Q6_K integer scale with device-side preparation, using acc32 narrowing
// instead of pack() (which hits a Peano legalizer bug in this combination).
#elif defined(BENCH_PROBE_Q6K_INT_FULL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const uint8_t* rec = in + i * kQ6KStride;
    aie::vector<int16, 8> mantissa;
    aie::vector<bfloat16, 8> factor8;
    decode_float16x8(rec, mantissa, factor8);
    const auto subgroup =
        aie::load_v<16>(reinterpret_cast<const int8*>(rec + 16));
    const auto meff = aie::mul(subgroup, aie::concat(mantissa, mantissa))
                          .template to_vector<int32>()
                          .template extract<8>(0);
    const auto mhi = aie::broadcast_vector<8, int16, kInt8Chunk>(
        aie::accum<acc32, 8>(aie::downshift(meff, 8))
            .template to_vector<int16>(),
        0);
    const auto mlo = aie::broadcast_vector<8, int16, kInt8Chunk>(
        aie::accum<acc32, 8>(aie::bit_and(meff, aie::broadcast<int32, 8>(0xff)))
            .template to_vector<int16>(),
        0);
    const auto factor =
        aie::broadcast_vector<8, bfloat16, kInt8Chunk>(factor8, 0);
    const auto q = aie::load_v<64>(reinterpret_cast<const int8*>(rec + 64));
    const auto phi = aie::mul(q, mhi).template to_vector<int32>();
    const auto plo = aie::mul(q, mlo).template to_vector<int32>();
    const auto p = aie::add(aie::upshift(phi, 8), plo);
    const auto rounded = aie::to_float<bfloat16>(p);
    const auto r = aie::mul(rounded, factor).template to_vector<bfloat16>();
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * kInt8Chunk, r);
  }
  event1();
}


#else
#error "no BENCH_PROBE_* selected"
#endif
