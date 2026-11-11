// Probe kernels for `fp16_ops.py`.
//
// Exactly one probe is compiled per translation unit, selected by
// `-DBENCH_PROBE_<NAME>`. Each probe has the same signature so the driver can
// reuse one IRON harness:
//
//   extern "C" void bench_probe(const uint8_t* in, uint8_t* out);
//
// The probes that are meant to fail are deliberately unsupported AIE API
// operations; their compiler diagnostics are the result.

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace {

constexpr int kIters = 32;
constexpr int kInt8Chunk = 64;
// One Q6_K probe record: 8 fp16 d, 8 int8 subgroup scales, 64 int8 weights.
constexpr int kQ6KStride = 128;

// Convert eight contiguous fp16 values to f32 from the raw bits. This mirrors
// `spiral::quants::load_fp16x8`; zero and subnormal inputs are returned as
// zero.
[[maybe_unused]] static inline
    __attribute__((always_inline)) aie::vector<float, 8>
    load_fp16x8(const uint8_t* p) {
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

// Shift-based fp16 -> f32 decode: zero-extend, shift the magnitude into the
// f32 mantissa field, restore the sign, then add the exponent bias. This is
// `((h & 0x7fff) << 13) | sign | 0x38000000`, with zero/subnormal flushed.
[[maybe_unused]] static inline
    __attribute__((always_inline)) aie::vector<float, 8>
    load_fp16x8_shift(const uint8_t* p) {
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

// Direct fp16 -> bf16 hi/lo split, no f32 intermediate. hi is the truncated
// top eight significand bits; lo is the low three bits scaled by the same
// power-of-two exponent. Both are exact.
[[maybe_unused]] static inline __attribute__((always_inline)) void
split_fp16x8_direct(const uint8_t* p,
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
[[maybe_unused]] static inline __attribute__((always_inline)) void
split_bf16_hi_lo(const aie::vector<float, 8>& scales,
                 aie::vector<bfloat16, 8>& hi,
                 aie::vector<bfloat16, 8>& lo) {
  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(scales));
  const auto hi_f = aie::accum<accfloat, 8>(hi).template to_vector<float>();
  lo =
      aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(aie::sub(scales, hi_f)));
}

// Q6_K effective scales need three bf16 terms. Mirrors
// `spiral::quants::q6_k_int8::detail::load_chunk_scales32`.
[[maybe_unused]] static inline __attribute__((always_inline)) void split3(
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

// Decode eight fp16 scales to integer mantissas (11 bits) and bf16
// power-of-two factors. The mantissas are positive; the factor carries sign.
[[maybe_unused]] static inline __attribute__((always_inline)) void
decode_fp16x8(const uint8_t* p,
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

}  // namespace

// Baseline: a 128-byte vector load/store. Always supported.
#if defined(BENCH_PROBE_U16_LOAD_STORE)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto v =
        aie::load_v<64>(reinterpret_cast<const uint16_t*>(in) + i * 64);
    aie::store_v(reinterpret_cast<uint16_t*>(out) + i * 64, v);
  }
  event1();
}

// Baseline: a 128-byte bf16 vector load/store.
#elif defined(BENCH_PROBE_BF16_LOAD_STORE)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto v =
        aie::load_v<64>(reinterpret_cast<const bfloat16*>(in) + i * 64);
    aie::store_v(reinterpret_cast<bfloat16*>(out) + i * 64, v);
  }
  event1();
}

// Expected to fail: scalar float16 arithmetic.
#elif defined(BENCH_PROBE_F16_SCALAR_ADD)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const float16 a = *reinterpret_cast<const float16*>(in);
  const float16 b = *reinterpret_cast<const float16*>(in + 1);
  const float16 c = a + b;
  *reinterpret_cast<float16*>(out) = c;
}

// Expected to fail: a float16 vector cannot be declared.
#elif defined(BENCH_PROBE_F16_VECTOR_DECL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::vector<float16, 32> v;
  (void)v;
  (void)in;
  out[0] = 0;
}

// Expected to fail: no float16 vector load/store.
#elif defined(BENCH_PROBE_F16_VECTOR_LOAD_STORE)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto v = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  aie::store_v(reinterpret_cast<float16*>(out), v);
}

// Expected to fail: float16 is not a 16-bit storage type on XDNA2.
#elif defined(BENCH_PROBE_F16_SIZE)
static_assert(sizeof(float16) == 2, "float16 must be 16 bits");
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  (void)in;
  out[0] = 0;
}

// Expected to fail: int8 -> float16 conversion.
#elif defined(BENCH_PROBE_F16_INT8_CONVERT)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto q = aie::load_v<64>(reinterpret_cast<const int8*>(in));
  const auto v = aie::to_float<float16>(q);
  aie::store_v(reinterpret_cast<float16*>(out), v);
}

// Expected to fail: accfloat -> float16 conversion.
#elif defined(BENCH_PROBE_F16_ACC_CONVERT)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto f = aie::load_v<32>(reinterpret_cast<const float*>(in));
  const auto v = aie::to_vector<float16>(aie::accum<accfloat, 32>(f));
  aie::store_v(reinterpret_cast<float16*>(out), v);
}

// Expected to fail: float16 vector addition.
#elif defined(BENCH_PROBE_F16_ADD)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto a = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  const auto b = aie::load_v<32>(reinterpret_cast<const float16*>(in + 64));
  aie::store_v(reinterpret_cast<float16*>(out), aie::add(a, b));
}

// Expected to fail: float16 vector multiply.
#elif defined(BENCH_PROBE_F16_MUL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto a = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  const auto b = aie::load_v<32>(reinterpret_cast<const float16*>(in + 64));
  const auto r = aie::to_vector<float16>(aie::mul(a, b));
  aie::store_v(reinterpret_cast<float16*>(out), r);
}

// Expected to fail: no float16 x float16 mmul.
#elif defined(BENCH_PROBE_F16_MMUL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  using MMUL = aie::mmul<4, 8, 8, float16, float16>;
  MMUL m;
  const auto a = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  const auto b = aie::load_v<64>(reinterpret_cast<const float16*>(in + 64));
  m.mac(a, b);
  const auto r = aie::to_vector<float16>(m.to_accum());
  aie::store_v(reinterpret_cast<float16*>(out), r);
}

// Supported fallback: raw fp16 bits -> f32.
#elif defined(BENCH_PROBE_FP16_BITS_F32)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto f = load_fp16x8(in + i * 16);
    aie::store_v(reinterpret_cast<float*>(out) + i * 8, f);
  }
  event1();
}

// Supported fallback: raw fp16 bits -> exact bf16 hi/lo pair.
#elif defined(BENCH_PROBE_FP16_BITS_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_bf16_hi_lo(load_fp16x8(in + i * 16), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
  }
  event1();
}

// Shift-based fp16 -> f32 decode.
#elif defined(BENCH_PROBE_FP16_SHIFT_F32)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  event0();
  for (int i = 0; i < kIters; ++i) {
    const auto f = load_fp16x8_shift(in + i * 16);
    aie::store_v(reinterpret_cast<float*>(out) + i * 8, f);
  }
  event1();
}

// Shift-based fp16 decode followed by the bf16 hi/lo split.
#elif defined(BENCH_PROBE_FP16_SHIFT_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_bf16_hi_lo(load_fp16x8_shift(in + i * 16), hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
  }
  event1();
}

// Direct fp16 -> bf16 hi/lo split, no f32 intermediate.
#elif defined(BENCH_PROBE_FP16_DIRECT_SPLIT_BF16)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    aie::vector<bfloat16, 8> hi;
    aie::vector<bfloat16, 8> lo;
    split_fp16x8_direct(in + i * 16, hi, lo);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32), hi);
    aie::store_v(reinterpret_cast<bfloat16*>(out + i * 32 + 16), lo);
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
  split_bf16_hi_lo(load_fp16x8(in), hi8, lo8);
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

// Exact baseline: fp16 scale split into bf16 hi/lo, two bf16 multiplies.
#elif defined(BENCH_PROBE_INT8_SCALE_BF16_2TERM)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::vector<bfloat16, 8> hi8;
  aie::vector<bfloat16, 8> lo8;
  split_bf16_hi_lo(load_fp16x8(in), hi8, lo8);
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

// Alternate exact decoder: fp16 = int16 mantissa * bf16 power-of-two. The
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

// Q6_K baseline: fp16 d * int8 subgroup scale, split into three bf16 terms.
// Scales are prepared once and reused for all payloads.
#elif defined(BENCH_PROBE_Q6K_3TERM_APPLY)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto d8 = load_fp16x8(in);
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
#elif defined(BENCH_PROBE_Q6K_3TERM_FULL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int i = 0; i < kIters; ++i) {
    const uint8_t* rec = in + i * kQ6KStride;
    const auto d8 = load_fp16x8(rec);
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
    decode_fp16x8(rec, mantissa, factor8);
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