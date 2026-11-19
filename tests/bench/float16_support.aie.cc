// Capability kernels for `float16_support.py`.
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
#elif defined(BENCH_PROBE_FLOAT16_SCALAR_ADD)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const float16 a = *reinterpret_cast<const float16*>(in);
  const float16 b = *reinterpret_cast<const float16*>(in + 1);
  const float16 c = a + b;
  *reinterpret_cast<float16*>(out) = c;
}

// Expected to fail: a float16 vector cannot be declared.
#elif defined(BENCH_PROBE_FLOAT16_VECTOR_DECL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  aie::vector<float16, 32> v;
  (void)v;
  (void)in;
  out[0] = 0;
}

// Expected to fail: no float16 vector load/store.
#elif defined(BENCH_PROBE_FLOAT16_VECTOR_LOAD_STORE)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto v = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  aie::store_v(reinterpret_cast<float16*>(out), v);
}

// Expected to fail: float16 is not a 16-bit storage type on XDNA2.
#elif defined(BENCH_PROBE_FLOAT16_SIZE)
static_assert(sizeof(float16) == 2, "float16 must be 16 bits");
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  (void)in;
  out[0] = 0;
}

// Expected to fail: int8 -> float16 conversion.
#elif defined(BENCH_PROBE_FLOAT16_INT8_CONVERT)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto q = aie::load_v<64>(reinterpret_cast<const int8*>(in));
  const auto v = aie::to_float<float16>(q);
  aie::store_v(reinterpret_cast<float16*>(out), v);
}

// Expected to fail: accfloat -> float16 conversion.
#elif defined(BENCH_PROBE_FLOAT16_ACC_CONVERT)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto f = aie::load_v<32>(reinterpret_cast<const float*>(in));
  const auto v = aie::to_vector<float16>(aie::accum<accfloat, 32>(f));
  aie::store_v(reinterpret_cast<float16*>(out), v);
}

// Expected to fail: float16 vector addition.
#elif defined(BENCH_PROBE_FLOAT16_ADD)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto a = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  const auto b = aie::load_v<32>(reinterpret_cast<const float16*>(in + 64));
  aie::store_v(reinterpret_cast<float16*>(out), aie::add(a, b));
}

// Expected to fail: float16 vector multiply.
#elif defined(BENCH_PROBE_FLOAT16_MUL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  const auto a = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  const auto b = aie::load_v<32>(reinterpret_cast<const float16*>(in + 64));
  const auto r = aie::to_vector<float16>(aie::mul(a, b));
  aie::store_v(reinterpret_cast<float16*>(out), r);
}

// Expected to fail: no float16 x float16 mmul.
#elif defined(BENCH_PROBE_FLOAT16_MMUL)
extern "C" void bench_probe(const uint8_t* in, uint8_t* out) {
  using MMUL = aie::mmul<4, 8, 8, float16, float16>;
  MMUL m;
  const auto a = aie::load_v<32>(reinterpret_cast<const float16*>(in));
  const auto b = aie::load_v<64>(reinterpret_cast<const float16*>(in + 64));
  m.mac(a, b);
  const auto r = aie::to_vector<float16>(m.to_accum());
  aie::store_v(reinterpret_cast<float16*>(out), r);
}

#else
#error "no BENCH_PROBE_* selected"
#endif
