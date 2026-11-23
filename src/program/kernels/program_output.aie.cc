// Common tile-local output epilogue: transpose + optional softcap or GEGLU +
// bf16.

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef PROGRAM_OUTPUT_TILE_ELEMS
#define PROGRAM_OUTPUT_TILE_ELEMS 64
#endif
#ifndef PROGRAM_COLS_TILE
#define PROGRAM_COLS_TILE 16
#endif
#ifndef PROGRAM_EPILOGUE_NONE
#define PROGRAM_EPILOGUE_NONE 0
#endif
#ifndef PROGRAM_EPILOGUE_SOFTCAP
#define PROGRAM_EPILOGUE_SOFTCAP 1
#endif
#ifndef PROGRAM_EPILOGUE_GEGLU
#define PROGRAM_EPILOGUE_GEGLU 2
#endif

namespace program {

using HalfVecF = aie::vector<float, PROGRAM_OUTPUT_TILE_ELEMS / 2>;
using HalfVecBf = aie::vector<bfloat16, PROGRAM_OUTPUT_TILE_ELEMS / 2>;

constexpr int32_t kEpilogueTileElems = PROGRAM_OUTPUT_TILE_ELEMS;
constexpr int32_t kEpilogueHalfTile = kEpilogueTileElems / 2;
constexpr int32_t kEpilogueColsTile = PROGRAM_COLS_TILE;
constexpr int32_t kEpilogueHalfCols = kEpilogueColsTile / 2;

// GELU tanh approximation, matching ggml_gelu_f32:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
// The tanh uses the native bf16 vtanh; the surrounding arithmetic stays f32.
constexpr float kGeluC = 0.7978845608028654f;
constexpr float kGeluA = 0.044715f;

// The stored tile keeps the native [half][vec][8] order, which is also the
// next matmul's [K/8][vec][8] input block order.
inline HalfVecBf to_bf16(const HalfVecF& v) {
  return aie::accum<accfloat, kEpilogueHalfTile>(v)
      .template to_vector<bfloat16>();
}

inline void store_half(const HalfVecBf& v, bfloat16* out, int half) {
  const int base = half * kEpilogueHalfTile;
  aie::store_v(out + base + 0 * kEpilogueHalfCols,
               v.template extract<kEpilogueHalfCols>(0));
  aie::store_v(out + base + 1 * kEpilogueHalfCols,
               v.template extract<kEpilogueHalfCols>(1));
  aie::store_v(out + base + 2 * kEpilogueHalfCols,
               v.template extract<kEpilogueHalfCols>(2));
  aie::store_v(out + base + 3 * kEpilogueHalfCols,
               v.template extract<kEpilogueHalfCols>(3));
}

struct OutputStore {
  static constexpr int kTileElems = kEpilogueTileElems;
  static constexpr int kHalfTile = kEpilogueHalfTile;
  static constexpr int kColsTile = kEpilogueColsTile;
  static_assert(kTileElems == 4 * kColsTile,
                "output tile must be inputs_vecs * cols_tile");

  static inline void plain(const HalfVecF& lo, const HalfVecF& hi,
                           bfloat16* out) {
    store_half(to_bf16(lo), out, 0);
    store_half(to_bf16(hi), out, 1);
  }

  static inline void softcap_store(const HalfVecF& lo, const HalfVecF& hi,
                                   bfloat16* out, int32_t cap_word,
                                   int32_t inv_word) {
    const float cap = bf16_word_lo(cap_word) + bf16_word_hi(cap_word);
    const float inv = bf16_word_lo(inv_word) + bf16_word_hi(inv_word);
    store_half(softcap(lo, cap, inv), out, 0);
    store_half(softcap(hi, cap, inv), out, 1);
  }

 private:
  static inline float bf16_bits_to_float(uint32_t bits) {
    union {
      uint32_t bits;
      float value;
    } v{bits << 16};
    return v.value;
  }

  static inline float bf16_word_lo(int32_t word) {
    return bf16_bits_to_float(static_cast<uint32_t>(word) & 0xffffu);
  }

  static inline float bf16_word_hi(int32_t word) {
    return bf16_bits_to_float(static_cast<uint32_t>(word) >> 16);
  }

  static inline HalfVecBf softcap(const HalfVecF& v, float cap, float inv) {
    const HalfVecBf t = aie::tanh(aie::mul(v, inv).template to_vector<float>());
    return aie::mul(t, aie::broadcast<bfloat16, kHalfTile>(bfloat16(cap)))
        .template to_vector<bfloat16>();
  }
};

// GEGLU over one 32-element vector.  f32 vector arithmetic is emulated on
// AIE2P with multiple bf16 products and conversions, so keep the polynomial
// and the activation in bf16 (the native width) and only widen for the native
// tanh argument.  The output is bf16 anyway, and the two-launch path also
// rounds gate/up to bf16 before GELU.
struct GluStore {
  static inline HalfVecBf narrow(const HalfVecF& v) {
    return aie::accum<accfloat, kEpilogueHalfTile>(v)
        .template to_vector<bfloat16>();
  }

  static inline HalfVecF widen(const HalfVecBf& v) {
    return aie::accum<accfloat, kEpilogueHalfTile>(v)
        .template to_vector<float>();
  }

  static inline HalfVecBf bmul(const HalfVecBf& a, const HalfVecBf& b) {
    return aie::mul(a, b).template to_vector<bfloat16>();
  }

  static inline HalfVecBf run(const HalfVecF& gate_f, const HalfVecF& up_f) {
    const HalfVecBf one =
        aie::broadcast<bfloat16, kEpilogueHalfTile>(bfloat16(1.0f));
    const HalfVecBf half =
        aie::broadcast<bfloat16, kEpilogueHalfTile>(bfloat16(0.5f));
    const HalfVecBf a =
        aie::broadcast<bfloat16, kEpilogueHalfTile>(bfloat16(kGeluA));
    const HalfVecBf c =
        aie::broadcast<bfloat16, kEpilogueHalfTile>(bfloat16(kGeluC));
    const HalfVecBf gate = narrow(gate_f);
    const HalfVecBf up = narrow(up_f);
    const HalfVecBf x2 = bmul(gate, gate);
    const HalfVecBf poly = aie::add(one, bmul(a, x2));
    const HalfVecBf inner = bmul(c, bmul(gate, poly));
    const HalfVecBf t = aie::tanh(widen(inner));
    const HalfVecBf gelu = bmul(bmul(gate, aie::add(one, t)), half);
    return bmul(gelu, up);
  }
};

}  // namespace program

// `epilogue` selects the tile epilogue kind.  EPILOGUE_GEGLU expects a combined
// gate+up call: the first `cols_bands / 2` bands are the gate and the second
// half is up.  It stores the product in the gate bands so the caller's packed
// output buffer holds the next matmul's input at offset 0.
extern "C" void program_output(const float* restrict accum,
                               bfloat16* restrict outputs,
                               int32_t cols_bands,
                               int32_t softcap_hi_lo,
                               int32_t softinv_hi_lo,
                               int32_t epilogue) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  if (epilogue == PROGRAM_EPILOGUE_GEGLU) {
    const int32_t gate_bands = cols_bands / 2;
    for (int32_t band = 0; band < gate_bands; ++band) {
      const float* gate = accum + band * program::kEpilogueTileElems;
      const float* up =
          accum + (band + gate_bands) * program::kEpilogueTileElems;
      bfloat16* out = outputs + band * program::kEpilogueTileElems;
      // Keep one copy of the GEGLU body: the unroll would otherwise duplicate
      // the vector tail for the two halves and grow the resident image.
#pragma clang loop unroll(disable)
      for (int half = 0; half < 2; ++half) {
        const program::HalfVecF gate_v = aie::load_v<
            program::kEpilogueHalfTile>(
            gate + half * program::kEpilogueHalfTile);
        const program::HalfVecF up_v = aie::load_v<
            program::kEpilogueHalfTile>(
            up + half * program::kEpilogueHalfTile);
        program::store_half(program::GluStore::run(gate_v, up_v), out, half);
      }
    }
    return;
  }

  if (epilogue == PROGRAM_EPILOGUE_SOFTCAP) {
    for (int32_t band = 0; band < cols_bands; ++band) {
      const float* src = accum + band * program::kEpilogueTileElems;
      const program::HalfVecF lo =
          aie::load_v<program::kEpilogueHalfTile>(src);
      const program::HalfVecF hi =
          aie::load_v<program::kEpilogueHalfTile>(
              src + program::kEpilogueHalfTile);
      program::OutputStore::softcap_store(
          lo, hi, outputs + band * program::kEpilogueTileElems,
          softcap_hi_lo, softinv_hi_lo);
    }
    return;
  }

  for (int32_t band = 0; band < cols_bands; ++band) {
    const float* src = accum + band * program::kEpilogueTileElems;
    const program::HalfVecF lo =
        aie::load_v<program::kEpilogueHalfTile>(src);
    const program::HalfVecF hi =
        aie::load_v<program::kEpilogueHalfTile>(
            src + program::kEpilogueHalfTile);
    program::OutputStore::plain(
        lo, hi, outputs + band * program::kEpilogueTileElems);
  }
}
