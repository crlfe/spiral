// Common tile-local output epilogue: transpose + optional softcap + bf16.

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef PROGRAM_OUTPUT_TILE_ELEMS
#define PROGRAM_OUTPUT_TILE_ELEMS 64
#endif
#ifndef PROGRAM_COLS_TILE
#define PROGRAM_COLS_TILE 16
#endif

namespace program {

// The mmul accumulates f32 into the native [col_half][vec][8] tile.  The
// stored tile keeps that order ([half][vec][8]), which is also the next
// matmul's [K/8][vec][8] input block order.
//
// The optional tanh softcap uses the native bf16 vtanh.  The host passes cap
// and 1/cap as bf16 hi/lo pairs (two 16-bit values per config word), so the
// device reconstructs f32 constants without a division.  cap_lo mainly refines
// the rounding of the bf16 post-scale; inv_lo feeds the f32 pre-scale.
struct OutputStore {
  static constexpr int kTileElems = PROGRAM_OUTPUT_TILE_ELEMS;
  static constexpr int kHalfTile = kTileElems / 2;
  static constexpr int kColsTile = PROGRAM_COLS_TILE;
  static constexpr int kHalfCols = kColsTile / 2;
  static_assert(kTileElems == 4 * kColsTile,
                "output tile must be inputs_vecs * cols_tile");

  using HalfVecF = aie::vector<float, kHalfTile>;
  using HalfVecBf = aie::vector<bfloat16, kHalfTile>;

  static inline void run(const HalfVecF& lo, const HalfVecF& hi,
                         bfloat16* out, int32_t cap_word,
                         int32_t inv_word) {
    if (cap_word == 0) {
      store_half(to_bf16(lo), out, 0);
      store_half(to_bf16(hi), out, 1);
      return;
    }

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

  static inline HalfVecBf to_bf16(const HalfVecF& v) {
    return aie::accum<accfloat, kHalfTile>(v).template to_vector<bfloat16>();
  }

  static inline HalfVecBf softcap(const HalfVecF& v, float cap, float inv) {
    const HalfVecBf t = aie::tanh(aie::mul(v, inv).template to_vector<float>());
    return aie::mul(t, aie::broadcast<bfloat16, kHalfTile>(bfloat16(cap)))
        .template to_vector<bfloat16>();
  }

  static inline void store_half(const HalfVecBf& v, bfloat16* out,
                                int half) {
    const int base = half * kHalfTile;
    aie::store_v(out + base + 0 * kHalfCols,
                 v.template extract<kHalfCols>(0));
    aie::store_v(out + base + 1 * kHalfCols,
                 v.template extract<kHalfCols>(1));
    aie::store_v(out + base + 2 * kHalfCols,
                 v.template extract<kHalfCols>(2));
    aie::store_v(out + base + 3 * kHalfCols,
                 v.template extract<kHalfCols>(3));
  }
};

}  // namespace program

extern "C" void program_output(const float* restrict accum,
                               bfloat16* restrict outputs,
                               int32_t cols_bands,
                               int32_t softcap_hi_lo,
                               int32_t softinv_hi_lo) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  for (int32_t band = 0; band < cols_bands; ++band) {
    const float* src = accum + band * program::OutputStore::kTileElems;
    const aie::vector<float, program::OutputStore::kHalfTile> lo =
        aie::load_v<program::OutputStore::kHalfTile>(src);
    const aie::vector<float, program::OutputStore::kHalfTile> hi =
        aie::load_v<program::OutputStore::kHalfTile>(
            src + program::OutputStore::kHalfTile);
    program::OutputStore::run(
        lo, hi, outputs + band * program::OutputStore::kTileElems,
        softcap_hi_lo, softinv_hi_lo);
  }
}
