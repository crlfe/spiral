#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/common/tile.h"
#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

namespace detail {

// Load the per-column fp16 `d` values once per record and split them into
// exact bf16 hi/lo terms. The product of `d` and an 8-bit subgroup scale needs
// up to 18 significand bits, so the product is split into three bf16 terms:
// two terms are not always enough to round to the same bf16 as an fp32
// multiply.
static inline __attribute__((always_inline)) void load_d16(
    const uint8_t* record,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& d_hi32,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& d_lo32) {
  aie::vector<bfloat16, 8> d_hi8;
  aie::vector<bfloat16, 8> d_lo8;
  split_bf16_hi_lo(load_fp16x8(record_scales(record)), d_hi8, d_lo8);
  d_hi32 = aie::broadcast_vector<kTileCols, bfloat16, kPackedChunkScalesBytes>(
      d_hi8, 0);
  d_lo32 = aie::broadcast_vector<kTileCols, bfloat16, kPackedChunkScalesBytes>(
      d_lo8, 0);
}

static inline __attribute__((always_inline)) void load_chunk_scales32(
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& d_hi32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& d_lo32,
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& hi,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& mid,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& lo) {
  const auto subgroup_scales = aie::to_float<bfloat16>(
      aie::load_unaligned_v<kPackedChunkScalesBytes>(packed_scales, 16)
          .template cast_to<int8>());
  auto accumulator = aie::mul(d_hi32, subgroup_scales);
  accumulator = aie::mac(accumulator, d_lo32, subgroup_scales);
  const auto product = accumulator.template to_vector<float>();

  hi = aie::to_vector<bfloat16>(
      aie::accum<accfloat, kPackedChunkScalesBytes>(product));
  const auto hi_f32 =
      aie::accum<accfloat, kPackedChunkScalesBytes>(hi).to_vector<float>();
  const auto residual1 = aie::sub(product, hi_f32);

  mid = aie::to_vector<bfloat16>(
      aie::accum<accfloat, kPackedChunkScalesBytes>(residual1));
  const auto mid_f32 =
      aie::accum<accfloat, kPackedChunkScalesBytes>(mid).to_vector<float>();
  const auto residual2 = aie::sub(residual1, mid_f32);

  lo = aie::to_vector<bfloat16>(
      aie::accum<accfloat, kPackedChunkScalesBytes>(residual2));
}

// Expand one subgroup's eight per-column scale terms to 64 lanes.
static inline __attribute__((always_inline)) void load_subgroup_scales64(
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& hi32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& mid32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& lo32,
    int subgroup,
    aie::vector<bfloat16, kMmulWeightsElems>& hi,
    aie::vector<bfloat16, kMmulWeightsElems>& mid,
    aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  hi = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(hi32,
                                                                     subgroup);
  mid = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(mid32,
                                                                      subgroup);
  lo = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(lo32,
                                                                     subgroup);
}

// Apply the three exact bf16 scale terms to one 64-lane values vector.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    apply_scales(const aie::vector<bfloat16, kMmulWeightsElems>& values,
                 const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                 const aie::vector<bfloat16, kMmulWeightsElems>& mid,
                 const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  auto accumulator = aie::mul(values, hi);
  accumulator = aie::mac(accumulator, values, mid);
  accumulator = aie::mac(accumulator, values, lo);
  return accumulator.template to_vector<bfloat16>();
}

// Decode two adjacent 8-row x 8-column tiles together. The two tiles are
// contiguous in both planes, so this halves the payload loads and shares the
// high-plane masks.
static inline __attribute__((always_inline)) void dequantize_pair(
    const uint8_t* packed_chunk,
    int weights_rows_tile,
    const aie::vector<bfloat16, kMmulWeightsElems>& hi,
    const aie::vector<bfloat16, kMmulWeightsElems>& mid,
    const aie::vector<bfloat16, kMmulWeightsElems>& lo,
    aie::vector<bfloat16, kMmulWeightsElems>& values0,
    aie::vector<bfloat16, kMmulWeightsElems>& values1) {
  constexpr int kPairElems = 2 * kTileRows * kTileCols;
  const uint8_t* packed_low = packed_chunk;
  const uint8_t* packed_high = packed_low + kPackedChunkLowBytes;
  const int low_offset = weights_rows_tile * (kTileRows * kTileCols / 2);
  const int high_offset = weights_rows_tile * (kTileRows * kTileCols / 4);
  const uint8_t* low = packed_low + low_offset;
  const uint8_t* high = packed_high + high_offset;

  const auto low_bytes = aie::load_unaligned_v<kPairElems / 2>(low, 16);
  const auto low4 = low_bytes.template cast_to<uint4>();

  const auto high_bytes = aie::load_unaligned_v<kPairElems / 4>(high, 16);
  const auto high8 =
      high_bytes.template cast_to<uint4>().template unpack<uint8>();
  const auto mask8 =
      aie::broadcast<uint8_t, kPairElems / 2>(static_cast<uint8_t>(0x03));

  const auto h_even = aie::bit_and(high8, mask8);
  const auto h_odd = aie::logical_downshift(high8, 2);
  const auto [c0, c1] = aie::interleave_zip(h_even, h_odd, 1);
  const auto high_u8 = aie::concat(c0, c1);

  const auto low_u8 = low4.template unpack<uint8>();
  const auto values_u8 = aie::bit_or(low_u8, aie::upshift(high_u8, 4));
  const auto centered =
      aie::add(values_u8,
               aie::broadcast<uint8_t, kPairElems>(static_cast<uint8_t>(0xE0)));
  const auto quant = centered.template cast_to<int8>();
  const auto values = aie::to_float<bfloat16>(quant);

  values0 =
      apply_scales(values.template extract<kMmulWeightsElems>(0), hi, mid, lo);
  values1 =
      apply_scales(values.template extract<kMmulWeightsElems>(1), hi, mid, lo);
}

}  // namespace detail

// Unpack one 64-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* record,
                                 const uint8_t* __restrict packed_chunk,
                                 bfloat16* __restrict weights_tile) {
  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);
  constexpr int kTilesPerChunk = kUnpackTileRows / kTileRows;
  constexpr int kTilesPerSubgroup = 16 / kTileRows;

  aie::vector<bfloat16, kPackedChunkScalesBytes> hi32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> mid32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> lo32;
  const int chunk_index =
      static_cast<int>((packed_chunk - record) / kPackedChunkBytes);
  aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> d_lo32;
  detail::load_d16(record, d_hi32, d_lo32);
  detail::load_chunk_scales32(d_hi32, d_lo32, chunk_scales(record, chunk_index),
                              hi32, mid32, lo32);

  for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
       ++subgroup) {
    aie::vector<bfloat16, kMmulWeightsElems> hi;
    aie::vector<bfloat16, kMmulWeightsElems> mid;
    aie::vector<bfloat16, kMmulWeightsElems> lo;
    detail::load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi, mid, lo);

    static_assert(kTilesPerSubgroup == 2,
                  "Q6_K pair decode expects two tiles per subgroup");
    const int tile0 = subgroup * kTilesPerSubgroup;
    aie::vector<bfloat16, kMmulWeightsElems> values0;
    aie::vector<bfloat16, kMmulWeightsElems> values1;
    detail::dequantize_pair(packed_chunk, tile0, hi, mid, lo, values0, values1);
    output_tiles[tile0] = values0;
    output_tiles[tile0 + 1] = values1;
  }
}

}  // namespace spiral::quants::q6_k
