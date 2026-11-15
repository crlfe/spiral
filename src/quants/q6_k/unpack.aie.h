#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/q6_scale.aie.h"
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
    const uint8_t* tile,
    int local_col,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& d_hi32,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& d_lo32) {
  q6_scale::load_d16_terms(record_scales(tile, local_col), d_hi32, d_lo32);
}

static inline __attribute__((always_inline)) void load_chunk_scales32(
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& d_hi32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& d_lo32,
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& hi,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& mid,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& lo) {
  q6_scale::compute_chunk_scale_terms(d_hi32, d_lo32, packed_scales, hi, mid,
                                      lo);
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
  q6_scale::load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi, mid, lo);
}

// Apply the three exact bf16 scale terms to one 64-lane values vector.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    apply_scales(const aie::vector<bfloat16, kMmulWeightsElems>& values,
                 const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                 const aie::vector<bfloat16, kMmulWeightsElems>& mid,
                 const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  return q6_scale::apply_scales(values, hi, mid, lo);
}

// Decode four adjacent 8-row x 8-column tiles. The 32-row group shares one
// 64-byte high-plane load across both pairs; the low plane needs two 64-byte
// loads. This cuts the high-plane load count in half relative to pair decode.
static inline __attribute__((always_inline)) void dequantize_quad(
    const uint8_t* packed_chunk,
    int weights_rows_tile,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& hi32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& mid32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& lo32,
    int subgroup,
    aie::vector<bfloat16, kMmulWeightsElems>& values0,
    aie::vector<bfloat16, kMmulWeightsElems>& values1,
    aie::vector<bfloat16, kMmulWeightsElems>& values2,
    aie::vector<bfloat16, kMmulWeightsElems>& values3) {
  constexpr int kPairElems = 2 * kTileRows * kTileCols;
  constexpr int kQuadElems = 2 * kPairElems;
  const uint8_t* packed_low = packed_chunk;
  const uint8_t* packed_high = packed_low + kPackedChunkLowBytes;
  const int low_offset = weights_rows_tile * (kTileRows * kTileCols / 2);
  const int high_offset = weights_rows_tile * (kTileRows * kTileCols / 4);
  const uint8_t* low = packed_low + low_offset;
  const uint8_t* high = packed_high + high_offset;

  const auto low0 = aie::load_v<kPairElems / 2>(low);
  const auto low1 = aie::load_v<kPairElems / 2>(low + kPairElems / 2);
  const auto high_bytes = aie::load_v<kPairElems / 2>(high);

  // low0/low1 hold the 4-bit planes for 256 weights. high_bytes holds two
  // 2-bit fields per nibble; separate the even/odd fields and interleave them
  // back into weight order before merging with the low nibbles.
  const auto low_nibbles = aie::concat(low0, low1).template cast_to<uint4>();
  const auto high_pairs =
      high_bytes.template cast_to<uint4>().template unpack<uint8>();
  const auto field_mask =
      aie::broadcast<uint8_t, kQuadElems / 2>(static_cast<uint8_t>(0x03));
  const auto high_even = aie::bit_and(high_pairs, field_mask);
  const auto high_odd = aie::logical_downshift(high_pairs, 2);
  const auto [high_lo, high_hi] = aie::interleave_zip(high_even, high_odd, 1);
  const auto high_bits = aie::concat(high_lo, high_hi);
  const auto low_nibbles_u8 = low_nibbles.template unpack<uint8>();
  const auto six_bit_values =
      aie::bit_or(low_nibbles_u8, aie::upshift(high_bits, 4));
  const auto signed_bytes =
      aie::add(six_bit_values,
               aie::broadcast<uint8_t, kQuadElems>(static_cast<uint8_t>(0xE0)));
  const auto values_i8 = signed_bytes.template cast_to<int8>();
  const auto values_bf16 = aie::to_float<bfloat16>(values_i8);

  aie::vector<bfloat16, kMmulWeightsElems> hi0, mid0, lo0;
  aie::vector<bfloat16, kMmulWeightsElems> hi1, mid1, lo1;
  load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi0, mid0, lo0);
  load_subgroup_scales64(hi32, mid32, lo32, subgroup + 1, hi1, mid1, lo1);

  values0 = apply_scales(values_bf16.template extract<kMmulWeightsElems>(0),
                         hi0, mid0, lo0);
  values1 = apply_scales(values_bf16.template extract<kMmulWeightsElems>(1),
                         hi0, mid0, lo0);
  values2 = apply_scales(values_bf16.template extract<kMmulWeightsElems>(2),
                         hi1, mid1, lo1);
  values3 = apply_scales(values_bf16.template extract<kMmulWeightsElems>(3),
                         hi1, mid1, lo1);
}

}  // namespace detail

// Unpack one chunk with the record's d terms already loaded.
inline void unpack_chunk_with_d(
    const uint8_t* tile,
    int local_col,
    int chunk_index,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& d_hi32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& d_lo32,
    bfloat16* __restrict weights_tile) {
  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);
  constexpr int kTilesPerChunk = kUnpackTileRows / kTileRows;
  constexpr int kTilesPerQuad = 4;

  const uint8_t* packed_chunk =
      chunk(record_payloads(tile, local_col), chunk_index);

  aie::vector<bfloat16, kPackedChunkScalesBytes> hi32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> mid32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> lo32;
  detail::load_chunk_scales32(d_hi32, d_lo32,
                              chunk_scales(tile, local_col, chunk_index), hi32,
                              mid32, lo32);

  for (int quad = 0; quad < kTilesPerChunk / kTilesPerQuad; ++quad) {
    const int tile0 = quad * kTilesPerQuad;
    aie::vector<bfloat16, kMmulWeightsElems> values0;
    aie::vector<bfloat16, kMmulWeightsElems> values1;
    aie::vector<bfloat16, kMmulWeightsElems> values2;
    aie::vector<bfloat16, kMmulWeightsElems> values3;
    detail::dequantize_quad(packed_chunk, tile0, hi32, mid32, lo32, quad * 2,
                            values0, values1, values2, values3);
    output_tiles[tile0] = values0;
    output_tiles[tile0 + 1] = values1;
    output_tiles[tile0 + 2] = values2;
    output_tiles[tile0 + 3] = values3;
  }
}

// Unpack one 64-row x 8-column chunk from a tile-local record.
inline void unpack_tile_chunk(const uint8_t* tile,
                              int local_col,
                              int chunk_index,
                              bfloat16* __restrict weights_tile) {
  aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> d_lo32;
  detail::load_d16(tile, local_col, d_hi32, d_lo32);
  unpack_chunk_with_d(tile, local_col, chunk_index, d_hi32, d_lo32,
                      weights_tile);
}

// Unpack a whole record from a tile pointer, loading d once.
inline void unpack_record_tile(const uint8_t* tile,
                               int local_col,
                               bfloat16* __restrict weights_tile) {
  aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> d_lo32;
  detail::load_d16(tile, local_col, d_hi32, d_lo32);

  constexpr int kChunkElems = kUnpackTileRows * kTileCols;
  for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
    unpack_chunk_with_d(tile, local_col, chunk_index, d_hi32, d_lo32,
                        weights_tile + chunk_index * kChunkElems);
  }
}

// Unpack a whole record from a contiguous packed stream, loading d once.
inline void unpack_record(const uint8_t* packed,
                          int weights_cols_tile,
                          int weights_rows_record,
                          int cols_pairs,
                          bfloat16* __restrict weights_tile) {
  const int local_col = weights_cols_tile % 2;
  const uint8_t* tile = transport_tile(packed, weights_cols_tile,
                                       weights_rows_record, cols_pairs);
  unpack_record_tile(tile, local_col, weights_tile);
}

// Unpack one 64-row x 8-column chunk from a contiguous packed stream.
inline void unpack_weights_chunk(const uint8_t* packed,
                                 int weights_cols_tile,
                                 int weights_rows_record,
                                 int cols_pairs,
                                 int chunk_index,
                                 bfloat16* __restrict weights_tile) {
  const int local_col = weights_cols_tile % 2;
  const uint8_t* tile = transport_tile(packed, weights_cols_tile,
                                       weights_rows_record, cols_pairs);
  unpack_tile_chunk(tile, local_col, chunk_index, weights_tile);
}

}  // namespace spiral::quants::q6_k
