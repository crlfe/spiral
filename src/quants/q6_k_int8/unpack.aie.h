#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/common/tile.h"
#include "quants/q6_k_int8/layout.h"

namespace spiral::quants::q6_k_int8 {

namespace detail {

// Load the per-column fp16 `d` values once per record and expand them to 32
// lanes. The product of `d` and an 8-bit subgroup scale needs up to 18
// significand bits, so the product is split into three bf16 terms: two terms
// are not always enough to round to the same bf16 as an fp32 multiply.
static inline
    __attribute__((always_inline)) aie::vector<float, kPackedChunkScalesBytes>
    load_d32(const uint8_t* record) {
  const auto d8 = load_fp16x8(record_scales(record));
  return aie::concat(d8, d8, d8, d8);
}

static inline __attribute__((always_inline)) void load_chunk_scales32(
    const aie::vector<float, kPackedChunkScalesBytes>& d32,
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& hi,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& mid,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& lo) {
  const auto subgroup_scales =
      aie::to_float<float>(aie::load_v<kPackedChunkScalesBytes>(packed_scales)
                               .template cast_to<int8>());
  const auto product =
      aie::mul(d32, subgroup_scales).template to_vector<float>();

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

// Dequantize one 8-row x 8-column tile from signed int8 weights.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    dequantize_tile(const uint8_t* packed_chunk,
                    int weights_rows_tile,
                    const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                    const aie::vector<bfloat16, kMmulWeightsElems>& mid,
                    const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  const uint8_t* packed = packed_chunk + weights_rows_tile * kMmulWeightsElems;
  const auto quant =
      aie::load_v<kMmulWeightsElems>(reinterpret_cast<const int8*>(packed));
  const auto values = aie::to_float<bfloat16>(quant);

  auto accumulator = aie::mul(values, hi);
  accumulator = aie::mac(accumulator, values, mid);
  accumulator = aie::mac(accumulator, values, lo);
  return accumulator.template to_vector<bfloat16>();
}

}  // namespace detail

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
  const auto d32 = detail::load_d32(record);
  detail::load_chunk_scales32(d32, chunk_scales(record, chunk_index), hi32,
                              mid32, lo32);

#pragma unroll
  for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
       ++subgroup) {
    aie::vector<bfloat16, kMmulWeightsElems> hi;
    aie::vector<bfloat16, kMmulWeightsElems> mid;
    aie::vector<bfloat16, kMmulWeightsElems> lo;
    detail::load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi, mid, lo);

#pragma unroll
    for (int tile = 0; tile < kTilesPerSubgroup; ++tile) {
      const int weights_rows_tile = subgroup * kTilesPerSubgroup + tile;
      output_tiles[weights_rows_tile] =
          detail::dequantize_tile(packed_chunk, weights_rows_tile, hi, mid, lo);
    }
  }
}

}  // namespace spiral::quants::q6_k_int8
