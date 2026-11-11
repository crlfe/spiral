#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/q4_0/layout.h"

namespace spiral::quants::q4_0 {

namespace detail {

// Load, split, and expand one 32-row block's scales to one 64-lane vector per
// bf16 component.
static inline __attribute__((always_inline)) void load_block_scales64(
    const uint8_t* packed_chunk,
    int weights_rows_block,
    aie::vector<bfloat16, kMmulWeightsElems>& hi,
    aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  const auto scales =
      load_fp16x8(block_scales(packed_chunk, weights_rows_block));
  aie::vector<bfloat16, 8> hi8;
  aie::vector<bfloat16, 8> lo8;
  split_bf16_hi_lo(scales, hi8, lo8);
  hi = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(hi8, 0);
  lo = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(lo8, 0);
}

// Dequantize one 8x8 tile from the packed chunk.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    dequantize_tile(const uint8_t* packed_chunk,
                    int weights_rows_block,
                    int tile_index,
                    const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                    const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  using PayloadVec = aie::vector<uint8_t, kTilePayloadBytes>;
  const auto payload_tiles = reinterpret_cast<const PayloadVec*>(
      block_payloads(packed_chunk, weights_rows_block));
  const auto quant = payload_tiles[tile_index].template cast_to<int4>();
  const auto quant_float = aie::to_float<bfloat16>(quant, 0);
  auto accumulator = aie::mul(quant_float, hi);
  accumulator = aie::mac(accumulator, quant_float, lo);
  return accumulator.template to_vector<bfloat16>();
}

}  // namespace detail

// Unpack one 64-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* record,
                                 const uint8_t* __restrict packed_chunk,
                                 bfloat16* __restrict weights_tile) {
  (void)record;

  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);

  constexpr int kTilesPerBlock = kBlockRows / kMmulWeightsRows;
  for (int weights_rows_block = 0; weights_rows_block < kBlocksPerRecord;
       ++weights_rows_block) {
    aie::vector<bfloat16, kMmulWeightsElems> hi;
    aie::vector<bfloat16, kMmulWeightsElems> lo;
    detail::load_block_scales64(packed_chunk, weights_rows_block, hi, lo);

#pragma unroll
    for (int tile_index = 0; tile_index < kTilesPerBlock; ++tile_index) {
      const int weights_rows_tile =
          weights_rows_block * kTilesPerBlock + tile_index;
      const auto values = detail::dequantize_tile(
          packed_chunk, weights_rows_block, tile_index, hi, lo);
      output_tiles[weights_rows_tile] = values;
    }
  }
}

}  // namespace spiral::quants::q4_0
