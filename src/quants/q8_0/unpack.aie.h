#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/common/tile.h"
#include "quants/q8_0/layout.h"

namespace spiral::quants::q8_0 {

namespace detail {

// Split the eight per-column fp16 `d` values of one 32-row block into an
// exact bf16 hi/lo pair. An fp16 `d` has an 11-bit significand, so the top
// eight bits fit in `hi` and the remaining bits fit exactly in `lo`.
static inline __attribute__((always_inline)) void load_block_scales(
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kTileCols>& hi,
    aie::vector<bfloat16, kTileCols>& lo) {
  const auto d = load_fp16x8(packed_scales);
  split_bf16_hi_lo(d, hi, lo);
}

// Expand eight per-column scale terms to 64 lanes.
static inline __attribute__((always_inline)) void expand_scales64(
    const aie::vector<bfloat16, kTileCols>& hi8,
    const aie::vector<bfloat16, kTileCols>& lo8,
    aie::vector<bfloat16, kMmulWeightsElems>& hi,
    aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  hi = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(hi8, 0);
  lo = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(lo8, 0);
}

// Dequantize one 8-row x 8-column tile from its 64-byte int8 payload.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    dequantize_tile(const uint8_t* packed_chunk,
                    int block,
                    int weights_rows_tile,
                    const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                    const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  const auto weights = aie::to_float<bfloat16>(
      aie::load_v<kTilePayloadBytes>(block_payloads(packed_chunk, block) +
                                     weights_rows_tile * kTilePayloadBytes)
          .template cast_to<int8>());

  auto accumulator = aie::mul(weights, hi);
  accumulator = aie::mac(accumulator, weights, lo);
  return accumulator.template to_vector<bfloat16>();
}

}  // namespace detail

// Unpack one 256-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* record,
                                 const uint8_t* __restrict packed_chunk,
                                 bfloat16* __restrict weights_tile) {
  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);
  (void)record;

  for (int block = 0; block < kBlocksPerRecord; ++block) {
    aie::vector<bfloat16, kTileCols> hi8;
    aie::vector<bfloat16, kTileCols> lo8;
    detail::load_block_scales(block_scales(packed_chunk, block), hi8, lo8);

    aie::vector<bfloat16, kMmulWeightsElems> hi;
    aie::vector<bfloat16, kMmulWeightsElems> lo;
    detail::expand_scales64(hi8, lo8, hi, lo);

#pragma unroll
    for (int tile = 0; tile < kTilesPerBlock; ++tile) {
      const int weights_rows_tile = block * kTilesPerBlock + tile;
      output_tiles[weights_rows_tile] =
          detail::dequantize_tile(packed_chunk, block, tile, hi, lo);
    }
  }
}

}  // namespace spiral::quants::q8_0
