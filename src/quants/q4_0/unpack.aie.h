#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/q4_0/layout.h"

namespace spiral::quants::q4_0 {

namespace detail {

// Expand one eight-column scale chunk to the 64-lane mmul vector.
template <unsigned Elems>
static inline __attribute__((always_inline)) void expand_scale64(
    const aie::vector<bfloat16, Elems>& terms,
    int chunk,
    aie::vector<bfloat16, kMmulWeightsElems>& out) {
  const auto short_terms = terms.template extract<kTileCols>(chunk);
  out = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(
      short_terms, 0);
}

// Dequantize one 8x8 tile from the packed chunk.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    dequantize_tile(const uint8_t* payloads,
                    int weights_rows_block,
                    int tile_index,
                    const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                    const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  using PayloadVec = aie::vector<uint8_t, kTilePayloadBytes>;
  const auto payload_tiles = reinterpret_cast<const PayloadVec*>(
      block_payloads(payloads, weights_rows_block));
  const auto quant = payload_tiles[tile_index].template cast_to<int4>();
  const auto quant_float = aie::to_float<bfloat16>(quant, 0);
  auto accumulator = aie::mul(quant_float, hi);
  accumulator = aie::mac(accumulator, quant_float, lo);
  return accumulator.template to_vector<bfloat16>();
}

}  // namespace detail

// Standalone unpack loops are test-only. The fused mmul path is the
// production consumer and compiles only the detail helpers above.
#if defined(QUANTS_BUILD_UNPACK_TEST)
// Unpack one 64-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* packed,
                                 int weights_cols_tile,
                                 int weights_rows_record,
                                 int cols_pairs,
                                 int chunk_index,
                                 bfloat16* __restrict weights_tile) {
  (void)chunk_index;

  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);

  const int col_pair = weights_cols_tile / 2;
  const int local_col = weights_cols_tile % 2;
  const int local_row = weights_rows_record % kRowsRecordsPerTile;
  const int rows_tile = weights_rows_record / kRowsRecordsPerTile;
  const uint8_t* tile =
      packed + (rows_tile * cols_pairs + col_pair) * kPackedTransportTileBytes;
  const uint8_t* payloads = record_payloads(tile, local_col, local_row);
  const uint8_t* scales = record_scales(tile, local_col, local_row);

  // One record carries two blocks of eight column scales, contiguous as 16
  // canonical fp16 values. Decode them once, then expand a block at a time.
  aie::vector<bfloat16, 16> hi16;
  aie::vector<bfloat16, 16> lo16;
  split_fp16x16_exact(scales, hi16, lo16);

  constexpr int kTilesPerBlock = kBlockRows / kMmulWeightsRows;
  for (int weights_rows_block = 0; weights_rows_block < kBlocksPerRecord;
       ++weights_rows_block) {
    aie::vector<bfloat16, kMmulWeightsElems> hi;
    aie::vector<bfloat16, kMmulWeightsElems> lo;
    detail::expand_scale64(hi16, weights_rows_block, hi);
    detail::expand_scale64(lo16, weights_rows_block, lo);

#pragma unroll
    for (int tile_index = 0; tile_index < kTilesPerBlock; ++tile_index) {
      const int weights_rows_tile =
          weights_rows_block * kTilesPerBlock + tile_index;
      const auto values = detail::dequantize_tile(payloads, weights_rows_block,
                                                  tile_index, hi, lo);
      output_tiles[weights_rows_tile] = values;
    }
  }
}

// Unpack all chunks of one record.
inline void unpack_record(const uint8_t* packed,
                          int weights_cols_tile,
                          int weights_rows_record,
                          int cols_pairs,
                          bfloat16* __restrict weights_tile) {
  unpack_weights_chunk(packed, weights_cols_tile, weights_rows_record,
                       cols_pairs, 0, weights_tile);
}

#endif

}  // namespace spiral::quants::q4_0
