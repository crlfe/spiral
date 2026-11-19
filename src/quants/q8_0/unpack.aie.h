#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/common/tile.h"
#include "quants/q8_0/layout.h"

namespace spiral::quants::q8_0 {

// Scale decode batches four 32-row blocks into one native 512-bit bf16 pair.
// Scales stay canonical fp16, so subnormals decode exactly.
constexpr int kScaleLanes = 32;
constexpr int kBlocksPerDecode = kScaleLanes / kTileCols;
constexpr int kScaleBytesPerDecode = kBlocksPerDecode * kScaleBytesPerBlock;

static_assert(kScaleBytesPerDecode == 2 * kScaleLanes,
              "Q8_0 scale batch must be one 64-byte load");
static_assert(kBlocksPerRecord % kBlocksPerDecode == 0,
              "Q8_0 record must contain whole scale batches");

namespace detail {

// Decode 32 canonical fp16 scales into exact bf16 hi/lo terms.
static inline __attribute__((always_inline)) void load_scales32(
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kScaleLanes>& hi,
    aie::vector<bfloat16, kScaleLanes>& lo) {
  split_fp16_exact<kScaleLanes>(aie::load_v<2 * kScaleLanes>(packed_scales),
                                hi, lo);
}

// Expand one block's eight per-column terms to the 64-lane mmul vector.
static inline __attribute__((always_inline)) void expand_scales64(
    const aie::vector<bfloat16, kScaleLanes>& hi32,
    const aie::vector<bfloat16, kScaleLanes>& lo32,
    int block_in_batch,
    aie::vector<bfloat16, kMmulWeightsElems>& hi,
    aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  hi = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(
      hi32.template extract<kTileCols>(block_in_batch), 0);
  lo = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(
      lo32.template extract<kTileCols>(block_in_batch), 0);
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

// Standalone unpack loops are test-only. The fused mmul path is the
// production consumer and compiles only the detail helpers above.
#if defined(QUANTS_BUILD_UNPACK_TEST)
// Unpack one 256-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* packed,
                                 int weights_cols_tile,
                                 int weights_rows_record,
                                 int cols_pairs,
                                 int chunk_index,
                                 bfloat16* __restrict weights_tile) {
  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);

  const int col_pair = weights_cols_tile / 2;
  const int local_col = weights_cols_tile % 2;
  const uint8_t* tile = packed + (weights_rows_record * cols_pairs + col_pair) *
                                     kPackedTransportTileBytes;
  const uint8_t* record = record_ptr(tile, local_col);
  const uint8_t* packed_chunk = chunk(record, chunk_index);

#pragma unroll
  for (int batch = 0; batch < kBlocksPerRecord / kBlocksPerDecode; ++batch) {
    aie::vector<bfloat16, kScaleLanes> hi32;
    aie::vector<bfloat16, kScaleLanes> lo32;
    detail::load_scales32(
        block_scales(packed_chunk, batch * kBlocksPerDecode), hi32, lo32);

#pragma unroll
    for (int block_in_batch = 0; block_in_batch < kBlocksPerDecode;
         ++block_in_batch) {
      const int block = batch * kBlocksPerDecode + block_in_batch;
      aie::vector<bfloat16, kMmulWeightsElems> hi;
      aie::vector<bfloat16, kMmulWeightsElems> lo;
      detail::expand_scales64(hi32, lo32, block_in_batch, hi, lo);

#pragma unroll
      for (int tile_index = 0; tile_index < kTilesPerBlock; ++tile_index) {
        const int weights_rows_tile = block * kTilesPerBlock + tile_index;
        output_tiles[weights_rows_tile] = detail::dequantize_tile(
            packed_chunk, block, tile_index, hi, lo);
      }
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

}  // namespace spiral::quants::q8_0
