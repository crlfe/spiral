#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/q4_0/layout.h"

namespace spiral::quants::q4_0 {

namespace detail {

static inline __attribute__((always_inline)) aie::vector<int4, 32>
load_packed_payload(const uint8_t* unit, int payload_index) {
  const uint8_t* payload = unit + payload_index * kPayloadBytes;
  return aie::load_v<16>(payload).template cast_to<int4>();
}

}  // namespace detail

// Unpack one 64-row x 8-column chunk into the canonical bf16 weights tile.
// The fp16 scales are split into bf16 hi/lo components so arbitrary fp16
// scales remain bit-identical to the float32 dequant path.
inline void unpack_weights_chunk(const uint8_t* record,
                                 const uint8_t* __restrict packed_chunk,
                                 bfloat16* __restrict weights_tile) {
  (void)record;
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};

  constexpr int kWeightsRowsBlocks = kUnpackTileRows / kBlockRows;
#pragma unroll
  for (int weights_rows_block = 0;
       weights_rows_block < kWeightsRowsBlocks; ++weights_rows_block) {
    const uint8_t* unit = packed_chunk + weights_rows_block * kUnitBytes;
    const auto scales = load_fp16x8(unit + kBlockPayloadRegionBytes);
    aie::vector<bfloat16, 8> weights_hi;
    aie::vector<bfloat16, 8> weights_lo;
    split_bf16_hi_lo(scales, weights_hi, weights_lo);
    const auto weights_hi_rows =
        aie::concat(weights_hi, weights_hi, weights_hi, weights_hi);
    const auto weights_lo_rows =
        aie::concat(weights_lo, weights_lo, weights_lo, weights_lo);

#pragma unroll
    for (int payload_index = 0; payload_index < kPayloadsPerBlock;
         ++payload_index) {
      const auto quant = detail::load_packed_payload(unit, payload_index);
      const auto quant_float = aie::to_float<bfloat16>(quant, 0);
      auto accumulator = aie::mul(quant_float, weights_hi_rows);
      accumulator = aie::mac(accumulator, quant_float, weights_lo_rows);
      const auto values = accumulator.template to_vector<bfloat16>();

      const int weights_rows_tile =
          weights_rows_block * (kBlockRows / kMmulWeightsRows) +
          payload_index / 2;
      const int tile_row_base = (payload_index % 2) * kPayloadRows;
      aie::store_v(weights_tile +
                       weights_rows_tile * kMmulWeightsElems +
                       tile_row_base * kTileCols,
                   values);
    }
  }
}

// Q4_0 has one chunk per record, so a whole-record wrapper is unambiguous.
inline void unpack_weights_tile(const uint8_t* record,
                                bfloat16* weights_tile) {
  unpack_weights_chunk(record, chunk(record, 0), weights_tile);
}

}  // namespace spiral::quants::q4_0
