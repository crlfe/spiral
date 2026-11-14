#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/q4_0/layout.h"
#include "quants/q4_0/unpack.aie.h"

namespace spiral::quants::q4_0 {

// Fused unpack and mmul for a 256-row x 16-column transport tile. Two column
// tiles are processed together to expose independent mmul accumulators. The
// caller must zero outputs and select aie::rounding_mode::conv_even before
// calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kTransportTileRows == 0,
                "weights_rows must be a multiple of the transport tile");
  static_assert(weights_cols % kTransportTileCols == 0,
                "weights_cols must cover whole 16-column tiles");

  constexpr int kColsTiles = weights_cols / kTileCols;
  constexpr int kColsPairs = weights_cols / kTransportTileCols;
  constexpr int kRowsRecords = weights_rows / kRecordRows;
  constexpr int kTilesPerBlock = kBlockRows / spiral::quants::kMmulWeightsRows;
  constexpr int kMmulInputsElems = spiral::quants::kMmulInputsElems;
  constexpr int kMmulOutputsElems = spiral::quants::kMmulOutputsElems;
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  for (int weights_cols_tile = 0; weights_cols_tile < kColsTiles;
       weights_cols_tile += 2) {
    OutputVec* outputs_tile0 = output_tiles + weights_cols_tile;
    OutputVec* outputs_tile1 = outputs_tile0 + 1;
    const int col_pair = weights_cols_tile / 2;

    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const int local_row = weights_rows_record % kRowsRecordsPerTile;
      const int rows_tile = weights_rows_record / kRowsRecordsPerTile;
      const uint8_t* tile =
          packed + (rows_tile * kColsPairs + col_pair) *
                       kPackedTransportTileBytes;
      const uint8_t* payloads0 = record_payloads(tile, 0, local_row);
      const uint8_t* payloads1 = record_payloads(tile, 1, local_row);
      const uint8_t* scales0 = record_scales(tile, 0, local_row);
      const uint8_t* scales1 = record_scales(tile, 1, local_row);

      const int weights_rows_start = weights_rows_record * kRecordRows;
      const InputVec* inputs_slice =
          input_tiles + weights_rows_start / spiral::quants::kMmulWeightsRows;

      MMUL accumulator0(*outputs_tile0);
      MMUL accumulator1(*outputs_tile1);

      for (int block = 0; block < kBlocksPerRecord; ++block) {
        aie::vector<bfloat16, kMmulWeightsElems> hi0;
        aie::vector<bfloat16, kMmulWeightsElems> lo0;
        aie::vector<bfloat16, kMmulWeightsElems> hi1;
        aie::vector<bfloat16, kMmulWeightsElems> lo1;
        detail::load_block_scales64(scales0, block, hi0, lo0);
        detail::load_block_scales64(scales1, block, hi1, lo1);

#pragma unroll
        for (int tile_index = 0; tile_index < kTilesPerBlock; ++tile_index) {
          const int absolute_tile = block * kTilesPerBlock + tile_index;
          const InputVec& inputs_tile = inputs_slice[absolute_tile];

          spiral::quants::mmul_weights_tile(
              accumulator0, inputs_tile,
              detail::dequantize_tile(payloads0, block, tile_index, hi0, lo0));
          spiral::quants::mmul_weights_tile(
              accumulator1, inputs_tile,
              detail::dequantize_tile(payloads1, block, tile_index, hi1, lo1));
        }
      }

      *outputs_tile0 = accumulator0.template to_vector<float>();
      *outputs_tile1 = accumulator1.template to_vector<float>();
    }
  }
}

}  // namespace spiral::quants::q4_0
