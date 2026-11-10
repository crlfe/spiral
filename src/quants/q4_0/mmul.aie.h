#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/q4_0/layout.h"
#include "quants/q4_0/unpack.aie.h"

namespace spiral::quants::q4_0 {

// Fused unpack and mmul for a 64-row weights tile. Two column tiles are
// processed together to expose independent mmul accumulators. The caller must
// zero outputs and select aie::rounding_mode::conv_even before calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kRecordRows == 0,
                "weights_rows must be a multiple of the record depth");
  static_assert(weights_cols % (2 * kTileCols) == 0,
                "weights_cols must cover an even number of tile columns");

  constexpr int kColsTiles = weights_cols / kTileCols;
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

    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const int record_index0 =
          weights_cols_tile * kRowsRecords + weights_rows_record;
      const int record_index1 =
          (weights_cols_tile + 1) * kRowsRecords + weights_rows_record;
      const uint8_t* record0 = packed + record_index0 * kPackedRecordBytes;
      const uint8_t* record1 = packed + record_index1 * kPackedRecordBytes;

      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const uint8_t* packed_chunk0 = chunk(record0, chunk_index);
        const uint8_t* packed_chunk1 = chunk(record1, chunk_index);
        const int weights_rows_start =
            weights_rows_record * kRecordRows + chunk_index * kUnpackTileRows;
        const InputVec* inputs_slice =
            input_tiles + weights_rows_start / spiral::quants::kMmulWeightsRows;

        MMUL accumulator0(*outputs_tile0);
        MMUL accumulator1(*outputs_tile1);

        for (int block = 0; block < kBlocksPerRecord; ++block) {
          aie::vector<bfloat16, kMmulWeightsElems> hi0;
          aie::vector<bfloat16, kMmulWeightsElems> lo0;
          aie::vector<bfloat16, kMmulWeightsElems> hi1;
          aie::vector<bfloat16, kMmulWeightsElems> lo1;
          detail::load_block_scales64(packed_chunk0, block, hi0, lo0);
          detail::load_block_scales64(packed_chunk1, block, hi1, lo1);

#pragma unroll
          for (int tile_index = 0; tile_index < kTilesPerBlock; ++tile_index) {
            const int absolute_tile = block * kTilesPerBlock + tile_index;
            const InputVec& inputs_tile = inputs_slice[absolute_tile];

            spiral::quants::mmul_weights_tile(
                accumulator0, inputs_tile,
                detail::dequantize_tile(packed_chunk0, block, tile_index, hi0,
                                        lo0));
            spiral::quants::mmul_weights_tile(
                accumulator1, inputs_tile,
                detail::dequantize_tile(packed_chunk1, block, tile_index, hi1,
                                        lo1));
          }
        }

        *outputs_tile0 = accumulator0.template to_vector<float>();
        *outputs_tile1 = accumulator1.template to_vector<float>();
      }
    }
  }
}

}  // namespace spiral::quants::q4_0
