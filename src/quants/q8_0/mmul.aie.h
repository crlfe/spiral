#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/common/tile.h"
#include "quants/q8_0/layout.h"
#include "quants/q8_0/unpack.aie.h"

namespace spiral::quants::q8_0 {

// Fused Q8_0 unpack and mmul for 256-row x 16-column transport tiles. The
// output accumulators start at zero; the caller must select
// aie::rounding_mode::conv_even before calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kRecordRows == 0,
                "weights_rows must be a multiple of the record depth");
  static_assert(weights_cols % kTransportTileCols == 0,
                "Q8_0 transport requires whole 16-column tiles");
  static_assert(kChunksPerRecord == 1,
                "Q8_0 fused mmul assumes one chunk per record");

  constexpr int kColsTiles = weights_cols / kTileCols;
  constexpr int kColsPairs = weights_cols / kTransportTileCols;
  constexpr int kRowsRecords = weights_rows / kRecordRows;
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  for (int weights_cols_tile = 0; weights_cols_tile < kColsTiles;
       weights_cols_tile += 2) {
    const int col_pair = weights_cols_tile / 2;
    OutputVec* outputs_tile0 = output_tiles + weights_cols_tile;
    OutputVec* outputs_tile1 = outputs_tile0 + 1;
    MMUL accumulator0(aie::zeros<float, kMmulOutputsElems>());
    MMUL accumulator1(aie::zeros<float, kMmulOutputsElems>());

#pragma unroll
    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* tile =
          packed + (weights_rows_record * kColsPairs + col_pair) *
                       kPackedTransportTileBytes;
      const uint8_t* record0 = record_ptr(tile, 0);
      const uint8_t* record1 = record_ptr(tile, 1);
      const uint8_t* packed_chunk0 = chunk(record0, 0);
      const uint8_t* packed_chunk1 = chunk(record1, 0);
      const InputVec* inputs_slice =
          input_tiles + weights_rows_record * kRecordRows / kTileRows;

      const uint8_t* scales0 = block_scales(packed_chunk0, 0);
      const uint8_t* scales1 = block_scales(packed_chunk1, 0);

#pragma unroll
      for (int batch = 0; batch < kBlocksPerRecord / kBlocksPerDecode;
           ++batch) {
        // Decode four blocks' canonical fp16 scales into one exact 32-lane
        // batch for each column record.
        aie::vector<bfloat16, kScaleLanes> hi32_0, lo32_0;
        aie::vector<bfloat16, kScaleLanes> hi32_1, lo32_1;
        detail::load_scales32(scales0 + batch * kScaleBytesPerDecode, hi32_0,
                              lo32_0);
        detail::load_scales32(scales1 + batch * kScaleBytesPerDecode, hi32_1,
                              lo32_1);

#pragma unroll
        for (int block_in_batch = 0; block_in_batch < kBlocksPerDecode;
             ++block_in_batch) {
          const int block = batch * kBlocksPerDecode + block_in_batch;
          aie::vector<bfloat16, kMmulWeightsElems> hi0, lo0;
          aie::vector<bfloat16, kMmulWeightsElems> hi1, lo1;
          detail::expand_scales64(hi32_0, lo32_0, block_in_batch, hi0, lo0);
          detail::expand_scales64(hi32_1, lo32_1, block_in_batch, hi1, lo1);

#pragma unroll
          for (int tile_index = 0; tile_index < kTilesPerBlock; ++tile_index) {
            const int weights_rows_tile = block * kTilesPerBlock + tile_index;
            const InputVec& inputs_tile = inputs_slice[weights_rows_tile];
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
      }
    }
    *outputs_tile0 = accumulator0.template to_vector<float>();
    *outputs_tile1 = accumulator1.template to_vector<float>();
  }
}

}  // namespace spiral::quants::q8_0
