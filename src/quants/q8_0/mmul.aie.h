#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/common/tile.h"
#include "quants/q8_0/layout.h"
#include "quants/q8_0/unpack.aie.h"

namespace spiral::quants::q8_0 {

// Fused Q8_0 unpack and mmul. The output accumulators start at zero; the
// caller must select aie::rounding_mode::conv_even before calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kRecordRows == 0,
                "weights_rows must be a multiple of the record depth");
  static_assert(weights_cols % kTileCols == 0,
                "weights_cols must be a multiple of the tile width");
  static_assert(kChunksPerRecord == 1,
                "Q8_0 fused mmul assumes one chunk per record");

  constexpr int kColsTiles = weights_cols / kTileCols;
  constexpr int kRowsRecords = weights_rows / kRecordRows;
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  int weights_cols_tile = 0;
#pragma unroll
  for (; weights_cols_tile + 1 < kColsTiles; weights_cols_tile += 2) {
    const uint8_t* cols_tile0 =
        packed + weights_cols_tile * kRowsRecords * kPackedRecordBytes;
    const uint8_t* cols_tile1 = cols_tile0 + kRowsRecords * kPackedRecordBytes;
    MMUL accumulator0(aie::zeros<float, kMmulOutputsElems>());
    MMUL accumulator1(aie::zeros<float, kMmulOutputsElems>());

#pragma unroll
    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* record0 =
          cols_tile0 + weights_rows_record * kPackedRecordBytes;
      const uint8_t* record1 =
          cols_tile1 + weights_rows_record * kPackedRecordBytes;
      const uint8_t* packed_chunk0 = chunk(record0, 0);
      const uint8_t* packed_chunk1 = chunk(record1, 0);
      const InputVec* inputs_slice =
          input_tiles + weights_rows_record * kRecordRows / kTileRows;

#pragma unroll
      for (int block = 0; block < kBlocksPerRecord; ++block) {
        aie::vector<bfloat16, kTileCols> hi8_0, lo8_0;
        aie::vector<bfloat16, kTileCols> hi8_1, lo8_1;
        detail::load_block_scales(block_scales(packed_chunk0, block), hi8_0,
                                  lo8_0);
        detail::load_block_scales(block_scales(packed_chunk1, block), hi8_1,
                                  lo8_1);

        aie::vector<bfloat16, kMmulWeightsElems> hi0, lo0;
        aie::vector<bfloat16, kMmulWeightsElems> hi1, lo1;
        detail::expand_scales64(hi8_0, lo8_0, hi0, lo0);
        detail::expand_scales64(hi8_1, lo8_1, hi1, lo1);

#pragma unroll
        for (int tile = 0; tile < kTilesPerBlock; ++tile) {
          const int weights_rows_tile = block * kTilesPerBlock + tile;
          const InputVec& inputs_tile = inputs_slice[weights_rows_tile];
          spiral::quants::mmul_weights_tile(
              accumulator0, inputs_tile,
              detail::dequantize_tile(packed_chunk0, block, tile, hi0, lo0));
          spiral::quants::mmul_weights_tile(
              accumulator1, inputs_tile,
              detail::dequantize_tile(packed_chunk1, block, tile, hi1, lo1));
        }
      }
    }
    output_tiles[weights_cols_tile] = accumulator0.template to_vector<float>();
    output_tiles[weights_cols_tile + 1] =
        accumulator1.template to_vector<float>();
  }

  if constexpr (kColsTiles % 2 == 1) {
    const uint8_t* cols_tile =
        packed + weights_cols_tile * kRowsRecords * kPackedRecordBytes;
    MMUL accumulator(aie::zeros<float, kMmulOutputsElems>());

#pragma unroll
    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* record =
          cols_tile + weights_rows_record * kPackedRecordBytes;
      const uint8_t* packed_chunk = chunk(record, 0);
      const InputVec* inputs_slice =
          input_tiles + weights_rows_record * kRecordRows / kTileRows;

#pragma unroll
      for (int block = 0; block < kBlocksPerRecord; ++block) {
        aie::vector<bfloat16, kTileCols> hi8, lo8;
        detail::load_block_scales(block_scales(packed_chunk, block), hi8, lo8);
        aie::vector<bfloat16, kMmulWeightsElems> hi, lo;
        detail::expand_scales64(hi8, lo8, hi, lo);

#pragma unroll
        for (int tile = 0; tile < kTilesPerBlock; ++tile) {
          const int weights_rows_tile = block * kTilesPerBlock + tile;
          spiral::quants::mmul_weights_tile(
              accumulator, inputs_slice[weights_rows_tile],
              detail::dequantize_tile(packed_chunk, block, tile, hi, lo));
        }
      }
    }
    output_tiles[weights_cols_tile] = accumulator.template to_vector<float>();
  }
}

}  // namespace spiral::quants::q8_0
