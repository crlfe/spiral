#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/common/tile.h"
#include "quants/q6_k_int8/layout.h"
#include "quants/q6_k_int8/unpack.aie.h"

namespace spiral::quants::q6_k_int8 {

// Fused Q6_K unpack and mmul. The caller must zero outputs and select
// aie::rounding_mode::conv_even before calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kRecordRows == 0,
                "weights_rows must be a multiple of the record depth");
  static_assert(weights_cols % kTileCols == 0,
                "weights_cols must be a multiple of the tile width");

  constexpr int kColsTiles = weights_cols / kTileCols;
  constexpr int kRowsRecords = weights_rows / kRecordRows;
  constexpr int kTilesPerChunk = kUnpackTileRows / kTileRows;
  constexpr int kTilesPerSubgroup = 16 / kTileRows;
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  int weights_cols_tile = 0;
  for (; weights_cols_tile + 1 < kColsTiles; weights_cols_tile += 2) {
    const uint8_t* cols_tile0 =
        packed + weights_cols_tile * kRowsRecords * kPackedRecordBytes;
    const uint8_t* cols_tile1 = cols_tile0 + kRowsRecords * kPackedRecordBytes;
    MMUL accumulator0;
    MMUL accumulator1;

#pragma unroll
    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* record0 =
          cols_tile0 + weights_rows_record * kPackedRecordBytes;
      const uint8_t* record1 =
          cols_tile1 + weights_rows_record * kPackedRecordBytes;
      const auto d32_0 = detail::load_d32(record0);
      const auto d32_1 = detail::load_d32(record1);

#pragma unroll
      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const uint8_t* packed_chunk0 = chunk(record0, chunk_index);
        const uint8_t* packed_chunk1 = chunk(record1, chunk_index);
        const InputVec* inputs_slice =
            input_tiles + (weights_rows_record * kRecordRows +
                           chunk_index * kUnpackTileRows) /
                              kTileRows;

        aie::vector<bfloat16, kPackedChunkScalesBytes> hi32_0, mid32_0, lo32_0;
        aie::vector<bfloat16, kPackedChunkScalesBytes> hi32_1, mid32_1, lo32_1;
        detail::load_chunk_scales32(d32_0, chunk_scales(record0, chunk_index),
                                    hi32_0, mid32_0, lo32_0);
        detail::load_chunk_scales32(d32_1, chunk_scales(record1, chunk_index),
                                    hi32_1, mid32_1, lo32_1);

#pragma unroll
        for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
             ++subgroup) {
          aie::vector<bfloat16, kMmulWeightsElems> hi0, mid0, lo0;
          aie::vector<bfloat16, kMmulWeightsElems> hi1, mid1, lo1;
          detail::load_subgroup_scales64(hi32_0, mid32_0, lo32_0, subgroup, hi0,
                                         mid0, lo0);
          detail::load_subgroup_scales64(hi32_1, mid32_1, lo32_1, subgroup, hi1,
                                         mid1, lo1);

#pragma unroll
          for (int tile = 0; tile < kTilesPerSubgroup; ++tile) {
            const int weights_rows_tile = subgroup * kTilesPerSubgroup + tile;
            const InputVec& inputs_tile = inputs_slice[weights_rows_tile];
            spiral::quants::mmul_weights_tile(
                accumulator0, inputs_tile,
                detail::dequantize_tile(packed_chunk0, weights_rows_tile, hi0,
                                        mid0, lo0));
            spiral::quants::mmul_weights_tile(
                accumulator1, inputs_tile,
                detail::dequantize_tile(packed_chunk1, weights_rows_tile, hi1,
                                        mid1, lo1));
          }
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
    MMUL accumulator;

#pragma unroll
    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* record =
          cols_tile + weights_rows_record * kPackedRecordBytes;
      const auto d32 = detail::load_d32(record);

      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const uint8_t* packed_chunk = chunk(record, chunk_index);
        const InputVec* inputs_slice =
            input_tiles + (weights_rows_record * kRecordRows +
                           chunk_index * kUnpackTileRows) /
                              kTileRows;

        aie::vector<bfloat16, kPackedChunkScalesBytes> hi32, mid32, lo32;
        detail::load_chunk_scales32(d32, chunk_scales(record, chunk_index),
                                    hi32, mid32, lo32);

#pragma unroll
        for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
             ++subgroup) {
          aie::vector<bfloat16, kMmulWeightsElems> hi, mid, lo;
          detail::load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi, mid,
                                         lo);

#pragma unroll
          for (int tile = 0; tile < kTilesPerSubgroup; ++tile) {
            const int weights_rows_tile = subgroup * kTilesPerSubgroup + tile;
            spiral::quants::mmul_weights_tile(
                accumulator, inputs_slice[weights_rows_tile],
                detail::dequantize_tile(packed_chunk, weights_rows_tile, hi,
                                        mid, lo));
          }
        }
      }
    }
    output_tiles[weights_cols_tile] = accumulator.template to_vector<float>();
  }
}

}  // namespace spiral::quants::q6_k_int8
