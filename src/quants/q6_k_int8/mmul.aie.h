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
  static_assert(weights_cols % kTransportTileCols == 0,
                "Q6_K_int8 transport requires whole 16-column tiles");

  constexpr int kColsTiles = weights_cols / kTileCols;
  constexpr int kColsPairs = weights_cols / kTransportTileCols;
  constexpr int kRowsRecords = weights_rows / kRecordRows;
  constexpr int kTilesPerChunk = kUnpackTileRows / kTileRows;
  constexpr int kTilesPerSubgroup = 16 / kTileRows;
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  for (int weights_cols_tile = 0; weights_cols_tile < kColsTiles;
       weights_cols_tile += 2) {
    const int col_pair = weights_cols_tile / 2;
    MMUL accumulator0;
    MMUL accumulator1;

#pragma unroll
    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* tile =
          packed + (weights_rows_record * kColsPairs + col_pair) *
                       kPackedTransportTileBytes;
      aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32_0, d_lo32_0;
      aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32_1, d_lo32_1;
      detail::load_d16(tile, 0, d_hi32_0, d_lo32_0);
      detail::load_d16(tile, 1, d_hi32_1, d_lo32_1);

#pragma clang loop unroll(disable)
      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const uint8_t* packed_chunk0 =
            chunk(record_payloads(tile, 0), chunk_index);
        const uint8_t* packed_chunk1 =
            chunk(record_payloads(tile, 1), chunk_index);
        const InputVec* inputs_slice =
            input_tiles + (weights_rows_record * kRecordRows +
                           chunk_index * kUnpackTileRows) /
                              kTileRows;

        aie::vector<bfloat16, kPackedChunkScalesBytes> hi32_0, mid32_0, lo32_0;
        aie::vector<bfloat16, kPackedChunkScalesBytes> hi32_1, mid32_1, lo32_1;
        detail::load_chunk_scales32(d_hi32_0, d_lo32_0,
                                    chunk_scales(tile, 0, chunk_index), hi32_0,
                                    mid32_0, lo32_0);
        detail::load_chunk_scales32(d_hi32_1, d_lo32_1,
                                    chunk_scales(tile, 1, chunk_index), hi32_1,
                                    mid32_1, lo32_1);

#pragma unroll
        for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
             ++subgroup) {
#pragma unroll
          for (int tile_index = 0; tile_index < kTilesPerSubgroup;
               ++tile_index) {
            const int weights_rows_tile =
                subgroup * kTilesPerSubgroup + tile_index;
            const InputVec& inputs_tile = inputs_slice[weights_rows_tile];
            aie::vector<bfloat16, kMmulWeightsElems> hi, mid, lo;

            detail::load_subgroup_scales64(hi32_0, mid32_0, lo32_0, subgroup,
                                           hi, mid, lo);
            spiral::quants::mmul_weights_tile(
                accumulator0, inputs_tile,
                detail::dequantize_tile(packed_chunk0, weights_rows_tile, hi,
                                        mid, lo));

            detail::load_subgroup_scales64(hi32_1, mid32_1, lo32_1, subgroup,
                                           hi, mid, lo);
            spiral::quants::mmul_weights_tile(
                accumulator1, inputs_tile,
                detail::dequantize_tile(packed_chunk1, weights_rows_tile, hi,
                                        mid, lo));
          }
        }
      }
    }
    output_tiles[weights_cols_tile] = accumulator0.template to_vector<float>();
    output_tiles[weights_cols_tile + 1] =
        accumulator1.template to_vector<float>();
  }
}

}  // namespace spiral::quants::q6_k_int8
