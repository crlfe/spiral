#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/common/tile.h"
#include "quants/q6_k/layout.h"
#include "quants/q6_k/unpack.aie.h"

namespace spiral::quants::q6_k {

// Fused Q6_K unpack and mmul using the native six-bit payload. The vector
// bit-plane expansion is still the dominant cost; this variant is a
// comparison point for the int8-expanded layout. The caller must zero outputs
// and select aie::rounding_mode::conv_even before calling this.
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
  static_assert(kTilesPerSubgroup == 2,
                "Q6_K pair decode expects two tiles per subgroup");
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  for (int weights_cols_tile = 0; weights_cols_tile < kColsTiles;
       ++weights_cols_tile) {
    OutputVec* output_tile = output_tiles + weights_cols_tile;
    MMUL accumulator(*output_tile);

    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* record =
          packed + (weights_cols_tile * kRowsRecords + weights_rows_record) *
                       kPackedRecordBytes;

      aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32, d_lo32;
      detail::load_d16(record, d_hi32, d_lo32);

      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const uint8_t* packed_chunk = chunk(record, chunk_index);
        const InputVec* inputs_slice =
            input_tiles + (weights_rows_record * kRecordRows +
                           chunk_index * kUnpackTileRows) /
                              kTileRows;

        aie::vector<bfloat16, kPackedChunkScalesBytes> hi32, mid32, lo32;
        detail::load_chunk_scales32(d_hi32, d_lo32,
                                    chunk_scales(record, chunk_index), hi32,
                                    mid32, lo32);

        for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
             ++subgroup) {
          const int tile0 = subgroup * kTilesPerSubgroup;
          aie::vector<bfloat16, kMmulWeightsElems> hi, mid, lo;
          aie::vector<bfloat16, kMmulWeightsElems> values0, values1;
          detail::load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi, mid,
                                         lo);
          detail::dequantize_pair(packed_chunk, tile0, hi, mid, lo, values0,
                                  values1);
          spiral::quants::mmul_weights_tile(accumulator, inputs_slice[tile0],
                                            values0);
          spiral::quants::mmul_weights_tile(accumulator,
                                            inputs_slice[tile0 + 1], values1);
        }
      }
    }

    *output_tile = accumulator.template to_vector<float>();
  }
}

}  // namespace spiral::quants::q6_k
