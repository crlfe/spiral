#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.aie.h"
#include "quants/common/tile.h"
#include "quants/q6_k/layout.h"
#include "quants/q6_k/unpack.aie.h"

namespace spiral::quants::q6_k {

// Fused Q6_K unpack and mmul using the native six-bit payload. Both 8-column
// records of each 256x16 transport tile are processed together so they share
// input vectors and expose two independent mmul accumulators. The caller must
// zero outputs and select aie::rounding_mode::conv_even before calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kRecordRows == 0,
                "weights_rows must be a multiple of the record depth");
  static_assert(weights_cols % kTransportTileCols == 0,
                "Q6_K transport requires whole 16-column tiles");

  constexpr int kColsTiles = weights_cols / kTileCols;
  constexpr int kColsPairs = weights_cols / kTransportTileCols;
  constexpr int kRowsRecords = weights_rows / kRecordRows;
  constexpr int kTilesPerChunk = kUnpackTileRows / kTileRows;
  constexpr int kTilesPerQuad = 4;
  static_assert(kTilesPerChunk % kTilesPerQuad == 0,
                "Q6_K quad decode expects a multiple of four tiles per chunk");
  using InputVec = aie::vector<bfloat16, kMmulInputsElems>;
  using OutputVec = aie::vector<float, kMmulOutputsElems>;

  const auto input_tiles = reinterpret_cast<const InputVec*>(inputs);
  auto output_tiles = reinterpret_cast<OutputVec*>(outputs);

  for (int weights_cols_tile = 0; weights_cols_tile < kColsTiles;
       weights_cols_tile += 2) {
    OutputVec* output_tile0 = output_tiles + weights_cols_tile;
    OutputVec* output_tile1 = output_tile0 + 1;
    MMUL accumulator0(*output_tile0);
    MMUL accumulator1(*output_tile1);

    for (int weights_rows_record = 0; weights_rows_record < kRowsRecords;
         ++weights_rows_record) {
      const uint8_t* tile = transport_tile(packed, weights_cols_tile,
                                           weights_rows_record, kColsPairs);

      aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32_0, d_lo32_0;
      aie::vector<bfloat16, kPackedChunkScalesBytes> d_hi32_1, d_lo32_1;
      detail::load_d16(tile, 0, d_hi32_0, d_lo32_0);
      detail::load_d16(tile, 1, d_hi32_1, d_lo32_1);

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
        for (int quad = 0; quad < kTilesPerChunk / kTilesPerQuad; ++quad) {
          const int tile0 = quad * kTilesPerQuad;
          aie::vector<bfloat16, kMmulWeightsElems> values0;
          aie::vector<bfloat16, kMmulWeightsElems> values1;
          aie::vector<bfloat16, kMmulWeightsElems> values2;
          aie::vector<bfloat16, kMmulWeightsElems> values3;
          aie::vector<bfloat16, kMmulWeightsElems> values4;
          aie::vector<bfloat16, kMmulWeightsElems> values5;
          aie::vector<bfloat16, kMmulWeightsElems> values6;
          aie::vector<bfloat16, kMmulWeightsElems> values7;

          detail::dequantize_quad(packed_chunk0, tile0, hi32_0, mid32_0, lo32_0,
                                  quad * 2, values0, values1, values2, values3);
          detail::dequantize_quad(packed_chunk1, tile0, hi32_1, mid32_1, lo32_1,
                                  quad * 2, values4, values5, values6, values7);
          spiral::quants::mmul_weights_tile(accumulator0, inputs_slice[tile0],
                                            values0);
          spiral::quants::mmul_weights_tile(accumulator0,
                                            inputs_slice[tile0 + 1], values1);
          spiral::quants::mmul_weights_tile(accumulator0,
                                            inputs_slice[tile0 + 2], values2);
          spiral::quants::mmul_weights_tile(accumulator0,
                                            inputs_slice[tile0 + 3], values3);
          spiral::quants::mmul_weights_tile(accumulator1, inputs_slice[tile0],
                                            values4);
          spiral::quants::mmul_weights_tile(accumulator1,
                                            inputs_slice[tile0 + 1], values5);
          spiral::quants::mmul_weights_tile(accumulator1,
                                            inputs_slice[tile0 + 2], values6);
          spiral::quants::mmul_weights_tile(accumulator1,
                                            inputs_slice[tile0 + 3], values7);
        }
      }
    }

    *output_tile0 = accumulator0.template to_vector<float>();
    *output_tile1 = accumulator1.template to_vector<float>();
  }
}

}  // namespace spiral::quants::q6_k
