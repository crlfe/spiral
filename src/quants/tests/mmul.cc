#ifndef QUANTS_FORMAT
#error "QUANTS_FORMAT must be defined"
#endif
#ifndef QUANTS_WEIGHTS_ROWS
#error "QUANTS_WEIGHTS_ROWS must be defined"
#endif
#ifndef QUANTS_WEIGHTS_COLS
#error "QUANTS_WEIGHTS_COLS must be defined"
#endif

#define QUANTS_STRINGIFY_(x) #x
#define QUANTS_STRINGIFY(x) QUANTS_STRINGIFY_(x)

#include <stdint.h>

#include "quants/common/mmul.h"
#include "quants/common/tile.h"
#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/layout.h)
#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/unpack.h)

extern "C" void quants_unpack_mmul_all(const uint8_t* packed,
                                       const bfloat16* inputs,
                                       float* outputs) {
  using namespace spiral::quants::QUANTS_FORMAT;
  const int32_t cols_tiles = QUANTS_WEIGHTS_COLS / kTileCols;
  const int32_t rows_records = QUANTS_WEIGHTS_ROWS / kRecordRows;
  constexpr int kWeightsRowsTiles =
      kUnpackTileRows / spiral::quants::kMmulWeightsRows;

  for (int32_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {
    float* outputs_tile =
        outputs + weights_cols_tile * spiral::quants::kMmulOutputsElems;
    for (int i = 0; i < spiral::quants::kMmulOutputsElems; ++i) {
      outputs_tile[i] = 0.0f;
    }

    for (int32_t weights_rows_record = 0;
         weights_rows_record < rows_records; ++weights_rows_record) {
      const int32_t record_index =
          weights_cols_tile * rows_records + weights_rows_record;
      const uint8_t* record = packed + record_index * kPackedRecordBytes;

      for (int chunk_index = 0; chunk_index < kChunksPerRecord;
           ++chunk_index) {
        alignas(64) bfloat16 weights_tile[kUnpackTileRows * kTileCols];
        event0();
        unpack_weights_chunk(record, chunk(record, chunk_index), weights_tile);
        event1();

        const int32_t weights_rows_start =
            weights_rows_record * kRecordRows +
            chunk_index * kUnpackTileRows;
        const bfloat16* inputs_slice =
            inputs +
            (weights_rows_start / spiral::quants::kMmulWeightsRows) *
                spiral::quants::kMmulInputsElems;
        event0();
        spiral::quants::mmul_weights_tile<kWeightsRowsTiles>(
            inputs_slice, weights_tile, outputs_tile);
        event1();
      }
    }
  }
}
