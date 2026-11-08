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

// clang-format off
#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/layout.h)
#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/unpack.aie.h)
// clang-format on

extern "C" void quants_unpack_all(const uint8_t* packed, bfloat16* out) {
  using namespace spiral::quants::QUANTS_FORMAT;
  const int32_t cols_tiles = QUANTS_WEIGHTS_COLS / kTileCols;
  const int32_t rows_records = QUANTS_WEIGHTS_ROWS / kRecordRows;
  const int32_t chunk_elems = kUnpackTileRows * kTileCols;

  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int32_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {
    for (int32_t weights_rows_record = 0; weights_rows_record < rows_records;
         ++weights_rows_record) {
      const int32_t record_index =
          weights_cols_tile * rows_records + weights_rows_record;
      const uint8_t* record = packed + record_index * kPackedRecordBytes;

      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        bfloat16* tile =
            out + (record_index * kChunksPerRecord + chunk_index) * chunk_elems;
        unpack_weights_chunk(record, chunk(record, chunk_index), tile);
      }
    }
  }
  event1();
}
