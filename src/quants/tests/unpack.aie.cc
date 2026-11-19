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
#define QUANTS_BUILD_UNPACK_TEST 1
#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/unpack.aie.h)
// clang-format on

extern "C" void quants_unpack_all(const uint8_t* packed, bfloat16* out) {
  using namespace spiral::quants::QUANTS_FORMAT;
  const int32_t cols_tiles = QUANTS_WEIGHTS_COLS / kTileCols;
  const int32_t rows_records = QUANTS_WEIGHTS_ROWS / kRecordRows;
  const int32_t cols_pairs = QUANTS_WEIGHTS_COLS / kTransportTileCols;
  const int32_t chunk_elems = kUnpackTileRows * kTileCols;

  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  for (int32_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {
    for (int32_t weights_rows_record = 0; weights_rows_record < rows_records;
         ++weights_rows_record) {
      const int32_t out_index =
          weights_cols_tile * rows_records + weights_rows_record;
      bfloat16* record_out = out + out_index * kChunksPerRecord * chunk_elems;
      unpack_record(packed, weights_cols_tile, weights_rows_record, cols_pairs,
                    record_out);
    }
  }
  event1();
}
