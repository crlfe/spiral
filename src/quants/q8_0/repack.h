#pragma once

#include <stdint.h>

#include "quants/common/repack.h"
#include "quants/q8_0/layout.h"

namespace spiral::quants::q8_0 {

constexpr int64_t record_count(int64_t weights_rows, int64_t weights_cols) {
  if (weights_rows <= 0 || weights_cols <= 0) {
    return 0;
  }
  if (weights_rows % kRecordRows != 0 || weights_cols % kTileCols != 0) {
    return 0;
  }
  return (weights_rows / kRecordRows) * (weights_cols / kTileCols);
}

constexpr int64_t packed_bytes(int64_t weights_rows, int64_t weights_cols) {
  return record_count(weights_rows, weights_cols) * kPackedRecordBytes;
}

// Repack canonical column-major Q8_0 into 256-row x 8-column records. Each
// 64-byte tile payload holds one 8x8 matrix in row-major order.
inline int64_t repack_weights(const uint8_t* src,
                              int64_t src_col_stride_bytes,
                              uint8_t* dst,
                              int64_t dst_capacity,
                              int64_t weights_rows,
                              int64_t weights_cols) {
  const int64_t records = record_count(weights_rows, weights_cols);
  const int64_t bytes = records * kPackedRecordBytes;
  if (records == 0 || bytes > dst_capacity || src == nullptr ||
      dst == nullptr) {
    return 0;
  }
  const int64_t blocks_per_col = weights_rows / kBlockRows;
  if (src_col_stride_bytes < blocks_per_col * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  const int64_t rows_records = weights_rows / kRecordRows;
  spiral::quants::detail::RepackWarnings warnings;

  for (int64_t cols_tile = 0; cols_tile < cols_tiles; ++cols_tile) {
    for (int64_t rows_record = 0; rows_record < rows_records; ++rows_record) {
      const int64_t record_index = cols_tile * rows_records + rows_record;
      uint8_t* record = dst + record_index * kPackedRecordBytes;
      uint8_t* packed_chunk = record + kCommonScaleBytes;

      for (int block = 0; block < kBlocksPerRecord; ++block) {
        uint8_t* payloads =
            packed_chunk + kPayloadsOffset + block * kBlockPayloadRegionBytes;
        uint8_t* scales =
            packed_chunk + kScalesOffset + block * kScaleBytesPerBlock;

        for (int col = 0; col < kTileCols; ++col) {
          const int64_t global_col = cols_tile * kTileCols + col;
          const int64_t block_index = rows_record * kBlocksPerRecord + block;
          const uint8_t* block_ptr = src + global_col * src_col_stride_bytes +
                                     block_index * kBlockBytes;

          uint8_t* scale = scales + col * kBlockScaleBytes;
          uint16_t scale_bits = load_u16(block_ptr);
          if (fp16_is_subnormal(scale_bits)) {
            scale_bits = 0;
            warnings.subnormal_scale = true;
          }
          store_u16(scale, scale_bits);

          const uint8_t* quant = block_ptr + kBlockScaleBytes;
          for (int row = 0; row < kBlockRows; ++row) {
            const uint8_t value = quant[row];
            if (value == 0x80u) {
              warnings.min_int8 = true;
            }
            const int tile = row / kTileRows;
            const int tile_row = row % kTileRows;
            payloads[tile * kTilePayloadBytes + tile_row * kTileCols + col] =
                value;
          }
        }
      }
    }
  }

  warnings.warn("Q8_0");
  return bytes;
}

}  // namespace spiral::quants::q8_0
