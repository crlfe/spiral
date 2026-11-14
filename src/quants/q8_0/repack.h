#pragma once

#include <stdint.h>
#include <string.h>

#include "quants/common/repack.h"
#include "quants/q8_0/layout.h"

namespace spiral::quants::q8_0 {

constexpr int64_t tile_count(int64_t weights_rows, int64_t weights_cols) {
  if (weights_rows <= 0 || weights_cols <= 0) {
    return 0;
  }
  if (weights_rows % kTransportTileRows != 0 ||
      weights_cols % kTransportTileCols != 0) {
    return 0;
  }
  return (weights_rows / kTransportTileRows) *
         (weights_cols / kTransportTileCols);
}

// Number of 256-row x 8-column records (legacy count).
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
  return tile_count(weights_rows, weights_cols) * kPackedTransportTileBytes;
}

// Repack canonical column-major Q8_0 into 256-row x 16-column transport tiles.
// Each record is payload-first, and each 64-byte tile payload holds one 8x8
// matrix in row-major order.
inline int64_t repack_weights(const uint8_t* src,
                              int64_t src_col_stride_bytes,
                              uint8_t* dst,
                              int64_t dst_capacity,
                              int64_t weights_rows,
                              int64_t weights_cols) {
  const int64_t tiles = tile_count(weights_rows, weights_cols);
  const int64_t bytes = tiles * kPackedTransportTileBytes;
  if (tiles == 0 || bytes > dst_capacity || src == nullptr ||
      dst == nullptr) {
    return 0;
  }
  const int64_t blocks_per_col = weights_rows / kBlockRows;
  if (src_col_stride_bytes < blocks_per_col * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  const int64_t cols_pairs = weights_cols / kTransportTileCols;
  const int64_t rows_records = weights_rows / kRecordRows;
  memset(dst, 0, bytes);
  spiral::quants::detail::RepackWarnings warnings;

  for (int64_t rows_record = 0; rows_record < rows_records; ++rows_record) {
    for (int64_t cols_tile = 0; cols_tile < cols_tiles; ++cols_tile) {
      const int64_t col_pair = cols_tile / 2;
      const int local_col = static_cast<int>(cols_tile % 2);
      uint8_t* tile =
          dst + (rows_record * cols_pairs + col_pair) *
                    kPackedTransportTileBytes;
      uint8_t* record = record_ptr(tile, local_col);
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
            const int tile_index = row / kTileRows;
            const int tile_row = row % kTileRows;
            payloads[tile_index * kTilePayloadBytes +
                     tile_row * kTileCols + col] = value;
          }
        }
      }
    }
  }

  warnings.warn("Q8_0");
  return bytes;
}

}  // namespace spiral::quants::q8_0
