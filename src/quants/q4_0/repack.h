#pragma once

#include <stdint.h>

#include "quants/q4_0/layout.h"

namespace spiral::quants::q4_0 {

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

// Repack canonical column-major Q4_0 tensor bytes into
// [weights_cols_tile][weights_rows_record][packed_record] order.
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
  if (src_col_stride_bytes <
      (weights_rows / kBlockRows) * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  const int64_t rows_records = weights_rows / kRecordRows;
  const int64_t blocks_per_record = kRecordRows / kBlockRows;

  for (int64_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {
    for (int64_t weights_rows_record = 0;
         weights_rows_record < rows_records; ++weights_rows_record) {
      const int64_t record_index =
          weights_cols_tile * rows_records + weights_rows_record;
      uint8_t* record = dst + record_index * kPackedRecordBytes;
      uint8_t* packed_chunk = record + kCommonScaleBytes;

      for (int i = 0; i < kPackedChunkBytes; ++i) {
        packed_chunk[i] = 0;
      }

      for (int weights_rows_block = 0;
           weights_rows_block < blocks_per_record; ++weights_rows_block) {
        uint8_t* unit = packed_chunk + weights_rows_block * kUnitBytes;

        for (int weights_cols_in_tile = 0;
             weights_cols_in_tile < kTileCols; ++weights_cols_in_tile) {
          const int64_t weights_cols_index =
              weights_cols_tile * kTileCols + weights_cols_in_tile;
          const int64_t weights_rows_block_index =
              weights_rows_record * blocks_per_record + weights_rows_block;
          const uint8_t* block =
              src + weights_cols_index * src_col_stride_bytes +
              weights_rows_block_index * kBlockBytes;

          uint8_t* scale =
              unit + kBlockPayloadRegionBytes +
              weights_cols_in_tile * kBlockScaleBytes;
          scale[0] = block[0];
          scale[1] = block[1];

          for (int payload_index = 0; payload_index < kPayloadsPerBlock;
               ++payload_index) {
            uint8_t* payload = unit + payload_index * kPayloadBytes;
            for (int payload_row = 0; payload_row < kPayloadRows;
                 ++payload_row) {
              const int block_row =
                  payload_index * kPayloadRows + payload_row;
              const uint8_t quant =
                  block_row < 16
                      ? static_cast<uint8_t>(block[2 + block_row] & 0x0F)
                      : static_cast<uint8_t>(block[2 + block_row - 16] >> 4);
              const uint8_t signed_quant =
                  static_cast<uint8_t>((quant - 8) & 0x0F);
              const int byte_index =
                  payload_row * 4 + weights_cols_in_tile / 2;
              if ((weights_cols_in_tile & 1) == 0) {
                payload[byte_index] =
                    static_cast<uint8_t>((payload[byte_index] & 0xF0) |
                                         signed_quant);
              } else {
                payload[byte_index] = static_cast<uint8_t>(
                    (payload[byte_index] & 0x0F) | (signed_quant << 4));
              }
            }
          }
        }
      }
    }
  }

  return bytes;
}

}  // namespace spiral::quants::q4_0
