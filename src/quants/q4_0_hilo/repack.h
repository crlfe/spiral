#pragma once

#include <stdint.h>
#include <string.h>

#include "quants/common/repack.h"
#include "quants/q4_0_hilo/layout.h"

namespace spiral::quants::q4_0_hilo {

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

// Number of 64-row x 8-column records (legacy count).
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

// Repack canonical column-major Q4_0 tensor bytes into 256-row x 16-column
// transport tiles. Values are reordered inside each tile so payloads and
// host-precomputed bf16 hi/lo scale pairs are contiguous. The host split is
// exact for normal and subnormal fp16 scales.
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
  if (src_col_stride_bytes < (weights_rows / kBlockRows) * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  const int64_t cols_pairs = weights_cols / kTransportTileCols;
  const int64_t rows_records = weights_rows / kRecordRows;
  const int64_t blocks_per_record = kRecordRows / kBlockRows;
  memset(dst, 0, bytes);

  for (int64_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {
    const int64_t col_pair = weights_cols_tile / 2;
    const int local_col = static_cast<int>(weights_cols_tile % 2);
    for (int64_t weights_rows_record = 0; weights_rows_record < rows_records;
         ++weights_rows_record) {
      const int local_row =
          static_cast<int>(weights_rows_record % kRowsRecordsPerTile);
      const int64_t rows_tile = weights_rows_record / kRowsRecordsPerTile;
      uint8_t* tile =
          dst + (rows_tile * cols_pairs + col_pair) *
                    kPackedTransportTileBytes;
      uint8_t* record_payload_base =
          record_payloads(tile, local_col, local_row);
      uint8_t* record_scale_base =
          record_scales(tile, local_col, local_row);

      for (int weights_rows_block = 0; weights_rows_block < blocks_per_record;
           ++weights_rows_block) {
        uint8_t* payloads =
            block_payloads(record_payload_base, weights_rows_block);
        uint8_t* scales =
            block_scales(record_scale_base, weights_rows_block);

        for (int weights_cols_in_tile = 0; weights_cols_in_tile < kTileCols;
             ++weights_cols_in_tile) {
          const int64_t weights_cols_index =
              weights_cols_tile * kTileCols + weights_cols_in_tile;
          const int64_t weights_rows_block_index =
              weights_rows_record * blocks_per_record + weights_rows_block;
          const uint8_t* block = src +
                                 weights_cols_index * src_col_stride_bytes +
                                 weights_rows_block_index * kBlockBytes;

          const uint16_t scale_bits = load_u16(block);
          const float scale_f32 = detail::fp16_to_f32(scale_bits);
          const uint16_t hi = detail::f32_to_bf16(scale_f32);
          const uint16_t lo =
              detail::f32_to_bf16(scale_f32 - detail::bf16_to_f32(hi));
          uint8_t* scale = scales + weights_cols_in_tile * kBlockScaleBytes;
          store_u16(scale, hi);
          store_u16(scale + kTileCols * kBlockScaleBytes, lo);

          for (int payload_index = 0; payload_index < kPayloadsPerBlock;
               ++payload_index) {
            uint8_t* payload = payloads + payload_index * kPayloadBytes;
            for (int payload_row = 0; payload_row < kPayloadRows;
                 ++payload_row) {
              const int block_row = payload_index * kPayloadRows + payload_row;
              const uint8_t quant =
                  block_row < 16
                      ? static_cast<uint8_t>(block[2 + block_row] & 0x0F)
                      : static_cast<uint8_t>(block[2 + block_row - 16] >> 4);
              const uint8_t signed_quant =
                  static_cast<uint8_t>((quant - 8) & 0x0F);
              const int byte_index = payload_row * 4 + weights_cols_in_tile / 2;
              if ((weights_cols_in_tile & 1) == 0) {
                payload[byte_index] = static_cast<uint8_t>(
                    (payload[byte_index] & 0xF0) | signed_quant);
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

}  // namespace spiral::quants::q4_0_hilo
