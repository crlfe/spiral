#pragma once

#include <stdint.h>
#include <string.h>

#include "quants/common/repack.h"
#include "quants/q6_k_int8/layout.h"

namespace spiral::quants::q6_k_int8 {

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

// Repack canonical column-major Q6_K into 256-row x 16-column transport tiles.
// Each 6-bit weight is expanded to a signed int8 (q - 32), so the device no
// longer decodes low/high bit planes.
inline int64_t repack_weights(const uint8_t* src,
                              int64_t src_col_stride_bytes,
                              uint8_t* dst,
                              int64_t dst_capacity,
                              int64_t weights_rows,
                              int64_t weights_cols) {
  const int64_t tiles = tile_count(weights_rows, weights_cols);
  const int64_t bytes = tiles * kPackedTransportTileBytes;
  if (tiles == 0 || bytes > dst_capacity || dst == nullptr || src == nullptr) {
    return 0;
  }
  const int64_t blocks_per_col = weights_rows / kBlockRows;
  if (src_col_stride_bytes < blocks_per_col * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  const int64_t cols_pairs = weights_cols / kTransportTileCols;
  memset(dst, 0, bytes);

  for (int64_t rows_record = 0; rows_record < blocks_per_col; ++rows_record) {
    for (int64_t cols_tile = 0; cols_tile < cols_tiles; ++cols_tile) {
      const int64_t col_pair = cols_tile / 2;
      const int local_col = static_cast<int>(cols_tile % 2);
      uint8_t* tile = dst + (rows_record * cols_pairs + col_pair) *
                                kPackedTransportTileBytes;
      uint8_t* record_d = record_scales(tile, local_col);

      for (int col = 0; col < kTileCols; ++col) {
        const int64_t global_col = cols_tile * kTileCols + col;
        const uint8_t* block =
            src + global_col * src_col_stride_bytes + rows_record * kBlockBytes;
        const uint8_t* ql = block;
        const uint8_t* qh = block + kBlockQlBytes;
        const uint8_t* scales = block + kBlockQlBytes + kBlockQhBytes;

        const uint16_t d_bits =
            load_u16(block + kBlockBytes - kBlockScaleBytes);
        const float d_f32 = spiral::quants::detail::fp16_to_f32(d_bits);
        const uint16_t d_hi = spiral::quants::detail::f32_to_bf16(d_f32);
        const uint16_t d_lo = spiral::quants::detail::f32_to_bf16(
            d_f32 - spiral::quants::detail::bf16_to_f32(d_hi));
        store_u16(record_d + col * kBlockScaleBytes, d_hi);
        store_u16(record_d + kCommonScalePlaneBytes + col * kBlockScaleBytes,
                  d_lo);

        for (int chunk_index = 0; chunk_index < kChunksPerRecord;
             ++chunk_index) {
          const int half = chunk_index / 2;
          const int pair = chunk_index % 2;
          uint8_t* packed_payload =
              chunk(record_payloads(tile, local_col), chunk_index);
          uint8_t* packed_scales = chunk_scales(tile, local_col, chunk_index);

          for (int subgroup = 0; subgroup < 4; ++subgroup) {
            packed_scales[subgroup * kTileCols + col] =
                scales[half * 8 + pair * 4 + subgroup];
          }

          for (int row = 0; row < kUnpackTileRows; ++row) {
            const int group = row / 32;
            const int local_row = row % 32;
            const uint8_t ql_byte = ql[half * 64 + group * 32 + local_row];
            const uint8_t low_nibble =
                pair == 0 ? (ql_byte & 0x0F) : ((ql_byte >> 4) & 0x0F);
            const uint8_t high_bits =
                (qh[half * 32 + local_row] >> (4 * pair + 2 * group)) & 0x03;
            const uint8_t value = low_nibble | (high_bits << 4);

            const int index = row * kTileCols + col;
            packed_payload[index] =
                static_cast<uint8_t>(static_cast<int8_t>(value) - 32);
          }
        }
      }
    }
  }

  return bytes;
}

}  // namespace spiral::quants::q6_k_int8
