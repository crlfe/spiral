#pragma once

#include <stdint.h>
#include <string.h>

#include "quants/common/repack.h"
#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

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

// Repack canonical column-major Q6_K into whole 256-row x 8-column records,
// preserving the native six-bit payload. The quantized values are split into
// a low-nibble plane and a two-bit high plane inside each 64-row chunk.
inline int64_t repack_weights(const uint8_t* src,
                              int64_t src_col_stride_bytes,
                              uint8_t* dst,
                              int64_t dst_capacity,
                              int64_t weights_rows,
                              int64_t weights_cols) {
  const int64_t records = record_count(weights_rows, weights_cols);
  const int64_t bytes = records * kPackedRecordBytes;
  if (records == 0 || bytes > dst_capacity || dst == nullptr ||
      src == nullptr) {
    return 0;
  }
  const int64_t blocks_per_col = weights_rows / kBlockRows;
  if (src_col_stride_bytes < blocks_per_col * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  spiral::quants::detail::RepackWarnings warnings;

  for (int64_t cols_tile = 0; cols_tile < cols_tiles; ++cols_tile) {
    for (int64_t rows_record = 0; rows_record < blocks_per_col; ++rows_record) {
      uint8_t* record =
          dst + (cols_tile * blocks_per_col + rows_record) * kPackedRecordBytes;
      const uint8_t* ql[kTileCols];
      const uint8_t* qh[kTileCols];
      const uint8_t* col_scales[kTileCols];

      for (int col = 0; col < kTileCols; ++col) {
        const int64_t global_col = cols_tile * kTileCols + col;
        const uint8_t* block =
            src + global_col * src_col_stride_bytes + rows_record * kBlockBytes;
        ql[col] = block;
        qh[col] = block + kBlockQlBytes;
        col_scales[col] = block + kBlockQlBytes + kBlockQhBytes;

        uint16_t d_bits = load_u16(block + kBlockBytes - kBlockScaleBytes);
        if (fp16_is_subnormal(d_bits)) {
          d_bits = 0;
          warnings.subnormal_scale = true;
        }
        store_u16(record_scales(record) + col * kBlockScaleBytes, d_bits);
      }

      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const int half = chunk_index / 2;
        const int pair = chunk_index % 2;
        uint8_t* packed_payload = chunk(record, chunk_index);
        uint8_t* packed_low = packed_payload;
        uint8_t* packed_high = packed_low + kPackedChunkLowBytes;
        uint8_t* packed_scales = chunk_scales(record, chunk_index);
        memset(packed_payload, 0, kPackedChunkPayloadBytes);

        for (int col = 0; col < kTileCols; ++col) {
          const uint8_t* ql_col = ql[col];
          const uint8_t* qh_col = qh[col];
          const uint8_t* scales = col_scales[col];

          for (int subgroup = 0; subgroup < 4; ++subgroup) {
            packed_scales[subgroup * kTileCols + col] =
                scales[half * 8 + pair * 4 + subgroup];
          }

          for (int row = 0; row < kUnpackTileRows; ++row) {
            const int group = row / 32;
            const int local_row = row % 32;
            const uint8_t ql_byte = ql_col[half * 64 + group * 32 + local_row];
            const uint8_t low_nibble =
                pair == 0 ? (ql_byte & 0x0F) : ((ql_byte >> 4) & 0x0F);
            const uint8_t high_bits =
                (qh_col[half * 32 + local_row] >> (4 * pair + 2 * group)) &
                0x03;
            const uint8_t value = low_nibble | (high_bits << 4);
            const int index = row * kTileCols + col;

            packed_low[index / 2] |=
                static_cast<uint8_t>((value & 0x0F) << (4 * (index % 2)));
            packed_high[index / 4] |= static_cast<uint8_t>(
                ((value >> 4) & 0x03) << (2 * (index % 4)));
          }
        }
      }
    }
  }

  warnings.warn("Q6_K");
  return bytes;
}

}  // namespace spiral::quants::q6_k
