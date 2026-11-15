#pragma once

#include <stdint.h>
#include <string.h>

#include "quants/common/repack.h"
#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

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

// Repack canonical column-major Q6_K into 256-row x 16-column transport tiles,
// preserving the native six-bit payload. Values are reordered inside each tile
// so both records' payloads, scales, and d values are contiguously aligned.
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
  constexpr int kSubgroupsPerHalf = 4;
  constexpr int kScalesPerHalf = 8;
  constexpr int kRowsPerGroup = 32;
  constexpr int kQlHalfBytes = kBlockQlBytes / 2;
  constexpr int kQhHalfBytes = kBlockQhBytes / 2;
  memset(dst, 0, bytes);
  spiral::quants::detail::RepackWarnings warnings;

  for (int64_t rows_record = 0; rows_record < blocks_per_col; ++rows_record) {
    for (int64_t cols_tile = 0; cols_tile < cols_tiles; ++cols_tile) {
      const int64_t col_pair = cols_tile / 2;
      const int local_col = static_cast<int>(cols_tile % 2);
      uint8_t* tile = dst + (rows_record * cols_pairs + col_pair) *
                                kPackedTransportTileBytes;
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
        store_u16(record_scales(tile, local_col) + col * kBlockScaleBytes,
                  d_bits);
      }

      // Canonical ql is 128 bytes: two 64-byte halves. qh is 64 bytes: two
      // 32-byte halves. Each ql nibble selects one 6-bit value, with its top
      // two bits selected from the matching qh byte. The repack splits those
      // values back into the low-nibble and high-two-bit planes.
      for (int chunk_index = 0; chunk_index < kChunksPerRecord; ++chunk_index) {
        const int ql_half = chunk_index / 2;
        const int ql_nibble = chunk_index % 2;
        uint8_t* packed_payload =
            chunk(record_payloads(tile, local_col), chunk_index);
        uint8_t* packed_low = packed_payload;
        uint8_t* packed_high = packed_low + kPackedChunkLowBytes;
        uint8_t* packed_scales = chunk_scales(tile, local_col, chunk_index);

        for (int col = 0; col < kTileCols; ++col) {
          const uint8_t* ql_col = ql[col];
          const uint8_t* qh_col = qh[col];
          const uint8_t* scales = col_scales[col];

          for (int subgroup = 0; subgroup < kSubgroupsPerHalf; ++subgroup) {
            packed_scales[subgroup * kTileCols + col] =
                scales[ql_half * kScalesPerHalf +
                       ql_nibble * kSubgroupsPerHalf + subgroup];
          }

          for (int row = 0; row < kUnpackTileRows; ++row) {
            const int group = row / kRowsPerGroup;
            const int local_row = row % kRowsPerGroup;
            const uint8_t ql_byte = ql_col[ql_half * kQlHalfBytes +
                                           group * kRowsPerGroup + local_row];
            const uint8_t low_nibble =
                ql_nibble == 0 ? (ql_byte & 0x0F) : ((ql_byte >> 4) & 0x0F);
            const uint8_t high_bits =
                (qh_col[ql_half * kQhHalfBytes + local_row] >>
                 (4 * ql_nibble + 2 * group)) &
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
