#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "quants/common/repack.h"
#include "quants/q4_0_hilo/layout.h"

namespace spiral::quants::q4_0_hilo {

namespace detail {

static inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    const float value = ldexpf(static_cast<float>(mant), -24);
    return (sign != 0) ? -value : value;
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static inline uint16_t f32_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static inline float bf16_to_f32(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

}  // namespace detail

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
// `dst` must be 32-byte aligned for the device's aligned tile loads.
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
  if (src_col_stride_bytes < (weights_rows / kBlockRows) * kBlockBytes) {
    return 0;
  }

  const int64_t cols_tiles = weights_cols / kTileCols;
  const int64_t rows_records = weights_rows / kRecordRows;
  const int64_t blocks_per_record = kRecordRows / kBlockRows;
  spiral::quants::detail::RepackWarnings warnings;

  for (int64_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {
    for (int64_t weights_rows_record = 0; weights_rows_record < rows_records;
         ++weights_rows_record) {
      const int64_t record_index =
          weights_cols_tile * rows_records + weights_rows_record;
      uint8_t* record = dst + record_index * kPackedRecordBytes;
      uint8_t* packed_chunk = record + kCommonScaleBytes;

      for (int weights_rows_block = 0; weights_rows_block < blocks_per_record;
           ++weights_rows_block) {
        uint8_t* payloads = packed_chunk + kPayloadsOffset +
                            weights_rows_block * kBlockPayloadRegionBytes;
        uint8_t* scales = packed_chunk + kScalesOffset +
                          weights_rows_block * kScaleBytesPerBlock;

        for (int weights_cols_in_tile = 0; weights_cols_in_tile < kTileCols;
             ++weights_cols_in_tile) {
          const int64_t weights_cols_index =
              weights_cols_tile * kTileCols + weights_cols_in_tile;
          const int64_t weights_rows_block_index =
              weights_rows_record * blocks_per_record + weights_rows_block;
          const uint8_t* block = src +
                                 weights_cols_index * src_col_stride_bytes +
                                 weights_rows_block_index * kBlockBytes;

          uint16_t scale_bits = load_u16(block);
          if (fp16_is_subnormal(scale_bits)) {
            scale_bits = 0;
            warnings.subnormal_scale = true;
          }
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

  warnings.warn("Q4_0 hilo");
  return bytes;
}

}  // namespace spiral::quants::q4_0_hilo
