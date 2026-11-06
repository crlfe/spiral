#pragma once

#include <stdint.h>
#include <string.h>

#include <aie_api/aie.hpp>

#include "quants/common/trace.h"
#include "quants/q4_0/layout.h"

namespace spiral::quants::q4_0 {

namespace detail {

static inline float fp16_to_f32(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1Fu;
  uint32_t mantissa = value & 0x03FFu;
  uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = -14;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --shift;
      }
      mantissa &= 0x03FFu;
      bits = sign | (static_cast<uint32_t>(shift + 127) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static inline int8_t signed_quant(uint8_t value) {
  return value < 8 ? static_cast<int8_t>(value)
                   : static_cast<int8_t>(value - 16);
}

}  // namespace detail

// Unpack one 64-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* common_scales,
                                 const uint8_t* packed_chunk,
                                 bfloat16* weights_tile) {
  (void)common_scales;
  TRACE_ENTER();
  // static_cast<bfloat16> follows the tile rounding mode; the default truncates.
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};

  const int weights_rows_blocks = kUnpackTileRows / kBlockRows;
  for (int weights_rows_block = 0;
       weights_rows_block < weights_rows_blocks; ++weights_rows_block) {
    const uint8_t* unit = packed_chunk + weights_rows_block * kUnitBytes;
    const uint8_t* scales = unit + kBlockPayloadRegionBytes;

    for (int payload_index = 0; payload_index < kPayloadsPerBlock;
         ++payload_index) {
      const uint8_t* payload = unit + payload_index * kPayloadBytes;
      const int weights_rows_tile =
          weights_rows_block * (kBlockRows / kMmulWeightsRows) +
          payload_index / 2;
      const int tile_row_base = (payload_index % 2) * kPayloadRows;

      for (int weights_cols_in_tile = 0;
           weights_cols_in_tile < kTileCols; ++weights_cols_in_tile) {
        const float scale = detail::fp16_to_f32(
            static_cast<uint16_t>(scales[weights_cols_in_tile * 2]) |
            (static_cast<uint16_t>(scales[weights_cols_in_tile * 2 + 1])
             << 8));

        for (int payload_row = 0; payload_row < kPayloadRows;
             ++payload_row) {
          const int byte_index =
              payload_row * 4 + weights_cols_in_tile / 2;
          const uint8_t packed = payload[byte_index];
          const uint8_t quant =
              (weights_cols_in_tile & 1) == 0
                  ? static_cast<uint8_t>(packed & 0x0F)
                  : static_cast<uint8_t>(packed >> 4);
          const float value =
              static_cast<float>(detail::signed_quant(quant)) * scale;
          const int output_index =
              weights_rows_tile * kMmulWeightsElems +
              (tile_row_base + payload_row) * kTileCols +
              weights_cols_in_tile;
          weights_tile[output_index] = static_cast<bfloat16>(value);
        }
      }
    }
  }

  TRACE_LEAVE();
}

// Q4_0 has one chunk per record, so a whole-record wrapper is unambiguous.
inline void unpack_weights_tile(const uint8_t* record,
                                bfloat16* weights_tile) {
  unpack_weights_chunk(common_scales(record), chunk(record, 0), weights_tile);
}

}  // namespace spiral::quants::q4_0
