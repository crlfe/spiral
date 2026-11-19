#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "program_block_common.h"

// Shared fused block body: assemble transport tiles from the streamed weights
// and hand each complete tile to a format-specific `Tile` policy. The policy
// is a template parameter, not a runtime function pointer: every format needs
// one compiled copy anyway (linking more than one overflows program memory),
// and a direct call lets the whole dequantize+mmul body inline.
namespace program_block {

// Fixed layout of the per-core state object shared with host/test code.
enum State {
  kCarryLen = 0,
  kBlockPos = 1,
  kBlockDone = 2,
  kTileIndex = 3,
  kReady = 4,
};

// `Tile` must provide:
//   static void apply(const uint8_t* tile,
//                     const bfloat16* inputs,
//                     float* outputs);
template <typename Tile>
inline __attribute__((always_inline)) void run(
    const uint8_t* restrict block,
    uint8_t* restrict carry,
    const uint16_t* restrict inputs,
    float* restrict outputs,
    int32_t* restrict state,
    int32_t tile_bytes,
    int32_t row_limit,
    int32_t cols_bands) {
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};

  int32_t pos = state[kBlockPos];
  int32_t carry_len = state[kCarryLen];
  int32_t tile_index = state[kTileIndex];
  bool block_done = state[kBlockDone] != 0;

  for (int32_t attempt = 0; attempt < PROGRAM_MAX_TILES; ++attempt) {
    if (block_done || tile_index >= row_limit) {
      break;
    }
    const uint8_t* tile = nullptr;
    if (assemble_one(block, carry, tile_bytes, pos, carry_len, block_done,
                     tile)) {
      const int32_t band = tile_index % cols_bands;
      const int32_t row = tile_index / cols_bands;
      const int32_t local_row = row % PROGRAM_INPUT_GROUP_ROWS;
      const bfloat16* row_inputs =
          reinterpret_cast<const bfloat16*>(inputs) +
          local_row * PROGRAM_INPUT_ELEMS;
      Tile::apply(tile, row_inputs, outputs + band * 64);
      ++tile_index;
    }
  }

  state[kCarryLen] = carry_len;
  state[kBlockPos] = pos;
  state[kBlockDone] = block_done ? 1 : 0;
  state[kTileIndex] = tile_index;
  state[kReady] = 0;
}

}  // namespace program_block
