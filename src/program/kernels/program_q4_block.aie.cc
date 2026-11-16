// Fused Q4_0 block: assemble/carry loop plus tile mmul in one kernel.

#include <stdint.h>

#include "program_block_common.h"
#include "quants/q4_0/mmul.aie.h"

namespace {
enum State {
  kCarryLen = 0,
  kBlockPos = 1,
  kBlockDone = 2,
  kTileIndex = 3,
  kReady = 4,
};
}  // namespace

extern "C" void program_q4_block(const uint8_t* restrict block,
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
    if (program_block::assemble_one(block, carry, tile_bytes, pos, carry_len,
                                   block_done, tile)) {
      const int32_t band = tile_index % cols_bands;
      const int32_t row = tile_index / cols_bands;
      const int32_t local_row = row % PROGRAM_INPUT_GROUP_ROWS;
      const bfloat16* row_inputs =
          reinterpret_cast<const bfloat16*>(inputs) +
          local_row * PROGRAM_INPUT_ELEMS;
      spiral::quants::q4_0::unpack_mmul<256, 16>(
          tile, row_inputs, outputs + band * 64);
      ++tile_index;
    }
  }

  state[kCarryLen] = carry_len;
  state[kBlockPos] = pos;
  state[kBlockDone] = block_done ? 1 : 0;
  state[kTileIndex] = tile_index;
  state[kReady] = 0;
}
