#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "program_state.h"

#ifndef PROGRAM_BLOCK_BYTES
#define PROGRAM_BLOCK_BYTES 8192
#endif
#ifndef PROGRAM_MAX_TILES
#define PROGRAM_MAX_TILES 5
#endif
#ifndef PROGRAM_OUTPUT_TILE_ELEMS
#define PROGRAM_OUTPUT_TILE_ELEMS 64
#endif

// Shared fused block body: assemble transport tiles from the streamed weights
// and hand each complete tile to a format-specific policy. `run` is
// non-template and only defined once (COMDAT), so adding a format does not
// duplicate the assembly loop; the policy is called through a function
// pointer. The pointer call is the price of sharing the loop and is measured
// against the old per-format inline copy.
namespace program_block {

using program::kBlockDone;
using program::kBlockPos;
using program::kCarryLen;
using program::kReady;
using program::kTileIndex;

// Assemble one transport tile, or point at it directly when it is complete
// inside the current block.  A tile can straddle a block boundary, in which
// case the prefix is copied into the carry buffer and completed from the next
// block.
inline bool assemble_one(const uint8_t* restrict block,
                         uint8_t* restrict carry,
                         int32_t tile_bytes,
                         int32_t& pos,
                         int32_t& carry_len,
                         bool& block_done,
                         const uint8_t*& tile) {
  if (block_done) {
    return false;
  }

  if (carry_len == 0 && pos + tile_bytes <= PROGRAM_BLOCK_BYTES) {
    tile = block + pos;
    pos += tile_bytes;
    if (pos == PROGRAM_BLOCK_BYTES) {
      block_done = true;
    }
    return true;
  }

  const int32_t need = tile_bytes - carry_len;
  const int32_t avail = PROGRAM_BLOCK_BYTES - pos;
  const int32_t n = need < avail ? need : avail;

  const uint64_t* src = reinterpret_cast<const uint64_t*>(block + pos);
  uint64_t* dst = reinterpret_cast<uint64_t*>(carry + carry_len);
  const int32_t words = n / 8;
  const uint64_t* src_end = src + words;
  while (src < src_end) {
    *dst++ = *src++;
  }
  for (int32_t i = words * 8; i < n; ++i) {
    carry[carry_len + i] = block[pos + i];
  }
  pos += n;

  if (carry_len + n == tile_bytes) {
    carry_len = 0;
    if (pos == PROGRAM_BLOCK_BYTES) {
      block_done = true;
    }
    tile = carry;
    return true;
  }

  carry_len += n;
  block_done = true;
  return false;
}

// The block kernels accumulate into the private f32 accumulator at
// `outputs + band * PROGRAM_OUTPUT_TILE_ELEMS`; the separate program_output
// epilogue transposes, optionally softcaps, and stores bf16.
using ApplyFn = void (*)(const uint8_t* tile,
                         const bfloat16* inputs,
                         float* outputs);

inline __attribute__((noinline)) void run(
    const uint8_t* restrict block,
    uint8_t* restrict carry,
    const uint16_t* restrict inputs,
    float* restrict outputs,
    int32_t* restrict state,
    int32_t tile_bytes,
    int32_t row_limit,
    int32_t cols_bands,
    ApplyFn apply) {
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
      apply(tile, row_inputs, outputs + band * PROGRAM_OUTPUT_TILE_ELEMS);
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
