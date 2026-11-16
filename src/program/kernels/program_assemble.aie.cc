// Assemble one transport tile, or point at it directly when it is complete
// inside the current L1 block.
//
// A tile can straddle a weight block boundary, so in that case the prefix is
// copied into the carry buffer and completed from the next block.  When the
// whole tile already lies in the acquired block, no copy is needed: the tile
// compute reads it in place.
//
// The function reports tile completion through state[kReady]; the worker then
// runs the format's tile compute and releases the block.

#include <stdint.h>

#ifndef PROGRAM_BLOCK_BYTES
#define PROGRAM_BLOCK_BYTES 4096
#endif
#ifndef PROGRAM_CARRY_BYTES
#define PROGRAM_CARRY_BYTES 4096
#endif

namespace {
enum State {
  kCarryLen = 0,
  kBlockPos = 1,
  kBlockDone = 2,
  kTileIndex = 3,
  kReady = 4,
  kTileSource = 5,
  kTileOffset = 6,
};
}  // namespace

extern "C" void program_assemble(const uint8_t* restrict block,
                                 uint8_t* restrict carry,
                                 int32_t* restrict state,
                                 int32_t tile_bytes,
                                 int32_t total_tiles) {
  if (state[kBlockDone] != 0 || state[kTileIndex] >= total_tiles) {
    state[kReady] = 0;
    return;
  }

  const int32_t pos = state[kBlockPos];
  const int32_t carry_len = state[kCarryLen];

  if (carry_len == 0 && pos + tile_bytes <= PROGRAM_BLOCK_BYTES) {
    state[kTileSource] = 1;
    state[kTileOffset] = pos;
    state[kReady] = 1;
    state[kBlockPos] = pos + tile_bytes;
    if (pos + tile_bytes == PROGRAM_BLOCK_BYTES) {
      state[kBlockDone] = 1;
    }
    return;
  }

  const int32_t need = tile_bytes - carry_len;
  const int32_t avail = PROGRAM_BLOCK_BYTES - pos;
  const int32_t n = need < avail ? need : avail;

  // All block positions, carry lengths, and tile sizes are multiples of 8
  // here, so copy whole 64-bit words instead of bytes.
  const uint64_t* src =
      reinterpret_cast<const uint64_t*>(block + pos);
  uint64_t* dst = reinterpret_cast<uint64_t*>(carry + carry_len);
  const int32_t words = n / 8;
  const uint64_t* src_end = src + words;
  while (src < src_end) {
    *dst++ = *src++;
  }
  for (int32_t i = words * 8; i < n; ++i) {
    carry[carry_len + i] = block[pos + i];
  }
  state[kBlockPos] = pos + n;
  state[kTileSource] = 0;

  if (carry_len + n == tile_bytes) {
    state[kCarryLen] = 0;
    state[kReady] = 1;
    if (pos + n == PROGRAM_BLOCK_BYTES) {
      state[kBlockDone] = 1;
    }
  } else {
    // The block is exhausted and the tile is still incomplete.
    state[kCarryLen] = carry_len + n;
    state[kBlockDone] = 1;
    state[kReady] = 0;
  }
}
