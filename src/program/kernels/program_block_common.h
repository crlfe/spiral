#pragma once

#include <stdint.h>

#ifndef PROGRAM_BLOCK_BYTES
#define PROGRAM_BLOCK_BYTES 8192
#endif
#ifndef PROGRAM_MAX_TILES
#define PROGRAM_MAX_TILES 5
#endif

namespace program_block {

// Assemble one tile attempt using local block state.
// Returns true when a complete tile is available through `tile`.
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

}  // namespace program_block
