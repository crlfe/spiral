// No-op tile consumer: checksum the assembled tile and its input object.
//
// The result is written as a raw uint32 bit pattern into the bf16 output
// object, so the host can compare it exactly.  This is the format-0 probe
// used to validate the data flow independently of the quantization kernels.

#include <stdint.h>

#include "program_state.h"

#ifndef PROGRAM_INPUT_ELEMS
#define PROGRAM_INPUT_ELEMS 1024
#endif
#ifndef PROGRAM_INPUT_GROUP_ROWS
#define PROGRAM_INPUT_GROUP_ROWS 4
#endif

using program::kReady;
using program::kTileIndex;
using program::kTileOffset;
using program::kTileSource;

extern "C" void program_probe(const uint8_t* restrict block,
                              const uint8_t* restrict carry,
                              const uint16_t* restrict inputs,
                              uint8_t* restrict outputs,
                              int32_t* restrict state,
                              int32_t tile_bytes,
                              int32_t cols_bands) {
  const uint8_t* tile =
      state[kTileSource] ? block + state[kTileOffset] : carry;
  uint32_t sum = 0;

  int32_t i = 0;
  for (; i + 4 <= tile_bytes; i += 4) {
    uint32_t value = 0;
    value |= static_cast<uint32_t>(tile[i + 0]) << 0;
    value |= static_cast<uint32_t>(tile[i + 1]) << 8;
    value |= static_cast<uint32_t>(tile[i + 2]) << 16;
    value |= static_cast<uint32_t>(tile[i + 3]) << 24;
    sum += value;
  }
  for (; i < tile_bytes; ++i) {
    sum += tile[i];
  }

  const int32_t index = state[kTileIndex];
  const int32_t row = index / cols_bands;
  const int32_t local_row = row % PROGRAM_INPUT_GROUP_ROWS;
  inputs += local_row * PROGRAM_INPUT_ELEMS;

  for (int32_t j = 0; j < PROGRAM_INPUT_ELEMS; ++j) {
    sum += inputs[j];
  }

  reinterpret_cast<uint32_t*>(outputs)[index] = sum;
  state[kTileIndex] = index + 1;
  state[kReady] = 0;
}
