// Minimal stream probe: touch one tile word and one input word.
//
// The weight DMA and carry assembly still move the full tile; this kernel
// performs almost no arithmetic so timing isolates the stream path.

#include <stdint.h>

#ifndef PROGRAM_INPUT_ELEMS
#define PROGRAM_INPUT_ELEMS 1024
#endif
#ifndef PROGRAM_INPUT_GROUP_ROWS
#define PROGRAM_INPUT_GROUP_ROWS 4
#endif

namespace {
enum State {
  kTileIndex = 3,
  kReady = 4,
  kTileSource = 5,
  kTileOffset = 6,
};
}  // namespace

extern "C" void program_stream(const uint8_t* restrict block,
                               const uint8_t* restrict carry,
                               const uint16_t* restrict inputs,
                               float* restrict outputs,
                               int32_t* restrict state,
                               int32_t cols_bands) {
  const uint8_t* tile =
      state[kTileSource] ? block + state[kTileOffset] : carry;
  const int32_t index = state[kTileIndex];
  const int32_t band = index % cols_bands;
  const int32_t row = index / cols_bands;
  const int32_t local_row = row % PROGRAM_INPUT_GROUP_ROWS;
  inputs += local_row * PROGRAM_INPUT_ELEMS;
  outputs[band * 64] = static_cast<float>(tile[0]) +
                       static_cast<float>(inputs[0]);
  state[kTileIndex] = index + 1;
  state[kReady] = 0;
}
