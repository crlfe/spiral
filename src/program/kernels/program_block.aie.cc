// Prepare the state for a newly acquired weight block.

#include <stdint.h>

namespace {
enum State {
  kBlockPos = 1,
  kBlockDone = 2,
  kReady = 4,
};
}  // namespace

extern "C" void program_block_begin(int32_t* restrict state) {
  state[kBlockPos] = 0;
  state[kBlockDone] = 0;
  state[kReady] = 0;
}
