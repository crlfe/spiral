// Prepare the state for a newly acquired weight block.

#include <stdint.h>

#include "program_state.h"

extern "C" void program_block_begin(int32_t* restrict state) {
  state[program::kBlockPos] = 0;
  state[program::kBlockDone] = 0;
  state[program::kReady] = 0;
}
