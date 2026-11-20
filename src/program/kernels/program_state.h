// Shared per-core state layout.  Keep in sync with design.py's STATE_* values
// and the host's state object; the kernels and the host both index this array.

#pragma once

#include <stdint.h>

namespace program {

enum State : int32_t {
  kCarryLen = 0,
  kBlockPos = 1,
  kBlockDone = 2,
  kTileIndex = 3,
  kReady = 4,
  kTileSource = 5,
  kTileOffset = 6,
  kCurrentRow = 7,
};

}  // namespace program
