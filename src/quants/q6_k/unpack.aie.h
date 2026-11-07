#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/tile.h"
#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

// No-op for this pass: zero-fill one chunk-sized weights tile.
inline void unpack_weights_chunk(const uint8_t* record,
                                 const uint8_t* packed_chunk,
                                 bfloat16* weights_tile) {
  (void)record;
  (void)packed_chunk;
  constexpr int kWeightsElems = kUnpackTileRows * kTileCols;
  for (int i = 0; i < kWeightsElems; ++i) {
    weights_tile[i] = static_cast<bfloat16>(0.0f);
  }
}

}  // namespace spiral::quants::q6_k
