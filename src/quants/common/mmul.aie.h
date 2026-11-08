#pragma once

#include <aie_api/aie.hpp>

#include "quants/common/tile.h"

namespace spiral::quants {

using MMUL = aie::mmul<kMmulInputsVecs,
                       kMmulWeightsRows,
                       kMmulWeightsCols,
                       bfloat16,
                       bfloat16>;

// Accumulate one 8x8 weights tile into an mmul accumulator. Fused kernels use
// this to reuse a preloaded input vector across multiple weight tiles.
static inline void mmul_weights_tile(
    MMUL& accumulator,
    const aie::vector<bfloat16, kMmulInputsElems>& inputs,
    const aie::vector<bfloat16, kMmulWeightsElems>& weights) {
  accumulator.mac(inputs, weights);
}

}  // namespace spiral::quants
