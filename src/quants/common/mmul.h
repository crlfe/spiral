#pragma once

#include <aie_api/aie.hpp>

#include "quants/common/tile.h"

namespace spiral::quants {

using MMUL = aie::mmul<kMmulInputsVecs, kMmulWeightsRows, kMmulWeightsCols,
                       bfloat16, bfloat16>;

// outputs[outputs_vecs][outputs_len] +=
//     inputs[weights_rows_tile][inputs_vecs][inputs_len] *
//     weights[weights_rows_tile][weights_rows][weights_cols]
//
// The outer weights_rows_tile index tiles the shared dot-product dimension.
// It is not an inputs_vecs tile.
template <int weights_rows_tiles>
static inline void mmul_weights_tile(const bfloat16* inputs,
                                     const bfloat16* weights,
                                     float* outputs) {
  MMUL accumulator(aie::load_v<kMmulOutputsElems>(outputs));
#pragma unroll
  for (int weights_rows_tile = 0; weights_rows_tile < weights_rows_tiles;
       ++weights_rows_tile) {
    accumulator.mac(
        aie::load_v<kMmulInputsElems>(
            inputs + weights_rows_tile * kMmulInputsElems),
        aie::load_v<kMmulWeightsElems>(
            weights + weights_rows_tile * kMmulWeightsElems));
  }
  aie::store_v(outputs, accumulator.template to_vector<float>());
}

}  // namespace spiral::quants
