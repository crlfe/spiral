#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

// TODO: implement fused Q6_K unpack and mmul. The caller must zero outputs
// and select aie::rounding_mode::conv_even before calling this.
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs) {
  static_assert(weights_rows % kRecordRows == 0,
                "weights_rows must be a multiple of the record depth");
  static_assert(weights_cols % kTileCols == 0,
                "weights_cols must be a multiple of the tile width");

  (void)packed;
  (void)inputs;
  (void)outputs;
}

}  // namespace spiral::quants::q6_k
