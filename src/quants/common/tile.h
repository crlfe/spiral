#pragma once

namespace spiral::quants {

// Shape of one aie::mmul<4, 8, 8> tile, named by inputs/weights/outputs.
constexpr int kMmulInputsVecs = 4;
constexpr int kMmulInputsLen = 8;
constexpr int kMmulWeightsRows = 8;
constexpr int kMmulWeightsCols = 8;
constexpr int kMmulOutputsVecs = 4;
constexpr int kMmulOutputsLen = 8;

constexpr int kMmulInputsElems = kMmulInputsVecs * kMmulInputsLen;
constexpr int kMmulWeightsElems = kMmulWeightsRows * kMmulWeightsCols;
constexpr int kMmulOutputsElems = kMmulOutputsVecs * kMmulOutputsLen;

static_assert(kMmulInputsLen == kMmulWeightsRows,
              "mmul inputs and weights must share the dot-product dimension");

}  // namespace spiral::quants
