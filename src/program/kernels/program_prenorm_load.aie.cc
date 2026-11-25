#include "program_prenorm_impl.h"

// Copy the norm weight from the config object into the stable L1 buffer the
// scale pass reads.  The config object is a stream object that the input DMA
// reuses, so the weight cannot stay there across the input stream.
extern "C" void program_prenorm_load(const bfloat16* restrict config,
                                     bfloat16* restrict weight,
                                     int32_t elems) {
  using namespace program_prenorm;
  const bfloat16* src = config + kWeightOffset;
  int32_t i = 0;
  for (; i + 32 <= elems; i += 32) {
    aie::store_v(weight + i, aie::load_v<32>(src + i));
  }
  for (; i < elems; ++i) {
    weight[i] = src[i];
  }
}
