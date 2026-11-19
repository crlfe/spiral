// Fused Q6_K block: assemble/carry loop plus tile mmul in one kernel.

#include "program_block_impl.h"
#include "quants/q6_k/mmul.aie.h"

namespace {

struct Q6Tile {
  static inline __attribute__((always_inline)) void apply(
      const uint8_t* tile, const bfloat16* inputs, float* outputs) {
    spiral::quants::q6_k::unpack_mmul<256, 16>(tile, inputs, outputs);
  }
};

}  // namespace

extern "C" void program_q6_block(const uint8_t* restrict block,
                                 uint8_t* restrict carry,
                                 const uint16_t* restrict inputs,
                                 float* restrict outputs,
                                 int32_t* restrict state,
                                 int32_t tile_bytes,
                                 int32_t row_limit,
                                 int32_t cols_bands) {
  program_block::run<Q6Tile>(block, carry, inputs, outputs, state, tile_bytes,
                             row_limit, cols_bands);
}
