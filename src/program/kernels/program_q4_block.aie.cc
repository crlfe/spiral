// Fused Q4_0 block: assemble/carry loop plus tile mmul in one kernel.

#include "program_block_impl.h"
#include "quants/q4_0/mmul.aie.h"

namespace {

struct Q4Tile {
  static inline __attribute__((always_inline)) void apply(
      const uint8_t* tile, const bfloat16* inputs, float* outputs) {
    spiral::quants::q4_0::unpack_mmul<256, 16>(tile, inputs, outputs);
  }
};

}  // namespace

extern "C" void program_q4_block(const uint8_t* restrict block,
                                 uint8_t* restrict carry,
                                 const uint16_t* restrict inputs,
                                 float* restrict outputs,
                                 int32_t* restrict state,
                                 int32_t tile_bytes,
                                 int32_t row_limit,
                                 int32_t cols_bands) {
  program_block::run<Q4Tile>(block, carry, inputs, outputs, state, tile_bytes,
                             row_limit, cols_bands);
}
