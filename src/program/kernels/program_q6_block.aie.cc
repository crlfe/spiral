// Fused Q6_K block: the shared assembly loop plus the tile mmul.

#include "program_block_impl.h"

#include "quants/q6_k/mmul.aie.h"

extern "C" void program_q6_block(const uint8_t* restrict block,
                                 uint8_t* restrict carry,
                                 const uint16_t* restrict inputs,
                                 float* restrict outputs,
                                 int32_t* restrict state,
                                 int32_t tile_bytes,
                                 int32_t row_limit,
                                 int32_t cols_bands) {
  program_block::run(block, carry, inputs, outputs, state, tile_bytes,
                     row_limit, cols_bands,
                     &spiral::quants::q6_k::unpack_mmul<256, 16>);
}
