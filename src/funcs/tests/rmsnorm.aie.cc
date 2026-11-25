#include <stdint.h>

#include <aie_api/aie.hpp>

#include "funcs/rmsnorm/rmsnorm.aie.h"

// Standalone test entry: copy the input to a separate output buffer, then run
// the full-K RMS norm in place.  Keeping the trace events around only the norm
// makes the cycle count comparable with the program's prelude.
extern "C" void funcs_rmsnorm(const bfloat16* input, const bfloat16* weights,
                              bfloat16* output, int32_t rows_steps,
                              int32_t eps_bits, int32_t inv_k_bits) {
  using namespace spiral::funcs::rmsnorm;
  const int groups = (rows_steps + kRowsPerGroup - 1) / kRowsPerGroup;
  const int elems = groups * kGroupInputElems;
  for (int i = 0; i < elems; ++i) {
    output[i] = input[i];
  }

  const float eps = bits_to_float(eps_bits);
  const float inv_k = bits_to_float(inv_k_bits);
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};

  float sums[kVecs] = {0.0f, 0.0f, 0.0f, 0.0f};
  event0();
  for (int g = 0; g < groups; ++g) {
    sum_squares_group_at(output + g * kGroupInputElems,
                         g * kRowsPerGroup, rows_steps, sums);
  }
  event1();

  bfloat16 r_hi[kVecs];
  bfloat16 r_lo[kVecs];
  finish_r(sums, eps, inv_k, r_hi, r_lo);

  event0();
  for (int g = 0; g < groups; ++g) {
    scale_group_at(output + g * kGroupInputElems, g * kRowsPerGroup,
                   rows_steps, weights, r_hi, r_lo);
  }
  event1();
}
