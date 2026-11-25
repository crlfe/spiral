#include "program_prenorm_impl.h"

extern "C" void program_prenorm_finish(int32_t* restrict state,
                                       int32_t eps_bits,
                                       int32_t inv_k_bits) {
  using namespace program_prenorm;
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  bfloat16* r_hi = r_hi_of(state);
  bfloat16* r_lo = r_hi + kVecs;
  finish_r(sums_of(state), bits_to_float(eps_bits),
           bits_to_float(inv_k_bits), r_hi, r_lo);
}
