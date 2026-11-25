#include "program_prenorm_impl.h"

extern "C" void program_prenorm_scale(bfloat16* restrict group,
                                      int32_t base_row,
                                      int32_t rows_steps,
                                      const bfloat16* restrict weight,
                                      const int32_t* restrict state) {
  using namespace program_prenorm;
  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  const bfloat16* r_hi = r_hi_of(state);
  const bfloat16* r_lo = r_hi + kVecs;
  scale_group_at(group, base_row, rows_steps, weight, r_hi, r_lo);
}
