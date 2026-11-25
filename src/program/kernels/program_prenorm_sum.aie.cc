#include "program_prenorm_impl.h"

extern "C" void program_prenorm_sum(const bfloat16* restrict group,
                                    int32_t base_row,
                                    int32_t rows_steps,
                                    int32_t* restrict state) {
  using namespace program_prenorm;
  float* sums = sums_of(state);
  if (base_row == 0) {
    for (int32_t v = 0; v < kVecs; ++v) {
      sums[v] = 0.0f;
    }
  }
  sum_squares_group_at(group, base_row, rows_steps, sums);
}
