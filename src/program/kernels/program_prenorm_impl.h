#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "funcs/rmsnorm/rmsnorm.aie.h"

#ifndef PROGRAM_PRENORM_WEIGHT_OFFSET_BYTES
#define PROGRAM_PRENORM_WEIGHT_OFFSET_BYTES 64
#endif

namespace program_prenorm {

using spiral::funcs::rmsnorm::bits_to_float;
using spiral::funcs::rmsnorm::finish_r;
using spiral::funcs::rmsnorm::kVecs;
using spiral::funcs::rmsnorm::scale_group_at;
using spiral::funcs::rmsnorm::sum_squares_group_at;

inline float* sums_of(int32_t* state) {
  return reinterpret_cast<float*>(state);
}

inline bfloat16* r_hi_of(int32_t* state) {
  return reinterpret_cast<bfloat16*>(state + 4);
}

inline const bfloat16* r_hi_of(const int32_t* state) {
  return reinterpret_cast<const bfloat16*>(state + 4);
}

inline constexpr int32_t kWeightOffset =
    PROGRAM_PRENORM_WEIGHT_OFFSET_BYTES / static_cast<int32_t>(sizeof(bfloat16));

}  // namespace program_prenorm
