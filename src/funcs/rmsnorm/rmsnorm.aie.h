#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "funcs/rmsnorm/layout.h"

namespace spiral::funcs::rmsnorm {

using InputVec = aie::vector<bfloat16, kVecs * kChunk>;
using SqVec = aie::vector<float, kVecs * kChunk>;

// Repeat the 128-bit lane (8 bf16) selected by the constant `Lane` across all
// 32 lanes.  On XDNA2 the compiler emits one vextbcst.128 for any lane;
// aie::broadcast_vector() first shifts the lane into position and costs an
// extra vshift per chunk.
#if defined(__AIE_ARCH__) && __AIE_ARCH__ == 21
template <int Lane>
inline InputVec repeat_lane(const InputVec& v) {
  return (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)v, Lane);
}
#else
template <int Lane>
inline InputVec repeat_lane(const InputVec& v) {
  return aie::broadcast_vector<kChunk>(v, Lane);
}
#endif

inline float bits_to_float(int32_t bits) {
  union {
    int32_t bits;
    float value;
  } v{bits};
  return v.value;
}

inline int weight_pair_index(int row, int chunk) {
  return (row * kRowStep + chunk * kChunk) * 2;
}

// Add one input group's sum of squares to `sums`, one value per input vector.
// `group_input` points at exactly one group: [row_step][chunk][vec][8].
// `base_row` is the group's first global row-step index.
inline void sum_squares_group_at(const bfloat16* group_input, int base_row,
                                 int rows_steps, float sums[kVecs]) {
  // Four f32 partials keep the per-chunk mac off a single dependency chain;
  // one accumulator measured 1059 cycles versus 765 for this pass.
  aie::accum<accfloat, kVecs * kChunk> p0(
      aie::zeros<float, kVecs * kChunk>());
  aie::accum<accfloat, kVecs * kChunk> p1(
      aie::zeros<float, kVecs * kChunk>());
  aie::accum<accfloat, kVecs * kChunk> p2(
      aie::zeros<float, kVecs * kChunk>());
  aie::accum<accfloat, kVecs * kChunk> p3(
      aie::zeros<float, kVecs * kChunk>());

#pragma clang loop unroll(disable)
  for (int within = 0; within < kRowsPerGroup; ++within) {
    const int row = base_row + within;
    if (row >= rows_steps) {
      break;
    }
    const bfloat16* row_input = group_input + within * kRowStep * kVecs;

    for (int chunk = 0; chunk < kChunksPerGroup; chunk += 8) {
      const bfloat16* p = row_input + chunk * (kVecs * kChunk);
      const InputVec x0 = aie::load_v<kVecs * kChunk>(p);
      const InputVec x1 = aie::load_v<kVecs * kChunk>(p + kVecs * kChunk);
      const InputVec x2 =
          aie::load_v<kVecs * kChunk>(p + 2 * kVecs * kChunk);
      const InputVec x3 =
          aie::load_v<kVecs * kChunk>(p + 3 * kVecs * kChunk);
      const InputVec x4 =
          aie::load_v<kVecs * kChunk>(p + 4 * kVecs * kChunk);
      const InputVec x5 =
          aie::load_v<kVecs * kChunk>(p + 5 * kVecs * kChunk);
      const InputVec x6 =
          aie::load_v<kVecs * kChunk>(p + 6 * kVecs * kChunk);
      const InputVec x7 =
          aie::load_v<kVecs * kChunk>(p + 7 * kVecs * kChunk);
      p0 = aie::mac(p0, x0, x0);
      p1 = aie::mac(p1, x1, x1);
      p2 = aie::mac(p2, x2, x2);
      p3 = aie::mac(p3, x3, x3);
      p0 = aie::mac(p0, x4, x4);
      p1 = aie::mac(p1, x5, x5);
      p2 = aie::mac(p2, x6, x6);
      p3 = aie::mac(p3, x7, x7);
    }
  }

  const aie::accum<accfloat, kVecs * kChunk> partial =
      aie::add(aie::add(p0, p1), aie::add(p2, p3));
  const SqVec sum_sq = partial.template to_vector<float>();
#pragma clang loop unroll(disable)
  for (int v = 0; v < kVecs; ++v) {
    sums[v] += aie::reduce_add(sum_sq.template extract<kChunk>(v));
  }
}

// Full-K wrapper: `input` points at the first group.
inline void sum_squares_group(const bfloat16* input, int group, int rows_steps,
                             float sums[kVecs]) {
  sum_squares_group_at(input + group * kGroupInputElems,
                       group * kRowsPerGroup, rows_steps, sums);
}

// Turn the accumulated squares into per-vector reciprocal RMS hi/lo pairs.
inline void finish_r(const float sums[kVecs], float eps, float inv_k,
                     bfloat16 r_hi[kVecs], bfloat16 r_lo[kVecs]) {
#pragma clang loop unroll(disable)
  for (int v = 0; v < kVecs; ++v) {
    const float mean = sums[v] * inv_k + eps;
    const float r = aie::invsqrt(mean);
    const bfloat16 hi = bfloat16(r);
    r_hi[v] = hi;
    r_lo[v] = bfloat16(r - static_cast<float>(hi));
  }
}

// Scale one input group in place with r * w.  `group_input` points at exactly
// one group; `base_row` selects its first weight row-step.  The weight is the
// full-K per-chunk hi/lo vector; this group reads its four row-step slices.
// Combining r and w in f32 before splitting avoids rounding x*r to bf16 before
// the weight multiply.
//
// Two chunks are processed per iteration so their independent scale/split
// chains overlap.  The [hi8][lo8] pairs of two consecutive chunks are
// contiguous, so both chunks' weights come from one 512-bit load.
inline void scale_group_at(bfloat16* group_input, int base_row, int rows_steps,
                           const bfloat16* weight,
                           const bfloat16 r_hi[kVecs],
                           const bfloat16 r_lo[kVecs]) {
  const InputVec rh = aie::concat(
      aie::broadcast<bfloat16, kChunk>(r_hi[0]),
      aie::broadcast<bfloat16, kChunk>(r_hi[1]),
      aie::broadcast<bfloat16, kChunk>(r_hi[2]),
      aie::broadcast<bfloat16, kChunk>(r_hi[3]));
  const InputVec rl = aie::concat(
      aie::broadcast<bfloat16, kChunk>(r_lo[0]),
      aie::broadcast<bfloat16, kChunk>(r_lo[1]),
      aie::broadcast<bfloat16, kChunk>(r_lo[2]),
      aie::broadcast<bfloat16, kChunk>(r_lo[3]));

#pragma clang loop unroll(disable)
  for (int within = 0; within < kRowsPerGroup; ++within) {
    const int row = base_row + within;
    if (row >= rows_steps) {
      break;
    }
    bfloat16* row_input = group_input + within * kRowStep * kVecs;

#pragma clang loop unroll_count(4)
    for (int chunk = 0; chunk < kChunksPerGroup; chunk += 2) {
      bfloat16* p0 = row_input + chunk * (kVecs * kChunk);
      bfloat16* p1 = p0 + kVecs * kChunk;
      const InputVec quad =
          aie::load_v<kVecs * kChunk>(weight + weight_pair_index(row, chunk));
      const InputVec x0 = aie::load_v<kVecs * kChunk>(p0);
      const InputVec x1 = aie::load_v<kVecs * kChunk>(p1);

      const InputVec wh0 = repeat_lane<0>(quad);
      const InputVec wl0 = repeat_lane<1>(quad);
      const InputVec wh1 = repeat_lane<2>(quad);
      const InputVec wl1 = repeat_lane<3>(quad);

      auto s0 = aie::mul(rh, wh0);
      auto s1 = aie::mul(rh, wh1);
      s0 = aie::mac(s0, rh, wl0);
      s1 = aie::mac(s1, rh, wl1);
      s0 = aie::mac(s0, rl, wh0);
      s1 = aie::mac(s1, rl, wh1);

      const InputVec hi0 = s0.template to_vector<bfloat16>();
      const InputVec hi1 = s1.template to_vector<bfloat16>();
      const InputVec lo0 = aie::accum<accfloat, kVecs * kChunk>(
          aie::sub(s0.template to_vector<float>(),
                   aie::accum<accfloat, kVecs * kChunk>(hi0)
                       .template to_vector<float>()))
                               .template to_vector<bfloat16>();
      const InputVec lo1 = aie::accum<accfloat, kVecs * kChunk>(
          aie::sub(s1.template to_vector<float>(),
                   aie::accum<accfloat, kVecs * kChunk>(hi1)
                       .template to_vector<float>()))
                               .template to_vector<bfloat16>();

      auto o0 = aie::mul(x0, hi0);
      auto o1 = aie::mul(x1, hi1);
      o0 = aie::mac(o0, x0, lo0);
      o1 = aie::mac(o1, x1, lo1);
      aie::store_v(p0, o0.template to_vector<bfloat16>());
      aie::store_v(p1, o1.template to_vector<bfloat16>());
    }
  }
}

inline void scale_group(bfloat16* input, int group, int rows_steps,
                        const bfloat16* weight, const bfloat16 r_hi[kVecs],
                        const bfloat16 r_lo[kVecs]) {
  scale_group_at(input + group * kGroupInputElems, group * kRowsPerGroup,
                 rows_steps, weight, r_hi, r_lo);
}

// Full-K reference path for the standalone test and for callers that already
// hold the whole activation.  `data` is updated in place.
inline void normalize(bfloat16* data, const bfloat16* weight, int rows_steps,
                      float eps, float inv_k) {
  float sums[kVecs] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int groups = (rows_steps + kRowsPerGroup - 1) / kRowsPerGroup;
  for (int g = 0; g < groups; ++g) {
    sum_squares_group(data, g, rows_steps, sums);
  }

  bfloat16 r_hi[kVecs];
  bfloat16 r_lo[kVecs];
  finish_r(sums, eps, inv_k, r_hi, r_lo);

  for (int g = 0; g < groups; ++g) {
    scale_group(data, g, rows_steps, weight, r_hi, r_lo);
  }
}

}  // namespace spiral::funcs::rmsnorm
