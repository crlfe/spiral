#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/tile.h"

// Shared exact scale path for the native and int8-expanded Q6_K variants.
namespace spiral::quants::q6_scale {

constexpr int kScaleLanes = 32;
constexpr int kScaleCols = 8;
constexpr int kScaleBytes = kScaleCols * 2;

// Load the host-precomputed exact bf16 hi/lo terms for the per-column d values
// and expand them to all subgroup scale lanes. Canonical fp16 d values,
// including subnormals, are split exactly on the host; the device never decodes
// fp16 here.
static inline __attribute__((always_inline)) void load_d16_terms(
    const uint8_t* packed_d,
    aie::vector<bfloat16, kScaleLanes>& d_hi,
    aie::vector<bfloat16, kScaleLanes>& d_lo) {
  const auto d_hi8 =
      aie::load_v<kScaleCols>(reinterpret_cast<const bfloat16*>(packed_d));
  const auto d_lo8 = aie::load_v<kScaleCols>(
      reinterpret_cast<const bfloat16*>(packed_d + kScaleBytes));
  d_hi = aie::broadcast_vector<kScaleCols, bfloat16, kScaleLanes>(d_hi8, 0);
  d_lo = aie::broadcast_vector<kScaleCols, bfloat16, kScaleLanes>(d_lo8, 0);
}

// Form d * subgroup_scale and split it into three exact bf16 terms. The
// product needs up to 18 significand bits; two terms are not always enough to
// round to the same bf16 as an fp32 multiply.
static inline __attribute__((always_inline)) void compute_chunk_scale_terms(
    const aie::vector<bfloat16, kScaleLanes>& d_hi,
    const aie::vector<bfloat16, kScaleLanes>& d_lo,
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kScaleLanes>& hi,
    aie::vector<bfloat16, kScaleLanes>& mid,
    aie::vector<bfloat16, kScaleLanes>& lo) {
  const auto subgroup_scales = aie::to_float<bfloat16>(
      aie::load_v<kScaleLanes>(packed_scales).template cast_to<int8>());
  auto accumulator = aie::mul(d_hi, subgroup_scales);
  accumulator = aie::mac(accumulator, d_lo, subgroup_scales);
  const auto product = accumulator.template to_vector<float>();

  hi = aie::to_vector<bfloat16>(aie::accum<accfloat, kScaleLanes>(product));
  const auto hi_f32 = aie::accum<accfloat, kScaleLanes>(hi).to_vector<float>();
  const auto residual1 = aie::sub(product, hi_f32);

  mid = aie::to_vector<bfloat16>(aie::accum<accfloat, kScaleLanes>(residual1));
  const auto mid_f32 =
      aie::accum<accfloat, kScaleLanes>(mid).to_vector<float>();
  const auto residual2 = aie::sub(residual1, mid_f32);

  lo = aie::to_vector<bfloat16>(aie::accum<accfloat, kScaleLanes>(residual2));
}

// Expand one subgroup's eight per-column scale terms to 64 lanes.
static inline __attribute__((always_inline)) void load_subgroup_scales64(
    const aie::vector<bfloat16, kScaleLanes>& hi32,
    const aie::vector<bfloat16, kScaleLanes>& mid32,
    const aie::vector<bfloat16, kScaleLanes>& lo32,
    int subgroup,
    aie::vector<bfloat16, kMmulWeightsElems>& hi,
    aie::vector<bfloat16, kMmulWeightsElems>& mid,
    aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  hi = aie::broadcast_vector<kScaleCols, bfloat16, kMmulWeightsElems>(hi32,
                                                                      subgroup);
  mid = aie::broadcast_vector<kScaleCols, bfloat16, kMmulWeightsElems>(
      mid32, subgroup);
  lo = aie::broadcast_vector<kScaleCols, bfloat16, kMmulWeightsElems>(lo32,
                                                                      subgroup);
}

// Apply the three exact bf16 scale terms to one 64-lane values vector.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    apply_scales(const aie::vector<bfloat16, kMmulWeightsElems>& values,
                 const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                 const aie::vector<bfloat16, kMmulWeightsElems>& mid,
                 const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  auto accumulator = aie::mul(values, hi);
  accumulator = aie::mac(accumulator, values, mid);
  accumulator = aie::mac(accumulator, values, lo);
  return accumulator.template to_vector<bfloat16>();
}

}  // namespace spiral::quants::q6_scale
