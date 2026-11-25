#pragma once

#include <aie_api/aie.hpp>

// Device vector helpers shared by the quant unpack code.
namespace spiral::quants {

// Repeat the 128-bit lane (8 bf16 values) selected by `lane` across a 512-bit
// vector.  On XDNA2 the hardware takes the lane as a register operand, so this
// is one `vextbcst.128` for any lane.  aie::broadcast_vector() instead lowers to
// a `vshift` that moves the lane into position followed by a lane-0 broadcast,
// which costs an extra instruction whenever the lane is not zero.
#if defined(__AIE_ARCH__) && __AIE_ARCH__ == 21
template <unsigned InElems>
static inline __attribute__((always_inline)) aie::vector<bfloat16, 32>
repeat_lane32(const aie::vector<bfloat16, InElems>& v, int lane) {
  if constexpr (InElems >= 32) {
    return (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)v, lane);
  } else {
    // grow() leaves the upper lanes undefined but keeps lanes 0..InElems/8-1,
    // which is where the selectable 128-bit lanes live.
    const aie::vector<bfloat16, 32> wide = v.template grow<32>();
    return (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)wide, lane);
  }
}
#else
template <unsigned InElems>
static inline __attribute__((always_inline)) aie::vector<bfloat16, 32>
repeat_lane32(const aie::vector<bfloat16, InElems>& v, int lane) {
  return aie::broadcast_vector<8, bfloat16, 32>(v, lane);
}
#endif

// Same, expanded to 64 lanes (two 512-bit halves holding the same pattern).
template <unsigned InElems>
static inline __attribute__((always_inline)) aie::vector<bfloat16, 64>
repeat_lane64(const aie::vector<bfloat16, InElems>& v, int lane) {
  return repeat_lane32(v, lane).template grow_replicate<64>();
}

}  // namespace spiral::quants
