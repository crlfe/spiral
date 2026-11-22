// Clear one compute's f32 accumulator before a call.

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" void program_zero(float* restrict outputs, int32_t count) {
  const aie::vector<float, 16> zero = aie::zeros<float, 16>();
  float* p = outputs;
  float* end = outputs + count;
  while (p + 16 <= end) {
    aie::store_v(p, zero);
    p += 16;
  }
  while (p < end) {
    *p++ = 0.0f;
  }
}
