// Clear the output object for one compute tile.

#include <stdint.h>

extern "C" void stream_zero(float * restrict outputs, int32_t vec_steps) {
    const int32_t n = vec_steps * 64;
    for (int32_t i = 0; i < n; ++i) {
        outputs[i] = 0.0f;
    }
}
