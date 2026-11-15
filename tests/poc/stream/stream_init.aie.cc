// Reset the per-compute stream state before each repetition.

#include <stdint.h>

extern "C" void stream_init(int32_t * restrict state) {
    state[0] = 0;  // carry length
    state[1] = 0;  // tile index
    state[2] = 0;  // band index
    state[3] = 0;  // row-step index
}
