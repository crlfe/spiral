// Reset the per-compute stream state before a call.

#include <stdint.h>

extern "C" void program_init(int32_t* restrict state) {
  for (int32_t i = 0; i < 8; ++i) {
    state[i] = 0;
  }
}
