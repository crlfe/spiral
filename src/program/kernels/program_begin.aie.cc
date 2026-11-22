// Copy the broadcast config object into per-compute config, reset stream state.

#include <stdint.h>

#ifndef PROGRAM_CONFIG_LEN
#define PROGRAM_CONFIG_LEN 16
#endif
#ifndef PROGRAM_STATE_WORDS
#define PROGRAM_STATE_WORDS 8
#endif

extern "C" void program_begin(const int32_t* restrict config,
                              int32_t* restrict out,
                              int32_t* restrict state) {
  for (int32_t i = 0; i < PROGRAM_CONFIG_LEN; ++i) {
    out[i] = config[i];
  }
  for (int32_t i = 0; i < PROGRAM_STATE_WORDS; ++i) {
    state[i] = 0;
  }
}
