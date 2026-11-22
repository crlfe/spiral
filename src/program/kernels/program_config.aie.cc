// Copy the broadcast config object into a per-compute state buffer.

#include <stdint.h>

#ifndef PROGRAM_CONFIG_LEN
#define PROGRAM_CONFIG_LEN 16
#endif

extern "C" void program_config_decode(const int32_t* restrict config,
                                      int32_t* restrict out) {
  for (int32_t i = 0; i < PROGRAM_CONFIG_LEN; ++i) {
    out[i] = config[i];
  }
}
