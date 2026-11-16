// Copy the broadcast config object into a per-compute state buffer.

#include <stdint.h>

extern "C" void program_config_decode(const int32_t* restrict config,
                                      int32_t* restrict out) {
  for (int32_t i = 0; i < 8; ++i) {
    out[i] = config[i];
  }
}
