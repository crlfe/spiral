// Diagnostic mode: checksum one delivered L1 stream block.
//
// No assembly or carry: the block is treated as the transport unit.  The
// checksum is written into the compute's output object at `index` so the host
// can compare it with the bytes it packed.

#include <stdint.h>

extern "C" void program_raw_check(const uint8_t* restrict block,
                                  float* restrict outputs,
                                  int32_t index,
                                  int32_t bytes) {
  const uint64_t* words = reinterpret_cast<const uint64_t*>(block);
  const int32_t count = bytes / 8;
  uint32_t sum = 0;
  for (int32_t i = 0; i < count; ++i) {
    const uint64_t value = words[i];
    sum += static_cast<uint32_t>(value);
    sum += static_cast<uint32_t>(value >> 32);
  }
  for (int32_t i = count * 8; i < bytes; ++i) {
    sum += block[i];
  }
  reinterpret_cast<uint32_t*>(outputs)[index] = sum;
}
