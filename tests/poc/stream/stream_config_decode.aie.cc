// Decode the broadcast config object into a small per-compute buffer.

#include <stdint.h>

#include <aie_api/aie.hpp>

// The config travels as the first broadcast input object.  Decode it into a
// small per-compute buffer so the worker can use the counts as runtime values.
extern "C" void stream_config_decode(
        const bfloat16 * restrict config,
        int32_t * restrict out)
{
    const int32_t * values = reinterpret_cast<const int32_t *>(config);
    for (int32_t i = 0; i < 8; ++i) {
        out[i] = values[i];
    }
}
