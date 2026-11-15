// Shared helpers for the 256x16 streaming PoC kernels.
//
// This header is part of the @iron.jit source_files list, so editing it
// invalidates the cached design automatically.

#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

static inline void stream_copy_bytes(
        uint8_t * restrict dst,
        const uint8_t * restrict src,
        int32_t n)
{
#ifdef STREAM_SKIP_COPY
    (void)dst;
    (void)src;
    (void)n;
#else
    // All weight tiles and stream blocks are 64-byte aligned.
    int32_t i = 0;
    for (; i + 64 <= n; i += 64) {
        aie::store_v(
            reinterpret_cast<uint32_t *>(dst + i),
            aie::load_v<16>(
                reinterpret_cast<const uint32_t *>(src + i)));
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
#endif
}
