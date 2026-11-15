// Drain one L1 weight block: multiply complete tiles in place from the
// transport buffer, copying only a straddling tile's prefix into the carry
// buffer.

#include "stream_common.h"

using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

constexpr int32_t ROW_GROUPS = 32;

static inline void mmul_tile(
        const bfloat16 * restrict weights,
        const bfloat16 * restrict all_inputs,
        float * restrict all_outputs,
        int32_t band,
        int32_t row_step,
        int32_t vec_steps,
        int32_t tile_vectors)
{
#ifdef STREAM_SKIP_MMUL
    (void)weights;
    (void)all_inputs;
    (void)all_outputs;
    (void)band;
    (void)row_step;
    (void)vec_steps;
    (void)tile_vectors;
    return;
#endif
    for (int32_t vec = 0; vec < vec_steps; ++vec) {
        const bfloat16 * inputs =
            all_inputs
            + (row_step * vec_steps + vec) * STREAM_INPUT_ELEMS;
        float * tile =
            all_outputs + (band * vec_steps + vec) * 64;

        MMUL accumulator0(aie::load_v<32>(tile));
        MMUL accumulator1(aie::load_v<32>(tile + 32));
        int32_t vector0 = 0;
        int32_t vector1 = 1;
        if (tile_vectors == 1) {
            vector1 = 0;
        }

        for (int32_t row_tile = 0; row_tile < ROW_GROUPS; ++row_tile) {
            const aie::vector<bfloat16, 32> inputs_tile =
                aie::load_v<32>(inputs + row_tile * 32);
            accumulator0.mac(
                inputs_tile,
                aie::load_v<64>(weights + vector0 * 64));
            accumulator1.mac(
                inputs_tile,
                aie::load_v<64>(weights + vector1 * 64));

            vector0 += 2;
            if (vector0 >= tile_vectors) {
                vector0 -= tile_vectors;
            }
            vector1 += 2;
            if (vector1 >= tile_vectors) {
                vector1 -= tile_vectors;
            }
        }

        aie::store_v(tile, accumulator0.template to_vector<float>());
        aie::store_v(
            tile + 32, accumulator1.template to_vector<float>());
    }
}

extern "C" void stream_fused(
        const uint8_t * restrict block,
        uint8_t * restrict carry,
        int32_t * restrict state,
        const bfloat16 * restrict all_inputs,
        float * restrict all_outputs,
        int32_t tile_bytes,
        int32_t tile_vectors,
        int32_t row_steps,
        int32_t col_steps,
        int32_t vec_steps)
{
    // Each call receives a fresh L1 block, so the block position is local.
    int32_t pos = 0;
    int32_t carry_len = state[0];
    int32_t tile_index = state[1];
    int32_t band = state[2];
    int32_t row_step = state[3];
    const int32_t total_tiles = row_steps * col_steps;

    while (tile_index < total_tiles) {
        if (carry_len == 0 && pos + tile_bytes <= STREAM_BLOCK_BYTES) {
            // Complete tile already sits in the transport buffer.
            mmul_tile(
                reinterpret_cast<const bfloat16 *>(block + pos),
                all_inputs, all_outputs, band, row_step, vec_steps,
                tile_vectors);
            pos += tile_bytes;
            ++tile_index;
            ++row_step;
            if (row_step == row_steps) {
                row_step = 0;
                ++band;
            }
        } else {
            const int32_t need = tile_bytes - carry_len;
            const int32_t avail = STREAM_BLOCK_BYTES - pos;
            const int32_t n = need < avail ? need : avail;
            stream_copy_bytes(carry + carry_len, block + pos, n);
            pos += n;
            carry_len += n;
            if (carry_len == tile_bytes) {
                mmul_tile(
                    reinterpret_cast<const bfloat16 *>(carry),
                    all_inputs, all_outputs, band, row_step, vec_steps,
                    tile_vectors);
                carry_len = 0;
                ++tile_index;
                ++row_step;
                if (row_step == row_steps) {
                    row_step = 0;
                    ++band;
                }
            }
        }

        if (pos == STREAM_BLOCK_BYTES) {
            break;
        }
    }

    state[0] = carry_len;
    state[1] = tile_index;
    state[2] = band;
    state[3] = row_step;
}
