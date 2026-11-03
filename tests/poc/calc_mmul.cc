#include <aie_api/aie.hpp>

constexpr int ROWS = 64;
constexpr int COLS = 64;
constexpr int ROW_TILES = ROWS / 8;
constexpr int COL_TILES = COLS / 8;

using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

extern "C" void calc_mmul(
    const bfloat16 * restrict weights,
    const float * restrict inputs,
    float * restrict accum
) {
    // Inputs arrive as (rows/8, vecs, 8) f32; convert them once into the bf16
    // layout the mmul expects.
    alignas(64) bfloat16 a_bf[ROW_TILES * 32];
    for (int row_tile = 0; row_tile < ROW_TILES; row_tile++) {
        aie::vector<float, 32> af = aie::load_v<32>(inputs + row_tile * 32);
        aie::accum<accfloat, 32> acc(af);
        aie::store_v(a_bf + row_tile * 32, acc.template to_vector<bfloat16>());
    }

    // Weights arrive as (rows/8, cols/8, 8, 8) bf16; the accumulator is
    // (cols/8, vecs, 8) f32.
    for (int col_tile = 0; col_tile < COL_TILES; col_tile++) {
        MMUL c(aie::load_v<32>(accum + col_tile * 32));
        for (int row_tile = 0; row_tile < ROW_TILES; row_tile++) {
            c.mac(
                aie::load_v<32>(a_bf + row_tile * 32),
                aie::load_v<64>(
                    weights + (row_tile * COL_TILES + col_tile) * 64));
        }
        aie::store_v(accum + col_tile * 32, c.template to_vector<float>());
    }
}
