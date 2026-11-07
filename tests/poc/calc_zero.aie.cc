#include <aie_api/aie.hpp>

constexpr int VECS = 4;
constexpr int COLS = 64;

extern "C" void calc_zero(float * restrict accum) {
    for (int i = 0; i < VECS * COLS; i++) {
        accum[i] = 0.0f;
    }
}
