#ifndef QUANTS_FORMAT
#error "QUANTS_FORMAT must be defined"
#endif
#ifndef QUANTS_WEIGHTS_ROWS
#error "QUANTS_WEIGHTS_ROWS must be defined"
#endif
#ifndef QUANTS_WEIGHTS_COLS
#error "QUANTS_WEIGHTS_COLS must be defined"
#endif

#define QUANTS_STRINGIFY_(x) #x
#define QUANTS_STRINGIFY(x) QUANTS_STRINGIFY_(x)

#include <stdint.h>

// clang-format off
#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/mmul.aie.h)
// clang-format on

extern "C" void quants_unpack_mmul_all(const uint8_t* packed,
                                       const bfloat16* inputs,
                                       float* outputs) {
  using namespace spiral::quants::QUANTS_FORMAT;
  constexpr int kOutputElems =
      (QUANTS_WEIGHTS_COLS / kTileCols) * spiral::quants::kMmulOutputsElems;
  for (int i = 0; i < kOutputElems; ++i) {
    outputs[i] = 0.0f;
  }

  aie::scoped_mode rounding{aie::rounding_mode::conv_even};
  event0();
  unpack_mmul<QUANTS_WEIGHTS_ROWS, QUANTS_WEIGHTS_COLS>(packed, inputs,
                                                        outputs);
  event1();
}
