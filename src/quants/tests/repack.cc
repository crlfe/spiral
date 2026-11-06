#ifndef QUANTS_FORMAT
#error "QUANTS_FORMAT must be defined"
#endif

#define QUANTS_STRINGIFY_(x) #x
#define QUANTS_STRINGIFY(x) QUANTS_STRINGIFY_(x)

#include <stdint.h>

#include QUANTS_STRINGIFY(quants/QUANTS_FORMAT/repack.h)

extern "C" {
int64_t quants_record_count(int64_t weights_rows, int64_t weights_cols) {
  return spiral::quants::QUANTS_FORMAT::record_count(weights_rows,
                                                      weights_cols);
}

int64_t quants_packed_bytes(int64_t weights_rows, int64_t weights_cols) {
  return spiral::quants::QUANTS_FORMAT::packed_bytes(weights_rows,
                                                      weights_cols);
}

int64_t quants_repack(const uint8_t* src,
                      int64_t src_col_stride_bytes,
                      uint8_t* dst,
                      int64_t dst_capacity,
                      int64_t weights_rows,
                      int64_t weights_cols) {
  return spiral::quants::QUANTS_FORMAT::repack_weights(
      src, src_col_stride_bytes, dst, dst_capacity, weights_rows,
      weights_cols);
}
}
