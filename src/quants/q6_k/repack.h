#pragma once

#include <stdint.h>
#include <string.h>

#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

constexpr int64_t record_count(int64_t weights_rows, int64_t weights_cols) {
  if (weights_rows <= 0 || weights_cols <= 0) {
    return 0;
  }
  if (weights_rows % kRecordRows != 0 || weights_cols % kTileCols != 0) {
    return 0;
  }
  return (weights_rows / kRecordRows) * (weights_cols / kTileCols);
}

constexpr int64_t packed_bytes(int64_t weights_rows, int64_t weights_cols) {
  return record_count(weights_rows, weights_cols) * kPackedRecordBytes;
}

// No-op for this pass: validate the shape, zero-fill the destination, and
// report the packed size so the test runner fails on accuracy, not setup.
inline int64_t repack_weights(const uint8_t* src,
                              int64_t src_col_stride_bytes,
                              uint8_t* dst,
                              int64_t dst_capacity,
                              int64_t weights_rows,
                              int64_t weights_cols) {
  (void)src;
  (void)src_col_stride_bytes;
  const int64_t records = record_count(weights_rows, weights_cols);
  const int64_t bytes = records * kPackedRecordBytes;
  if (records == 0 || bytes > dst_capacity || dst == nullptr) {
    return 0;
  }
  memset(dst, 0, static_cast<size_t>(bytes));
  return bytes;
}

}  // namespace spiral::quants::q6_k
