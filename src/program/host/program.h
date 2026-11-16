#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace spiral {

// One matmul call.  Dimensions use the model vocabulary, not M/N/K.
struct MatmulShape {
  int32_t format = 1;        // 0 = probe, 1 = Q4_0, 2 = Q6_K
  int32_t weights_rows = 0;  // multiple of 256
  int32_t weights_cols = 0;  // multiple of 512
  int32_t inputs_vecs = 4;   // must match the fixed design
  int32_t tile_bytes = 0;    // format 0 only; ignored for 1/2
};

// One weight block used to build a wider prepared tensor.  The segments are
// concatenated along weights_cols, which lets one call compute parallel column
// blocks such as FFN gate + up or Q + K + V.
struct WeightSegment {
  const uint8_t* raw_weights = nullptr;
  int64_t raw_stride = 0;
  int32_t weights_col_offset = 0;
  int32_t weights_cols = 0;
};

// Optional per-call timing breakdown, in microseconds.
struct RunTimings {
  double prepare_us = 0;  // repack + host stream build + txn generation
  double upload_us = 0;   // weight host -> BO
  double txn_us = 0;      // instruction BO write
  double input_us = 0;    // input packing + BO sync
  double kernel_us = 0;   // start + wait
  double unpack_us = 0;   // device output -> logical output
  double total_us = 0;
};

// Host-side weights prepared for repeated calls.
//
// Preparing repacks the canonical quant tensor and builds the block-major
// stream once.  Keep one object per weight tensor (or per head chunk) and
// reuse it across token steps.
class PreparedWeights {
 public:
  PreparedWeights();
  ~PreparedWeights();

  PreparedWeights(PreparedWeights&&) noexcept;
  PreparedWeights& operator=(PreparedWeights&&) noexcept;
  PreparedWeights(const PreparedWeights&) = delete;
  PreparedWeights& operator=(const PreparedWeights&) = delete;

 private:
  friend class Program;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A loaded Spiral NPU program.
//
// The device design, instruction stream, and buffer objects are created once.
// Each run() call packs its inputs, executes the matmul, and unpacks the
// result into a logical output matrix.
class Program {
 public:
  Program();
  ~Program();

  Program(Program&&) noexcept;
  Program& operator=(Program&&) noexcept;
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  // Repack and stage one weight tensor (or one head chunk) for reuse.
  PreparedWeights prepare(const MatmulShape& shape,
                          const uint8_t* raw_weights, int64_t raw_stride,
                          int32_t weights_col_offset);

  // Repack several weight tensors and concatenate them along weights_cols.
  // `shape.weights_cols` must equal the sum of the segment column counts.
  // Useful for FFN gate+up or Q+K+V in one launch.
  PreparedWeights prepare(const MatmulShape& shape,
                          const WeightSegment* segments,
                          int32_t segment_count);

  // Run one matmul with already prepared weights.
  //
  // Inputs are bfloat16 bit patterns in [inputs_vecs][weights_rows] order.
  // outputs is a float matrix in [inputs_vecs][outputs_cols] order; the result
  // for this call starts at column `outputs_col_offset`.
  void run(const MatmulShape& shape, const PreparedWeights& weights,
           const uint16_t* inputs, float* outputs, int32_t outputs_cols,
           int32_t outputs_col_offset, RunTimings* timings = nullptr);

  // One-shot convenience: prepare, run, and discard the repacked weights.
  //
  // raw_weights points at canonical quant bytes with `raw_stride` bytes per
  // weights column.  `weights_col_offset` selects the first 512-column band
  // of the chunk.
  void run(const MatmulShape& shape, const uint8_t* raw_weights,
           int64_t raw_stride, int32_t weights_col_offset,
           const uint16_t* inputs, float* outputs, int32_t outputs_cols,
           int32_t outputs_col_offset, RunTimings* timings = nullptr);

 private:
  PreparedWeights prepare_packed(const MatmulShape& shape,
                                 std::vector<uint8_t> packed);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spiral
