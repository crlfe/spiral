#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace spiral {

// One matmul call.  Dimensions use the model vocabulary, not M/N/K.
struct MatmulShape {
  int32_t format = 2;        // ggml_type; 2 = Q4_0, 14 = Q6_K
  int32_t weights_rows = 0;  // multiple of 256
  int32_t weights_cols = 0;  // multiple of 512
  int32_t inputs_vecs = 4;   // must match the fixed design
  int32_t tile_bytes = 0;    // diagnostic modes only; ignored for quant formats
  // Tanh softcap parameter: cap * tanh(x / cap).  Used only when epilogue is
  // kEpilogueSoftcap; it must be zero for the other kinds.
  float soft_cap = 0.0f;
  // Tile epilogue selector.  The kind is exclusive; modifier flags will get
  // their own field when the kernel has the operands for them.
  //   kEpilogueNone    plain bf16
  //   kEpilogueSoftcap cap * tanh(x / cap), using soft_cap
  //   kEpilogueGeglu   paired gate/up bands, gelu(gate) * up; requires a
  //                    two-segment prepare with equal widths, stores the gate
  //                    half at output offset 0
  int32_t epilogue = 0;
};

enum : int32_t {
  kEpilogueNone = 0,
  kEpilogueSoftcap = 1,
  kEpilogueGeglu = 2,
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
  double kernel_us = 0;   // start + wait + output sync
  double start_us = 0;    // argument setup + run.start
  double wait_us = 0;     // waiting for the device
  double sync_us = 0;     // output BO sync
  double unpack_us = 0;   // device output -> logical output
  double total_us = 0;
};

// One submitted call that has been started but not waited on.
//
// Submit several tickets before finishing the first to keep the device queue
// full and hide launch latency.  A ticket owns its timing record; read it
// after finish().
struct RunTicket {
  int32_t slot = -1;
  int32_t cols_bands = 0;
  uint16_t* outputs = nullptr;
  int32_t outputs_cols = 0;
  int32_t outputs_col_offset = 0;
  RunTimings timings;
  double start_time = 0.0;
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

// One operation in a chained transaction.  The PreparedWeights handle must
// outlive the chain.
struct ChainOp {
  MatmulShape shape;
  const PreparedWeights* weights = nullptr;
};

// A prepared multi-operation transaction.  Operations run back to back in one
// dispatch; each operation's input can read the previous output in place.  The
// chain copies the prepared weight streams, so the per-operation handles need
// not outlive it.
class PreparedChain {
 public:
  PreparedChain();
  ~PreparedChain();

  PreparedChain(PreparedChain&&) noexcept;
  PreparedChain& operator=(PreparedChain&&) noexcept;
  PreparedChain(const PreparedChain&) = delete;
  PreparedChain& operator=(const PreparedChain&) = delete;

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
  // outputs is bfloat16 in [inputs_vecs][outputs_cols] order; the result for
  // this call starts at column `outputs_col_offset`.  The layout matches the
  // next run's input, so the buffer can be chained without conversion.
  void run(const MatmulShape& shape, const PreparedWeights& weights,
           const uint16_t* inputs, uint16_t* outputs, int32_t outputs_cols,
           int32_t outputs_col_offset, RunTimings* timings = nullptr);

  // Number of calls that can be in flight at once (>= 1).
  int32_t pipeline_slots() const;

  // High-water mark of submitted-but-not-finished calls (diagnostics).
  int32_t max_outstanding() const;

  // Stage, start, and return a call without waiting.  At most
  // pipeline_slots() tickets may be outstanding.  `inputs` may be reused as
  // soon as submit() returns; `outputs` is written by finish().
  RunTicket submit(const MatmulShape& shape, const PreparedWeights& weights,
                   const uint16_t* inputs, uint16_t* outputs,
                   int32_t outputs_cols, int32_t outputs_col_offset);

  // Wait for a submitted call, unpack it, and release its slot.
  void finish(RunTicket& ticket);

  // Prepare one TXN with `count` operations.  Later operations read the
  // previous output buffer in place; the final operation's output is the
  // chain result.  The weight streams are copied into the chain's BO.
  PreparedChain prepare_chain(const ChainOp* ops, int32_t count);

  // Run a prepared chain synchronously and unpack the final operation's
  // output.  `inputs` feeds the first operation.
  void run_chain(const PreparedChain& chain, const uint16_t* inputs,
                 uint16_t* outputs, int32_t outputs_cols,
                 int32_t outputs_col_offset, RunTimings* timings = nullptr);

  // One-shot convenience: prepare, run, and discard the repacked weights.
  //
  // raw_weights points at canonical quant bytes with `raw_stride` bytes per
  // weights column.  `weights_col_offset` selects the first 512-column band
  // of the chunk.
  void run(const MatmulShape& shape, const uint8_t* raw_weights,
           int64_t raw_stride, int32_t weights_col_offset,
           const uint16_t* inputs, uint16_t* outputs, int32_t outputs_cols,
           int32_t outputs_col_offset, RunTimings* timings = nullptr);

 private:
  PreparedWeights prepare_packed(const MatmulShape& shape,
                                 std::vector<uint8_t> packed,
                                 int32_t segment_count = 1);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spiral
