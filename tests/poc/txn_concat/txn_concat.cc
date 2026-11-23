// Prototype: concatenate two fixed-program matmuls into one NPU command and
// chain the first output directly into the second input.
//
// Build/run through run.py.  The executable is started in the production
// build directory so "aie.xclbin" and "golden/" resolve.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "design_meta.h"
#include "generate_txn.h"
#include "program/host/program.h"
#include "quants/q4_0/repack.h"
#include "quants/q6_k/repack.h"
#include "txn_concat.h"

namespace {

constexpr uint32_t kRunOpcode = 3;
constexpr int kRunTimeoutMs = 60000;
constexpr int32_t kInputParts = 4;
constexpr int32_t kWeightParts = 2;
constexpr size_t kInputObjectBytes =
    design::kInputGroupElems * sizeof(uint16_t);
constexpr size_t kConfigObjectBytes = kInputObjectBytes;

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}

struct Golden {
  int32_t format = 0;
  int32_t rows_steps = 0;
  int32_t cols_bands = 0;
  int32_t weights_rows = 0;
  int32_t weights_cols = 0;
  int32_t raw_stride = 0;
  size_t raw_bytes = 0;
  size_t inputs_elems = 0;
  std::vector<uint8_t> raw;
  std::vector<uint16_t> inputs;
};

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                              std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> read_words(const std::string& path, size_t count) {
  auto bytes = read_file(path);
  if (bytes.size() < count * sizeof(T)) {
    throw std::runtime_error("short file: " + path);
  }
  std::vector<T> out(count);
  std::memcpy(out.data(), bytes.data(), count * sizeof(T));
  return out;
}

Golden load_golden(const std::string& dir, const std::string& name) {
  const std::string prefix = dir + "/" + name;
  std::ifstream meta(prefix + ".meta");
  if (!meta) {
    throw std::runtime_error("cannot open " + prefix + ".meta");
  }
  std::map<std::string, long long> values;
  std::string key;
  long long value;
  while (meta >> key >> value) {
    values[key] = value;
  }

  Golden golden;
  golden.format = static_cast<int32_t>(values.at("format"));
  golden.rows_steps = static_cast<int32_t>(values.at("rows_steps"));
  golden.cols_bands = static_cast<int32_t>(values.at("cols_bands"));
  golden.weights_rows = static_cast<int32_t>(values.at("weights_rows"));
  golden.weights_cols = static_cast<int32_t>(values.at("weights_cols"));
  golden.raw_stride = static_cast<int32_t>(values.at("raw_stride"));
  golden.raw_bytes = static_cast<size_t>(values.at("raw_bytes"));
  golden.inputs_elems = static_cast<size_t>(values.at("inputs_elems"));
  golden.raw = read_file(prefix + ".raw");
  golden.inputs = read_words<uint16_t>(prefix + ".inputs", golden.inputs_elems);
  if (golden.raw.size() != golden.raw_bytes) {
    throw std::runtime_error("raw size mismatch: " + name);
  }
  return golden;
}

void split_parts(int32_t total, int32_t parts, int32_t* out, int32_t& padded) {
  padded = std::max(total, parts);
  const int32_t base = padded / parts;
  const int32_t extra = padded % parts;
  for (int32_t i = 0; i < parts; ++i) {
    out[i] = base + (i < extra ? 1 : 0);
  }
}

void write_config(uint32_t* buf,
                  int32_t format,
                  int32_t tile_bytes,
                  int32_t rows_steps,
                  int32_t cols_bands,
                  int32_t weights_blocks,
                  int32_t total_tiles,
                  int32_t input_pad,
                  int32_t weight_pad,
                  int32_t epilogue = design::kEpilogueNone) {
  std::memset(buf, 0, design::kConfigLen * sizeof(uint32_t));
  buf[design::kCfgFormat] = static_cast<uint32_t>(format);
  buf[design::kCfgTileBytes] = static_cast<uint32_t>(tile_bytes);
  buf[design::kCfgRowsSteps] = static_cast<uint32_t>(rows_steps);
  buf[design::kCfgColsBands] = static_cast<uint32_t>(cols_bands);
  buf[design::kCfgWeightsBlocks] = static_cast<uint32_t>(weights_blocks);
  buf[design::kCfgTotalTiles] = static_cast<uint32_t>(total_tiles);
  buf[design::kCfgInputPad] = static_cast<uint32_t>(input_pad);
  buf[design::kCfgWeightPad] = static_cast<uint32_t>(weight_pad);
  buf[design::kCfgEpilogue] = static_cast<uint32_t>(epilogue);
}

// Host packing helpers mirror src/program/host/program.cc; keep them in
// sync when the production layout changes.

// Copy repacked transport tiles into the block-major host layout:
// [group][block][compute][L1].
void fill_weights(uint8_t* dst,
                  const uint8_t* packed,
                  int32_t rows_steps,
                  int32_t cols_bands,
                  int32_t tile_bytes,
                  int32_t group_bytes) {
  const int64_t cols_pairs =
      static_cast<int64_t>(cols_bands) * design::kBandCols / design::kColsTile;
  const int32_t tiles_per_band = design::kBandCols / design::kColsTile;
  for (int32_t g = 0; g < design::kGroups; ++g) {
    for (int32_t c = 0; c < design::kComputesPerGroup; ++c) {
      for (int32_t row = 0; row < rows_steps; ++row) {
        for (int32_t band = 0; band < cols_bands; ++band) {
          const int64_t global_tile = static_cast<int64_t>(row) * cols_pairs +
                                      band * tiles_per_band +
                                      g * design::kComputesPerGroup + c;
          const uint8_t* tile = packed + global_tile * tile_bytes;
          const int64_t stream_offset =
              static_cast<int64_t>(row * cols_bands + band) * tile_bytes;
          for (int64_t copied = 0; copied < tile_bytes;) {
            const int64_t block =
                (stream_offset + copied) / design::kL1StreamBytes;
            const int64_t within =
                (stream_offset + copied) % design::kL1StreamBytes;
            const int64_t count = std::min<int64_t>(
                tile_bytes - copied,
                static_cast<int64_t>(design::kL1StreamBytes) - within);
            uint8_t* out = dst + static_cast<int64_t>(g) * group_bytes +
                           block * design::kL2StreamBytes +
                           c * design::kL1StreamBytes + within;
            std::memcpy(out, tile + copied, count);
            copied += count;
          }
        }
      }
    }
  }
}

// Pack [inputs_vecs][weights_rows] bfloat16 bits into one object per row group.
void fill_inputs(uint32_t* buf,
                 int32_t rows_steps,
                 int32_t weights_rows,
                 const uint16_t* inputs) {
  auto* objects = reinterpret_cast<uint16_t*>(buf);
  constexpr int32_t kColsPerChunk = 8;
  for (int32_t row = 0; row < rows_steps; ++row) {
    const int32_t group = row / design::kInputGroupRows;
    const int32_t within = row % design::kInputGroupRows;
    uint16_t* dst = objects + group * design::kInputGroupElems +
                    within * design::kInputElems;
    for (int32_t chunk = 0; chunk < design::kRowsStep / kColsPerChunk;
         ++chunk) {
      for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
        const int64_t src = static_cast<int64_t>(vec) * weights_rows +
                            row * design::kRowsStep + chunk * kColsPerChunk;
        uint16_t* chunk_dst = dst +
                              chunk * design::kInputsVecs * kColsPerChunk +
                              vec * kColsPerChunk;
        std::memcpy(chunk_dst, inputs + src, kColsPerChunk * sizeof(uint16_t));
      }
    }
  }
}

// Convert the device [N/8][vec][8] chain layout to [vec][cols].
void unpack_outputs(const uint16_t* raw,
                    uint16_t* outputs,
                    int32_t cols_bands,
                    int32_t outputs_cols,
                    int32_t outputs_col_offset) {
  constexpr int32_t kHalfCols = design::kColsTile / 2;
  constexpr int32_t kBlockElems = design::kInputsVecs * kHalfCols;
  constexpr int32_t kBlocksPerBand = design::kBandCols / kHalfCols;
  constexpr int32_t kBlocksPerCompute = design::kColsTile / kHalfCols;
  constexpr int32_t kBlocksPerGroup =
      design::kComputesPerGroup * kBlocksPerCompute;
  for (int32_t g = 0; g < design::kGroups; ++g) {
    for (int32_t c = 0; c < design::kComputesPerGroup; ++c) {
      for (int32_t band = 0; band < cols_bands; ++band) {
        const int32_t col_base =
            band * design::kBandCols +
            (g * design::kComputesPerGroup + c) * design::kColsTile;
        for (int32_t half = 0; half < 2; ++half) {
          const int32_t block = band * kBlocksPerBand + g * kBlocksPerGroup +
                                c * kBlocksPerCompute + half;
          const int32_t raw_base = block * kBlockElems;
          const int32_t half_base = half * kHalfCols;
          for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
            const int64_t out_index = static_cast<int64_t>(vec) * outputs_cols +
                                      outputs_col_offset + col_base + half_base;
            std::memcpy(outputs + out_index, raw + raw_base + vec * kHalfCols,
                        static_cast<size_t>(kHalfCols) * sizeof(uint16_t));
          }
        }
      }
    }
  }
}

float bf16_to_float(uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// Round-to-nearest-even, matching the device epilogue's conv_even.
uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0xFFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float gelu_f32(float x) {
  constexpr float kC = 0.7978845608028654f;
  constexpr float kA = 0.044715f;
  return 0.5f * x * (1.0f + std::tanh(kC * x * (1.0f + kA * x * x)));
}

// Host reference GEGLU for a [vecs][2*ffn_cols] gate+up matrix.
std::vector<uint16_t> host_geglu(const std::vector<uint16_t>& gate_up,
                                 int32_t ffn_cols) {
  const int32_t vecs = design::kInputsVecs;
  const int32_t stride = 2 * ffn_cols;
  std::vector<uint16_t> out(
      static_cast<size_t>(vecs) * static_cast<size_t>(ffn_cols), 0);
  for (int32_t vec = 0; vec < vecs; ++vec) {
    for (int32_t col = 0; col < ffn_cols; ++col) {
      const float gate = bf16_to_float(gate_up[vec * stride + col]);
      const float up = bf16_to_float(gate_up[vec * stride + ffn_cols + col]);
      out[vec * ffn_cols + col] = float_to_bf16(gelu_f32(gate) * up);
    }
  }
  return out;
}

struct Op {
  Golden golden;
  int32_t epilogue = design::kEpilogueNone;
  int32_t rows_steps = 0;
  int32_t cols_bands = 0;
  int32_t total_tiles = 0;
  int32_t weights_blocks = 0;
  int32_t input_parts[kInputParts] = {};
  int32_t weight_parts[kWeightParts] = {};
  int32_t input_pad = 0;
  int32_t weight_pad = 0;
  int32_t group_stride_words = 0;
  size_t group_bytes = 0;
  size_t config_offset = 0;
  size_t weight_offset = 0;
  std::vector<uint32_t> base_txn;
  std::vector<uint8_t> weight_stream;
  std::vector<uint8_t> config;
};

Op prepare_op(const Golden& golden,
              int32_t epilogue = design::kEpilogueNone) {
  const bool is_q4 = golden.format == design::kFormatQ4_0;
  const bool is_q6 = golden.format == design::kFormatQ6K;
  if (!is_q4 && !is_q6) {
    throw std::runtime_error("prototype supports Q4_0 and Q6_K only");
  }
  const int32_t tile_bytes =
      is_q4 ? spiral::quants::q4_0::kPackedTransportTileBytes
            : spiral::quants::q6_k::kPackedTransportTileBytes;
  Op op;
  op.golden = golden;
  op.epilogue = epilogue;
  op.rows_steps = golden.weights_rows / design::kRowsStep;
  op.cols_bands = golden.weights_cols / design::kBandCols;
  op.total_tiles = op.rows_steps * op.cols_bands;
  op.weights_blocks =
      static_cast<int32_t>((static_cast<int64_t>(op.total_tiles) * tile_bytes +
                            design::kL1StreamBytes - 1) /
                           design::kL1StreamBytes);
  if (op.rows_steps != golden.rows_steps ||
      op.cols_bands != golden.cols_bands) {
    throw std::runtime_error("golden geometry does not match the shape");
  }
  if (op.rows_steps > design::kMaxRowsSteps ||
      op.cols_bands > design::kMaxBandsPerChunk ||
      op.weights_blocks > design::kMaxWeightBlocks) {
    throw std::runtime_error("shape exceeds the fixed design limits");
  }
  if (golden.inputs_elems !=
      static_cast<size_t>(design::kInputsVecs) * golden.weights_rows) {
    throw std::runtime_error("golden input size does not match weights_rows");
  }

  const int64_t packed_size =
      is_q4 ? spiral::quants::q4_0::packed_bytes(golden.weights_rows,
                                                 golden.weights_cols)
            : spiral::quants::q6_k::packed_bytes(golden.weights_rows,
                                                 golden.weights_cols);
  std::vector<uint8_t> packed(static_cast<size_t>(packed_size));
  const int64_t written =
      is_q4 ? spiral::quants::q4_0::repack_weights(
                  golden.raw.data(), golden.raw_stride, packed.data(),
                  packed_size, golden.weights_rows, golden.weights_cols)
            : spiral::quants::q6_k::repack_weights(
                  golden.raw.data(), golden.raw_stride, packed.data(),
                  packed_size, golden.weights_rows, golden.weights_cols);
  if (written != packed_size) {
    throw std::runtime_error("weight repack failed");
  }

  op.group_bytes =
      static_cast<size_t>(op.weights_blocks) * design::kL2StreamBytes;
  op.weight_stream.assign(design::kGroups * op.group_bytes, 0);
  fill_weights(op.weight_stream.data(), packed.data(), op.rows_steps,
               op.cols_bands, tile_bytes, static_cast<int32_t>(op.group_bytes));

  const int32_t input_groups =
      (op.rows_steps + design::kInputGroupRows - 1) / design::kInputGroupRows;
  int32_t input_padded = 0;
  int32_t weight_padded = 0;
  split_parts(input_groups, kInputParts, op.input_parts, input_padded);
  split_parts(op.weights_blocks, kWeightParts, op.weight_parts, weight_padded);
  op.input_pad = input_padded - input_groups;
  op.weight_pad = weight_padded - op.weights_blocks;
  op.group_stride_words = op.weights_blocks * (design::kL2StreamBytes / 4);

  auto base = generate_txn_main_sequence(op.input_parts[0], op.input_parts[1],
                                         op.input_parts[2], op.input_parts[3],
                                         op.weight_parts[0], op.weight_parts[1],
                                         op.group_stride_words);
  if (!base.has_value()) {
    throw std::runtime_error("transaction generation failed");
  }
  op.base_txn = std::move(*base);

  op.config.assign(kConfigObjectBytes, 0);
  write_config(reinterpret_cast<uint32_t*>(op.config.data()), golden.format,
               tile_bytes, op.rows_steps, op.cols_bands, op.weights_blocks,
               op.total_tiles, op.input_pad, op.weight_pad, op.epilogue);
  return op;
}

struct LoadedDevice {
  xrt::device device;
  xrt::xclbin xclbin;

  LoadedDevice() : device(0), xclbin(std::string("aie.xclbin")) {
    device.register_xclbin(xclbin);
  }
};

class Runner {
 public:
  Runner(size_t instr_words, size_t config_objects, size_t weight_bytes)
      : loaded_(),
        context_(loaded_.device,
                 loaded_.xclbin.get_uuid(),
                 xrt::hw_context::access_mode::exclusive),
        kernel_(context_, "MLIR_AIE"),
        instr_(loaded_.device,
               instr_words * sizeof(uint32_t),
               xrt::bo::flags::cacheable,
               kernel_.group_id(1)),
        inputs_(loaded_.device,
                design::kInputWords * sizeof(uint32_t),
                xrt::bo::flags::host_only,
                kernel_.group_id(4)),
        configs_(loaded_.device,
                 config_objects * kConfigObjectBytes,
                 xrt::bo::flags::host_only,
                 kernel_.group_id(5)),
        outputs_(loaded_.device,
                 design::kMaxOutputElems * sizeof(uint16_t),
                 xrt::bo::flags::host_only,
                 kernel_.group_id(6)),
        weights_(loaded_.device,
                 weight_bytes,
                 xrt::bo::flags::host_only,
                 kernel_.group_id(3)),
        run_(kernel_) {}

  void upload_weights(const std::vector<uint8_t>& combined) {
    if (combined.size() > weights_.size()) {
      throw std::runtime_error("combined weights exceed the BO");
    }
    weights_.write(combined.data(), combined.size(), 0);
    weights_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void upload_configs(const std::vector<uint8_t>& configs) {
    if (configs.size() > configs_.size()) {
      throw std::runtime_error("configs exceed the BO");
    }
    configs_.write(configs.data(), configs.size(), 0);
    configs_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void load_inputs(const Op& op, const uint16_t* input, int32_t input_objects) {
    fill_inputs(inputs_.map<uint32_t*>(), op.rows_steps, op.golden.weights_rows,
                input);
    inputs_.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                 static_cast<size_t>(input_objects) * kInputObjectBytes, 0);
  }

  double run_txn(const std::vector<uint32_t>& txn, double* kernel_us) {
    if (txn.size() > instr_.size() / sizeof(uint32_t)) {
      throw std::runtime_error("TXN exceeds the instruction BO");
    }
    auto stage = Clock::now();
    instr_.write(txn.data(), txn.size() * sizeof(uint32_t), 0);
    instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    run_.set_arg(0, kRunOpcode);
    run_.set_arg(1, instr_);
    run_.set_arg(2, static_cast<uint32_t>(txn.size()));
    run_.set_arg(3, weights_);
    run_.set_arg(4, inputs_);
    run_.set_arg(5, configs_);
    run_.set_arg(6, outputs_);
    const double write_us = elapsed_us(stage);
    stage = Clock::now();
    run_.start();
    run_.wait2(std::chrono::milliseconds(kRunTimeoutMs));
    outputs_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    *kernel_us = elapsed_us(stage);
    return write_us + *kernel_us;
  }

  void unpack(const Op& op, std::vector<uint16_t>& out) {
    out.assign(
        static_cast<size_t>(design::kInputsVecs) * op.golden.weights_cols, 0);
    unpack_outputs(outputs_.map<const uint16_t*>(), out.data(), op.cols_bands,
                   op.golden.weights_cols, 0);
  }

  // Unpack a prefix of the packed output buffer, used for the GEGLU
  // intermediate check (the gate half is the first cols_bands).
  std::vector<uint16_t> unpack_cols(int32_t cols_bands,
                                    int32_t outputs_cols) {
    std::vector<uint16_t> out(
        static_cast<size_t>(design::kInputsVecs) * outputs_cols, 0);
    unpack_outputs(outputs_.map<const uint16_t*>(), out.data(), cols_bands,
                   outputs_cols, 0);
    return out;
  }

 private:
  LoadedDevice loaded_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  xrt::bo instr_;
  xrt::bo inputs_;
  xrt::bo configs_;
  xrt::bo outputs_;
  xrt::bo weights_;
  xrt::run run_;
};

txn_concat::OpPatch patch_for(const Op& op, bool chained) {
  txn_concat::OpPatch patch;
  patch.config_offset = op.config_offset;
  patch.weight_offset = op.weight_offset;
  patch.output_offset = 0;
  patch.input_from_output = chained;
  return patch;
}

int32_t input_objects(const Op& op) {
  return (op.rows_steps + design::kInputGroupRows - 1) /
             design::kInputGroupRows +
         op.input_pad;
}

struct Result {
  std::vector<uint16_t> outputs;
  double total_us = 0.0;
  double kernel_us = 0.0;
};

// Prebuilt TXNs.  Building them inside the timing loop would charge the host
// for TXN assembly, not just for the dispatch path.
struct Plan {
  std::vector<const Op*> ops;
  std::vector<std::vector<uint32_t>> singles;
  std::vector<uint32_t> chain;
};

Plan make_plan(std::vector<const Op*> ops) {
  Plan plan;
  plan.ops = std::move(ops);
  if (plan.ops.empty()) {
    throw std::invalid_argument("plan is empty");
  }
  std::vector<std::vector<uint32_t>> bases;
  std::vector<txn_concat::OpPatch> patches;
  plan.singles.reserve(plan.ops.size());
  bases.reserve(plan.ops.size());
  patches.reserve(plan.ops.size());
  for (size_t i = 0; i < plan.ops.size(); ++i) {
    const Op& op = *plan.ops[i];
    plan.singles.push_back(
        txn_concat::single(op.base_txn, patch_for(op, false)));
    bases.push_back(op.base_txn);
    patches.push_back(patch_for(op, i > 0));
  }
  plan.chain = txn_concat::concat(bases, patches);
  return plan;
}

Result run_independent(Runner& runner, const Plan& plan, const Op& input_op) {
  Result result;
  runner.load_inputs(*plan.ops.front(), input_op.golden.inputs.data(),
                     input_objects(*plan.ops.front()));
  for (size_t i = 0; i < plan.ops.size(); ++i) {
    const Op& op = *plan.ops[i];
    double kernel = 0.0;
    result.total_us += runner.run_txn(plan.singles[i], &kernel);
    result.kernel_us += kernel;
    if (i + 1 < plan.ops.size()) {
      std::vector<uint16_t> activation;
      runner.unpack(op, activation);
      runner.load_inputs(*plan.ops[i + 1], activation.data(),
                         input_objects(*plan.ops[i + 1]));
    } else {
      runner.unpack(op, result.outputs);
    }
  }
  return result;
}

Result run_chain(Runner& runner, const Plan& plan, const Op& input_op) {
  Result result;
  runner.load_inputs(*plan.ops.front(), input_op.golden.inputs.data(),
                     input_objects(*plan.ops.front()));
  double kernel = 0.0;
  result.total_us = runner.run_txn(plan.chain, &kernel);
  result.kernel_us = kernel;
  runner.unpack(*plan.ops.back(), result.outputs);
  return result;
}

std::vector<const Op*> alternating(const Op& a, const Op& b, int count) {
  std::vector<const Op*> ops;
  ops.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    ops.push_back((i % 2 == 0) ? &a : &b);
  }
  return ops;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void compare(const std::vector<uint16_t>& independent,
             const std::vector<uint16_t>& chain) {
  if (independent.size() != chain.size()) {
    throw std::runtime_error("output sizes differ");
  }
  size_t bit_mismatches = 0;
  double max_abs = 0.0;
  size_t first = 0;
  for (size_t i = 0; i < independent.size(); ++i) {
    if (independent[i] != chain[i]) {
      if (bit_mismatches == 0) {
        first = i;
      }
      ++bit_mismatches;
    }
    const double diff =
        std::fabs(bf16_to_float(independent[i]) - bf16_to_float(chain[i]));
    if (diff > max_abs) {
      max_abs = diff;
    }
  }
  std::printf("chain vs independent: bit_mismatches=%zu max_abs=%.8f",
              bit_mismatches, max_abs);
  if (bit_mismatches != 0) {
    std::printf(" first=%zu ind=%u chain=%u", first,
                static_cast<unsigned>(independent[first]),
                static_cast<unsigned>(chain[first]));
  }
  std::printf("\n");
  if (bit_mismatches != 0) {
    throw std::runtime_error("chained output does not match independent path");
  }
}

// Tolerance comparison for paths that differ by an approximation (the bf16
// vtanh in the GEGLU epilogue).  Reports the global relative error against the
// reference magnitude, which is robust for near-zero elements.
void compare_tol(const std::vector<uint16_t>& reference,
                 const std::vector<uint16_t>& candidate,
                 double tol,
                 const char* label) {
  if (reference.size() != candidate.size()) {
    throw std::runtime_error("output sizes differ");
  }
  size_t bit_mismatches = 0;
  double max_abs = 0.0;
  double max_ref = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    if (reference[i] != candidate[i]) {
      ++bit_mismatches;
    }
    const double ref = bf16_to_float(reference[i]);
    const double got = bf16_to_float(candidate[i]);
    const double diff = std::fabs(ref - got);
    if (diff > max_abs) {
      max_abs = diff;
      worst = i;
    }
    max_ref = std::max(max_ref, std::fabs(ref));
  }
  const double rel = max_abs / std::max(1e-6, max_ref);
  std::printf("%s: bit_mismatches=%zu max_abs=%.6g rel=%.3g worst=%zu "
              "ref=%.6g got=%.6g\n",
              label, bit_mismatches, max_abs, rel, worst,
              bf16_to_float(reference[worst]),
              bf16_to_float(candidate[worst]));
  if (rel > tol) {
    throw std::runtime_error(std::string(label) + " exceeds tolerance");
  }
}

struct Timed {
  double total_us = 0.0;
  double kernel_us = 0.0;
};

Timed time_sequence(Runner& runner,
                    const Plan& plan,
                    const Op& input_op,
                    bool chained,
                    int iterations) {
  for (int i = 0; i < 3; ++i) {
    if (chained) {
      run_chain(runner, plan, input_op);
    } else {
      run_independent(runner, plan, input_op);
    }
  }
  std::vector<double> total;
  std::vector<double> kernel;
  for (int i = 0; i < iterations; ++i) {
    const Result result = chained ? run_chain(runner, plan, input_op)
                                  : run_independent(runner, plan, input_op);
    total.push_back(result.total_us);
    kernel.push_back(result.kernel_us);
  }
  return {median(total), median(kernel)};
}

// GEGLU chain: op0 is the combined gate+up matmul with the GEGLU epilogue,
// op1 is the down matmul reading op0's packed gate half in place.
void run_geglu_case(const std::string& golden_dir, int iterations) {
  constexpr const char* kGateUp = "q4_0_1536x12288";
  constexpr const char* kDown = "q4_0_6144x1536";
  const Golden gate_up_golden = load_golden(golden_dir, kGateUp);
  const Golden down_golden = load_golden(golden_dir, kDown);
  const int32_t ffn_cols = gate_up_golden.weights_cols / 2;
  if (ffn_cols != down_golden.weights_rows) {
    throw std::runtime_error("gate+up half width does not match down K");
  }

  Op raw = prepare_op(gate_up_golden, design::kEpilogueNone);
  Op glu = prepare_op(gate_up_golden, design::kEpilogueGeglu);
  Op down = prepare_op(down_golden, design::kEpilogueNone);
  raw.config_offset = 0;
  glu.config_offset = kConfigObjectBytes;
  down.config_offset = 2 * kConfigObjectBytes;
  raw.weight_offset = 0;
  glu.weight_offset = 0;  // same stream as raw
  down.weight_offset = raw.weight_stream.size();

  std::printf("geglu case: %s (gate+up) -> %s\n", kGateUp, kDown);
  std::printf("  gate+up: %dx%d, rows_steps=%d bands=%d blocks=%d\n",
              gate_up_golden.weights_rows, gate_up_golden.weights_cols,
              raw.rows_steps, raw.cols_bands, raw.weights_blocks);
  std::printf("  down:    %dx%d, rows_steps=%d bands=%d blocks=%d\n",
              down_golden.weights_rows, down_golden.weights_cols,
              down.rows_steps, down.cols_bands, down.weights_blocks);

  std::vector<uint8_t> combined(raw.weight_stream.size() +
                                    down.weight_stream.size() +
                                    design::kL2StreamBytes,
                                0);
  std::memcpy(combined.data(), raw.weight_stream.data(),
              raw.weight_stream.size());
  std::memcpy(combined.data() + down.weight_offset, down.weight_stream.data(),
              down.weight_stream.size());

  std::vector<uint8_t> configs(3 * kConfigObjectBytes, 0);
  std::memcpy(configs.data(), raw.config.data(), kConfigObjectBytes);
  std::memcpy(configs.data() + glu.config_offset, glu.config.data(),
              kConfigObjectBytes);
  std::memcpy(configs.data() + down.config_offset, down.config.data(),
              kConfigObjectBytes);

  Plan independent = make_plan({&raw, &down});
  Plan chained = make_plan({&glu, &down});
  size_t instr_words = 0;
  for (const Plan* plan : {&independent, &chained}) {
    instr_words = std::max(instr_words, plan->chain.size());
    for (const auto& single : plan->singles) {
      instr_words = std::max(instr_words, single.size());
    }
  }

  Runner runner(instr_words, 3, combined.size());
  runner.upload_weights(combined);
  runner.upload_configs(configs);

  // Reference: op0 without the epilogue, host GEGLU, then the down matmul.
  runner.load_inputs(raw, raw.golden.inputs.data(), input_objects(raw));
  double kernel_us = 0.0;
  runner.run_txn(independent.singles[0], &kernel_us);
  const std::vector<uint16_t> raw_output =
      runner.unpack_cols(raw.cols_bands, raw.golden.weights_cols);
  const std::vector<uint16_t> reference_glu = host_geglu(raw_output, ffn_cols);

  runner.load_inputs(down, reference_glu.data(), input_objects(down));
  runner.run_txn(independent.singles[1], &kernel_us);
  const std::vector<uint16_t> independent_output =
      runner.unpack_cols(down.cols_bands, down.golden.weights_cols);

  // Device GEGLU only, for an intermediate check.
  runner.load_inputs(glu, glu.golden.inputs.data(), input_objects(glu));
  runner.run_txn(chained.singles[0], &kernel_us);
  const std::vector<uint16_t> device_glu =
      runner.unpack_cols(glu.cols_bands / 2, ffn_cols);
  compare_tol(reference_glu, device_glu, 2e-2, "geglu intermediate");

  // Full chain.
  runner.load_inputs(glu, glu.golden.inputs.data(), input_objects(glu));
  runner.run_txn(chained.chain, &kernel_us);
  const std::vector<uint16_t> chained_output =
      runner.unpack_cols(down.cols_bands, down.golden.weights_cols);
  compare_tol(independent_output, chained_output, 1e-2, "geglu final");

  auto run_independent_geglu = [&]() {
    double total = 0.0;
    runner.load_inputs(raw, raw.golden.inputs.data(), input_objects(raw));
    total += runner.run_txn(independent.singles[0], &kernel_us);
    const auto raw_out =
        runner.unpack_cols(raw.cols_bands, raw.golden.weights_cols);
    const auto act = host_geglu(raw_out, ffn_cols);
    runner.load_inputs(down, act.data(), input_objects(down));
    total += runner.run_txn(independent.singles[1], &kernel_us);
    runner.unpack_cols(down.cols_bands, down.golden.weights_cols);
    return total;
  };
  auto run_chained_geglu = [&]() {
    runner.load_inputs(glu, glu.golden.inputs.data(), input_objects(glu));
    const double total = runner.run_txn(chained.chain, &kernel_us);
    runner.unpack_cols(down.cols_bands, down.golden.weights_cols);
    return total;
  };

  for (int i = 0; i < 3; ++i) {
    run_independent_geglu();
    run_chained_geglu();
  }
  std::vector<double> independent_times;
  std::vector<double> chained_times;
  for (int i = 0; i < iterations; ++i) {
    independent_times.push_back(run_independent_geglu());
    chained_times.push_back(run_chained_geglu());
  }
  const double ind_us = median(independent_times);
  const double chain_us = median(chained_times);
  std::printf("  total: independent %.1f us chain %.1f us speedup %.2fx\n",
              ind_us, chain_us, ind_us / chain_us);

  // Isolate the epilogue: same gate+up weights and input, none vs GEGLU.
  // Interleave the two modes so clock/thermal drift cancels from the delta.
  auto run_gate_up = [&](const Plan& plan, const Op& op) {
    runner.load_inputs(op, op.golden.inputs.data(), input_objects(op));
    return runner.run_txn(plan.singles[0], &kernel_us);
  };
  for (int i = 0; i < 3; ++i) {
    run_gate_up(independent, raw);
    run_gate_up(chained, glu);
  }
  std::vector<double> deltas;
  std::vector<double> plain_times;
  std::vector<double> geglu_times;
  for (int i = 0; i < iterations; ++i) {
    const double plain = run_gate_up(independent, raw);
    const double geglu = run_gate_up(chained, glu);
    plain_times.push_back(plain);
    geglu_times.push_back(geglu);
    deltas.push_back(geglu - plain);
  }
  std::printf(
      "  epilogue: plain %.1f us geglu %.1f us delta %.1f us (interleaved)\n",
      median(plain_times), median(geglu_times), median(deltas));
}

// Exercise the production Program path: prepare the two gate/up halves as one
// two-segment handle, run with the GEGLU epilogue, and compare against the host
// GEGLU of the plain run.  This covers the config word, the half-width output
// check, and the unpack.
void run_production_geglu_case(const std::string& golden_dir) {
  const Golden golden = load_golden(golden_dir, "q4_0_1536x12288");
  const int32_t half = golden.weights_cols / 2;
  spiral::WeightSegment segments[2] = {
      {golden.raw.data(), golden.raw_stride, 0, half},
      {golden.raw.data(), golden.raw_stride, half, half},
  };

  spiral::Program program;
  spiral::MatmulShape shape;
  shape.format = golden.format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;

  spiral::MatmulShape plain_shape = shape;
  plain_shape.epilogue = spiral::kEpilogueNone;
  spiral::PreparedWeights plain_weights =
      program.prepare(plain_shape, segments, 2);

  spiral::MatmulShape geglu_shape = shape;
  geglu_shape.epilogue = spiral::kEpilogueGeglu;
  spiral::PreparedWeights geglu_weights =
      program.prepare(geglu_shape, segments, 2);

  std::vector<uint16_t> plain_out(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols);
  std::vector<uint16_t> geglu_out(
      static_cast<size_t>(design::kInputsVecs) * half);
  program.run(plain_shape, plain_weights, golden.inputs.data(),
              plain_out.data(), golden.weights_cols, 0);
  program.run(geglu_shape, geglu_weights, golden.inputs.data(),
              geglu_out.data(), half, 0);

  const std::vector<uint16_t> reference = host_geglu(plain_out, half);
  compare_tol(reference, geglu_out, 2e-2, "production geglu");
}

void run_production_chain_case(const std::string& golden_dir) {
  const Golden gate_up = load_golden(golden_dir, "q4_0_1536x12288");
  const Golden down_golden = load_golden(golden_dir, "q4_0_6144x1536");
  const int32_t gate_cols = gate_up.weights_cols / 2;
  if (gate_cols != down_golden.weights_rows) {
    throw std::runtime_error("gate+up half width does not match down K");
  }
  spiral::WeightSegment segments[2] = {
      {gate_up.raw.data(), gate_up.raw_stride, 0, gate_cols},
      {gate_up.raw.data(), gate_up.raw_stride, gate_cols, gate_cols},
  };

  spiral::Program program;
  spiral::MatmulShape gate_shape;
  gate_shape.format = gate_up.format;
  gate_shape.weights_rows = gate_up.weights_rows;
  gate_shape.weights_cols = gate_up.weights_cols;
  gate_shape.inputs_vecs = design::kInputsVecs;

  spiral::MatmulShape down_shape;
  down_shape.format = down_golden.format;
  down_shape.weights_rows = down_golden.weights_rows;
  down_shape.weights_cols = down_golden.weights_cols;
  down_shape.inputs_vecs = design::kInputsVecs;

  spiral::PreparedWeights gate_weights =
      program.prepare(gate_shape, segments, 2);
  spiral::PreparedWeights down_weights =
      program.prepare(down_shape, down_golden.raw.data(),
                      down_golden.raw_stride, 0);

  // Reference: plain gate+up, host GEGLU, then down.
  std::vector<uint16_t> plain_out(
      static_cast<size_t>(design::kInputsVecs) * gate_up.weights_cols);
  program.run(gate_shape, gate_weights, gate_up.inputs.data(),
              plain_out.data(), gate_up.weights_cols, 0);
  const std::vector<uint16_t> reference_glu = host_geglu(plain_out, gate_cols);
  std::vector<uint16_t> reference_out(
      static_cast<size_t>(design::kInputsVecs) * down_golden.weights_cols);
  program.run(down_shape, down_weights, reference_glu.data(),
              reference_out.data(), down_golden.weights_cols, 0);

  // One TXN: gate+up (GEGLU) -> down, both in place.
  spiral::ChainOp ops[2];
  ops[0].shape = gate_shape;
  ops[0].shape.epilogue = spiral::kEpilogueGeglu;
  ops[0].weights = &gate_weights;
  ops[1].shape = down_shape;
  ops[1].shape.epilogue = spiral::kEpilogueNone;
  ops[1].weights = &down_weights;
  spiral::PreparedChain chain = program.prepare_chain(ops, 2);

  std::vector<uint16_t> chain_out(
      static_cast<size_t>(design::kInputsVecs) * down_golden.weights_cols);
  for (int i = 0; i < 3; ++i) {
    program.run_chain(chain, gate_up.inputs.data(), chain_out.data(),
                      down_golden.weights_cols, 0, nullptr);
  }
  std::vector<double> totals;
  std::vector<double> kernels;
  for (int i = 0; i < 20; ++i) {
    spiral::RunTimings timings;
    program.run_chain(chain, gate_up.inputs.data(), chain_out.data(),
                      down_golden.weights_cols, 0, &timings);
    totals.push_back(timings.total_us);
    kernels.push_back(timings.kernel_us);
  }
  compare_tol(reference_out, chain_out, 1e-2, "production chain");
  std::printf("production chain: total %.1f us kernel %.1f us\n",
              median(totals), median(kernels));

  // Two separate launches with the same device GEGLU intermediate.
  spiral::MatmulShape geglu_shape = gate_shape;
  geglu_shape.epilogue = spiral::kEpilogueGeglu;
  std::vector<uint16_t> glu_bf(
      static_cast<size_t>(design::kInputsVecs) * gate_cols);
  std::vector<uint16_t> two_out(
      static_cast<size_t>(design::kInputsVecs) * down_golden.weights_cols);
  auto run_two = [&](spiral::RunTimings* out) {
    spiral::RunTimings t1;
    spiral::RunTimings t2;
    program.run(geglu_shape, gate_weights, gate_up.inputs.data(),
                glu_bf.data(), gate_cols, 0, &t1);
    program.run(down_shape, down_weights, glu_bf.data(), two_out.data(),
                down_golden.weights_cols, 0, &t2);
    if (out != nullptr) {
      *out = t1;
      out->total_us += t2.total_us;
    }
  };
  for (int i = 0; i < 3; ++i) {
    run_two(nullptr);
  }
  std::vector<double> two_totals;
  for (int i = 0; i < 20; ++i) {
    spiral::RunTimings t;
    run_two(&t);
    two_totals.push_back(t.total_us);
  }
  compare_tol(chain_out, two_out, 1e-6, "chain vs two launches");
  std::printf("production two launches: total %.1f us\n", median(two_totals));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string golden_dir = argc > 1 ? argv[1] : "golden";
    const std::string name0 = argc > 2 ? argv[2] : "q4_0_1536x2048";
    const std::string name1 = argc > 3 ? argv[3] : "q4_0_2048x1536";
    const int iterations = argc > 4 ? std::stoi(argv[4]) : 20;
    if (iterations <= 0) {
      throw std::invalid_argument("iterations must be positive");
    }
    if (argc > 5 && std::string(argv[5]) == "geglu") {
      run_geglu_case(golden_dir, iterations);
      return 0;
    }
    if (argc > 5 && std::string(argv[5]) == "program") {
      run_production_geglu_case(golden_dir);
      return 0;
    }
    if (argc > 5 && std::string(argv[5]) == "chain") {
      run_production_chain_case(golden_dir);
      return 0;
    }

    const Golden golden0 = load_golden(golden_dir, name0);
    const Golden golden1 = load_golden(golden_dir, name1);
    if (golden0.weights_cols != golden1.weights_rows) {
      throw std::runtime_error("op0 outputs_cols must equal op1 weights_rows");
    }
    Op op0 = prepare_op(golden0);
    Op op1 = prepare_op(golden1);
    op0.config_offset = 0;
    op1.config_offset = kConfigObjectBytes;
    op0.weight_offset = 0;
    op1.weight_offset = op0.weight_stream.size();

    std::printf("case: %s -> %s\n", name0.c_str(), name1.c_str());
    std::printf("  op0: %dx%d, rows_steps=%d bands=%d blocks=%d\n",
                golden0.weights_rows, golden0.weights_cols, op0.rows_steps,
                op0.cols_bands, op0.weights_blocks);
    std::printf("  op1: %dx%d, rows_steps=%d bands=%d blocks=%d\n",
                golden1.weights_rows, golden1.weights_cols, op1.rows_steps,
                op1.cols_bands, op1.weights_blocks);

    std::vector<uint8_t> combined_weights(op0.weight_stream.size() +
                                              op1.weight_stream.size() +
                                              design::kL2StreamBytes,
                                          0);
    std::memcpy(combined_weights.data(), op0.weight_stream.data(),
                op0.weight_stream.size());
    std::memcpy(combined_weights.data() + op1.weight_offset,
                op1.weight_stream.data(), op1.weight_stream.size());

    std::vector<uint8_t> configs(2 * kConfigObjectBytes, 0);
    std::memcpy(configs.data(), op0.config.data(), kConfigObjectBytes);
    std::memcpy(configs.data() + kConfigObjectBytes, op1.config.data(),
                kConfigObjectBytes);

    Plan pair = make_plan({&op0, &op1});

    // When op1's output width matches op0's input width, alternate the pair
    // to measure the steady-state N-op chain.
    const bool repeatable = op1.golden.weights_cols == op0.golden.weights_rows;
    Plan repeated;
    size_t instr_words = pair.chain.size();
    for (const auto& single : pair.singles) {
      instr_words = std::max(instr_words, single.size());
    }
    if (repeatable) {
      repeated = make_plan(alternating(op0, op1, 8));
      instr_words = std::max(instr_words, repeated.chain.size());
      for (const auto& single : repeated.singles) {
        instr_words = std::max(instr_words, single.size());
      }
    }
    {
      Runner runner(instr_words, 2, combined_weights.size());
      runner.upload_weights(combined_weights);
      runner.upload_configs(configs);

      std::printf("  txn: single=%zu words chain=%zu words ops=%u\n",
                  pair.singles.front().size(), pair.chain.size(), pair.chain[2]);
      std::printf("  weights: op0=%zu op1=%zu guard=%zu bytes\n",
                  op0.weight_stream.size(), op1.weight_stream.size(),
                  design::kL2StreamBytes);

      const Result independent = run_independent(runner, pair, op0);
      const Result chain = run_chain(runner, pair, op0);
      compare(independent.outputs, chain.outputs);

      const Timed ind2 = time_sequence(runner, pair, op0, false, iterations);
      const Timed ch2 = time_sequence(runner, pair, op0, true, iterations);
      std::printf(
          "  2 ops: independent total %.1f us (kernel %.1f) "
          "chain total %.1f us (kernel %.1f) speedup %.2fx total %.2fx kernel\n",
          ind2.total_us, ind2.kernel_us, ch2.total_us, ch2.kernel_us,
          ind2.total_us / ch2.total_us, ind2.kernel_us / ch2.kernel_us);

      if (repeatable) {
        const Result ind8 = run_independent(runner, repeated, op0);
        const Result ch8 = run_chain(runner, repeated, op0);
        compare(ind8.outputs, ch8.outputs);
        const Timed ind8t =
            time_sequence(runner, repeated, op0, false, iterations);
        const Timed ch8t = time_sequence(runner, repeated, op0, true, iterations);
        std::printf(
            "  8 ops: independent %.1f us (%.1f us/op, kernel %.1f us/op) "
            "chain %.1f us (%.1f us/op, kernel %.1f us/op) speedup %.2fx\n",
            ind8t.total_us, ind8t.total_us / 8.0, ind8t.kernel_us / 8.0,
            ch8t.total_us, ch8t.total_us / 8.0, ch8t.kernel_us / 8.0,
            ind8t.total_us / ch8t.total_us);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
