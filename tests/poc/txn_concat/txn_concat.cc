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
                  int32_t weight_pad) {
  std::memset(buf, 0, design::kConfigLen * sizeof(uint32_t));
  buf[design::kCfgFormat] = static_cast<uint32_t>(format);
  buf[design::kCfgTileBytes] = static_cast<uint32_t>(tile_bytes);
  buf[design::kCfgRowsSteps] = static_cast<uint32_t>(rows_steps);
  buf[design::kCfgColsBands] = static_cast<uint32_t>(cols_bands);
  buf[design::kCfgWeightsBlocks] = static_cast<uint32_t>(weights_blocks);
  buf[design::kCfgTotalTiles] = static_cast<uint32_t>(total_tiles);
  buf[design::kCfgInputPad] = static_cast<uint32_t>(input_pad);
  buf[design::kCfgWeightPad] = static_cast<uint32_t>(weight_pad);
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

struct Op {
  Golden golden;
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

Op prepare_op(const Golden& golden) {
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
               op.total_tiles, op.input_pad, op.weight_pad);
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
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
