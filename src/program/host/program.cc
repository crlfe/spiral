#include "program.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "design_meta.h"
#include "generate_txn.h"
#include "quants/q4_0/repack.h"
#include "quants/q6_k/repack.h"

namespace spiral {
namespace {

constexpr uint32_t kRunOpcode = 3;
constexpr int kRunTimeoutMs = 60000;

using RepackFn = int64_t (*)(const uint8_t*, int64_t, uint8_t*, int64_t,
                             int64_t, int64_t);
using PackedBytesFn = int64_t (*)(int64_t, int64_t);

constexpr int32_t kInputParts = 4;
constexpr int32_t kWeightParts = 2;

void split_parts(int32_t total, int32_t parts, int32_t* out,
                 int32_t& padded) {
  padded = std::max(total, parts);
  const int32_t base = padded / parts;
  const int32_t extra = padded % parts;
  for (int32_t i = 0; i < parts; ++i) {
    out[i] = base + (i < extra ? 1 : 0);
  }
}

size_t max_instruction_words() {
  int32_t input_parts[kInputParts] = {};
  int32_t weight_parts[kWeightParts] = {};
  int32_t input_padded = 0;
  int32_t weight_padded = 0;
  split_parts(design::kMaxInputGroups, kInputParts, input_parts,
              input_padded);
  split_parts(design::kMaxWeightBlocks, kWeightParts, weight_parts,
              weight_padded);
  auto sequence = generate_txn_main_sequence(
      input_parts[0], input_parts[1], input_parts[2], input_parts[3],
      weight_parts[0], weight_parts[1]);
  if (!sequence.has_value()) {
    throw std::runtime_error("transaction generation failed");
  }
  return sequence->size();
}

RepackFn repack_fn(int32_t format) {
  return format == 1 ? spiral::quants::q4_0::repack_weights
                     : spiral::quants::q6_k::repack_weights;
}

PackedBytesFn packed_bytes_fn(int32_t format) {
  return format == 1 ? spiral::quants::q4_0::packed_bytes
                     : spiral::quants::q6_k::packed_bytes;
}

int32_t tile_bytes(int32_t format) {
  return format == 1
             ? spiral::quants::q4_0::kPackedTransportTileBytes
             : spiral::quants::q6_k::kPackedTransportTileBytes;
}

void write_config(uint32_t* buf, int32_t format, int32_t tile_bytes,
                  int32_t rows_steps, int32_t cols_bands,
                  int32_t weights_blocks, int32_t total_tiles,
                  int32_t input_pad, int32_t weight_pad) {
  buf[design::kCfgFormat] = static_cast<uint32_t>(format);
  buf[design::kCfgTileBytes] = static_cast<uint32_t>(tile_bytes);
  buf[design::kCfgRowsSteps] = static_cast<uint32_t>(rows_steps);
  buf[design::kCfgColsBands] = static_cast<uint32_t>(cols_bands);
  buf[design::kCfgWeightsBlocks] = static_cast<uint32_t>(weights_blocks);
  buf[design::kCfgTotalTiles] = static_cast<uint32_t>(total_tiles);
  buf[design::kCfgInputPad] = static_cast<uint32_t>(input_pad);
  buf[design::kCfgWeightPad] = static_cast<uint32_t>(weight_pad);
}

// Copy repacked transport tiles into the block-major host layout:
// [group][block][compute][L1].
void fill_weights(uint8_t* dst, const uint8_t* packed, int32_t rows_steps,
                  int32_t cols_bands, int32_t tile_bytes,
                  int32_t group_bytes) {
  const int64_t cols_pairs =
      static_cast<int64_t>(cols_bands) * design::kBandCols /
      design::kColsTile;
  const int32_t tiles_per_band = design::kBandCols / design::kColsTile;
  for (int32_t g = 0; g < design::kGroups; ++g) {
    for (int32_t c = 0; c < design::kComputesPerGroup; ++c) {
      for (int32_t row = 0; row < rows_steps; ++row) {
        for (int32_t band = 0; band < cols_bands; ++band) {
          const int64_t global_tile =
              static_cast<int64_t>(row) * cols_pairs +
              band * tiles_per_band + g * design::kComputesPerGroup + c;
          const uint8_t* tile = packed + global_tile * tile_bytes;
          const int64_t stream_offset =
              static_cast<int64_t>(row * cols_bands + band) * tile_bytes;
          for (int64_t copied = 0; copied < tile_bytes;) {
            const int64_t block = (stream_offset + copied) /
                                  design::kL1StreamBytes;
            const int64_t within = (stream_offset + copied) %
                                   design::kL1StreamBytes;
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

// Pack [inputs_vecs][weights_rows] bfloat16 bits into one object per row step.
void fill_inputs(uint32_t* buf, int32_t format, int32_t tile_bytes,
                 int32_t rows_steps, int32_t cols_bands,
                 int32_t weights_blocks, int32_t total_tiles,
                 int32_t input_pad, int32_t weight_pad,
                 int32_t weights_rows, const uint16_t* inputs) {
  write_config(buf, format, tile_bytes, rows_steps, cols_bands,
               weights_blocks, total_tiles, input_pad, weight_pad);
  const int32_t groups =
      (rows_steps + design::kInputGroupRows - 1) /
      design::kInputGroupRows;
  auto* objects =
      reinterpret_cast<uint16_t*>(buf) + design::kInputGroupElems;
  const int32_t cols_per_chunk = 8;
  for (int32_t row = 0; row < rows_steps; ++row) {
    const int32_t group = row / design::kInputGroupRows;
    const int32_t within = row % design::kInputGroupRows;
    uint16_t* dst = objects + group * design::kInputGroupElems +
                    within * design::kInputElems;
    for (int32_t chunk = 0; chunk < design::kRowsStep / cols_per_chunk;
         ++chunk) {
      for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
        for (int32_t col = 0; col < cols_per_chunk; ++col) {
          const int64_t src =
              static_cast<int64_t>(vec) * weights_rows +
              row * design::kRowsStep + chunk * cols_per_chunk + col;
          dst[chunk * design::kInputsVecs * cols_per_chunk +
              vec * cols_per_chunk + col] = inputs[src];
        }
      }
    }
  }
}

// Convert the kernel's [col_half][vec][8 cols] tiles to logical
// [inputs_vecs][weights_cols] rows.
void unpack_outputs(const float* raw, float* outputs, int32_t cols_bands,
                    int32_t outputs_cols, int32_t outputs_col_offset) {
  const int32_t half_cols = design::kColsTile / 2;
  const int32_t compute_stride = design::kOutputElems;
  for (int32_t g = 0; g < design::kGroups; ++g) {
    for (int32_t c = 0; c < design::kComputesPerGroup; ++c) {
      for (int32_t band = 0; band < cols_bands; ++band) {
        const int32_t raw_base = g * design::kGroupOutputs +
                                 c * compute_stride +
                                 band * design::kOutputTileElems;
        const int32_t col_base =
            band * design::kBandCols +
            (g * design::kComputesPerGroup + c) * design::kColsTile;
        for (int32_t half = 0; half < 2; ++half) {
          for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
            for (int32_t col = 0; col < half_cols; ++col) {
              const int32_t raw_index =
                  raw_base + half * design::kInputsVecs * half_cols +
                  vec * half_cols + col;
              const int32_t col_index =
                  outputs_col_offset + col_base + half * half_cols + col;
              outputs[static_cast<int64_t>(vec) * outputs_cols + col_index] =
                  raw[raw_index];
            }
          }
        }
      }
    }
  }
}

int32_t shape_tile_bytes(const MatmulShape& shape) {
  if (shape.format == 0 || shape.format == 3 || shape.format == 4) {
    if (design::kProduction) {
      throw std::invalid_argument(
          "diagnostic formats are disabled in the production build");
    }
    if (shape.tile_bytes <= 0 ||
        shape.tile_bytes > design::kL1StreamBytes) {
      throw std::invalid_argument(
          "probe tile_bytes must fit one L1 block");
    }
    return shape.tile_bytes;
  }
  if (shape.format == 1 || shape.format == 2) {
    return tile_bytes(shape.format);
  }
  throw std::invalid_argument("format must be 0, 1, 2, 3, or 4");
}

void check_shape(const MatmulShape& shape, int32_t& rows_steps,
                 int32_t& cols_bands, int32_t& total_tiles,
                 int32_t& weights_blocks) {
  const int32_t block_bytes = shape_tile_bytes(shape);
  if (shape.inputs_vecs != design::kInputsVecs) {
    throw std::invalid_argument("inputs_vecs does not match the design");
  }
  if (shape.weights_rows <= 0 ||
      shape.weights_rows % design::kRowsStep != 0) {
    throw std::invalid_argument("weights_rows must be a positive multiple of 256");
  }
  if (shape.weights_cols <= 0 ||
      shape.weights_cols % design::kBandCols != 0) {
    throw std::invalid_argument("weights_cols must be a positive multiple of 512");
  }
  rows_steps = shape.weights_rows / design::kRowsStep;
  cols_bands = shape.weights_cols / design::kBandCols;
  if (rows_steps > design::kMaxRowsSteps) {
    throw std::invalid_argument("weights_rows exceeds the design limit");
  }
  if (cols_bands > design::kMaxBandsPerChunk) {
    throw std::invalid_argument("weights_cols exceeds one band chunk");
  }
  total_tiles = rows_steps * cols_bands;
  if (total_tiles > design::kMaxTotalTiles) {
    throw std::invalid_argument("call needs too many input objects");
  }
  const int64_t bytes =
      static_cast<int64_t>(total_tiles) * block_bytes;
  weights_blocks = static_cast<int32_t>(
      (bytes + design::kL1StreamBytes - 1) / design::kL1StreamBytes);
  if (weights_blocks > design::kMaxWeightBlocks) {
    throw std::invalid_argument("call needs too many weight blocks");
  }
}

}  // namespace

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}

struct LoadedDevice {
  xrt::device device;
  xrt::xclbin xclbin;

  LoadedDevice() : device(0), xclbin(std::string("aie.xclbin")) {
    device.register_xclbin(xclbin);
  }
};

struct Program::Impl {
  size_t instr_words;
  uint64_t next_weight_id = 1;
  uint64_t last_weight_id = 0;
  LoadedDevice loaded;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::run run;
  xrt::bo instr;
  xrt::bo weights;
  xrt::bo inputs;
  xrt::bo outputs;

  Impl()
      : instr_words(max_instruction_words()),
        loaded(),
        context(loaded.device, loaded.xclbin.get_uuid(),
                xrt::hw_context::access_mode::exclusive),
        kernel(context, "MLIR_AIE"),
        run(kernel),
        instr(loaded.device, instr_words * sizeof(uint32_t),
              xrt::bo::flags::cacheable, kernel.group_id(1)),
        weights(loaded.device, design::kWeightWords * sizeof(uint32_t),
                xrt::bo::flags::host_only, kernel.group_id(3)),
        inputs(loaded.device, design::kInputWords * sizeof(uint32_t),
               xrt::bo::flags::host_only, kernel.group_id(4)),
        outputs(loaded.device, design::kOutputFloats * sizeof(float),
                xrt::bo::flags::host_only, kernel.group_id(5)) {}
};

struct PreparedWeights::Impl {
  const Program* owner = nullptr;
  uint64_t id = 0;
  MatmulShape shape;
  int32_t rows_steps = 0;
  int32_t cols_bands = 0;
  int32_t total_tiles = 0;
  int32_t weights_blocks = 0;
  int32_t tile_bytes = 0;
  int32_t input_pad = 0;
  int32_t weight_pad = 0;
  int32_t group_bytes = 0;
  std::vector<uint8_t> weights;
  std::vector<uint32_t> txn;
};

PreparedWeights::PreparedWeights() = default;
PreparedWeights::~PreparedWeights() = default;
PreparedWeights::PreparedWeights(PreparedWeights&&) noexcept = default;
PreparedWeights& PreparedWeights::operator=(PreparedWeights&&) noexcept =
    default;

Program::Program() : impl_(std::make_unique<Impl>()) {}
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

PreparedWeights Program::prepare(const MatmulShape& shape,
                                 const uint8_t* raw_weights,
                                 int64_t raw_stride,
                                 int32_t weights_col_offset) {
  const bool probe =
      shape.format == 0 || shape.format == 3 || shape.format == 4;
  if (!probe && raw_weights == nullptr) {
    throw std::invalid_argument("null weight buffer");
  }
  if (!probe && raw_stride <= 0) {
    throw std::invalid_argument("raw_stride must be positive");
  }
  if (weights_col_offset < 0 ||
      weights_col_offset % design::kBandCols != 0) {
    throw std::invalid_argument(
        "weights_col_offset must be a non-negative multiple of 512");
  }

  int32_t rows_steps = 0;
  int32_t cols_bands = 0;
  int32_t total_tiles = 0;
  int32_t weights_blocks = 0;
  check_shape(shape, rows_steps, cols_bands, total_tiles, weights_blocks);
  const int32_t bytes = shape_tile_bytes(shape);

  std::vector<uint8_t> packed;
  if (!probe) {
    const int64_t packed_size =
        packed_bytes_fn(shape.format)(shape.weights_rows, shape.weights_cols);
    if (packed_size <= 0) {
      throw std::invalid_argument("unsupported weight shape");
    }
    packed.resize(packed_size);
    const uint8_t* chunk =
        raw_weights + static_cast<int64_t>(weights_col_offset) * raw_stride;
    const int64_t written =
        repack_fn(shape.format)(chunk, raw_stride, packed.data(),
                                packed.size(), shape.weights_rows,
                                shape.weights_cols);
    if (written != packed_size) {
      throw std::runtime_error("weight repack failed");
    }
  }

  return prepare_packed(shape, std::move(packed));
}

PreparedWeights Program::prepare(const MatmulShape& shape,
                                 const WeightSegment* segments,
                                 int32_t segment_count) {
  if (segments == nullptr || segment_count <= 0) {
    throw std::invalid_argument("at least one weight segment is required");
  }
  if (shape.format != 1 && shape.format != 2) {
    throw std::invalid_argument(
        "combined weights require format 1 (Q4_0) or 2 (Q6_K)");
  }
  int32_t total_cols = 0;
  for (int32_t i = 0; i < segment_count; ++i) {
    const auto& seg = segments[i];
    if (seg.raw_weights == nullptr || seg.raw_stride <= 0 ||
        seg.weights_cols <= 0 || seg.weights_cols % 16 != 0) {
      throw std::invalid_argument("invalid weight segment");
    }
    if (seg.weights_col_offset < 0 ||
        seg.weights_col_offset % design::kBandCols != 0) {
      throw std::invalid_argument(
          "segment weights_col_offset must be a non-negative multiple of 512");
    }
    total_cols += seg.weights_cols;
  }
  if (total_cols != shape.weights_cols) {
    throw std::invalid_argument(
        "segment column count does not match shape.weights_cols");
  }

  const int32_t tile_bytes = shape_tile_bytes(shape);
  const int64_t rows_tiles = shape.weights_rows / design::kRowsStep;
  const int32_t total_pairs = shape.weights_cols / design::kColsTile;
  std::vector<uint8_t> combined(
      static_cast<size_t>(rows_tiles) * total_pairs * tile_bytes);
  int32_t pair_offset = 0;
  for (int32_t i = 0; i < segment_count; ++i) {
    const auto& seg = segments[i];
    const int64_t seg_size =
        packed_bytes_fn(shape.format)(shape.weights_rows, seg.weights_cols);
    if (seg_size <= 0) {
      throw std::invalid_argument("unsupported segment weight shape");
    }
    std::vector<uint8_t> packed(static_cast<size_t>(seg_size));
    const uint8_t* chunk =
        seg.raw_weights +
        static_cast<int64_t>(seg.weights_col_offset) * seg.raw_stride;
    const int64_t written =
        repack_fn(shape.format)(chunk, seg.raw_stride, packed.data(),
                                seg_size, shape.weights_rows,
                                seg.weights_cols);
    if (written != seg_size) {
      throw std::runtime_error("segment weight repack failed");
    }
    const int32_t seg_pairs = seg.weights_cols / design::kColsTile;
    for (int64_t row_tile = 0; row_tile < rows_tiles; ++row_tile) {
      const uint8_t* src =
          packed.data() +
          static_cast<size_t>(row_tile) * seg_pairs * tile_bytes;
      uint8_t* dst =
          combined.data() +
          (static_cast<size_t>(row_tile) * total_pairs + pair_offset) *
              tile_bytes;
      std::memcpy(dst, src, static_cast<size_t>(seg_pairs) * tile_bytes);
    }
    pair_offset += seg_pairs;
  }
  return prepare_packed(shape, std::move(combined));
}

PreparedWeights Program::prepare_packed(const MatmulShape& shape,
                                          std::vector<uint8_t> packed) {
  int32_t rows_steps = 0;
  int32_t cols_bands = 0;
  int32_t total_tiles = 0;
  int32_t weights_blocks = 0;
  check_shape(shape, rows_steps, cols_bands, total_tiles, weights_blocks);
  const int32_t bytes = shape_tile_bytes(shape);
  const bool probe =
      shape.format == 0 || shape.format == 3 || shape.format == 4;

  PreparedWeights prepared;
  prepared.impl_ = std::make_unique<PreparedWeights::Impl>();
  auto& impl = *prepared.impl_;
  impl.owner = this;
  impl.id = impl_->next_weight_id++;
  impl.shape = shape;
  impl.rows_steps = rows_steps;
  impl.cols_bands = cols_bands;
  impl.total_tiles = total_tiles;
  impl.weights_blocks = weights_blocks;
  impl.tile_bytes = bytes;
  impl.group_bytes = weights_blocks * design::kL2StreamBytes;
  impl.weights.assign(
      design::kGroups * static_cast<size_t>(impl.group_bytes), 0);
  if (!probe) {
    fill_weights(impl.weights.data(), packed.data(), rows_steps, cols_bands,
                 bytes, impl.group_bytes);
  }

  int32_t input_parts[kInputParts] = {};
  int32_t weight_parts[kWeightParts] = {};
  int32_t input_padded = 0;
  int32_t weight_padded = 0;
  const int32_t input_groups =
      (rows_steps + design::kInputGroupRows - 1) /
      design::kInputGroupRows;
  split_parts(input_groups, kInputParts, input_parts, input_padded);
  split_parts(weights_blocks, kWeightParts, weight_parts, weight_padded);
  impl.input_pad = input_padded - input_groups;
  impl.weight_pad = weight_padded - weights_blocks;

  auto sequence = generate_txn_main_sequence(
      input_parts[0], input_parts[1], input_parts[2], input_parts[3],
      weight_parts[0], weight_parts[1]);
  if (!sequence.has_value()) {
    throw std::runtime_error("transaction generation failed");
  }
  impl.txn = std::move(*sequence);
  return prepared;
}

void Program::run(const MatmulShape& shape, const PreparedWeights& weights,
                  const uint16_t* inputs, float* outputs,
                  int32_t outputs_cols, int32_t outputs_col_offset,
                  RunTimings* timings) {
  if (!impl_) {
    throw std::runtime_error("Program is not initialized");
  }
  if (!weights.impl_) {
    throw std::invalid_argument("PreparedWeights is empty");
  }
  if (inputs == nullptr || outputs == nullptr) {
    throw std::invalid_argument("null input or output buffer");
  }
  const auto& prepared = *weights.impl_;
  if (prepared.owner != this) {
    throw std::invalid_argument(
        "PreparedWeights belongs to a different Program");
  }
  if (shape.format != prepared.shape.format ||
      shape.weights_rows != prepared.shape.weights_rows ||
      shape.weights_cols != prepared.shape.weights_cols) {
    throw std::invalid_argument("shape does not match prepared weights");
  }
  if (outputs_col_offset < 0 ||
      outputs_cols < outputs_col_offset + shape.weights_cols) {
    throw std::invalid_argument("output buffer is too small");
  }
  if (timings) {
    *timings = RunTimings{};
  }
  const auto total_start = Clock::now();

  if (prepared.txn.size() > impl_->instr_words) {
    throw std::runtime_error("instruction buffer overflow");
  }
  auto stage = Clock::now();
  impl_->instr.write(prepared.txn.data(),
                     prepared.txn.size() * sizeof(uint32_t), 0);
  impl_->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (timings) {
    timings->txn_us = elapsed_us(stage);
  }

  stage = Clock::now();
  const size_t max_group_bytes =
      static_cast<size_t>(design::kMaxWeightBlocks) *
      design::kL2StreamBytes;
  auto* weight_dst = impl_->weights.map<uint8_t*>();
  if (prepared.id != impl_->last_weight_id) {
    for (int32_t g = 0; g < design::kGroups; ++g) {
      const size_t group_offset =
          static_cast<size_t>(g) * max_group_bytes;
      std::memcpy(weight_dst + group_offset,
                  prepared.weights.data() +
                      static_cast<size_t>(g) * prepared.group_bytes,
                  prepared.group_bytes);
      impl_->weights.sync(XCL_BO_SYNC_BO_TO_DEVICE, prepared.group_bytes,
                          group_offset);
    }
    impl_->last_weight_id = prepared.id;
  }
  if (timings) {
    timings->upload_us = elapsed_us(stage);
  }

  stage = Clock::now();
  fill_inputs(impl_->inputs.map<uint32_t*>(), shape.format,
              prepared.tile_bytes, prepared.rows_steps, prepared.cols_bands,
              prepared.weights_blocks, prepared.total_tiles,
              prepared.input_pad, prepared.weight_pad,
              shape.weights_rows, inputs);
  const int32_t input_groups =
      (prepared.rows_steps + design::kInputGroupRows - 1) /
      design::kInputGroupRows;
  const size_t input_objects =
      1 + input_groups + prepared.input_pad;
  impl_->inputs.sync(
      XCL_BO_SYNC_BO_TO_DEVICE,
      input_objects * design::kInputGroupElems * sizeof(uint16_t), 0);
  if (timings) {
    timings->input_us = elapsed_us(stage);
  }

  stage = Clock::now();
  auto& run = impl_->run;
  run.set_arg(0, kRunOpcode);
  run.set_arg(1, impl_->instr);
  run.set_arg(2, static_cast<uint32_t>(prepared.txn.size()));
  run.set_arg(3, impl_->weights);
  run.set_arg(4, impl_->inputs);
  run.set_arg(5, impl_->outputs);
  run.start();
  run.wait2(std::chrono::milliseconds(kRunTimeoutMs));
  impl_->outputs.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (timings) {
    timings->kernel_us = elapsed_us(stage);
  }

  stage = Clock::now();
  unpack_outputs(impl_->outputs.map<const float*>(), outputs,
                 prepared.cols_bands, outputs_cols, outputs_col_offset);
  if (timings) {
    timings->unpack_us = elapsed_us(stage);
    timings->total_us = elapsed_us(total_start);
  }
}

void Program::run(const MatmulShape& shape, const uint8_t* raw_weights,
                  int64_t raw_stride, int32_t weights_col_offset,
                  const uint16_t* inputs, float* outputs,
                  int32_t outputs_cols, int32_t outputs_col_offset,
                  RunTimings* timings) {
  const auto start = Clock::now();
  auto weights =
      prepare(shape, raw_weights, raw_stride, weights_col_offset);
  const double prepare_us = elapsed_us(start);
  run(shape, weights, inputs, outputs, outputs_cols, outputs_col_offset,
      timings);
  if (timings) {
    timings->prepare_us = prepare_us;
    timings->total_us += prepare_us;
  }
}

}  // namespace spiral
