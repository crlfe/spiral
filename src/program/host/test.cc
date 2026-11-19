#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "design_meta.h"
#include "program.h"

namespace {

struct Golden {
  int32_t format = 0;
  int32_t rows_steps = 0;
  int32_t cols_bands = 0;
  int32_t tile_bytes = 0;
  int32_t weights_blocks = 0;
  int32_t total_tiles = 0;
  int32_t weights_rows = 0;
  int32_t weights_cols = 0;
  int32_t raw_stride = 0;
  int32_t chunk_bands = 0;
  size_t raw_bytes = 0;
  size_t inputs_elems = 0;
  size_t expected_count = 0;
  std::vector<uint8_t> raw;
  std::vector<uint16_t> inputs;
  std::vector<float> expected;
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

Golden load_golden(const std::string& name) {
  const std::string dir = "golden/" + name;
  std::ifstream meta(dir + ".meta");
  if (!meta) {
    throw std::runtime_error("cannot open " + dir + ".meta");
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
  golden.tile_bytes = static_cast<int32_t>(values.at("tile_bytes"));
  golden.weights_blocks = static_cast<int32_t>(values.at("weights_blocks"));
  golden.total_tiles = static_cast<int32_t>(values.at("total_tiles"));
  golden.weights_rows = static_cast<int32_t>(values.at("weights_rows"));
  golden.weights_cols = static_cast<int32_t>(values.at("weights_cols"));
  golden.raw_stride = static_cast<int32_t>(values.at("raw_stride"));
  auto chunk = values.find("chunk_bands");
  if (chunk != values.end()) {
    golden.chunk_bands = static_cast<int32_t>(chunk->second);
  }
  golden.raw_bytes = static_cast<size_t>(values.at("raw_bytes"));
  golden.inputs_elems = static_cast<size_t>(values.at("inputs_elems"));
  golden.expected_count = static_cast<size_t>(values.at("expected_count"));

  golden.raw = read_file(dir + ".raw");
  golden.inputs = read_words<uint16_t>(dir + ".inputs", golden.inputs_elems);
  golden.expected =
      read_words<float>(dir + ".expected", golden.expected_count);
  if (golden.raw.size() != golden.raw_bytes) {
    throw std::runtime_error("raw size mismatch: " + name);
  }
  return golden;
}

// Expected values are stored per [group][compute][band][vec][16].  The
// wrapper's logical output is [vec][weights_cols].
bool check_outputs(const std::string& name, const Golden& golden,
                   const std::vector<float>& outputs,
                   int32_t outputs_cols = 0,
                   int32_t outputs_col_offset = 0) {
  if (outputs_cols <= 0) {
    outputs_cols = golden.weights_cols;
  }
  double max_error = 0.0;
  size_t worst = 0;
  int32_t worst_g = -1;
  int32_t worst_c = -1;
  int32_t worst_band = -1;
  for (int32_t g = 0; g < design::kGroups; ++g) {
    for (int32_t c = 0; c < design::kComputesPerGroup; ++c) {
      for (int32_t band = 0; band < golden.cols_bands; ++band) {
        const size_t exp_base =
            ((static_cast<size_t>(g) * design::kComputesPerGroup + c) *
                 golden.cols_bands +
             band) *
            design::kOutputTileElems;
        const int32_t col_base =
            band * design::kBandCols +
            (g * design::kComputesPerGroup + c) * design::kColsTile;
        for (int32_t half = 0; half < 2; ++half) {
          for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
            for (int32_t col = 0; col < design::kColsTile / 2; ++col) {
              const size_t exp_index =
                  exp_base + vec * design::kColsTile +
                  half * (design::kColsTile / 2) + col;
              const int32_t col_index =
                  outputs_col_offset + col_base + half * (design::kColsTile / 2) + col;
              const float got =
                  outputs[static_cast<size_t>(vec) * outputs_cols + col_index];
              const double diff = std::fabs(
                  static_cast<double>(got) -
                  static_cast<double>(golden.expected[exp_index]));
              if (diff > max_error) {
                max_error = diff;
                worst = static_cast<size_t>(vec) * outputs_cols + col_index;
                worst_g = g;
                worst_c = c;
                worst_band = band;
              }
              if (!std::isfinite(got)) {
                std::fprintf(stderr, "%s: non-finite output at %zu\n",
                             name.c_str(),
                             static_cast<size_t>(vec) * outputs_cols +
                                 col_index);
                return false;
              }
            }
          }
        }
      }
    }
  }

  const double tolerance = golden.format == 1 ? 0.01 : 0.02;
  const bool ok = max_error <= tolerance;
  std::printf("%s golden: %s (max error %.6f, %d tiles)\n", name.c_str(),
              ok ? "ok" : "FAILED", max_error, golden.total_tiles);
  if (!ok) {
    std::fprintf(stderr,
                 "%s: worst output index %zu (g=%d c=%d band=%d)\n",
                 name.c_str(), worst, worst_g, worst_c, worst_band);
  }
  return ok;
}

bool run_golden(spiral::Program& program, const std::string& name) {
  Golden golden = load_golden(name);
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());

  spiral::MatmulShape shape;
  shape.format = golden.format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;
  program.run(shape, golden.raw.data(), golden.raw_stride, 0,
              golden.inputs.data(), outputs.data(), golden.weights_cols, 0);

  return check_outputs(name, golden, outputs);
}

bool run_head(spiral::Program& program, const std::string& name) {
  Golden golden = load_golden(name);
  if (golden.chunk_bands <= 0) {
    throw std::runtime_error(name + " has no chunk_bands");
  }
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());

  for (int32_t start = 0; start < golden.cols_bands;
       start += golden.chunk_bands) {
    const int32_t bands =
        std::min(golden.chunk_bands, golden.cols_bands - start);
    spiral::MatmulShape shape;
    shape.format = golden.format;
    shape.weights_rows = golden.weights_rows;
    shape.weights_cols = bands * design::kBandCols;
    shape.inputs_vecs = design::kInputsVecs;
    const int32_t col_offset = start * design::kBandCols;
    program.run(shape, golden.raw.data(), golden.raw_stride, col_offset,
                golden.inputs.data(), outputs.data(), golden.weights_cols,
                col_offset);
  }

  return check_outputs(name, golden, outputs);
}

bool run_head_timing(spiral::Program& program, const std::string& name,
                     int iterations) {
  Golden golden = load_golden(name);
  if (golden.chunk_bands <= 0) {
    throw std::runtime_error(name + " has no chunk_bands");
  }
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());
  spiral::RunTimings total;
  int chunks = 0;
  for (int32_t start = 0; start < golden.cols_bands;
       start += golden.chunk_bands) {
    const int32_t bands =
        std::min(golden.chunk_bands, golden.cols_bands - start);
    spiral::MatmulShape shape;
    shape.format = golden.format;
    shape.weights_rows = golden.weights_rows;
    shape.weights_cols = bands * design::kBandCols;
    shape.inputs_vecs = design::kInputsVecs;
    const int32_t col_offset = start * design::kBandCols;
    auto prepared = program.prepare(shape, golden.raw.data(),
                                    golden.raw_stride, col_offset);
    // Warm-up run uploads the prepared weights; timed runs are steady state.
    program.run(shape, prepared, golden.inputs.data(), outputs.data(),
                golden.weights_cols, col_offset);
    for (int i = 0; i < iterations; ++i) {
      spiral::RunTimings timings;
      program.run(shape, prepared, golden.inputs.data(), outputs.data(),
                  golden.weights_cols, col_offset, &timings);
      total.prepare_us += timings.prepare_us;
      total.upload_us += timings.upload_us;
      total.txn_us += timings.txn_us;
      total.input_us += timings.input_us;
      total.kernel_us += timings.kernel_us;
      total.unpack_us += timings.unpack_us;
      total.total_us += timings.total_us;
    }
    ++chunks;
  }
  const double n = static_cast<double>(iterations);
  std::printf(
      "%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f  (%d chunks)\n",
      (name + " head").c_str(), total.prepare_us / n,
      total.upload_us / n, total.txn_us / n, total.input_us / n,
      total.kernel_us / n, total.unpack_us / n, total.total_us / n,
      chunks);
  return check_outputs(name + " head timing", golden, outputs);
}

bool run_prepared(spiral::Program& program, const std::string& name) {
  Golden golden = load_golden(name);
  spiral::MatmulShape shape;
  shape.format = golden.format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;

  auto prepared =
      program.prepare(shape, golden.raw.data(), golden.raw_stride, 0);
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<float> outputs(
        static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
        std::numeric_limits<float>::quiet_NaN());
    program.run(shape, prepared, golden.inputs.data(), outputs.data(),
                golden.weights_cols, 0);
    if (!check_outputs(name + " prepared", golden, outputs)) {
      return false;
    }
  }
  return true;
}

bool run_timing(spiral::Program& program, const std::string& name,
                int iterations) {
  Golden golden = load_golden(name);
  spiral::MatmulShape shape;
  shape.format = golden.format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;

  auto prepared =
      program.prepare(shape, golden.raw.data(), golden.raw_stride, 0);
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());
  spiral::RunTimings total;
  for (int i = 0; i < iterations; ++i) {
    spiral::RunTimings timings;
    program.run(shape, prepared, golden.inputs.data(), outputs.data(),
                golden.weights_cols, 0, &timings);
    total.prepare_us += timings.prepare_us;
    total.upload_us += timings.upload_us;
    total.txn_us += timings.txn_us;
    total.input_us += timings.input_us;
    total.kernel_us += timings.kernel_us;
    total.unpack_us += timings.unpack_us;
    total.total_us += timings.total_us;
  }
  const double n = static_cast<double>(iterations);
  std::printf(
      "%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f\n",
      name.c_str(), total.prepare_us / n, total.upload_us / n,
      total.txn_us / n, total.input_us / n, total.kernel_us / n,
      total.unpack_us / n, total.total_us / n);
  return check_outputs(name + " timing", golden, outputs);
}

bool run_cold_timing(spiral::Program& program, const std::string& name,
                     int iterations) {
  Golden golden = load_golden(name);
  spiral::MatmulShape shape;
  shape.format = golden.format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;
  // Distinct resident BOs so consecutive calls cannot keep the weights hot
  // in the last-level cache.  More than two also exercises IOMMU/TLB spread.
  int copies = 2;
  if (const char* env = std::getenv("SPIRAL_COLD_BOS")) {
    const int configured = std::atoi(env);
    if (configured > 0) {
      copies = configured;
    }
  }
  if (const char* env = std::getenv("SPIRAL_COLD_ITERS")) {
    const int configured = std::atoi(env);
    if (configured > 0) {
      iterations = configured;
    }
  }
  std::vector<spiral::PreparedWeights> prepared;
  prepared.reserve(copies);
  for (int i = 0; i < copies; ++i) {
    prepared.push_back(
        program.prepare(shape, golden.raw.data(), golden.raw_stride, 0));
  }
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());
  int gap_us = 0;
  if (const char* env = std::getenv("SPIRAL_COLD_GAP_US")) {
    gap_us = std::atoi(env);
  }
  spiral::RunTimings total;
  std::vector<double> kernels;
  kernels.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    if (gap_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
    }
    spiral::RunTimings timings;
    program.run(shape, prepared[static_cast<size_t>(i) % prepared.size()],
                golden.inputs.data(), outputs.data(), golden.weights_cols, 0,
                &timings);
    total.kernel_us += timings.kernel_us;
    total.unpack_us += timings.unpack_us;
    total.total_us += timings.total_us;
    kernels.push_back(timings.kernel_us);
  }
  const double n = static_cast<double>(iterations);
  std::printf("%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f\n",
              name.c_str(), 0.0, 0.0, 0.0, 0.0, total.kernel_us / n,
              total.unpack_us / n, total.total_us / n);
  std::sort(kernels.begin(), kernels.end());
  std::printf("%-22s kernel min %.1f  p50 %.1f  p95 %.1f  max %.1f us\n",
              name.c_str(), kernels.front(),
              kernels[kernels.size() / 2],
              kernels[kernels.size() * 95 / 100], kernels.back());
  return check_outputs(name + " cold timing", golden, outputs);
}

bool run_steady_timing(spiral::Program& program, const std::string& name,
                       int iterations) {
  Golden golden = load_golden(name);
  spiral::MatmulShape shape;
  shape.format = golden.format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;
  auto prepared =
      program.prepare(shape, golden.raw.data(), golden.raw_stride, 0);
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());
  // Warm-up run uploads and flushes the prepared weights.
  program.run(shape, prepared, golden.inputs.data(), outputs.data(),
              golden.weights_cols, 0);
  spiral::RunTimings total;
  for (int i = 0; i < iterations; ++i) {
    spiral::RunTimings timings;
    program.run(shape, prepared, golden.inputs.data(), outputs.data(),
                golden.weights_cols, 0, &timings);
    total.prepare_us += timings.prepare_us;
    total.upload_us += timings.upload_us;
    total.txn_us += timings.txn_us;
    total.input_us += timings.input_us;
    total.kernel_us += timings.kernel_us;
    total.unpack_us += timings.unpack_us;
    total.total_us += timings.total_us;
  }
  const double n = static_cast<double>(iterations);
  std::printf("%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f\n",
              name.c_str(), total.prepare_us / n, total.upload_us / n,
              total.txn_us / n, total.input_us / n, total.kernel_us / n,
              total.unpack_us / n, total.total_us / n);
  return check_outputs(name + " steady timing", golden, outputs);
}

bool run_combined_timing(spiral::Program& program, const std::string& label,
                         const std::vector<std::string>& names,
                         int iterations, bool verify) {
  std::vector<Golden> goldens;
  goldens.reserve(names.size());
  for (const auto& name : names) {
    goldens.push_back(load_golden(name));
  }
  const Golden& first = goldens[0];
  int32_t total_cols = 0;
  for (const auto& golden : goldens) {
    if (golden.format != first.format ||
        golden.weights_rows != first.weights_rows) {
      throw std::runtime_error("combined segments have different shapes");
    }
    total_cols += golden.weights_cols;
  }

  spiral::MatmulShape shape;
  shape.format = first.format;
  shape.weights_rows = first.weights_rows;
  shape.weights_cols = total_cols;
  shape.inputs_vecs = design::kInputsVecs;
  std::vector<spiral::WeightSegment> segments;
  segments.reserve(goldens.size());
  for (const auto& golden : goldens) {
    spiral::WeightSegment segment;
    segment.raw_weights = golden.raw.data();
    segment.raw_stride = golden.raw_stride;
    segment.weights_cols = golden.weights_cols;
    segments.push_back(segment);
  }

  auto prepared =
      program.prepare(shape, segments.data(),
                      static_cast<int32_t>(segments.size()));
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * total_cols,
      std::numeric_limits<float>::quiet_NaN());
  program.run(shape, prepared, first.inputs.data(), outputs.data(),
              total_cols, 0);
  spiral::RunTimings total;
  for (int i = 0; i < iterations; ++i) {
    spiral::RunTimings timings;
    program.run(shape, prepared, first.inputs.data(), outputs.data(),
                total_cols, 0, &timings);
    total.prepare_us += timings.prepare_us;
    total.upload_us += timings.upload_us;
    total.txn_us += timings.txn_us;
    total.input_us += timings.input_us;
    total.kernel_us += timings.kernel_us;
    total.unpack_us += timings.unpack_us;
    total.total_us += timings.total_us;
  }
  const double n = static_cast<double>(iterations);
  std::printf("%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f  (%d cols)\n",
              label.c_str(), total.prepare_us / n, total.upload_us / n,
              total.txn_us / n, total.input_us / n, total.kernel_us / n,
              total.unpack_us / n, total.total_us / n, total_cols);

  if (!verify) {
    return true;
  }
  bool ok = true;
  int32_t col_offset = 0;
  for (size_t i = 0; i < goldens.size(); ++i) {
    ok = check_outputs(label + " seg" + std::to_string(i), goldens[i],
                       outputs, total_cols, col_offset) &&
         ok;
    col_offset += goldens[i].weights_cols;
  }
  return ok;
}

bool run_pipeline_timing(spiral::Program& program, const std::string& name,
                         int iterations) {
  Golden golden = load_golden(name);
  if (golden.chunk_bands <= 0) {
    throw std::runtime_error(name + " has no chunk_bands");
  }
  struct Chunk {
    spiral::MatmulShape shape;
    int32_t col_offset = 0;
    spiral::PreparedWeights prepared;
  };
  std::vector<Chunk> chunks;
  for (int32_t start = 0; start < golden.cols_bands;
       start += golden.chunk_bands) {
    const int32_t bands =
        std::min(golden.chunk_bands, golden.cols_bands - start);
    Chunk chunk;
    chunk.shape.format = golden.format;
    chunk.shape.weights_rows = golden.weights_rows;
    chunk.shape.weights_cols = bands * design::kBandCols;
    chunk.shape.inputs_vecs = design::kInputsVecs;
    chunk.col_offset = start * design::kBandCols;
    chunk.prepared = program.prepare(chunk.shape, golden.raw.data(),
                                     golden.raw_stride, chunk.col_offset);
    chunks.push_back(std::move(chunk));
  }
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());
  auto pass = [&](double* kernel_sum) {
    std::vector<spiral::RunTicket> pending;
    const int32_t slots = program.pipeline_slots();
    pending.reserve(static_cast<size_t>(slots));
    for (const Chunk& chunk : chunks) {
      if (static_cast<int32_t>(pending.size()) >= slots) {
        spiral::RunTicket ticket = pending.front();
        pending.erase(pending.begin());
        program.finish(ticket);
        if (kernel_sum != nullptr) {
          *kernel_sum += ticket.timings.kernel_us;
        }
      }
      pending.push_back(program.submit(
          chunk.shape, chunk.prepared, golden.inputs.data(), outputs.data(),
          golden.weights_cols, chunk.col_offset));
    }
    while (!pending.empty()) {
      spiral::RunTicket ticket = pending.front();
      pending.erase(pending.begin());
      program.finish(ticket);
      if (kernel_sum != nullptr) {
        *kernel_sum += ticket.timings.kernel_us;
      }
    }
  };
  pass(nullptr);  // warm-up
  const auto start = std::chrono::steady_clock::now();
  double kernel_sum = 0.0;
  for (int i = 0; i < iterations; ++i) {
    pass(&kernel_sum);
  }
  const auto end = std::chrono::steady_clock::now();
  const double wall_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  const bool ok = check_outputs(name + " pipeline", golden, outputs);
  std::printf("%-22s slots=%d wall %.1f us  kernel_sum %.1f us  %s\n",
              name.c_str(), program.pipeline_slots(), wall_us / iterations,
              kernel_sum / iterations, ok ? "ok" : "FAIL");
  return ok;
}

bool run_manifest_timing(spiral::Program& program, int iterations) {
  std::ifstream index("golden/index.txt");
  if (!index) {
    throw std::runtime_error("cannot open golden/index.txt");
  }
  bool ok = true;
  std::string name;
  std::string mode;
  while (index >> name >> mode) {
    if (mode == "head") {
      ok = run_head_timing(program, name, 5) && ok;
    } else {
      ok = run_steady_timing(program, name, iterations) && ok;
    }
  }
  return ok;
}

bool run_probe_timing(spiral::Program& program, const std::string& name,
                      int iterations, int32_t format,
                      const std::string& label) {
  if (design::kProduction) {
    return true;
  }
  Golden golden = load_golden(name);
  spiral::MatmulShape shape;
  shape.format = format;
  shape.weights_rows = golden.weights_rows;
  shape.weights_cols = golden.weights_cols;
  shape.inputs_vecs = design::kInputsVecs;
  shape.tile_bytes = golden.tile_bytes;

  auto prepared = program.prepare(shape, nullptr, 0, 0);
  std::vector<float> outputs(
      static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
      std::numeric_limits<float>::quiet_NaN());
  spiral::RunTimings total;
  for (int i = 0; i < iterations; ++i) {
    spiral::RunTimings timings;
    program.run(shape, prepared, golden.inputs.data(), outputs.data(),
                golden.weights_cols, 0, &timings);
    total.prepare_us += timings.prepare_us;
    total.upload_us += timings.upload_us;
    total.txn_us += timings.txn_us;
    total.input_us += timings.input_us;
    total.kernel_us += timings.kernel_us;
    total.unpack_us += timings.unpack_us;
    total.total_us += timings.total_us;
  }
  const double n = static_cast<double>(iterations);
  std::printf(
      "%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f\n",
      (label + name).c_str(), total.prepare_us / n,
      total.upload_us / n, total.txn_us / n, total.input_us / n,
      total.kernel_us / n, total.unpack_us / n, total.total_us / n);
  return true;
}

bool run_manifest(spiral::Program& program) {
  std::ifstream index("golden/index.txt");
  if (!index) {
    throw std::runtime_error("cannot open golden/index.txt");
  }
  bool ok = true;
  std::string name;
  std::string mode;
  while (index >> name >> mode) {
    const bool passed =
        mode == "head" ? run_head(program, name) : run_golden(program, name);
    ok = passed && ok;
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    spiral::Program program;
    if (argc > 1 && std::string(argv[1]) == "--head-timing") {
      std::printf("%-22s %7s %7s %7s %7s %7s %7s %7s\n", "case", "prepare",
                  "upload", "txn", "input", "kernel", "unpack", "total");
      return run_head_timing(program, "q6_k_head", 10) ? 0 : 1;
    }
    if (argc > 2 && std::string(argv[1]) == "--head-soak") {
      // NPU-only soak: repeated head passes, one submission thread.
      return run_head_timing(program, "q6_k_head", std::stoi(argv[2])) ? 0 : 1;
    }
    if (argc > 1 && std::string(argv[1]) == "--cold-timing") {
      std::printf("%-22s %7s %7s %7s %7s %7s %7s %7s\n", "case", "prepare",
                  "upload", "txn", "input", "kernel", "unpack", "total");
      bool ok = true;
      ok = run_cold_timing(program, "q4_0_12288x1536", 50) && ok;
      ok = run_cold_timing(program, "q4_0_2048x1536", 20) && ok;
      ok = run_cold_timing(program, "q4_0_1536x12288", 20) && ok;
      ok = run_cold_timing(program, "q4_0_1536x6144", 20) && ok;
      return ok ? 0 : 1;
    }
    if (argc > 1 && std::string(argv[1]) == "--pipeline-timing") {
      std::printf("%-22s %7s %7s %7s %7s %7s %7s %7s\n", "case", "prepare",
                  "upload", "txn", "input", "kernel", "unpack", "total");
      bool ok = true;
      ok = run_pipeline_timing(program, "q6_k_head", 5) && ok;
      return ok ? 0 : 1;
    }
    if (argc > 1 && std::string(argv[1]) == "--all-timing") {
      std::printf("%-22s %7s %7s %7s %7s %7s %7s %7s\n", "case", "prepare",
                  "upload", "txn", "input", "kernel", "unpack", "total");
      return run_manifest_timing(program, 20) ? 0 : 1;
    }
    if (argc > 1 && std::string(argv[1]) == "--combined-timing") {
      std::printf("%-22s %7s %7s %7s %7s %7s %7s %7s\n", "case", "prepare",
                  "upload", "txn", "input", "kernel", "unpack", "total");
      bool ok = true;
      ok = run_combined_timing(
               program, "gate_up_6144",
               {"q4_0_1536x6144", "q4_0_1536x6144"}, 20, true) &&
           ok;
      ok = run_combined_timing(
               program, "gate_up_12288",
               {"q4_0_1536x12288", "q4_0_1536x12288"}, 20, true) &&
           ok;
      ok = run_combined_timing(
               program, "qkv_sliding",
               {"q4_0_1536x2048", "q4_0_1536x512", "q4_0_1536x512"}, 20,
               true) &&
           ok;
      ok = run_combined_timing(
               program, "qkv_full",
               {"q4_0_1536x4096", "q4_0_1536x512", "q4_0_1536x512"}, 20,
               true) &&
           ok;
      return ok ? 0 : 1;
    }
    if (argc > 1 && std::string(argv[1]) == "--timing") {
      std::printf("%-22s %7s %7s %7s %7s %7s %7s %7s\n", "case", "prepare",
                  "upload", "txn", "input", "kernel", "unpack", "total");
      if (!run_timing(program, "q4_0_1536x512", 50)) {
        return 1;
      }
      if (!run_probe_timing(program, "q4_0_1536x512", 50, 3, "probe ")) {
        return 1;
      }
      if (!run_probe_timing(program, "q4_0_1536x512", 50, 4, "cycle ")) {
        return 1;
      }
      if (!run_timing(program, "q4_0_1536x12288", 20)) {
        return 1;
      }
      if (!run_probe_timing(program, "q4_0_1536x12288", 20, 3, "probe ")) {
        return 1;
      }
      if (!run_probe_timing(program, "q4_0_1536x12288", 20, 4, "cycle ")) {
        return 1;
      }
      if (!run_timing(program, "q6_k_1536x6144", 20)) {
        return 1;
      }
      if (!run_timing(program, "q4_0_2048x2048", 50)) {
        return 1;
      }
      if (!run_probe_timing(program, "q4_0_2048x2048", 50, 3, "probe ")) {
        return 1;
      }
      if (!run_probe_timing(program, "q4_0_2048x2048", 50, 4, "cycle ")) {
        return 1;
      }
      if (!run_timing(program, "q6_k_2048x2048", 50)) {
        return 1;
      }
      if (!run_probe_timing(program, "q6_k_2048x2048", 50, 3, "probe ")) {
        return 1;
      }
      if (!run_probe_timing(program, "q6_k_2048x2048", 50, 4, "cycle ")) {
        return 1;
      }
      if (!run_probe_timing(program, "q6_k_1536x6144", 20, 3, "probe ")) {
        return 1;
      }
      if (!run_probe_timing(program, "q6_k_1536x6144", 20, 4, "cycle ")) {
        return 1;
      }
      return 0;
    }
    if (!run_manifest(program)) {
      return 1;
    }
    if (!run_prepared(program, "q4_0_1536x512")) {
      return 1;
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  std::printf("done\n");
  return 0;
}
